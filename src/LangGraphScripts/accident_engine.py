import os
import re
from typing import List, Dict, Literal, Optional, Any, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# --- 상태 정의 ---
class State(TypedDict, total=False):
    facts: Dict[str, Any]
    flags_red: List[str]
    flags_yellow: List[str]
    risk_score: int
    risk_bucket: Literal["GREEN", "YELLOW", "RED"]
    explanation_md: str
    followup_questions: List[str]
    final_answer: str

class AccidentDecisionEngine:
    def __init__(self):
        self.llm = self._build_llm()
        self.graph = self._compile_graph()
        self.forbidden_patterns = [
            r"\b하세요\b", r"\b하셔야\b", r"\b권장\b", r"\b반드시\b", 
            r"\b무조건\b", r"\b추천\b", r"\b필수\b", r"보험\s*처리\s*하세요"
        ]

    def _build_llm(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        return None

    def _compile_graph(self):
        workflow = StateGraph(State)

        # 노드 등록
        workflow.add_node("normalize", lambda s: s)
        workflow.add_node("rules", self._rule_score_node)       # 점수 및 플래그 생성
        workflow.add_node("bucket", self._risk_bucket_node)     # 등급 결정
        workflow.add_node("explain", self._llm_explain_node)
        workflow.add_node("questions", self._llm_questions_node)
        workflow.add_node("compose", self._compose_node)

        # 엣지 연결
        workflow.set_entry_point("normalize")
        workflow.add_edge("normalize", "rules")
        workflow.add_edge("rules", "bucket")
        workflow.add_edge("bucket", "explain")

        workflow.add_conditional_edges(
            "explain",
            self._need_questions_condition,
            {"questions": "questions", "compose": "compose"}
        )

        workflow.add_edge("questions", "compose")
        workflow.add_edge("compose", END)

        return workflow.compile()

    # --- 내부 노드 로직 (Private Methods) ---
    # --- 상세 규칙 및 점수 노드 ---
    def _rule_score_node(self, state: State) -> State:
        f = state["facts"]
        red: List[str] = []
        yellow: List[str] = []

        print("------- 분석 시작 (현재 상태) -------")

        # --- RED rules: 하나라도 해당하면 위험(RED) 신호 ---
        if f.get("injury") in ["애매", "있음"]:
            red.append("인명피해 가능성")
        if f.get("pain_now") in ["지속", "악화"]:
            red.append("통증 지속/악화")
        if f.get("hospital_visit") in ["예정", "완료"]:
            red.append("병원 방문/예정")

        if f.get("opponent_mentions_hospital") == "예":
            red.append("상대가 병원/통증 가능성 언급")
        if f.get("opponent_mentions_insurance") == "예":
            red.append("상대가 보험 처리 언급/요구")
        if f.get("evidence") == "없음":
            red.append("증거 부족(사진/블박 없음)")
        if f.get("vehicle_damage") in ["찌그러짐", "파손", "불명"]:
            red.append("손상 범위 불명확 또는 중대 가능")

        # --- YELLOW rules: 주의가 필요한 신호 ---
        if f.get("adas_sensor") in ["있음", "불명"]:
            yellow.append("센서/ADAS 영향 가능(있음/불명)")
        if f.get("vehicle_type") in ["수입", "전기차"]:
            yellow.append("수리비 변동성 큰 차종(수입/전기차)")
        if f.get("opponent_attitude") in ["애매", "공격적"]:
            yellow.append("상대 태도(애매/공격적)로 분쟁 리스크")
        if f.get("speed") == "불명":
            yellow.append("충돌 강도 불명")
        if f.get("evidence") == "일부":
            yellow.append("증거 일부만 확보")

        state["flags_red"] = red
        state["flags_yellow"] = yellow

        # 수식 교정: len(yellow) * 10
        state["risk_score"] = len(red) * 100 + (len(yellow) * 10)

        print(f"빨강 플래그: {red}")
        print(f"노랑 플래그: {yellow}")
        print(f"계산된 리스크 점수: {state['risk_score']}")

        return state
    
    def _risk_bucket_node(self, state: State) -> State:
        red_count  = len(state.get("flags_red", []))
        yellow_count = len(state.get("flags_yellow", []))

        # 판정 로직
        if len(state.get("flags_red", [])) >= 1:
            state["risk_bucket"] = "RED"
        elif len(state.get("flags_yellow", [])) >= 2:
            state["risk_bucket"] = "YELLOW"
        else:
            state["risk_bucket"] = "GREEN"
        
        return state
    
    def _llm_explain_node(self, state: State) -> State:
        if not self.llm:
            state["explanation_md"] = f"판단 등급: {state['risk_bucket']}"
            return state
        
        system = SystemMessage(content="지시형 결론 없이 상황의 리스크 요소만 설명하라.")
        human = HumanMessage(content=f"상황: {state['facts']}\n등급: {state['risk_bucket']}")
        resp = self.llm.invoke([system, human])

        # 지시형 문구 제거 안정장치
        text = resp.content
        for pat in self.forbidden_patterns:
            text = re.sub(pat, "", text)
        state["explanation_md"] = text.strip()
        return state
    
    def _need_questions_condition(self, state: State):
        unknowns = [v for v in state["facts"].values() if v == "불명"]
        return "questions" if state["risk_bucket"] != "GREEN" and unknowns else "compose"
    
    def _llm_questions_node(self, state: State) -> State:
        unknown_fields = [k for k, v in state["facts"].items() if v == "불명"]
        state["followup_questions"] = [f"'{field}' 항목이 '불명'입니다. 정확한 상황을 확인해 보시겠습니까?" for field in unknown_fields[:2]]
        return state
    
    def _compose_node(self, state: State) -> State:
        bucket_emoji = {"RED": "🔴 RED", "YELLOW": "🟡 YELLOW", "GREEN": "🟢 GREEN"}
        res = f"### 🚦 리스크 상태: {bucket_emoji.get(state['risk_bucket'], state['risk_bucket'])}\n\n"
        res += f"**[분석 요약]**\n{state['explanation_md']}\n\n"
        
        if state.get("flags_red"):
            res += "**🚨 감지된 위험:** " + ", ".join(state["flags_red"]) + "\n"
        
        if state.get("followup_questions"):
            res += "\n---\n**❓ 추가 확인 권장 사항:**\n- " + "\n- ".join(state["followup_questions"])
        
        state["final_answer"] = res
        return state
    
    # --- 외부 인터페이스 (Public Method) ---
    def run_analysis(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        initial_state = {"facts": facts}
        return self.graph.invoke(initial_state)