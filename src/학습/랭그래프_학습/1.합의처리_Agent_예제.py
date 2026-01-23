"""
보험처리 vs 개인합의 "판단 보조" LangGraph 파이프라인 (실행 가능한 단일 파일)

- Rule Engine이 리스크 버킷(🟢🟡🔴)을 "결정" (LLM이 못 바꿈)
- LLM은 "설명/체크리스트/질문"만 생성 (결론 지시 금지)
- LLM 없이도 동작하도록 fallback 포함

실행:
  pip install -U langgraph langchain-core langchain-openai langchain-ollama pydantic
  (둘 중 하나 택)
    - OpenAI:  set OPENAI_API_KEY=...
    - Ollama:  ollama pull llama3.1  (또는 사용 모델)
              set LLM_PROVIDER=ollama
  python accident_decision_graph.py
"""

from __future__ import annotations

import os
import re
from typing import TypedDict, List, Dict, Literal, Optional, Any

from langgraph.graph import StateGraph, END

import os
from dotenv import load_dotenv

#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

# --- LLM (선택) ---
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

# OpenAI / Ollama 둘 다 지원 (없으면 fallback)
def build_llm() -> Optional[BaseChatModel]:
    #provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
    if load_dotenv():
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        return llm
    # if provider == "openai":
    #     try:
    #         # 예: gpt-4.1-mini / gpt-4o-mini 등
    #         model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    #         temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    #         return ChatOpenAI(model=model, temperature=temperature)
    #     except Exception as e:
    #         print(f"[WARN] OpenAI LLM 초기화 실패: {e}")
    #
    # if provider == "ollama":
    #     try:
    #         from langchain_ollama import ChatOllama
    #
    #         model = os.getenv("OLLAMA_MODEL", "llama3.1")
    #         temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    #         return ChatOllama(model=model, temperature=temperature)
    #     except Exception as e:
    #         print(f"[WARN] Ollama LLM 초기화 실패: {e}")

    print("[INFO] LLM 미사용 모드로 실행합니다. (설명은 규칙 기반 텍스트로 출력)")
    return None


# ----------------------------
# 1) 스키마 / 상태 정의
# ----------------------------

RiskBucket = Literal["GREEN", "YELLOW", "RED"]

class Facts(TypedDict, total=False):
    accident_type: str  # 정차후출발|주차중|차선변경|후방추돌|기타|불명
    speed: str          # 저속|중속|고속|불명
    injury: str         # 없음|애매|있음|불명
    pain_now: str       # 없음|경미|지속|악화|불명
    hospital_visit: str # 없음|예정|완료|불명
    vehicle_damage: str # 없음|스크래치|찌그러짐|파손|불명
    adas_sensor: str    # 없음|있음|불명
    vehicle_type: str   # 국산|수입|전기차|불명
    evidence: str       # 충분|일부|없음|불명
    opponent_attitude: str            # 원만|애매|공격적|불명
    opponent_mentions_hospital: str   # 아니오|예|불명
    opponent_mentions_insurance: str  # 아니오|예|불명
    notes: str          # 자유메모

class State(TypedDict, total=False):
    raw_text: str
    facts: Facts

    flags_red: List[str]
    flags_yellow: List[str]
    risk_score: int
    risk_bucket: RiskBucket

    explanation_md: str
    followup_questions: List[str]
    final_answer: str


DEFAULTS: Facts = {
    "accident_type": "불명",
    "speed": "불명",
    "injury": "불명",
    "pain_now": "불명",
    "hospital_visit": "불명",
    "vehicle_damage": "불명",
    "adas_sensor": "불명",
    "vehicle_type": "불명",
    "evidence": "불명",
    "opponent_attitude": "불명",
    "opponent_mentions_hospital": "불명",
    "opponent_mentions_insurance": "불명",
    "notes": "",
}


# ----------------------------
# 2) 유틸
# ----------------------------

FORBIDDEN_IMPERATIVES = [
    r"\b하세요\b", r"\b하셔야\b", r"\b권장\b", r"\b반드시\b", r"\b무조건\b",
    r"\b추천\b", r"\b필수\b", r"\b결론\b", r"보험\s*처리\s*하세요", r"개인\s*합의\s*하세요"
]

def contains_imperative(text: str) -> bool:
    for pat in FORBIDDEN_IMPERATIVES:
        if re.search(pat, text):
            return True
    return False

def safe_strip_imperatives(text: str) -> str:
    # 완벽한 필터는 아니지만, 서비스용 안전장치로 유용
    for pat in FORBIDDEN_IMPERATIVES:
        text = re.sub(pat, "", text)
    return text


# ----------------------------
# 3) 노드 구현
# ----------------------------

def normalize_extract(state: State) -> State:
    facts = state.get("facts", {})
    for k, v in DEFAULTS.items():
        facts.setdefault(k, v)
    state["facts"] = facts
    return state

def rule_score(state: State) -> State:
    f = state["facts"]
    red: List[str] = []
    yellow: List[str] = []
    print('-------상태-------')
    print(f)
    # --- RED rules: 하나라도면 RED로 가는 강한 신호들 ---
    if f["injury"] in ["애매", "있음"]:
        red.append("인명피해/통증 가능성(‘애매/있음’)")
    if f["pain_now"] in ["지속", "악화"]:
        red.append("통증 지속/악화")
    if f["hospital_visit"] in ["예정", "완료"]:
        red.append("병원 방문/예정")
    if f["opponent_mentions_hospital"] == "예":
        red.append("상대가 병원/통증 가능성 언급")
    if f["opponent_mentions_insurance"] == "예":
        red.append("상대가 보험 처리 언급/요구")
    if f["evidence"] == "없음":
        red.append("증거 부족(사진/블박 없음)")
    if f["vehicle_damage"] in ["찌그러짐", "파손", "불명"]:
        red.append("손상 범위 불명확 또는 중대 가능")

    # --- YELLOW rules: 개인합의 시 주의가 필요한 신호들 ---
    if f["adas_sensor"] in ["있음", "불명"]:
        yellow.append("센서/ADAS 영향 가능(있음/불명)")
    if f["vehicle_type"] in ["수입", "전기차"]:
        yellow.append("수리비 변동성 큰 차종(수입/전기차)")
    if f["opponent_attitude"] in ["애매", "공격적"]:
        yellow.append("상대 태도(애매/공격적)로 분쟁 리스크")
    if f["speed"] == "불명":
        yellow.append("충돌 강도 불명")
    if f["evidence"] == "일부":
        yellow.append("증거 일부만 확보")

    state["flags_red"] = red
    state["flags_yellow"] = yellow
    state["risk_score"] = len(red) * 100 + len(yellow) * 10
    print( state["flags_red"] )
    return state

def risk_bucket(state: State) -> State:
    red = state.get("flags_red", [])
    yellow = state.get("flags_yellow", [])

    if len(red) >= 1:
        state["risk_bucket"] = "RED"
    elif len(yellow) >= 2:
        state["risk_bucket"] = "YELLOW"
    else:
        state["risk_bucket"] = "GREEN"
    return state

def llm_explain(state: State) -> State:
    """
    LLM은 '결정'을 하지 않고, bucket/flags를 근거로 설명만 생성.
    LLM이 없으면 규칙 기반 텍스트로 fallback.
    """
    f = state["facts"]
    bucket = state["risk_bucket"]
    red = state.get("flags_red", [])
    yellow = state.get("flags_yellow", [])

    llm = build_llm()

    # LLM 없는 경우 fallback
    if llm is None:
        md = []
        md.append(f"🚦 판단 상태: **{bucket}** (결정이 아닌 리스크 신호등)")
        md.append("")
        md.append("✅ 확인된 긍정 신호")
        positives = []
        if f["injury"] == "없음": positives.append("인명피해/통증 신호 없음")
        if f["pain_now"] == "없음": positives.append("현재 통증 없음")
        if f["hospital_visit"] == "없음": positives.append("병원 방문 계획/이력 없음")
        if f["evidence"] == "충분": positives.append("증거 충분(사진/블박 등)")
        if f["vehicle_damage"] in ["없음", "스크래치"]: positives.append("손상 범위가 외관 수준일 가능성")
        md.append("- " + (", ".join(positives) if positives else "정보가 부족하여 추가 확인 필요"))

        md.append("")
        md.append("⚠️ 감지된 위험 신호")
        if red: md.append("- 🔴 " + "; ".join(red))
        if yellow: md.append("- 🟡 " + "; ".join(yellow))
        if not red and not yellow: md.append("- 특이 위험 신호 없음")

        md.append("")
        md.append("📌 참고")
        md.append("- 본 결과는 일반적인 판단 보조 정보이며, 최종 선택과 책임은 사용자에게 있습니다.")
        state["explanation_md"] = "\n".join(md)
        return state

    system = SystemMessage(
        content=(
            "너는 교통사고 처리의 '결정'을 내리지 않는 보조자다.\n"
            "- 절대 '보험 처리하세요/개인 합의하세요' 같은 지시형 결론을 말하지 마라.\n"
            "- 반드시 bucket(RED/YELLOW/GREEN)과 flags를 그대로 근거로 삼아 설명만 해라.\n"
            "- 출력 포맷(마크다운):\n"
            "  1) 🚦 판단 상태(신호등) 2) ✅ 긍정 신호 3) ⚠️ 위험 신호\n"
            "  4) 🧾 개인합의 시 필수 기록 5) 🔁 보험 전환 트리거\n"
            "  6) 📌 최종 선택은 사용자 책임\n"
        )
    )

    human = HumanMessage(
        content=(
            f"[facts]\n{f}\n\n"
            f"[bucket]\n{bucket}\n\n"
            f"[red_flags]\n{red}\n\n"
            f"[yellow_flags]\n{yellow}\n\n"
            "위 정보를 바탕으로, 지시형 결론 없이 설명만 작성해라."
        )
    )

    resp = llm.invoke([system, human])
    text = resp.content if hasattr(resp, "content") else str(resp)
    print("설명 : ")
    print(text)
    # 안전장치: 혹시라도 지시형 문구가 섞이면 약하게 제거
    if contains_imperative(text):
        text = safe_strip_imperatives(text)

    state["explanation_md"] = text.strip()
    return state

def need_questions(state: State) -> bool:
    f = state["facts"]
    unknowns = [k for k, v in f.items() if v == "불명"]
    # GREEN이면 보통 질문 없이도 충분. YELLOW/RED면 불명값 보완 가치가 큼.
    return (state["risk_bucket"] != "GREEN") and (len(unknowns) > 0)

def llm_questions(state: State) -> State:
    """
    불명 필드를 좁히는 질문 최대 3개 생성.
    LLM 없으면 규칙 기반 질문으로 fallback.
    """
    f = state["facts"]
    unknowns = [k for k, v in f.items() if v == "불명"]
    llm = build_llm()

    # 우선순위(분쟁/리스크 큰 항목부터)
    priority = [
        "injury", "pain_now", "hospital_visit",
        "opponent_mentions_hospital", "opponent_mentions_insurance",
        "vehicle_damage", "evidence", "adas_sensor", "vehicle_type", "opponent_attitude", "speed"
    ]
    targets = [k for k in priority if k in unknowns][:3]

    if not targets:
        state["followup_questions"] = []
        return state

    # LLM 없는 경우 fallback
    if llm is None:
        qmap = {
            "injury": "몸 상태는 어떤가요? (없음/애매/있음)",
            "pain_now": "현재 통증은 어떤가요? (없음/경미/지속/악화)",
            "hospital_visit": "병원 방문 계획이나 진료가 있었나요? (없음/예정/완료)",
            "opponent_mentions_hospital": "상대가 병원/통증 가능성을 언급했나요? (아니오/예)",
            "opponent_mentions_insurance": "상대가 보험처리를 언급하거나 요구했나요? (아니오/예)",
            "vehicle_damage": "차량 손상은 어느 정도인가요? (없음/스크래치/찌그러짐/파손)",
            "evidence": "사진/블랙박스 등 증거 확보 상태는? (충분/일부/없음)",
            "adas_sensor": "접촉 부위 주변에 주차센서/레이더/카메라 등이 있나요? (없음/있음)",
            "vehicle_type": "차종은? (국산/수입/전기차)",
            "opponent_attitude": "상대 태도는? (원만/애매/공격적)",
            "speed": "충돌 속도/강도는? (저속/중속/고속)",
        }
        state["followup_questions"] = [qmap[t] for t in targets if t in qmap]
        return state

    system = SystemMessage(
        content=(
            "너는 결론을 내리지 않는다. 질문만 만든다.\n"
            "- 불명(unknown) 값을 좁히는 질문을 최대 3개.\n"
            "- 예/아니오 또는 선택지형으로 짧게.\n"
            "- 금액 질문 금지.\n"
        )
    )
    human = HumanMessage(
        content=f"unknown_fields={targets}\ncurrent_facts={f}\n질문을 최대 3개 생성해라."
    )
    resp = llm.invoke([system, human])
    text = resp.content if hasattr(resp, "content") else str(resp)

    # 줄 단위로 질문 추출
    qs = []
    for line in text.splitlines():
        line = line.strip("- ").strip()
        if not line:
            continue
        qs.append(line)
        if len(qs) >= 3:
            break

    state["followup_questions"] = qs
    return state

def compose(state: State) -> State:
    f = state["facts"]
    red = state.get("flags_red", [])
    yellow = state.get("flags_yellow", [])
    qs = state.get("followup_questions", [])
    bucket = state["risk_bucket"]

    lines: List[str] = []
    lines.append(state.get("explanation_md", f"🚦 판단 상태: **{bucket}**"))
    lines.append("")
    lines.append("—")
    lines.append("")
    lines.append("🧾 개인합의로 진행할 때 *공통 필수 기록*(권장)")
    lines.append("- 양 차량 번호판 포함 사진 + 접촉 부위 근접 사진 + 사고 위치 사진")
    lines.append("- 블랙박스 원본 보관(가능하면 별도 저장)")
    lines.append("- 문자/카톡으로 ‘인명피해 없음’ 및 ‘추가 청구 없음’ 상호 확인")
    lines.append("")
    lines.append("🔁 보험 전환 트리거(하나라도 발생하면 개인합의 리스크 급상승)")
    lines.append("- 통증/병원 언급 발생(당사자/상대 포함)")
    lines.append("- 수리비가 예상보다 커짐(센서/범퍼 내부/도색 범위 확대 등)")
    lines.append("- 상대 태도 변화(기록 거부, 과실 다툼, 과도한 요구)")
    lines.append("- 증거(사진/블박) 부족 또는 분실")
    lines.append("")
    if qs:
        lines.append("❓ 추가 확인 질문(답하면 판단 정확도↑)")
        lines.extend([f"- {q}" for q in qs])
        lines.append("")
    lines.append("📌 본 내용은 일반적인 판단 보조 정보이며, 최종 선택과 책임은 사용자에게 있습니다.")

    state["final_answer"] = "\n".join(lines)
    return state


# ----------------------------
# 4) 그래프 빌드/실행
# ----------------------------

def build_graph():
    graph = StateGraph(State)

    graph.add_node("normalize", normalize_extract)
    graph.add_node("rules", rule_score)
    graph.add_node("bucket", risk_bucket)
    graph.add_node("explain", llm_explain)
    graph.add_node("questions", llm_questions)
    graph.add_node("compose", compose)

    graph.set_entry_point("normalize")
    graph.add_edge("normalize", "rules")
    graph.add_edge("rules", "bucket")
    graph.add_edge("bucket", "explain")

    graph.add_conditional_edges(
        "explain",
        lambda s: "questions" if need_questions(s) else "compose",
        {"questions": "questions", "compose": "compose"},
    )

    graph.add_edge("questions", "compose")
    graph.add_edge("compose", END)

    return graph.compile()


def run_demo():
    app = build_graph()

    # ✅ 여기만 바꾸면 됩니다: 입력 Facts
    demo_facts: Facts = {
        "accident_type": "정차후출발",
        "speed": "저속",
        "injury": "없음",
        "pain_now": "없음",
        "hospital_visit": "없음",
        "vehicle_damage": "스크래치",
        "adas_sensor": "불명",             # 불명이라 질문 생성될 수 있음
        "vehicle_type": "국산",
        "evidence": "충분",
        "opponent_attitude": "원만",
        "opponent_mentions_hospital": "아니오",
        "opponent_mentions_insurance": "예",
        "notes": "사거리 정차 중 출발 시 아주 경미한 접촉",
    }

    state: State = {"facts": demo_facts}
    out = app.invoke(state)

    print("\n================= FINAL ANSWER =================\n")
    print(out["final_answer"])
    # print("\n================= DEBUG =================\n")
    # print("risk_bucket:", out.get("risk_bucket"))
    # print("flags_red:", out.get("flags_red"))
    # print("flags_yellow:", out.get("flags_yellow"))
    # print("risk_score:", out.get("risk_score"))


if __name__ == "__main__":
    run_demo()