import os
import sys
import operator
from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# -----------------------------------------------------------
# 경로 설정 (프로젝트 루트를 시스템 경로에 추가)
# -----------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)  # 현재 폴더 포함
sys.path.append(parent_dir)   # 상위 폴더 포함 (필요 시)

# 모듈 임포트
try:
    from STT.google_stt_handler import GoogleSTTHandler
    from RAG.AccidentRAGEngine import AccidentRAGEngine
    from LangGraphScripts.accident_engine import AccidentDecisionEngine
except ImportError as e:
    print(f"⚠️ 모듈 임포트 오류: {e}")
    print("폴더 구조와 파일명이 정확한지 확인해주세요.")

# -------------------------------------------------------------------------
# 1. 상태(State) 정의
# -------------------------------------------------------------------------
class AccidentCaseState(TypedDict):
    # [필수] 대화 기록 (Chat History) - 메시지 누적
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # [제어] 라우팅: 다음 실행할 에이전트 이름
    next: str

    # [입력] 사용자 데이터 및 모드
    input_mode: str                         # "UI" (상세선택) or "VOICE" (음성 녹음)
    audio_path: Optional[str]               # 음성 파일 경로 (녹음 시)
    image_data: Optional[List[str]]         # Base64 인코딩된 이미지 리스트

    # UI 모드일 때 전달되는 상세 정보 (road_type, traffic_light, faults 등)
    user_facts: Dict[str, Any]

    # [처리 결과] 에이전트들이 생성한 데이터
    stt_transcript: Optional[str]           # STT 변환 텍스트
    visual_analysis: Optional[str]          # 이미지 분석 결과 (Vision)

    legal_context: Optional[str]            # RAG 검색 결과 요약 (LLM 참조용)
    source_docs: Optional[List[Dict]]       # 화면 표시용 원본 판례 리스트

    # [최종 결과]
    final_report: Optional[str]             # 최종 분석 리포트
    risk_bucket: Optional[str]              # 위험 등급 (RED/YELLOW/GREEN)

# -------------------------------------------------------------------------
# 2. Supervisor Graph 클래스 (관제탑)
# -------------------------------------------------------------------------
class AccidentSupervisorGraph:
    def __init__(self):
        # 1. 외부 엔진 초기화
        self.stt_handler = GoogleSTTHandler()
        self.rag_engine = AccidentRAGEngine()
        self.decision_engine = AccidentDecisionEngine()

        # 2. Supervisor LLM 설정 (비용 효율적인 gpt-4o-mini 사용)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 3. 그래프 빌드 및 컴파일
        self.app = self._build_graph()

    def _build_graph(self):
        # 워커 에이전트 목록
        members = ["STT_Agent", "Vision_Agent", "Legal_Agent", "Solver_Agent"]

        # --- Supervisor Prompt (라우팅 로직) ---
        system_prompt = (
            "당신은 교통사고 대응 솔루션의 관리자(Supervisor) 입니다. "
            "현재 상태(State)를 분석하여 다음에 실행해야 할 작업자(Worker)를 지정하세요.\n\n"
            "**작업 순서 및 라우팅 규칙:**\n"
            "1. [음성 처리]: 'input_mode'가 'VOICE'이고, 'stt_transcript'가 아직 없다면 -> 'STT_Agent'\n"
            "2. [이미지 분석]: 'image_data'가 있고, 'visual_analysis'가 비어있다면 -> 'Vision_Agent'\n"
            "3. [법률 검색]: 사고 정황(STT 또는 UI입력 + 이미지분석)이 확보되었고, 'legal_context'가 없다면 -> 'Legal_Agent'\n"
            "4. [최종 판단]: 법률 정보('legal_context')가 확보되었다면 -> 'Solver_Agent'\n"
            "5. [종료]: 최종 리포트('final_report')가 생성되었다면 -> 'FINISH'"
        )

        function_def = {
            "name": "route",
            "description": "다음 작업을 수행할 에이전트를 선택합니다.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next Agent",
                        "anyOf": [{"enum": members + ["FINISH"]}],
                    }
                },
                "required": ["next"],
            },
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "현재 상태 점검:\n"
                "- 모드: {input_mode}\n"
                "- STT완료여부: {stt_transcript}\n"
                "- 시각분석여부: {visual_analysis}\n"
                "- 법률검색여부: {legal_context}\n"
                "- 최종리포트여부: {final_report}\n\n"
                "다음 작업자는 누구입니까?"
            )
        ])

        supervisor_chain = (
            prompt
            | self.llm.bind_functions(functions=[function_def], function_call="route")
            | JsonOutputFunctionsParser()
        )

        # --- Graph 구성 ---
        workflow = StateGraph(AccidentCaseState)

        # 노드 등록
        workflow.add_node("Supervisor", lambda state: {"next": supervisor_chain.invoke(state)["next"]})
        workflow.add_node("STT_Agent", self.stt_node)
        workflow.add_node("Vision_Agent", self.vision_node)
        workflow.add_node("Legal_Agent", self.legal_node)
        workflow.add_node("Solver_Agent", self.solver_node)

        # 엔트리 포인트 및 엣지 연결
        workflow.set_entry_point("Supervisor")

        for member in members:
            workflow.add_edge(member, "Supervisor")
        
        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END

        workflow.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)

        return workflow.compile()

    # -------------------------------------------------------------------------
    # 3. Worker Nodes 구현 (실제 기능 수행)
    # -------------------------------------------------------------------------

    def stt_node(self, state: AccidentCaseState):
        """[STT Agent] 음성 파일을 텍스트로 변환하거나 입력된 텍스트 확인"""
        # 1. 이미 텍스트로 입력된 경우 (Voice Simulation)
        if state["user_facts"].get("voice_transcript"):
            return {
                "stt_transcript": state["user_facts"]["voice_transcript"],
                "messages": [HumanMessage(content="[STT] 텍스트 입력 확인됨")]
            }
        
        # 2. 실제 오디오 파일 처리
        print("🎙️ [STT_Agent] 음성 파일 변환 시작...")
        if not state.get("audio_path"):
            return {"stt_transcript": "(음성 파일 없음)"}
        
        transcript = self.stt_handler.transcribe_audio(state["audio_path"])
        return {
            "stt_transcript": transcript,
            "messages": [HumanMessage(content=f"[STT 결과] {transcript}")]
        }

    async def vision_node(self, state: AccidentCaseState):
        """[Vision Agent] 사고 사진 분석"""
        print("👁️ [Vision_Agent] 이미지 분석 시작...")
        if not state.get("image_data"):
            return {"visual_analysis": "(이미지 없음)"}
        
        analysis_text = await self.rag_engine.analyze_image_async(state["image_data"])

        return {
            "visual_analysis": analysis_text,
            "messages": [HumanMessage(content=f"[이미지 분석] {analysis_text}")]
        }

    async def legal_node(self, state: AccidentCaseState):
        """[Legal Agent] UI/Voice 입력에 따른 맞춤형 쿼리 생성 및 RAG 검색"""
        print("⚖️ [Legal_Agent] 판례 검색 시작...")

        facts = state.get("user_facts", {})
        stt = state.get("stt_transcript", "")
        vision = state.get("visual_analysis", "")

        # --- 쿼리 생성 로직 (UI vs Voice 분기) --- 
        if state.get("input_mode") == "UI":
            my_faults_str = ", ".join(facts.get("my_faults", []))
            opp_faults_str = ", ".join(facts.get("opp_faults", []))

            query = (
                f"교통사고 판례 검색. "
                f"상황: {facts.get('road_type', '미상')} 도로, 신호 {facts.get('traffic_light', '미상')}. "
                f"당사자 행동: 본인({facts.get('my_action', '')}, 속도 {facts.get('my_speed', '')}), "
                f"상대방({facts.get('opp_action', '')}, 진입 {facts.get('opp_entry', '')}). "
                f"충돌: 본인 차량 {facts.get('collision_part', '')} 파손. "
                f"과실 의심: 본인[{my_faults_str}], 상대[{opp_faults_str}]. "
                f"시각적 정황: {vision}"
            )
        else:
            # Voice 모드
            query = (
                f"교통사고 과실 판례 검색. "
                f"운전자 진술: {stt}. "
                f"시각적 정황: {vision}."
            )

        # --- RAG 엔진 호출 --- 
        rag_summary, raw_docs = await self.rag_engine.ask_with_context(
            query=query,
            image_description=vision,
            image_base64_list=state.get("image_data")
        )

        # 화면 표시용 원본 문서 포맷팅
        formatted_docs = []
        if raw_docs:
            for doc, score in raw_docs:
                formatted_docs.append({
                    "content": getattr(doc, "page_content", str(doc)),
                    "source": getattr(doc, "metadata", {}).get("source", "법령/판례"),
                    "similarity": float(score) if score is not None else 0.0
                })

        return {
            "legal_context": rag_summary,
            "source_docs": formatted_docs,
            "messages": [HumanMessage(content=f"[법률 검색 완료] {len(formatted_docs)}건의 판례 확보")]
        }

    def solver_node(self, state: AccidentCaseState):
        """[Solver Agent] 최종 종합 판단 및 리포트 작성"""
        print("🧠 [Solver_Agent] 최종 리포트 작성 중...")

        combined_facts = state["user_facts"].copy()
        combined_facts["rag_context"] = state.get("legal_context", "")
        combined_facts["stt_context"] = state.get("stt_transcript", "")
        combined_facts["vision_context"] = state.get("visual_analysis", "")

        result = self.decision_engine.run_analysis(combined_facts)

        return {
            "final_report": result.get("final_answer"),
            "risk_bucket": result.get("risk_bucket"),
            "messages": [HumanMessage(content="최종 분석 완료")]
        }

    async def run(self, inputs: Dict[str, Any]):
        return await self.app.ainvoke(inputs)