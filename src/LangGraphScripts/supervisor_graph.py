import operator
from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from STT.google_stt_handler import GoogleSTTHandler
from RAG.AccidentRAGEngine import AccidentRAGEngine
from LangGraphScripts.accident_engine import AccidentDecisionEngine

# -------------------------------------------------------------------------
# 1. 상태(State) 정의
# -------------------------------------------------------------------------
class AccidentCaseState(TypedDict):
    # [필수] 대화 기록 (Chat History) - 메시지 누적
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # [제어] 라우딩: 다음 실행할 에이전트 이름
    next: str

    # [입력] 사용자 데이터 및 모드
    input_mode: str                     # "UI" (상세선택) or "VOICE" (음성 녹음)
    audio_path: Optional[str]           # 음성 파일 경로 (녹음 시)
    image_data: Optional[List[str]]     # Base64 인코딩된 이미지 리스트

    # UI 모드일 때 전달되는 상세 정보 (road_type, traffic_light, faults 등)
    # Voice 모드일 때 시뮬레이션 된 텍스트가 들어올 수도 있음
    user_facts: Dict[str, Any]

    # [처리 결과] 에이전트들이 생성한 데이터
    stt_transcript: Optional[str]       # STT 변환 텍스트
    visual_analysis: Optional[str]      # 이미지 분석 결과 (Vision)

    legal_context: Optional[str]        # RAG 검색 결과 요약 (LLM 참조용)
    source_docs: Optional[List[Dict]]   # 화면 표시용 원본 판례 리스트 ({content, source, similarity})

    # [최종 결과]
    final_report: Optional[str]         # 최종 분석 리포트
    risk_bucket: Optional[str]          # 위험 등급 (RED/YELLOW/GREEN)

# -------------------------------------------------------------------------
# 2. Supervisor Graph 클래스 (관제탑)
# -------------------------------------------------------------------------
class AccidentSupervisorGraph:
    pass