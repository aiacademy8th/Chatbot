import os
import sys
import operator
import logging
from typing import Annotated, List, TypedDict, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from dotenv import load_dotenv

load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# 현재 파일의 부모 디렉토리를 시스템 경로에 추가
# 그래야 옆 동네인 'RAG' 폴더를 찾을 수 있음
# ---------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))    # 현재 폴더 경로
parent_dir = os.path.dirname(current_dir)                   # 상위 프로젝트 루트 경로
sys.path.append(parent_dir)

# 엔진 및 데이터 모델 임포트
from RAG.AccidentRAGEngine import AccidentRAGEngine, AccidentAnalysisResult, FaultRatio

# --- 1. 상태(State3) 정의 ---
class AgentState(TypedDict):
    """
    LangGraph Agent의 상태 정의
    TypedVDict를 사용하여 타입 힌트 제공
    """

    # messages: 히스토리 누적 (add operator 사용)
    messages: Annotated[List[BaseMessage], operator.add]
    image_paths: List[str]

    # 워커들의 결과를 저장할 필드 (덮어쓰기)
    image_summary: str
    rag_context: str

    # 슈퍼바이저가 결정한 실행할 툴 리스트
    required_tools: List[str]

    # 최종 결과 (구조화된 데이터 - dict 형태로 저장)
    final_result: dict

# --- 2. 챗봇 전용 상태 정의
class ChatState(TypedDict):
    """
    챗봇 전용 상태 - 기존 분석 결과를 컨텍스트로 활용
    """
    
    messages: Annotated[List[BaseMessage], operator.add]

    # [컨텍스트] 초기 분석에서 가져온 정보
    accident_context: str       # 초기 사고 상황 설명
    vision_summary: str         # Vision 분석 결과
    initial_rag: str            # 초기 RAG 검색 결과
    fault_analysis: str         # 과실 비율 판단 결과

    # [실시간 검색] 추가 질문에 대한 RAG 검색
    additional_rag: str         # 추가 검색된 판례

    # [출력] 최종 답변
    response: str

# --- 3. 클래스 기반 Agent 정의 ---
class AccidentGraphAgent:
    """
    교통사고 분석 LangGraph Agent (클래스 기반)

    슈퍼바이저 패턴과 병렬처리 패턴을 구현한 Agent 클래스
    - Vision Tool: 이미지 분석
    - Search Tool: 법률 판례 검색
    - Final Solver: 종합 분석 및 구조화된 결과 생성
    """

    # 클래스 변수: 싱글톤 패턴으로 메모리 절약
    _shared_engine: Optional[AccidentRAGEngine] = None

    def __init__(self, engine: Optional[AccidentRAGEngine] = None):
        """
        Agent 초기화

        Args:
            engine: AccidentRAGEngine 인스턴스 (None이면 새로 생성 또는 공유 엔진 사용)
        """

        # 엔진 설정 (의존성 주입 지원)
        if engine:
            # 명시적으로 제공된 엔진 사용 (데스크용 Mock 주입 가능)
            self.engine = engine
        else:
            # 공유 엔진 사용 (싱글톤 패턴)
            if AccidentGraphAgent._shared_engine is None:
                AccidentGraphAgent._shared_engine = AccidentRAGEngine()
            self.engine = AccidentGraphAgent._shared_engine
        
        
        self.pool = None            # Connection Pool 변수 초기화 (나중에 닫기 위해 유지)
        self.checkpointer = None    # 체크포인터 참조 유지
        self.graph_app = None       # 컴파일전에는 None 상태
        self.workflow = StateGraph(AgentState)

        # 그래프 뼈대만 미리 구축
        self._build_structure()

    def _build_structure(self):
        """노드와 엣지만 정의 (DB 연결 없음)"""

        # 노드 등록
        self.workflow.add_node("supervisor", self.supervisor_node)
        self.workflow.add_node("vision_tool", self.vision_node)
        self.workflow.add_node("search_tool", self.search_node)
        self.workflow.add_node("join", self.join_node)
        self.workflow.add_node("final_solver", self.final_solver)

        # 진입점 설정
        self.workflow.set_entry_point("supervisor")

        # 슈퍼바이저 -> 각 툴 (Fan-out: 병렬 실행)
        # 주의: LangGraph 기본 동작은 순차 실행이며, 진정한 병렬 실행을 위해서는 Send API 사용 필요
        self.workflow.add_conditional_edges(
            "supervisor",
            self._route_tools,              # 라우팅 함수
            ["vision_tool", "search_tool"]  # 가능한 워커 목록
        )

        # 각 툴 -> Join 노드 (Fan-in: 동기화)
        self.workflow.add_edge("vision_tool", "join")
        self.workflow.add_edge("search_tool", "join")

        # Join -> Final Solver (단일 실행 보장)
        self.workflow.add_edge("join", "final_solver")

        # 종료
        self.workflow.add_edge("final_solver", END)

    async def setup(self):
        """
        [핵심] 서버 시작 시(Lifespan) 호출되어 실제 리소스를 할당하고 그래프를 컴파일함
        이 메서드는 이미 Event Loop 가 돌아가는 상황에서 호출되므로 에러가 발생하지 않음
        """
        logger.info("AccidentGraphAgent 리소스 설정 시작 (DB 연결)...")

        # 비동기 PostgresSaver 설정
        # 1. DB 연결 정보 구성
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME")
        DB_URL = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"

        # 2. Connection Pool 생성
        # 주의: 프로덕션에서는 FastAPI 수명 주기(Lifespan) 내에서 관리하는 것이 권장되나, 편의상 여기서 초기화
        self.pool = AsyncConnectionPool(
            conninfo=DB_URL,
            max_size=20,
            kwargs={"autocommit": True},
            open=False
        )

        await self.pool.open()
        logger.info("AsyncConnectionPool 연결 성공")

        # 3. PostgresSaver 생성 (이제 Loop 가 있으므로 안전)
        self.checkpointer = AsyncPostgresSaver(self.pool)
        await self.checkpointer.setup() # 테이블 생성
        logger.info("PostgresSaver 테이블 설정 완료")

        # 4. 그래프 컴파일 (이제 graph_app이 완료됨)
        self.graph_app = self.workflow.compile(checkpointer=self.checkpointer)
        logger.info("그래프 컴파일 완료 (Ready to serve)")

    def _route_tools(self, state: AgentState):
        """
        슈퍼바이저가 결정한 툴 리스트 반환 (라우팅 함수)

        Args:
            state: 현재 Agent 상태
        
        Returns:
            실행할 툴 이름 리스트

        """
        return state["required_tools"]
    
    # --- 노드(Node) 메서드 정의 ---
    async def supervisor_node(self, state: AgentState):
        """
        [슈퍼바이저] 입력값을 분석하여 어떤 워커를 가동할지 결정

        Args:
            state: 현재 Agent 상태

        Returns:
            실행할 툴 리스트를 포함한 상태 업데이트

        """
        tools = []

        # 1. 이미지가 있으면 Vision 워커 호출
        if state.get("image_paths") and len(state["image_paths"]) > 0:
            tools.append("vision_tool")

        # 2. 사용자 메시지가 있으면 Search 워커 호출 (기본값)
        if state["messages"]:
            tools.append("search_tool")

        # 방어 로직: 툴이 없으면 검색이라도 실행
        if not tools:
            tools.append("search_tool")

        logger.info(f"[Supervisor] 실행할 툴: {tools}")
        return {"required_tools": tools}

    async def vision_node(self, state: AgentState) -> dict:
        """
        [Worker A] 이미지 분석 -> 텍스트 변환 (휘발성)

        Args:
            state: 현재 Agent 상태

        Returns:
            이미지 요약 텍스트를 포함한 상태 업데이트
        """

        try:
            paths = state.get("image_paths", [])
            if not paths:
                return {"image_summary": "분석할 이미지가 없습니다."}

            logger.info(f"[Vision Node] 이미지 {len(paths)}개 분석 시작")
            # 여기서 생성된 'description'이 Final Solver의 유일한 시각 정보가 됨
            description = await self.engine.analyze_images_from_paths(paths)
            logger.info(f"[Vision Node] 분석 완료: {description[:100]}...")

            return {"image_summary": description}

        except Exception as e:
            logger.error(f"[Vision Node] 오류: {e}")
            return {"image_summary": f"이미지 분석 중 오류 발생: {str(e)}"}
        
    async def search_node(self, state: AgentState) -> dict:
        """
        [Search Worker] 법률 판례 검색

        Args:
            state: 현재 Agent 상태

        Returns:
            검색된 판례 정보를 포함한 상태 업데이트
        """
        try:
            # 가장 최근 사용자 발화 찾기
            last_human_msg = next(
                (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), 
                None
            )

            if not last_human_msg:
                return {"rag_context": "검색할 사용자 질문이 없습니다."}
            
            query = last_human_msg.content
            logger.info(f"[Search Node] 판례 검색 시작: {query[:50]}...")

            docs = await self.engine.search_legal_docs(query)
            logger.info(f"[Search Node] 검색 완료")

            return {"rag_context": docs}

        except Exception as e:
            logger.error(f"[Search Node] 오류: {e}")
            return {"rag_context": f"판례 검색 중 오류 발생: {str(e)}"}
    
    async def join_node(self, state: AgentState) -> dict:
        """
        [Join Node] 병렬 워커들의 동기화 지점.
        모든 워커가 끝날 때까지 기다렸다가 다음 단계로 넘어갑니다.

        Args:
            state: 현재 Agent 상태

        Returns:
            빈 딕셔너리 (상태 변경 없음, 동기화만 수행)
        """

        # [개선] 워커 결과 검증
        img_summary = state.get("image_summary", "")
        rag_context = state.get("rag_context", "")

        issues = []

        # 실패 키워드 확장
        FAILURE_KEYWORDS = ["오류", "실패", "없습니다.", "처리에 실패"]

        if any(keyword in img_summary for keyword in FAILURE_KEYWORDS):
            issues.append("Vision 워커 실패")
        if any(keyword in rag_context for keyword in FAILURE_KEYWORDS):
            issues.append("Search 워커 실패")

        if issues:
            logger.warning(f"[Join Node] 워커 이슈 감지: {', '.join(issues)}")
        else:
            logger.info("[Join Node] 모든 워커 정상 완료")

        return {}

    async def final_solver(self, state: AgentState) -> dict:
        """
        [Final Node] 모든 정보를 종합하여 구조화된 답변 생성
        [비용 절감] 이미지 파일은 넘기지 않고, Vision 요약 텍스트만 전달함

        Args:
            state: 현재 Agent 상태
        
        Returns:
            최종 분석 결과와 AI 조언을 포함한 상태 업데이트
        """

        try:
            # 워커 결과 수집
            img_summary = state.get("image_summary", "시각 정보 없음")
            rag_result = state.get("rag_context", "법률 정보 없음")

            # LLM에게 제공할 컨텍스트 메시지 구성
            # Vision 결과가 텍스트에 녹아들어감
            context_msg = SystemMessage(content=f"""
            [상황 분석 리포트]
            1. 현장 시각 정보 요약 (Vision AI 분석):
            {img_summary}
            2. 관련 판례 및 법규 (RAG 검색)
            {rag_result}               
            """)

            # 현재까지의 대화 + 컨텍스트 주입
            msgs_for_llm = state["messages"] + [context_msg]

            # [핵심 전략] image_paths=[] 전달 -> 이미지 재사용 안 함 (비용 절감)
            result: AccidentAnalysisResult = await self.engine.generate_final_solution_structured(
                    messages=msgs_for_llm,
                    image_paths=[]
            )

            logger.info(f"[Final Solver] 분석 완료 - 과실 비율: {result.fault_ratio.me}:{result.fault_ratio.opponent}")

            # 결과 반환
            return {
                "final_result": result.model_dump(),
                "messages": [AIMessage(content=result.advice)]      # 대화 기록에는 텍스트 조언만 남김
            }

        except Exception as e:
             # 로깅 + 사용자 친화적 메시지
            logger.error(f"[Final Solver] 치명적 오류: {e}")

            # Fallback 응답
            fallback_result = AccidentAnalysisResult(
                summary="시스템 오류로 분석을 완료하지 못했습니다.",
                fault_ratio=FaultRatio(me=50, opponent=50),     # Dict -> 모델 객체
                legal_basis=[],
                advice="일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                reasoning=str(e),
                action_guide="보험 처리 권장"
            )
            return {
                "final_result": fallback_result.model_dump(),
                "messages": [AIMessage(content=fallback_result.advice)]
            }
        
    
    async def run(self, initial_state: dict, config: dict = None) -> dict:
        """
        그래프 실행 헬퍼 메서드

        Args:
            initial_state: 초기 상태 딕셔너리
            config: 실행 설정 (thread_id 등)

        Returns:
            최종 상태 딕셔너리
        """
        
        if config is None:
            config = {}

        return await self.graph_app.ainvoke(initial_state, config=config)
    
# --- 4. 챗봇 전용 Agent 클래스 ---
class AccidentChatAgent:
    """
    교통사고 후속 질문 챗봇 Agent
    - 기존 분석 결과를 컨텍스트로 활용
    - 추가 질문에 대해 RAG 재검색
    - 스트리밍 응답 지원
    """

    _shared_engine: Optional[AccidentRAGEngine] = None

    def __init__(self, engine: Optional[AccidentRAGEngine] = None):
        if engine:
            self.engine = engine
        else:
            if AccidentChatAgent._shared_engine is None:
                AccidentChatAgent._shared_engine = AccidentRAGEngine()
            self.engine = AccidentChatAgent._shared_engine
        
        self.pool = None
        self.checkpointer = None
        self.graph_app = None
        self.workflow = StateGraph(ChatState)
        self._build_chat_structure()
    
    def _build_chat_structure(self):
        """챗봇 워크플로우 구조 정의"""
        # 노드 등록
        self.workflow.add_node("load_context", self.load_context_node)
        self.workflow.add_node("search_additional", self.search_additional_node)
        self.workflow.add_node("generate_response", self.generate_response_node)

        # 워크플로우: Context 로드 -> 추가 검색 -> 응답 생성
        self.workflow.set_entry_point("load_context")
        self.workflow.add_edge("load_context", "search_additional")

        # 검색 후 'generate_response' 로 이동하도록 변경
        self.workflow.add_edge("search_additional", "generate_response")

        # 응답 생성 후 종료
        self.workflow.add_edge("generate_response", END)

    async def setup(self):
        """리소스 설정 (기존 Agent와 동일한 DB 공유)"""
        logger.info("AccidentChatAgent 리소스 설정 시작...")

        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME")
        DB_URL = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"

        self.pool = AsyncConnectionPool(
            conninfo=DB_URL,
            max_size=20,
            kwargs={"autocommit": True},
            open=False
        )

        await self.pool.open()
        logger.info("[ChatAgent] AsyncConnectionPool 연결 성공")

        self.checkpointer = AsyncPostgresSaver(self.pool)
        await self.checkpointer.setup() # 테이블 생성
        logger.info("[ChatAgent] PostgresSaver 테이블 설정 완료")

        # 4. 그래프 컴파일 (이제 graph_app이 완료됨)
        self.graph_app = self.workflow.compile(checkpointer=self.checkpointer)
        logger.info("[ChatAgent] 그래프 컴파일 완료 (Ready to serve)")

    async def load_context_node(self, state: ChatState) -> dict:
        """
        [Node 1] 기존 분석 결과 로드
        - 체크포인터에서 초기 분석 상태 조회
        - accident_context, vision_summary 등 복원
        """
        logger.info("[ChatAgent] 컨텍스트 로드 시작")

        # 이미 state에 컨텍스트가 있다면 그대로 사용 (초기 주입)
        if state.get("accident_context"):
            logger.info("[ChatAgent] 컨텍스트가 이미 존재 (스킵)")
            return {}
    
        # 컨텍스트가 없으면 에러 (API에서 주입해야 함)
        logger.warning("[ChatAgent] 컨텍스트 미제공")
        return {
            "accident_context": "사고 상황 정보 없음",
            "vision_summary": "이미지 분석 정보 없음",
            "initial_rag": "초기 판례 정보 없음",
            "fault_analysis": "과실 비율 정보 없음"
        }

    async def search_additional_node(self, state: ChatState) -> dict:
        """
        [Node 2] 추가 질문에 대한 RAG 재검색
        - 사용자 질문 + 사고 상황을 조합하여 검색
        """
        try:
            # 가장 최근 사용자 질문 추출
            last_human_msg = next(
                (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
                None
            )

            if not last_human_msg:
                return {"additional_rag": "검색할 질문이 없습니다."}
            
            user_question = last_human_msg.content

            # [핵심] 사고 상황 + 질문을 조합하여 검색 정확도 향상
            combined_query = f"""
            [사고 상황]
            {state.get("accident_context", '')}

            [사용자 질문]
            {user_question}            
            """

            logger.info(f"[ChatAgent] 추가 검색 시작: {user_question[:50]}...")

            # RAG 검색 실행
            additional_docs = await self.engine.search_legal_docs(combined_query)

            logger.info("[ChatAgent] 추가 검색 완료")
            return {"additional_rag": additional_docs}

        except Exception as e:
            logger.error(f"[ChatAgent] 추가 검색 오류")
            return {"additional_rag": "추가 검색 중 오류 발생"}

    async def generate_response_node(self, state: ChatState) -> dict:
        """
        [Node 3] 최종 응답 생성
        - 현재 상담 단계(current_stage) 내부 자동 판별
        - 이전 분석 결과(과실 비율 + 판례)를 컨텍스트로 활용
        """

        try:
            # --- [내부 단계 판별] 사용자 메시지 키워드 기반 자동 분류 ---
            last_human = next(
                (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
                ""
            )

            # 단계 판별 (statement_writing > case_confirmation > clarification)
            if any(kw in last_human for kw in ["진술서", "요약", "정리", "작성", "사고경위서", "보고서"]):
                current_stage = "statement_writing"
            elif any(kw in last_human for kw in ["과실", "비율", "판례", "합의", "보험", "보상", "책임", "할증"]):
                current_stage = "case_confirmation"
            else:
                current_stage = "clarification"
            
            logger.info(f"[ChatAgent] 상담 단계 판별: {current_stage}")

            # --- {rag_result} 조립 ---\
            rag_result_parts = []

            accident_ctx = state.get("accident_context", "")
            if accident_ctx:
                rag_result_parts.append(f"[사고 상황 요약]\n{accident_ctx}")

            vision_ctx = state.get("vision_summary", "")
            if vision_ctx:
                rag_result_parts.append(f"[현장 이미지 분석 결과]\n{vision_ctx}")

            fault_ctx = state.get("fault_analysis", "")
            if fault_ctx:
                rag_result_parts.append(f"[초기 과실 비율 판단]\n{fault_ctx}")

            initial_rag_ctx = state.get("initial_rag", "")
            if initial_rag_ctx:
                rag_result_parts.append(f"[초기 참조 판례]\n{initial_rag_ctx}")

            additional_rag_ctx = state.get("additional_rag", "")
            if additional_rag_ctx:
                rag_result_parts.append(f"[추가 검색된 관련 판례]\n{additional_rag_ctx}")

            rag_result = "\n\n".join(rag_result_parts) if rag_result_parts else "분석 결과 정보 없음"

            # 시스템 프롬프트 구성
            system_prompt = f"""
            당신은 교통사고 전문 상담 AI입니다.

            [이전 분석 결과]
            {rag_result}

            위 분석 결과를 바탕으로 사용자의 추가 질문에 답변합니다.
            답변 시 다음 사항을 준수하세요:
            1. 상위로 분석된 과실 비율과 판례 3가지 케이스를 바탕으로 답변
            2. 사용자가 이해하기 쉬운 용어 사용
            3. 필요시 추가 정보 요청

            현재 단계: {current_stage}
            - clarification: 사고 상황 명확화 질문
            - case_confirmation: 판례 확정을 위한 질문
            - statement_writing: 진술서 및 사고 요약 작성 안내

            [단계별 상세 지침]
            {self._get_stage_instruction(current_stage)}

            [답변 지침]
            1. 사용자의 질문에 직접적으로 답변하세요.
            2. 위 컨텍스트 정보를 근거로 활용하되, 자연스럽게 녹여서 설명하세요.
            3. 법률 용어는 쉽게 풀어서 설명하세요.
            4. 필요시 추가 질문을 유도하세요.
            5. 답변은 3-5 문단으로 구성하세요.
            """

            # 메시지 구성
            messages = [
                SystemMessage(content=system_prompt)
            ] + state["messages"]

            # LLM 호출 (gpt-4o-mini 사용 - 챗봇은 비용 효율적으로)
            response = await self.engine.llm_fast.ainvoke(messages)

            logger.info("[ChatAgent] 응답 생성 완료")

            return {
                "response": response.content,
                "messages": [AIMessage(content=response.content)]
            }

        except Exception as e:
            logger.error(f"[ChatAgent] 응답 생성 오류: {e}")
            fallback_response = "죄송합니다. 일시적인 오류가 발생했습니다. 질문을 다시 입력해주세요."
            return {
                "response": fallback_response,
                "messages": [AIMessage(content=fallback_response)]
            }
        
    def _get_stage_instruction(self, stage: str) -> str:
        """단계별 상세 응답 지침 반환"""
        instructions = {
            "clarification": (
                "- 사고 상황이 불명확한 부분이 있다면 구체적으로 질문하세요.\n"
                "- 예: \"사고 당시 신호등 상태는 어땠나요?\", \"상대 차량의 진행 방향을 알 수 있나요?\"\n"
                "- 충분한 정보가 이미 있다면, 분석 결과를 바탕으로 직접 답변하세요."
            ),
            "case_confirmation": (
                "- 이전 분석에서 산정된 과실 비율과 판례를 근거로 답변하세요.\n"
                "- 과실 비율이 달라질 수 있는 핵심 변수(속도, 신호, 진입 시점 등)를 안내하세요.\n"
                "- 합의/보험 처리 시 주의사항 등 실질적 조언을 제공하세요.\n"
                "- \"판례에 따르면...\" 형태로 근거를 명시하세요."
            ),
            "statement_writing": (
                "- 진술서 또는 사고 요약문 작성을 안내하세요.\n"
                "- 육하원칙(언제, 어디서, 누가, 무엇을, 어떻게, 왜) 형식으로 구성을 잡아주세요.\n"
                "- 사용자에게 유리한 표현과 불리한 표현을 구분하여 조언하세요.\n"
                "- 사실 관계를 왜곡하지 않도록 주의하세요."
            ),
        }
        return instructions.get(stage, instructions["clarification"])

    async def run(self, initial_state: dict, config: dict = None) -> dict:
        """그래프 실행 헬퍼"""
        if config is None:
            config = {}
        return await self.graph_app.ainvoke(initial_state, config=config)

    async def stream_response(self, initial_state: dict, config: dict = None):
        """
        [스트리밍 응답 생성]
        실시간으로 응답을 반환하여 사용자 경험 향상
        """
        if config is None:
            config = {}

        # 스트리밍 실행
        async for event in self.graph_app.astream(initial_state, config=config):
            # 각 노드의 출력을 yield
            for node_name, node_output in event.items():
                if node_name == "generate_response" and "response" in node_output:
                    yield node_output["response"]
    
# 전역 인스턴스 생성
agent_instance = AccidentGraphAgent()
chat_agent_instance = AccidentChatAgent()
    