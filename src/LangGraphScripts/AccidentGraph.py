import os
import sys
import operator
from typing import Annotated, List, TypedDict, Union
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------
# 현재 파일의 부모 디렉토리를 시스템 경로에 추가
# 그래야 옆 동네인 'RAG' 폴더를 찾을 수 있음
# ---------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))    # 현재 폴더 경로
parent_dir = os.path.dirname(current_dir)                   # 상위 프로젝트 루트 경로
sys.path.append(parent_dir)

# 엔진 및 데이터 모델 임포트
from RAG.AccidentRAGEngine import AccidentRAGEngine, AccidentAnalysisResult

# 엔진 로드
engine = AccidentRAGEngine()

# --- 1. 상태(State3) 정의 ---
class AgentState(TypedDict):
    # messages: 히스토리 누적 (add operator 사용)
    messages: Annotated[List[BaseMessage], operator.add]
    image_paths: List[str]

    # 워커들의 결과를 저장할 필드 (덮어쓰기)
    image_summary: str
    rag_context: str

    # 슈퍼바이저가 결정한 실행항 틀 리스트
    required_tools: List[str]

    # 최종 결과 (구조화된 데이터 - dict 형태로 저장)
    final_result: dict

# --- 2. 노드(Node) 정의 ---
async def supervisor_node(state: AgentState):
    """[슈퍼바이저] 입력값을 분석하여 어떤 워커를 가동할지 결정"""
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

    return {"required_tools": tools}

async def vision_node(state: AgentState):
    """[Worker A] 이미지 분석 -> 텍스트 변환 (휘발성)"""
    try:
        paths = state.get("image_paths", [])
        if not paths:
            return {"image_summary": "분석할 이미지가 없습니다."}

        # 여기서 생성된 'description'이 Final Solver의 유일한 시각 정보가 됨
        description = await engine.analyze_images_from_paths(paths)
        return {"image_summary": description}

    except Exception as e:
        return {"image_summary": f"이미지 분석 중 오류 발생: {str(e)}"}
    
async def search_node(state: AgentState):
    """[Worker B] 법률 검색"""
    try:
        # 가장 최근 사용자 발화 찾기
        last_human_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)

        if not last_human_msg:
            return {"rag_context": "검색할 사용자 질문이 없습니다."}
        
        query = last_human_msg.content
        docs = await engine.search_legal_docs(query)
        return {"rag_context": docs}

    except Exception as e:
        return {"rag_context": f"판례 검색 중 오류 발생: {str(e)}"}
    
async def join_node(state: AgentState):
    """
    [Join Node] 병렬 워커들의 동기화 지점.
    모든 워커가 끝날 때까지 기다렸다가 다음 단계로 넘어갑니다.
    """
    return {}

async def final_solver(state: AgentState):
    """
    [Final Node] 모든 정보를 종합하여 구조화된 답변 생성
    [비용 절감] 이미지 파일은 넘기지 않고, Vision 요약 텍스트만 전달함
    """

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
    result: AccidentAnalysisResult = await engine.generate_final_solution_structured(
        messages=msgs_for_llm,
        image_paths=[]
    )

    # 결과 반환
    return {
        "final_result": result.model_dump(),
        "messages": [AIMessage(content=result.advice)]      # 대화 기록에는 텍스트 조언만 남김
    }

# --- 3. 그래프 조립 ---
workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("vision_tool", vision_node)
workflow.add_node("search_tool", search_node)
workflow.add_node("join", join_node)
workflow.add_node("final_solver", final_solver)

# 엣지 연결
workflow.set_entry_point("supervisor")

def route_tools(state: AgentState):
    return state["required_tools"]

# 슈퍼바이저 -> 각 툴 (Fan-out)
workflow.add_conditional_edges(
    "supervisor",
    route_tools,
    ["vision_tool", "search_tool"]
)

# 각 툴 -> Join 노드 (Fan-in)
workflow.add_edge("vision_tool", "join")
workflow.add_edge("search_tool", "join")

# Join -> Final Solver (단일 실행 보장)
workflow.add_edge("join", "final_solver")

# 종료
workflow.add_edge("final_solver", END)

memory = MemorySaver()
graph_app = workflow.compile(checkpointer=memory)