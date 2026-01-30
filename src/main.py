from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# 엔진 임포트
try:
    from LangGraphScripts.accident_engine import AccidentDecisionEngine
    from RAG.AccidentRAGEngine import AccidentRAGEngine
except ImportError as e:
    logging.error(f"Import Error: {e}")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

load_dotenv()

# --- 데이터 스키마 ---
class AnalysisRequest(BaseModel):
    accident_type: str
    speed: str
    injury: str
    pain_now: str
    hospital_visit: str
    vehicle_damage: str
    adas_sensor: str
    vehicle_type: str
    evidence: str
    opponent_attitude: str
    opponent_mentions_hospital: str
    opponent_mentions_insurance: str
    notes: str

class SourceDoc(BaseModel):
    content: str
    similarity: float
    source: str

class AnalysisResponse(BaseModel):
    risk_bucket: str                    # LangGraph 결과: GREEN|YELLOW|RED
    final_answer: str                   # LangGraph의 최종 판단 리스트
    flags_red: List[str]                # 고위험 요소 리스트
    flags_yellow: List[str]             # 주의 요소 리스트
    relevant_sources: List[SourceDoc]   # RAG가 찾은 근거 문서들

# --- 유틸리티 ---
def build_query_from_request(req: AnalysisRequest) -> str:
    return (
        f"사고 유형: {req.accident_type}, 속도: {req.speed}, 부상: {req.injury}, "
        f"통증: {req.pain_now}, 파손: {req.vehicle_damage}, 상대 태도: {req.opponent_attitude}. "
        f"메모: {req.notes}. 관련 대응법과 판례 알려줘."
    )

# --- Lifespan: 엔진은 여기서 한 번만 생성합니다 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 엔진 초기화 중 (RAG + LangGraph)...")
    try:
        app.state.rag_engine = AccidentRAGEngine()
        app.state.decision_engine = AccidentDecisionEngine()
        logging.info("✅ 모든 엔진 로드 완료")
    except Exception as e:
        logging.error(f"❌ 엔진 초기화 실패: {e}")
    yield
    # 종료 시 리소스 정리
    if hasattr(app.state, "rag_engine"):
        del app.state.rag_engine
    if hasattr(app.state, "decision_engine"):
        del app.state.decision_engine
    logging.info("🛑 엔진 리소스 해제 완료")

# FastAPI 앱 생성 (lifespan 전달)
app = FastAPI(title="교통사고 판단 보조 API", lifespan=lifespan)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Accident Engine is healthy."}

# main.py (핵심 로직 부분)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(data: AnalysisRequest, request: Request):
    try:
        rag_engine = request.app.state.rag_engine
        decision_engine = request.app.state.decision_engine

        # STEP 1: RAG 검색 (법률/판례 지식 추출)
        search_query = build_query_from_request(data)
        rag_answer, docs = rag_engine.ask(search_query) # rag_answer는 요약문, docs는 원문 리스트
        
        # STEP 2: 데이터 결합 (사용자 정보 + RAG 지식)
        facts = data.model_dump()
        facts["rag_context"] = rag_answer # 랭그래프가 읽을 참고 지식 주입

        # STEP 3: LangGraph 실행 (추론 및 최종 판단)
        graph_result = decision_engine.run_analysis(facts)

        # STEP 4: 데이터 포맷팅
        formatted_sources = [
            SourceDoc(
                content=getattr(doc, "page_content", str(doc)),
                similarity=float(score) if score is not None else 0.0,
                source=getattr(doc, "metadata", {}).get("source", "법규/판례")
            ) for doc, score in docs
        ]

        # STEP 5: 결합된 결과 반환
        return AnalysisResponse(
            risk_bucket=graph_result.get("risk_bucket", "정보부족"),
            final_answer=graph_result.get("final_answer", "분석 실패"),
            flags_red=graph_result.get("flags_red", []),
            flags_yellow=graph_result.get("flags_yellow", []),
            relevant_sources=formatted_sources # 화면에 먼저 보여줄 근거 자료
        )
    except Exception as e:
        logging.error(f"❌ 분석 에러: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)