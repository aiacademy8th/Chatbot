from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from LangGraphScripts.accident_engine import AccidentDecisionEngine     # LangGraphScripts 폴더의 accident_engine 파일의 AccidentDecisionEngine 클래스 참조cket

app = FastAPI(title="교통사고 판단 보조 API")

# API 서버 시작 시 엔진 객체를 한 번만 생성 (메모리 효율)
engine = AccidentDecisionEngine()

# --- 데이터 스키마 ---
class AnalysisRequest(BaseModel):
    accident_type: str               # 정차후출발|주차중|차선변경|후방추돌|기타|불명
    speed: str                       # 저속|중속|고속|불명
    injury: str                      # 없음|애매|있음|불명
    pain_now: str                    # 없음|경미|지속|악화|불명
    hospital_visit: str              # 없음|예정|완료|불명
    vehicle_damage: str              # 없음|스크래치|찌그러짐|파손|불명
    adas_sensor: str                 # 없음|있음|불명
    vehicle_type: str                # 국산|수입|전기차|불명
    evidence: str                    # 충분|일부|없음|불명
    opponent_attitude: str           # 원만|애매|공격적|불명
    opponent_mentions_hospital: str  # 아니오|예|불명
    opponent_mentions_insurance: str # 아니오|예|불명
    notes: str                       # 자유메모

class AnalysisResponse(BaseModel):
    risk_bucket: str
    final_answer: str
    flags_red: List[str]
    flags_yellow: List[str]

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Accident Decision Engine is running."}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    try:
        # Pydantic 모델을 딕셔너리로 반환하여 엔진에 전달
        facts_dict = request.model_dump()
        result = engine.run_analysis(facts_dict)

        return AnalysisResponse(
            risk_bucket=result.get("risk_bucket", "정보 부족"),
            final_answer=result.get("final_answer", "분석 결과를 생성할 수 없습니다."),
            flags_red=result.get("flags_red", []),
            flags_yellow=result.get("flags_yellow", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)