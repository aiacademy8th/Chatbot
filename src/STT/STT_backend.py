from fastapi import FastAPI, UploadFile, File, HTTPException
from google_stt_handler import GoogleSTTHandler
import uvicorn

app = FastAPI(title="Traffic AI Voice Server")

# STT 핸들러 초기화 (서버 시작 시 1회 로드)
try:
    stt_handler = GoogleSTTHandler()
except Exception as e:
    print(f"❌ STT 핸들러 로드 실패: {e}")
    stt_handler = None

@app.post("/stt")
async def process_voice(file: UploadFile = File(...)):
    """
    클라이언트(Streamlit)로 부터 오디오 파일을 받아 텍스트로 변환
    """
    if not stt_handler:
        raise HTTPException(status_code=500, detail="STT 엔진이 초기화되지 않았습니다.")
    
    print(f"📥 오디오 수신 중: {file.filename}")

    try:
        # 1. 파일 읽기
        audio_bytes = await file.read()

        # 2. 변환 요청
        transcript = stt_handler.transcribe_audio(audio_bytes)

        print(f"👉 변환 결과: {transcript}")

        # 3. JSON 응답
        return {"text": transcript}
    
    except Exception as e:
        print(f"⚠️ 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    # 직접 실행 시 디버그 모드로 서버 구동
    uvicorn.run(app, host="0.0.0.0", port=8000)