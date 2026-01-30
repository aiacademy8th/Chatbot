import os
import sys

# -----------------------------------------------------------
# 현재 파일(STT폴더)의 부모 디렉토리를 시스템 경로에 추가
# 그래야 옆 동네인 'RAG' 폴더를 찾을 수 있습니다.
# -----------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))  # STT 폴더 경로
parent_dir = os.path.dirname(current_dir)                 # 상위 프로젝트 루트 경로
sys.path.append(parent_dir)                               # 파이썬에게 루트 경로를 알려줌

import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from google_stt_handler import GoogleSTTHandler
from RAG.AccidentRAGEngine import AccidentRAGEngine


app = FastAPI(title="Traffic AI Integrated Server")

# --- 엔진 초기화 ---
print("⏳ 엔진 로딩 중...")
stt_handler = GoogleSTTHandler()
rag_engine = AccidentRAGEngine()
print("✅ 모든 엔진 준비 완료!")

# 임시 파일 저장 디렉토리
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_accident(
    voice_file: UploadFile = File(...),     # 필수: 음성 파일
    image_file: UploadFile = File(None)     # 선택: 이미지 파일
):
    """
    1. 음성 -> 텍스트 변환 (STT)
    2. 이미지 저장
    3. RAG 엔진 실행 (텍스트 + 이미지)
    """

    voice_path = None
    image_path = None
    transcript = ""

    try:
        # --- 1. 음성 처리 (STT) ---
        print(f"🎤 음성 파일 수신: {voice_file.filename}")
        voice_bytes = await voice_file.read()
        transcript = stt_handler.transcribe_audio(voice_bytes)
        print(f"📝 STT 결과: {transcript}")

        if not transcript or transcript == "(인식된 음성 내용 없음)":
            return {"text": transcript, "answer": "음성이 명확하지 않아 분석할 수 없습니다."}
        
        # --- 2. 이미지 처리 (저장) ---
        if image_file:
            image_path = os.path.join(TEMP_DIR, image_file.filename)
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image_file.file, buffer)
            print(f"🖼️ 이미지 저장 완료: {image_path}")

        # --- 3. RAG 엔진 실행 ---
        print("🧠 AI 분석 시작...")
        answer, docs = rag_engine.ask(query=transcript, image_path=image_path)

        # 참고 문헌 정리
        references = [doc.metadata.get("source", "Unknown") for doc, score in docs]

        return {
            "transcript": transcript,       # 사용자가 말한 내용
            "answer": answer,               # AI 답변
            "references": references        # 참고한 판례
        }

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 처리 후 임시 이미지 파일 삭제
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            print("🗑️ 임시 파일 삭제 완료")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)