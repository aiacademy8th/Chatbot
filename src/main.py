import os
import sys
import shutil
import asyncio
import uuid
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from langchain_core.messages import HumanMessage

# -------------------------------------------
# 현재 파일의 부모 디렉토리를 시스템 경로에 추가
# 그래야 옆 동네인 'RAG' 폴더를 찾을 수 있음
# -------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))    # 현재 폴더 경로
parent_dir = os.path.dirname(current_dir)                   # 상위 프로젝트 루트 경로
sys.path.append(parent_dir)

# 모듈 import
from STT.google_stt_handler import GoogleSTTHandler
from LangGraphScripts.AccidentGraph import graph_app

# FastAPI 앱 정의
app = FastAPI(title="교통사고 대응 AI (Supervisor Agent)")
stt_handler = GoogleSTTHandler()

# 임시 저장소 설정
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_accident(
    voice_file: UploadFile = File(None),
    text_query: str = Form(None),
    image_files: List[UploadFile] = File(None),
    thread_id: str = Form(None)     # 멀티턴 세션 ID
):
    temp_voice_path = None
    temp_image_paths = []
    transcript = ""

    try:
        # --- 1. 입력 소스 처리 ---
        if voice_file:
            ext = voice_file.filename.split('.')[-1]
            temp_voice_path = os.path.join(TEMP_DIR, f"voice_{uuid.uuid4()}.{ext}")

            with open(temp_voice_path, "wb") as buffer:
                shutil.copyfileobj(voice_file.file, buffer)

            transcript = await asyncio.to_thread(stt_handler.transcribe_audio, temp_voice_path)
            print(f"📝 [STT]: {transcript}")

            if not transcript or transcript == "(인식된 음성 내용 없음)":
                if not text_query:
                    return {"answer": "음성 인식 실패. 텍스트로 입력해주세요."}
                
        # 텍스트 입력 병합
        if text_query:
            if transcript:
                transcript = f"{transcript} (추가 메모: {text_query})"
            else:
                transcript = text_query
        if not transcript and not image_files:
            raise HTTPException(status_code=400, detail="입력 데이터(음성, 텍스트, 사진)가 없습니다.")
        
        # --- 2. 이미지 파일 저장 ---
        if image_files:
            for img_file in image_files:
                ext = img_file.filename.split(".")[-1]
                path = os.path.join(TEMP_DIR, f"img_{uuid.uuid4()}.{ext}")
                with open(path, "wb") as buffer:
                    shutil.copyfileobj(img_file.file, buffer)
                temp_image_paths.append(path)

        # --- 3. Supervisor Graph 실행 ---
        # (1) Thread ID 관리
        if not thread_id:
            thread_id = str(uuid.uuid4())
            print(f"🆕 새로운 대화 세션 생성: {thread_id}")

        # (2) Config 설정 (Memory 사용)
        config = {"configurable": {"thread_id": thread_id}}

        # (3) 초기 상태 주입
        initial_state = {
            "messages": [HumanMessage(content=transcript)] if transcript else [],
            "image_paths": temp_image_paths,
            "image_summary": "",
            "rag_context": ""
        }

        print(f"🤖 Agent 실행 (Thread: {thread_id})...")
        final_state = await graph_app.ainvoke(initial_state, config=config)

        # --- 4. 결과 추출 ---
        structured_result = final_state.get("final_result", {})
        rag_context = final_state.get("rag_context", "검색 결과 없음")

        return {
            "status": "success",
            "thread_id": thread_id,
            "transcript": transcript,
            "rag_context": rag_context,
            "result": structured_result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 리소스 정리
        paths_to_remove = temp_image_paths + ([temp_voice_path] if temp_voice_path else [])
        for path in paths_to_remove:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

if __name__ == "__main__":
    import uvicorn
    print("🚀 벡엔드 서버를 시작합니다 (http://0.0.0.0:8000)...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)