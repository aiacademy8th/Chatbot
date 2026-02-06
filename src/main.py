import os
import sys
import shutil
import uuid
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from langchain_core.messages import HumanMessage

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------
# 현재 파일의 부모 디렉토리를 시스템 경로에 추가
# 그래야 옆 동네인 'RAG' 폴더를 찾을 수 있음
# -------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))    # 현재 폴더 경로
parent_dir = os.path.dirname(current_dir)                   # 상위 프로젝트 루트 경로
sys.path.append(parent_dir)

# 모듈 import
# from STT.google_stt_handler import GoogleSTTHandler       # 기능 제외로 주석 처리
from LangGraphScripts.AccidentGraph import agent_instance

# Lifespan(수명 주기) 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DB 연결 풀 및 체크포인터 초기화 중...")
    # agent의 setup() 메서드 하나만 호출하면 내부에서 다 처리됨
    try:
        await agent_instance.setup()
    except Exception as e:
        logger.error(f"초기화 실퍄: {e}")
    
    yield
    # 종료시 연결 닫기
    logger.info("DB 연결 종료 중...")
    if agent_instance.pool:
        await agent_instance.pool.close()

# FastAPI 앱 정의
app = FastAPI(
    title="교통사고 대응 AI (Supervisor Agent)",
    lifespan=lifespan)      # lifespan 등록

# 임시 저장소 설정
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_accident(
    text_query: str = Form(...),                            # 텍스트 입력은 필수가 됨
    image_files: List[UploadFile] = File(None),
    thread_id: str = Form(None)     # 멀티턴 세션 ID
):
    """
    교통사고 분석 엔드포인트

    Args:
        text_query: 사고 상황 텍스트 설명 (필수)
        image_files: 현장 사진 (선택)
        thread_id: 대화 세션 ID (선택, 없으면 자동 생성)

    Returns:
        분석 결과 JSON
    """
    temp_image_paths = []
    transcript = text_query                                 # 이제 transcript 는 순수하게 text_query 내용만 담음

    try:
        # --- 1. 입력 검증 ---
        if not transcript and not image_files:
            raise HTTPException(
                status_code=400, 
                detail="입력 데이터(텍스트 또는 사진)가 없습니다."
            )
        
        # --- 2. 이미지 파일 저장 ---
        if image_files:
            for img_file in image_files:
                ext = img_file.filename.split(".")[-1]
                path = os.path.join(TEMP_DIR, f"img_{uuid.uuid4()}.{ext}")
                with open(path, "wb") as buffer:
                    shutil.copyfileobj(img_file.file, buffer)
                temp_image_paths.append(path)
            logger.info(f"이미지 {len(temp_image_paths)}개 저장 완료")

        # --- 3. Thread ID 관리 ---
        if not thread_id:
            thread_id = str(uuid.uuid4())
            logger.info(f"새로운 대화 세션 생성: {thread_id}")

        # --- 4. Config 설정 (Memory 사용) ---
        config = {"configurable": {"thread_id": thread_id}}

        # --- 5. 초기 상태 주입 ---
        initial_state = {
            "messages": [HumanMessage(content=transcript)] if transcript else [],
            "image_paths": temp_image_paths,
            "image_summary": "",
            "rag_context": ""
        }

        logger.info(f"Agent 실행 시작 (Thread: {thread_id})")

        # agent_instance.graph_app을 사용하여 호출
        # (setup())이 완료되었으므로 이제 None이 아님
        if agent_instance.graph_app is None:
            raise HTTPException(status_code=500, detail="Graph not initalized")

        # --- 6. Supervisor Graph 실행 ---
        if agent_instance.graph_app is None:
            raise HTTPException(status_code=500, detail="Graph not initialized")
        
        final_state = await agent_instance.graph_app.ainvoke(initial_state, config=config)

        # --- 7. 결과 추출 ---
        structured_result = final_state.get("final_result", {})
        rag_context = final_state.get("rag_context", "검색 결과 없음")

        logger.info("분석 완료")

        return {
            "status": "success",
            "thread_id": thread_id,
            "transcript": transcript,
            "rag_context": rag_context,
            "result": structured_result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # --- 8. [개선] 임시 파일 안전하게 처리
        for path in temp_image_paths:
            with contextlib.suppress(FileNotFoundError, PermissionError):
                os.remove(path)

        if temp_image_paths:
            logger.info(f"임시 파일 {len(temp_image_paths)}개 삭제 완료")

if __name__ == "__main__":
    import uvicorn
    print("🚀 벡엔드 서버를 시작합니다 (http://0.0.0.0:8000)...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)