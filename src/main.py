import os
import sys
import shutil
import uuid
import contextlib
import logging
import json
import asyncio

from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# 모듈 import
# from STT.google_stt_handler import GoogleSTTHandler       # 기능 제외로 주석 처리
from src.LangGraphScripts.AccidentGraph import agent_instance, chat_agent_instance

# Lifespan(수명 주기) 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DB 연결 풀 및 체크포인터 초기화 중...")
    # agent의 setup() 메서드 하나만 호출하면 내부에서 다 처리됨
    try:
        # 기존 Agent 및 Chat Agent 동시 설정
        await agent_instance.setup()
        await chat_agent_instance.setup()
        logger.info("모든 Agent 초기화 완료")
    except Exception as e:
        logger.error(f"초기화 실퍄: {e}")
    
    yield

    # 종료시 연결 닫기
    logger.info("DB 연결 종료 중...")
    if agent_instance.pool:
        await agent_instance.pool.close()
    
    if chat_agent_instance.pool:
        await chat_agent_instance.pool.close()

# FastAPI 앱 정의
app = FastAPI(
    title="교통사고 대응 AI (Supervisor Agent + Chatbot)",
    lifespan=lifespan)      # lifespan 등록

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용 (개발 중)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        - status: 성공 여부
        - thread_id: 세션 ID (챗봇에서 재사용)
        - transcript: 입력 텍스트
        - rag_context: 초기 검색된 판례
        - result: 구조화된 분석 결과
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

        # --- 6. Supervisor Graph 실행 ---
        if agent_instance.graph_app is None:
            raise HTTPException(status_code=500, detail="Graph not initialized")
        
        final_state = await agent_instance.graph_app.ainvoke(initial_state, config=config)

        # --- 7. 결과 추출 ---
        structured_result = final_state.get("final_result", {})
        rag_context = final_state.get("rag_context", "검색 결과 없음")
        image_summary = final_state.get("image_summary", "이미지 분석 없음")

        logger.info("분석 완료")

        return {
            "status": "success",
            "thread_id": thread_id,
            "transcript": transcript,
            "image_summary": image_summary,
            "rag_context": rag_context,
            "result": structured_result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # --- 8. 임시 파일 안전하게 처리
        for path in temp_image_paths:
            with contextlib.suppress(FileNotFoundError, PermissionError):
                os.remove(path)

        if temp_image_paths:
            logger.info(f"임시 파일 {len(temp_image_paths)}개 삭제 완료")

# -----------------------------------------
# 챗봇 엔드 포인트
# -----------------------------------------
@app.post("/chat")
async def chat_with_bot(
    thread_id: str = Form(...),         # 필수: 초기 분석 세션 ID
    user_message: str = Form(...),      # 필수: 사용자 질문
    accident_context: str = Form(None), # 선택: 사고 상황 (없으면 DB에서 조회)
    vision_summary: str = Form(None),   # 선택: 초기 Vision 분석 결과
    initial_rag: str = Form(None),      # 선택: 초기 RAG 결과
    fault_analysis: str = Form(None)    # 선택: 과실 비율 분석
):
    """
    챗봇 대화 엔드포인트

    Args:
        thread_id: 초기 분석에서 받은 세션 ID
        user_message: 사용자의 추가 질문
        accident_context: 사고 상황 컨텍스트 (옵션)
        vision_summary: 이미지 분석 결과 (옵션)
        initial_rag: 초기 판례 검색 결과 (옵션)
        fault_analysis: 과실 비율 판단 (옵션)

    Returns:
        - response: AI 답변
        - thread_id: 세션 ID (계속 유지)
    """
    try:
        if not thread_id or not user_message:
            raise HTTPException(
                status_code=400,
                detail="thread_id와 user_message는 필수입니다."
            )
        
        logger.info(f"[Chat] 세션 {thread_id}에서 질문: {user_message[:50]}...")

        # Config 설정 (동일한 thread_id 사용)
        config = {"configurable": {"thread_id": thread_id}}

        # 초기 상태 구성
        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "accident_context": accident_context or "",
            "vision_summary": vision_summary or "",
            "initial_rag": initial_rag or "",
            "fault_analysis": fault_analysis or "",
            "additional_rag": "",
            "response": "" 
        }

        # Chat Agent 실행
        if chat_agent_instance.graph_app is None:
            raise HTTPException(status_code=500, detail="Chat Agent not initialized")
        
        final_state = await chat_agent_instance.graph_app.ainvoke(initial_state, config=config)

        # 응답 추출
        response_text = final_state.get("response", "응답 생성 실패")

        logger.info(f"[Chat] 응답 생성 완료: {response_text[:100]}...")

        return {
            "status": "success",
            "thread_id": thread_id,
            "response": response_text
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Chat] 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
# -----------------------------------------
# 스트리밍 챗봇 엔드 포인트
# -----------------------------------------
@app.post("/chat/stream")
async def chat_with_bot_stream(
    thread_id: str = Form(...),
    user_message: str = Form(...),
    accident_context: str = Form(None),
    vision_summary: str = Form(None),
    initial_rag: str = Form(None),
    fault_analysis: str = Form(None)
):
    """
    스트리밍 챗봇 엔드포인트
    - 실시간으로 응답을 반환하여 사용자 경험 향상

    Returns:
        StreamingResponse (Server-Sent Event 형식)
    """
    
    try:
        if not thread_id or not user_message:
            raise HTTPException(
                status_code=400,
                detail="thread_id와 user_message는 필수입니다."
            )
        
        logger.info(f"[Chat Stream] 세션 {thread_id}에서 스트리밍 시작")

        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "accident_context": accident_context or "",
            "vision_summary": vision_summary or "",
            "initial_rag": initial_rag or "",
            "fault_analysis": fault_analysis or "",
            "additional_rag": "",
            "response": "" 
        }

        # 스트리밍 생성기
        async def generate():
            try:
                async for chunk in chat_agent_instance.stream_response(initial_state, config):
                    # Server-Sent Event 형식으로 전송 (한글 깨짐 방지: ensure_ascii=False)
                    yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
                
                # 스트리밍 종료 신호
                yield "data: {\"done\": true}\n\n"
            except Exception as e:
                logger.error(f"[Chat Stream] 스트리밍 오류: {e}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Chat Stream] 초기화 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------
# 대화 히스토리 조회
# -----------------------------------------
@app.get("/chat/history/{thread_id}")
async def get_chat_history(thread_id: str):
    """
    특정 세션의 대화 기록 조회

    Args:
        thread_id: 세션 ID
    
    Returns
        대화 시스토리 (messages 리스트)
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}

        # 체크포인트에서 상태 조회
        state = await chat_agent_instance.checkpointer.aget(config)

        if not state:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        # 메시지 추출
        messages = state.get("messages", [])

        # 직렬화 가능한 형태로 변환
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})

        return {
            "status": "success",
            "thread_id": thread_id,
            "history": history            
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[History] 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# -----------------------------------------
# 헬스 체크
# -----------------------------------------
@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    from src.RAG.AccidentRAGEngine import AccidentRAGEngine
    engine = AccidentRAGEngine()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "analyze_agent": agent_instance.graph_app is not None,
            "chat_agent": chat_agent_instance.graph_app is not None
        },
        "database": {
            "rag-pool": engine.get_pool_status()
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 벡엔드 서버를 시작합니다 (http://0.0.0.0:8001)...")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)