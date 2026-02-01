import os
import sys
import asyncio
import shutil
from typing import List

# -----------------------------------------------------------
# 현재 파일(STT폴더)의 부모 디렉토리를 시스템 경로에 추가
# 그래야 옆 동네인 'RAG' 폴더를 찾을 수 있습니다.
# -----------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))  # STT 폴더 경로
parent_dir = os.path.dirname(current_dir)                 # 상위 프로젝트 루트 경로
sys.path.append(parent_dir)                               # 파이썬에게 루트 경로를 알려줌

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from google_stt_handler import GoogleSTTHandler
from RAG.AccidentRAGEngine import AccidentRAGEngine

app = FastAPI(title="교통사고 대응 AI (Multi-Modal)")

# --- 엔진 초기화 ---
print("⏳ 엔진 로딩 중...")
try:
    stt_handler = GoogleSTTHandler()
    rag_engine = AccidentRAGEngine()
    print("✅ 모든 엔진 준비 완료!")
except Exception as e:
    print(f"❌ 초기화 실패: {e}")

# 임시 파일 저장 디렉토리
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_accident(
    voice_file: UploadFile = File(...),             # 필수: 음성 파일
    image_files: List[UploadFile] = File(None)      # 선택: 이미지 파일 리스트 (여러장 처리 가능)
):
    """
    1. 음성 -> 텍스트 변환 (STT)
    2. 다중 이미지 저장 및 인코딩
    3. RAG 엔진 병렬 실행 (텍스트 + 여러 이미지)
    """

    voice_path = None
    temp_image_paths = []       # 이미지 분석 후 삭제 처리를 위한 경로 저장용 리스트
    image_base64_list = []      # 엔진에 전달할 Base64 문자열 리스트
    transcript = ""
    image_description = ""

    try:
        # 1. 파일 수신 및 저장 (I/O)
        print(f"📂 음성 파일 수신 중: {voice_file.filename}")
        voice_bytes = await voice_file.read()

        if image_files:
            print(f"📸 이미지 {len(image_files)}장 수신됨. 처리 시작...")
            for img_file in image_files:
                # 파일명 충돌 방지를 위해 간단한 처리 (필요시 uuid 사용)
                temp_path = os.path.join(TEMP_DIR, img_file.filename)

                # 파일 저장
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(img_file.file, buffer)
                
                temp_image_paths.append(temp_path)

                # RAG 엔진의 리사이징 로직을 이용해 인코딩
                # (CPU 작업이지만 빠르므로 여기서 순차적으로 처리해도 무방)
                base64_str = rag_engine._encode_image(temp_path)
                image_base64_list.append(base64_str)
        
        print("🚀 병렬 처리 시작 (STT ⚡ Vision)")

        # 2. 최적화, 병렬 작업 정의
        
        # Task A: STT 
        # (Google API는 동기 함수이므로 스레드로 분리하여 메인 루프 차단 방지)
        task_stt = asyncio.to_thread(stt_handler.transcribe_audio, voice_bytes)

        # Task B: Vision
        # (이미지가 있을 때만 비동기 분석 작업 생성)
        task_vision = None
        if image_base64_list:
            task_vision = rag_engine.analyze_image_async(image_base64_list)
        
        # 3. 최적화, 병렬 실행 및 대기 (Gather)
        # 두 작업 중 늦게 끝나는 것 기준으로 시간이 소요됨 (예: STT 3초, Vision 4초 -> 총 4초 소요)
        if task_vision:
            results = await asyncio.gather(task_stt, task_vision)
            transcript = results[0]
            image_description = results[1]
        else:
            # 이미지가 없다면 STT만 수행
            transcript = await task_stt
            image_description = ""

        print(f"✅ 병렬 처리 완료 | STT: {len(transcript)}자, Vision: {len(image_description)}자")

        # 예외 처리: 음성 인식이 안 된 경우
        if not transcript or transcript == "(인식된 음성 내용 없음)":
            return {
                "answer": "음성을 인식할 수 없습니다. 다시 시도해 주세요.", 
                "transcript": transcript, 
                "references": []
            }
        
        # 4. 최종 답변 생성 (검색 + LLM)
        # 검색 및 답변 생성 함수 호출 (이미지 리스트 전달)
        answer, docs = await rag_engine.ask_with_context(
            query=transcript,
            image_description=image_description,
            # 엔진 정의(AccidentRAGEngine.py)의 인자 이름과 일치
            image_base64_list=image_base64_list 
        )

        # [중요] docs는 (Document, score) 튜플 리스트이므로 언패킹 필요
        references = [doc.metadata.get("source", "Unknown") for doc, score in docs]

        return {
            "transcript": transcript,
            "answer": answer,
            "references": references
        }

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        # 디버깅을 위해 상세 에러 출력
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 5. 리소스 정리 (임시 이미지 파일 삭제)
        if temp_image_paths:
            print(f"🗑️ 임시 파일 {len(temp_image_paths)}개 삭제 중...")
            for path in temp_image_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as cleanup_error:
                        print(f"⚠️ 파일 삭제 실패 ({path}): {cleanup_error}")
            print("🗑️ 모든 임시 파일 삭제 완료")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)