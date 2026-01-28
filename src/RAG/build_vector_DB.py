import os
import logging
from pathlib import Path
import urllib.parse

import psycopg2
from dotenv import load_dotenv
from google.cloud import storage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# 기존 FAISS 를 사용하던 구조에서 PGVectorStore 를 사용하던 방식으로 변경
#from langchain_community.vectorstores import FAISS
from langchain_postgres import PGVector, PGEngine
from sqlalchemy import create_engine

# --- 1. 로깅 설정 (개선된 부분) ---
# print() 대신 표준 로깅 모듈을 사용하여 로그의 레벨 관리와 포맷팅을 체계화합니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_env_config():
    """환경 변수를 로드하고 필수 설정을 확인"""
    load_dotenv()
    required_vars = ["OPENAI_API_KEY", "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    for var in required_vars:
        if not os.getenv(var):
            logging.error(f"{var}가 .env 파일에 설정되지 않음.")
            raise ValueError(f"필수 설정 누락: {var}")
    
    # 비밀번호 특수문자(@) 처리를 위한 인코딩
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")
        
    # SQLAlchemy 스타일 연결 문자열 생성 (PGVectorStore용)
    conn_str = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db_name}"

    return conn_str

# --- 2. DB 및 GCS 연동 함수 
def get_pending_files():
    """DB에서 아직 벡터화 되지 않은(is_vectorized = false) 파일 목록을 가져옴"""

    raw_password = urllib.parse.unquote(os.getenv("DB_PASSWORD"))

    # psycopg2는 별도 인코딩 없이 비밀번호 바로 사용 가능
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=raw_password,
        port=os.getenv("DB_PORT", "5432")
    )

    cur = conn.cursor()
    cur.execute("SELECT file_name FROM pdf_documents WHERE is_vectorized = FALSE")
    files = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()

    return files

def update_db_status(file_name, status="completed"):
    """처리가 완료된 파일의 상태를 DB에 업데이트."""

    raw_password = urllib.parse.unquote(os.getenv("DB_PASSWORD"))

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=raw_password,
        port=os.getenv("DB_PORT", "5432")
    )

    cur = conn.cursor()
    cur.execute(
        "UPDATE pdf_documents SET is_vectorized = TRUE, status = %s WHERE file_name = %s",
        (status, file_name)
    )

    conn.commit()
    cur.close()
    conn.close()

def process_file(file_name, bucket_name, vector_store):
    """GCS에서 다운로드 후 텍스트 분할 및 벡터 저장을 수행"""
    local_path = Path(f"./temp_{file_name}")

    try:
        key_path = os.getenv("GCS_KEY_PATH")
        if not key_path:
            raise ValueError("GCS_KEY_PATH 가 .env에 설정되지 않음")

        # GCS 다운로드
        storage_client = storage.Client.from_service_account_json(key_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.download_to_filename(str(local_path))
        logging.info(f"📥 '{file_name}' 다운로드 완료")

        # 문서 로드 및 분할
        loader = PyMuPDFLoader(str(local_path))
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True
        )

        chunks = text_splitter.split_documents(pages)

        # 메타데이터 주입 (출처 추적용)
        for chunk in chunks:
            chunk.metadata["source"] = file_name

        # PGVectorStore 저장
        vector_store.add_documents(chunks)
        logging.info(f"✨ '{file_name}' 벡터 DB 주입 완료 ({len(chunks)} 청크)")

        return True

    except Exception as e:
        logging.error(f"❌ '{file_name}' 처리 중 오류: {e}")
        return False
    finally:
        if local_path.exists():
            local_path.unlink()

# --- 4. 메인 실행 구조 ---
def main():
    try:
        connection_string = load_env_config()
        pending_files = get_pending_files()

        if not pending_files:
            logging.info("💡 처리할 새로운 파일이 없습니다. 종료합니다.")
            return
        
        logging.info(f"🚀 총 {len(pending_files)}개의 파일 처리를 시작합니다.")

        async_engine = create_engine(connection_string.replace("postgresql+psycopg", "postgresql+psycopg2"))

        # PGVectorStore 초기화
        vector_store = PGVector(
            connection=async_engine,
            embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
            collection_name="accident_vectors",
            use_jsonb=True
        )

        # 버킷명 설정 
        bucket_name = "pdf-storage-2026"

        for file_name in pending_files:
            if process_file(file_name, bucket_name, vector_store):
                update_db_status(file_name)
                logging.info(f"✅ DB 상태 갱신 완료: {file_name}")
        
        logging.info("-" * 30)
        logging.info("🎉 모든 벡터화 작업이 성공적으로 끝났습니다!")
        logging.info("-" * 30)

    except Exception as e:
        logging.error(f"⚠️ 시스템 오류로 중단됨: {e}")

if __name__ == "__main__":
    main()