import os
import psycopg2
from dotenv import load_dotenv
from google.cloud import storage
import urllib.parse

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": urllib.parse.unquote(os.getenv("DB_PASSWORD")),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", "5432"),
}

BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def sync_gcs_to_db():
    """
    GCS PDF 업로드/삭제 시 DB 자동 동기화
    context.event_type을 사용하여 생성과 삭제를 구분
    """

    print(f"--- 동기화 시작: 버킷 '{BUCKET_NAME}' ---")
    
    key_path = os.getenv("GCS_KEY_PATH")
    # [디버깅용] 경로가 제대로 출력되는지 꼭 확인하세요!
    print(f"디버그 - 로드된 키 경로: {key_path}")


    if not key_path or not os.path.exists(key_path):
        print("🚨 오류: .env 파일의 GCS_KEY_PATH가 잘못되었거나 파일을 찾을 수 없습니다.")
        return

    # GCS 클라이언트 초기화
    storage_client = storage.Client.from_service_account_json(key_path)

    print(f"✅ 사용 중인 계정: {storage_client.get_service_account_email()}")

    bucket = storage_client.bucket(BUCKET_NAME)

    # GCS 버킷 내의 모든 PDF 파일 목록 가져오기
    blobs = storage_client.list_blobs(BUCKET_NAME)
    gcs_files = {blob.name: blob for blob in blobs if blob.name.lower().endswith(".pdf")}
    gcs_file_names = set(gcs_files.keys())
    
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # DB에 등록된 파일 목록 가져오기
        cur.execute("SELECT file_name FROM pdf_documents")
        db_file_names = set(row[0] for row in cur.fetchall())

        # [CASE 1] GCS에는 없는 데 DB에는 있는 파일 -> 삭제
        files_to_delete = db_file_names - gcs_file_names
        for file_name in files_to_delete:
            cur.execute("DELETE FROM pdf_documents WHERE file_name = %s", (file_name,))
            print(f"DB에서 삭제됨: {file_name}")
        
        # [CASE 2] GCS에 있는 파일들 -> 추가 또는 업데이트
        for file_name in gcs_file_names:
            gcs_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{file_name}"
            insert_query = """
                INSERT INTO pdf_documents (file_name, gcs_url, is_vectorized, status)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (file_name) DO UPDATE 
                SET gcs_url = EXCLUDED.gcs_url, status = 'pending';
            """

            # 새로 추가되는 파일만 
            cur.execute(insert_query, (file_name, gcs_url, False, "pending"))
            conn.commit()
            print(f"Successfully resistered/updated: {file_name}")
            
    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    # 스크립트 단독 실행 시 호출
    sync_gcs_to_db()