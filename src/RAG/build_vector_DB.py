import os
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. 로깅 설정 (개선된 부분) ---
# print() 대신 표준 로깅 모듈을 사용하여 로그의 레벨 관리와 포맷팅을 체계화합니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_api_key():
    """환경 변수에서 OpenAI API 키를 로드합니다."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        raise ValueError("API 키가 없습니다.")
    return api_key

def load_and_split_documents(file_path: Path) -> list:
    """PDF 문서를 로드하고 텍스트를 청크로 분할합니다."""
    if not file_path.exists():
        logging.error(f"'{file_path}' 파일을 찾을 수 없습니다.")
        raise FileNotFoundError(f"지정된 경로에 파일이 없습니다: {file_path}")
    
    logging.info(f"'{file_path}' 파일 로드를 시작합니다...")
    loader = PyMuPDFLoader(str(file_path))
    pages = loader.load()
    logging.info(f"로드 완료: 총 {len(pages)} 페이지")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(pages)
    logging.info(f"청킹 완료: 총 {len(chunks)} 청크 생성")
    return chunks

def create_and_save_vector_db(chunks: list, save_path: Path):
    """임베딩을 생성하고 벡터 DB를 로컬에 저장합니다."""
    logging.info("임베딩 생성 및 FAISS 벡터 DB 저장 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # 저장 경로가 존재하지 않으면 생성
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vector_db.save_local(str(save_path))
    logging.info(f"벡터 DB를 '{save_path}'에 성공적으로 저장했습니다.")
    return embeddings, save_path

def verify_db(save_path: Path, embeddings):
    """저장된 벡터 DB를 검증합니다."""
    logging.info("저장된 벡터 DB 검증 시작...")
    vector_db = FAISS.load_local(str(save_path), embeddings, allow_dangerous_deserialization=True)
    logging.info(f"전체 벡터 개수: {vector_db.index.ntotal}")
    
    # 일부 문서 내용 확인
    try:
        docstore_dict = vector_db.docstore._dict
        logging.info("--- 저장된 문서 샘플 (상위 5개) ---")
        for i, (key, doc) in enumerate(docstore_dict.items()):
            logging.info(f"문서 {i+1}: {doc.page_content[:100]}...")
            if i >= 4:
                break
    except Exception as e:
        logging.warning(f"문서 샘플 확인 중 오류 발생: {e}")

# --- 2. 커맨드라인 인자 처리 (개선된 부분) ---
# argparse를 사용하여 파일 경로를 하드코딩하는 대신,
# 스크립트 실행 시 동적으로 지정할 수 있도록 하여 재사용성을 높입니다.
def parse_arguments():
    """스크립트 실행을 위한 커맨드라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="PDF 문서를 처리하여 FAISS 벡터 데이터베이스를 생성합니다.")
    
    # 현재 파일 위치를 기준으로 기본 경로 설정
    project_root = Path(__file__).resolve().parent.parent.parent
    default_input = project_root / "data" / "P02_01_01_001_20210101.pdf"
    default_output = project_root / "vectorDB" / "faiss_index_samsung_fire"

    parser.add_argument(
        "--input",
        type=str,
        default=str(default_input),
        help="처리할 PDF 파일의 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help="생성된 벡터 DB를 저장할 경로"
    )
    return parser.parse_args()

# --- 3. 메인 로직 구조화 (개선된 부분) ---
# 각 기능(API 키 로드, 문서 처리, DB 생성, 검증)을 별도의 함수로 분리하여
# 코드의 가독성과 유지보수성을 향상시킵니다.
def main():
    """메인 실행 함수"""
    args = parse_arguments()
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        load_api_key()
        chunks = load_and_split_documents(input_path)
        embeddings, saved_path = create_and_save_vector_db(chunks, output_path)
        verify_db(saved_path, embeddings)
        
        logging.info("-" * 30)
        logging.info("🎉 모든 작업이 완료되었습니다!")
        logging.info(f"📂 벡터 DB 저장 위치: {saved_path.resolve()}")
        logging.info("-" * 30)

    except (ValueError, FileNotFoundError) as e:
        logging.error(f"작업 실패: {e}")
    except Exception as e:
        logging.error(f"예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()