import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

# 1. 환경 변수 로드 (.env 파일 읽기)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")

def run_ingestion():
    # 2. 파일 경로 설정 (data 폴더 명시)
    # 현재 파일(ragTest.py)의 절대 경로를 가져옵니다.
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent

    file_path = project_root / "data" / "P02_01_01_001_20210101.pdf"

    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"❌ 에러: '{file_path}' 파일을 찾을 수 없습니다.")
        print("💡 프로젝트 내에 'data' 폴더를 만들고 PDF 파일을 넣어주세요.")
        return
    
    print(f"✅ '{file_path}' 파일을 찾았습니다. 문서 처리 중...")

    # 3. PDF 문서 로드 및 파싱
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    print(f"✅ 로드 완료: 총 {len(pages)} 페이지")

    # 4. 청킹(Chunking) - 조항 단위 문맥 보존을 위해 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(pages)
    print(f"✅ 청킹 완료: 총 {len(chunks)} 청크 생성")

    # 5. 임베딩 및 벡터 DB 저장
    print("임베딩 생성 및 FAISS 벡터 DB 저장 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # FAISS 데이터베이스 생성
    vector_db = FAISS.from_documents(chunks, embeddings)

    # 6. 로컬에 저장
    save_path = project_root / "vectorDB" / "faiss_index_samsung_fire"
    vector_db.save_local(save_path)

    print("-" * 30)
    print(f"🎉 모든 작업이 완료되었습니다!")
    print(f"📂 벡터 DB 저장 위치: {os.path.abspath(save_path)}")
    print("-" * 30)

    # 7. 내용 확인하기
    vector_db = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    print(f"전체 벡터 개수: {vector_db.index.ntotal}")

    # 8. 저장된 원본 문서 내용 확인
    # 파일이 커서 일부만 확인
    docstore_dict = vector_db.docstore._dict
    for i, (key, doc) in enumerate(docstore_dict.items()):
        print(f"\n--- 문서 {i+1} ---")
        print(f"Content: {doc.page_content[:100]}...")  # 앞 100자만 출력
        print(f"Metadata: {doc.metadata}")
        if i >= 5:  # 처음 5개 문서만 출력
            break

if __name__ == "__main__":
    run_ingestion()