import os
import logging
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from sqlalchemy import create_engine

# readline 임포트 (한글 입력 및 백스페이스 지원)
try:
    import readline
except ImportError:
    # Window에서는 pyreadline3 사용
    try:
        import pyreadline3 as readline
    except ImportError:
        readline = None
        print("⚠️  readline을 사용할 수 없습니다. 한글 입력 시 백스페이스가 제대로 작동하지 않을 수 있습니다.")
        print("   해결 방법: pip install pyreadline3 (Windows) 또는 readline은 Linux/Mac에 기본 설치됨")

# 로깅 설정
logging.basicConfig(level=logging.INFO)

load_dotenv()

def get_vector_store():
    """벡터 스토어를 로드하는 함수"""

    # .env 설정 로드
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")

    # DB 연결 문자열 (psycopg2 사용)
    connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(connection_string)

    # 기존에 생성된 벡터 스토어 로드
    vector_store = PGVector(
        connection=engine,
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="accident_vectors",
        use_jsonb=True
    )

    return vector_store

def get_relevant_docs(vector_store, query, similarity_threshold=0.7):
    """
    유사도 점수를 기반으로 가장 관련성 높은 문서만 검색
    
    Args:
        vector_store: PGVector 벡터 스토어
        query: 검색 질의
        similarity_threshold: 유사도 임계값 (0.0 ~ 1.0)
    
    Returns:
        관련성 높은 문서 리스트와 유사도 점수
    """
    # similarity_search_with_score로 유사도 점수와 함께 검색
    docs_with_scores = vector_store.similarity_search_with_score(query, k=10)
    
    # 유사도가 임계값 이상인 문서만 필터링
    relevant_docs = []
    for doc, score in docs_with_scores:
        # 거리를 유사도로 변환 (거리가 작을수록 유사도 높음)
        similarity = 1 - (score / 2.0)  # 정규화
        
        if similarity >= similarity_threshold:
            relevant_docs.append((doc, similarity))
            logging.info(f"관련 문서 발견 - 유사도: {similarity:.3f}, 출처: {doc.metadata.get('source', '알 수 없음')}")
    
    # 유사도 순으로 정렬 (높은 순)
    relevant_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 3개만 사용 (가장 관련성 높은 문서)
    return relevant_docs[:3]

def format_docs_for_synthesis(docs_with_scores):
    """
    상위 3개 문서를 LLM이 통합하여 답변할 수 있도록 포맷팅
    각 문서의 전체 내용과 유사도를 명시
    """

    if not docs_with_scores:
        return "관련 문서를 찾을 수 없습니다."
    
    formatted_parts = []
    for idx, (doc, score) in enumerate(docs_with_scores, 1):
        source = doc.metadata.get("source", "알 수 없음")

        formatted_parts.append(
            f"=== 문서 {idx} (관련도: {score:.1%}, 출처: {source}) ===\n"
            f"{doc.page_content}\n"
        )

    return "\n\n".join(formatted_parts)

def retrieve_with_scores(vector_store, similarity_threshold=0.7):
    """유사도 기반 검색 함수를 반환"""
    def retriever_func(query):
        docs_with_scores = get_relevant_docs(vector_store, query, similarity_threshold)
        if not docs_with_scores:
            logging.warning(f"질의 '{query}'에 대한 관련 문서를 찾을 수 없습니다.")
        return docs_with_scores
    return retriever_func

def setup_rag_chain(vector_store, similarity_threshold=0.7):
    """
    유사도 기반 RAG 체인 설정

    Args:
        vector_store: PGVector 벡터 스토어
        similatiry_threshold: 유사도 임계값 (0.0 ~ 1.0, 기본값: 0.7)
    """

    # 1. LLM 설정
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 2. 유사도 기반 Retriever 생성
    retriever_func = retrieve_with_scores(vector_store, similarity_threshold)

    # 3. 프롬프트 템플릿 - 문서 통합 지시
    template = """당신은 교총 사고 대응 전문 AI 어시스텐트입니다.

아래에 질문과 가장 관련성이 높은 상위 3개의 문서가 제공됩니다.
각 문서에는 관련도(유사도 점수)가 표시되어 있습니다.

**답변 작성 지침:**
1. 제공된 모든 문서의 내용은 꼼꼼히 검토하세요.
2. 관련도가 높은 문서의 내용을 우선적으로 활용하되, 모든 문서의 정보를 종합하세요
3. 여러 문서에서 나온 정보를 자연스럽게 통합하여 하나의 일관된 답변을 작성하세요
4. 문서들 간에 내용이 중복되거나 보완적인 경우, 가장 완전하고 정확한 정보를 제공하세요.
5. 문서에 명확한 답변이 없다면 "제공된 문서에서 관련 내용을 찾을 수 없습니다"라고 답하세요
6. 문서에 없는 내용을 추측하거나 만들어내지 마세요
7. 답변은 정중하고 신뢰감 있는 말투로 작성하세요
8. 가능한 경우 단계별로 설명하거나 구조화하여 답변하세요

검색된 문서들:
{context}

질문: {question}

답변:"""

    prompt = ChatPromptTemplate.from_template(template)

    # 4. LCEL 체인 구성
    rag_chain = (
        RunnableParallel(
            context=lambda x: format_docs_for_synthesis(retriever_func(x)),
            question=RunnablePassthrough()
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever_func

def get_user_input(prompt_text):
    """
    한글 입력을 안정적으로 받기 위한 함수
    readline을 사용하여 백스페이스 지원
    """
    
    try:
        # UTF-8 인코딩 강제 설정
        if sys.stdout.encoding != "utf-8":
            import codecs
            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
            sys.stdin = codecs.getreader("utf-8")(sys.stdin.buffer, "strict")

        user_input = input(prompt_text).strip()
        return user_input
    except (EOFError, KeyboardInterrupt):
        return "q"
    except Exception as e:
        logging.error(f"입력 오류: {e}")
        return ""
    
def main():
    """메인 실행 함수"""

    # 유사도 임계값 설정 (0.0 ~ 1.0)
    SIMILARITY_THRESHOLD = 0.7

    try:
        # 벡터 스토어 및 RAG 체인 초기화
        print("\n🔧 시스템을 초기화하는 중...")
        vector_store = get_vector_store()
        rag_chain, retriever_func = setup_rag_chain(vector_store, SIMILARITY_THRESHOLD)

        print("\n" + "=" * 60)
        print("🚗 교통사고 대응 지식봇이 준비되었습니다!")
        print(f"   📊 유사도 임계값: {SIMILARITY_THRESHOLD:.1%}")
        print("   💡 상위 3개 문서를 조합하여 통합 답변을 제공합니다")
        print("   종료하려면 'q' 또는 'quit'를 입력하세요.")
        print("=" * 60)

        # readline 상태 확인
        if readline:
            print("   ✅ 한글 입력 지원: readline 활성화됨")
        else:
            print("   ⚠️  한글 입력 시 백스페이스가 제대로 작동하지 않을 수 있습니다")

        while True:
            # 개선된 입력 함수 사용
            query = get_user_input("\n💬 질문하세요: ")

            # 종료 명령 확인
            if query.lower() in ['q', 'quit', 'exit']:
                print("\n👋 프로그램을 종료합니다. 감사합니다!")
                break

            # 빈 입력 체크
            if not query:
                print("⚠️  질문을 입력해주세요.")
                continue

            try:
                print("\n🔍 가장 관련성 높은 문서를 검색 중입니다...")

                # 관련 문서 먼저 확인
                relevant_docs = retriever_func(query)

                if not relevant_docs:
                    print("\n" + "=" * 60)
                    print("⚠️  관련성 높은 문서를 찾을 수 없습니다.")
                    print("=" * 60)
                    print(f"유사도 {SIMILARITY_THRESHOLD:.1%} 이상인 문서가 없습니다.")
                    print("다른 질문을 시도해보세요.")
                    continue

                print(f"✅ {len(relevant_docs)}개의 관련 문서를 찾았습니다.")
                print("🔄 문서 내용을 통합하여 답변을 생성 중입니다...")

                # 답변 생성 (여러 문서 통합)
                answer = rag_chain.invoke(query)

                # 답변 출력
                print(f"\n{'='*60}")
                print("🤖 통합 답변:")
                print(f"{'='*60}")
                print(answer)

                # 참고한 문서 목록 출력
                print(f"\n{'=' * 60}")
                print("📚 참고 문서 (유사도 순):")
                print(f"{'-' * 60}")

                for idx, (doc, score) in enumerate(relevant_docs, 1):
                    source = doc.metadata.get("source", "알 수 없음")
                    print(f"{idx}. [{score:.1%} 관련도] {source}")

                print(f"\n💡 위 {len(relevant_docs)}개 문서의 내용이 통합되어 답변되었습니다.")

            except Exception as e:
                logging.error(f"질의 처리 중 오류 발생: {e}", exc_info=True)
                print(f"\n❌ 오류가 발생했습니다: {str(e)}")
                print("다시 시도해주세요.")
    except Exception as e:
        logging.error(f"초기화 중 오류 발생: {e}", exc_info=True)
        print(f"\n❌ 시스템 초기화 중 오류가 발생했습니다.")
        print(f"오류 내용: {str(e)}")
        print("\n다음 사항을 확인해주세요:")
        print("1. .env 파일에 데이터베이스 연결 정보가 올바르게 설정되어 있는지")
        print("2. PostgreSQL 데이터베이스가 실행 중인지")
        print("3. OPENAI_API_KEY가 .env 파일에 설정되어 있는지")
        print("4. pgvector 확장이 데이터베이스에 설치되어 있는지")

if __name__ == "__main__":
    main()