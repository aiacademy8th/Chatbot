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

class AccidentRAGEngine:
    def __init__(self, similarity_threshold=0.7):
        """
        기본의 초기화 로직 및 setuo_rag_chain의 기능을 클래스 생성 시 수행합니다.
        """
        load_dotenv()
        self.threshold = similarity_threshold

        # 1. 벡터 스토어 로드 
        self.vector_store = self._get_vector_store()

        # 2. LLM 설정
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 프롬프트와 체인을 분리하여 구성
        self.rag_chain = self._initialize_rag_chain()

        logging.info(f"AccidentRAGEngine 초기화 완료 (임계값: {self.threshold})")

    def _get_vector_store(self):
        """DB 연결 및 벡터 스토어 인스턴스 생성"""
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME")

        connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        engine = create_engine(connection_string)

        return PGVector(
            connection=engine,
            embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
            collection_name="accident_vectors",
            use_jsonb=True
        )

    def _get_relevant_docs(self, query):
        """get_relevant_docs 로직: 유사도 기반 필터링 및 상위 3개 추출"""
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=10)

        relevant_docs = []
        for doc, score in docs_with_scores:
            similarity = 1 - (score / 2.0)
            if similarity >= self.threshold:
                relevant_docs.append((doc, similarity))
                logging.info(f"문서 발견 - 유사도: {similarity:.3f}")

        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        return relevant_docs[:3]

    def _format_docs_for_synthesis(self, docs_with_scores):
        """format_docs_for_synthesis 로직: LLM 전달용 텍스트 변환"""
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

    def _initialize_rag_chain(self):
        """LCEL 체인 구성 로직"""
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

        검색된 문서들:
        {context}

        질문: {question}

        답변:"""

        self.prompt = ChatPromptTemplate.from_template(template)

        # 미리 검색된 context를 받는 체인
        return (
            RunnableParallel(
                context=lambda x: x["context"],
                question=lambda x: x["question"]
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, query):
        """
        UI에서 호출할 최종 인터페이스
        답변과 참고한 문서 리스트를 동시에 반환
        """

        # 먼저 문서 검색 (유사도 체크를 위해)
        relevant_docs = self._get_relevant_docs(query)

        if not relevant_docs:
            return "관련된 정보를 찾을 수 없습니다. (유사도 임계값 미달)", []
        
        context = self._format_docs_for_synthesis(relevant_docs)

        # 체인 실행
        answer = self.rag_chain.invoke({"context": context, "question": query})
        return answer, relevant_docs
