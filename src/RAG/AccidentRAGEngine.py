import os
import logging
import base64
from io import BytesIO
from PIL import Image  # 이미지 처리를 위해 추가 (pip install Pillow)
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine

# 로깅 설정
logging.basicConfig(level=logging.INFO)

class AccidentRAGEngine:
    def __init__(self, similarity_threshold=0.7):
        load_dotenv()
        self.threshold = similarity_threshold
        self.vector_store = self._get_vector_store()
        
        # [최적화] 온도 0, max_tokens 설정으로 응답 속도 및 일관성 확보
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vision_parser = StrOutputParser()
        logging.info(f"AccidentRAGEngine(Multimodal) 초기화 완료")

    def _get_vector_store(self):
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

    def _encode_image(self, image_path):
        """
        [최적화] 이미지 리사이징 후 Base64 인코딩
        원본 이미지가 너무 크면 전송/분석이 느려지므로 최대 크기(1024px)로 조정합니다.
        """
        try:
            with Image.open(image_path) as img:
                # RGB로 변환 (PNG 투명도 문제 방지)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # 이미지 리사이징 (최대 1024x1024 유지)
                img.thumbnail((1024, 1024))
                
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85) # JPEG 압축
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logging.error(f"이미지 전처리 실패: {e}")
            # 실패 시 원본 그대로 시도
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        
    def _analyze_image_for_search(self, image_base64):
        """
        [최적화] 검색용 키워드 추출 (간결하게 요청)
        """
        prompt = [
            SystemMessage(content="교통사고 분석가입니다. 핵심 키워드만 간결하게 나열하세요."),
            HumanMessage(content=[
                {"type": "text", "text": "차량 위치, 파손 부위, 신호등, 차선 등 사고 과실 판단에 필요한 핵심 요소만 3줄 이내로 요약해."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "low"}} # [최적화] detail="low"로 설정하여 토큰 절약
            ])
        ]
        # max_tokens를 제한하여 빠른 응답 유도
        return self.llm.invoke(prompt, config={"max_tokens": 150}).content

    def _get_relevant_docs(self, query):
        # [최적화] 검색 개수(k)를 5개로 줄여서 DB 부하 감소
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=5)
        relevant_docs = []
        for doc, score in docs_with_scores:
            similarity = 1 - (score / 2.0)
            if similarity >= self.threshold:
                relevant_docs.append((doc, similarity)) # 튜플 형태로 저장
        
        # 상위 3개만 반환
        return relevant_docs[:3]

    def _format_docs_for_synthesis(self, docs_with_scores):
        if not docs_with_scores: return "관련 문서를 찾을 수 없습니다."
        
        formatted_parts = []
        for idx, (doc, score) in enumerate(docs_with_scores, 1):
            source = doc.metadata.get("source", "알 수 없음")
            # [최적화] LLM에 들어갈 텍스트 양을 줄이기 위해 page_content 일부만 사용하거나 요약할 수 있음
            # 현재는 전체 사용하되 로그만 남김
            formatted_parts.append(
                f"=== 문서 {idx} (출처: {source}) ===\n{doc.page_content}\n"
            )
        return "\n\n".join(formatted_parts)

    def ask(self, query: str, image_path: str = None):
        search_query = query
        image_base64 = None
        image_description = ""

        # 1. 이미지 처리 (이미지가 있을 때만 수행)
        if image_path and os.path.exists(image_path):
            try:
                logging.info(f"이미지 처리 시작: {image_path}")
                image_base64 = self._encode_image(image_path)
                
                # 1-1. 검색용 텍스트 생성 (1차 LLM 호출 - 병목 지점)
                image_description = self._analyze_image_for_search(image_base64)
                search_query = f"상황: {image_description}\n질문: {query}"
                
            except Exception as e:
                logging.error(f"이미지 처리 오류: {e}")

        # 2. 문서 검색
        logging.info("RAG 문서 검색 시작...")
        relevant_docs = self._get_relevant_docs(search_query)
        context = self._format_docs_for_synthesis(relevant_docs)

        # 3. 최종 답변 생성
        system_prompt = f"""당신은 교통사고 판단 보조 AI입니다.
        [법률 정보] {context}
        
        위 정보를 바탕으로 핵심만 간결하게 답변하세요.
        """

        messages = [SystemMessage(content=system_prompt)]
        user_content = [{"type": "text", "text": query}]
        
        if image_description:
            user_content.append({"type": "text", "text": f"(이미지 분석 요약: {image_description})"})
        
        if image_base64:
            # [최적화] detail="auto" 또는 "low" 사용
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "auto"}})

        messages.append(HumanMessage(content=user_content))
        
        logging.info("최종 답변 생성 중...")
        # 스트리밍을 사용하지 않는다면 invoke 사용
        answer = self.llm.invoke(messages).content
        
        return answer, relevant_docs