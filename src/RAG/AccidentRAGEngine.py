import os
import logging
import base64
import asyncio
from io import BytesIO
from PIL import Image       # 이미지 처리를 위해 추가
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

        # temperature 0
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        logging.info(f"AccidentRAGEngine 초기화 완료")

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
            collection_name="accident_vectors", # [수정] 콤마 추가
            use_jsonb=True
        )
    
    def _encode_image(self, image_path):
        """
        최적화를 위해 이미지 리사이징 (Standard Method)
        원본이 클 경우 전송 속도 저하의 주원인이 되므로 1024px로 제한합니다.
        """
        try:
            with Image.open(image_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")        # JPG 변환을 위해 모드 변경

                # 비율 유지하며 최대 1024x1024 로 축소
                img.thumbnail((1024, 1024))

                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85)   # 압축률 85%
                return base64.b64encode(buffered.getvalue()).decode("utf-8")

        except Exception as e:
            logging.error(f"이미지 리사이징 실패, 원본 사용: {e}")
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
            
    async def analyze_image_async(self, image_base64_list):
        """
        최적화를 위해 비동기 이미지 분석 & 토큰 제한
        여러장의 이미지를 동시에 분석
        """

        # 기본 텍스트 메시지
        content_list = [
            {"type": "text", "text": "차량 위치, 파손 부위 신호등, 차선 등 과실 판단 핵심 요소 5줄 요약"}
        ]

        # 이미지 개수만큼 반복해서 추가
        for img_b64 in image_base64_list:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"}
            })

        prompt = [
            SystemMessage(content="교통사고 분석가 입니다. 여러 장의 사진을 종합하여 핵심 정황을 요약하세요."),
            HumanMessage(content=content_list)
        ]

        # ainvoke 사용 (비동기 호출) + max_tokens 제한
        response = await self.llm.ainvoke(prompt, config={"max_tokens": 400})
        return response.content
    
    def _get_relevant_docs(self, query):
        # 최적화 검색 후보군(k)을 줄여 DB 부하 감소
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=5)
        relevant_docs = []
        for doc, score in docs_with_scores:
            similarity = 1 - (score / 2.0) # [수정] 오타 수정 (similrarity -> similarity)
            if similarity >= self.threshold:
                relevant_docs.append((doc, similarity))

        return relevant_docs[:3]    # 상위 3개만 사용

    def _format_docs_for_synthesis(self, docs_with_scores):
        if not docs_with_scores: 
            return "관련 문서를 찾을 수 없습니다."
        
        formatted_parts = []
        for idx, (doc, score) in enumerate(docs_with_scores, 1):
            source = doc.metadata.get("source", "알 수 없음") # [수정] 오타 수정 (metadate -> metadata)
            formatted_parts.append(f"=== 문서 {idx} (출처: {source}) ===\n{doc.page_content}\n")
        
        return "\n\n".join(formatted_parts)

    async def ask_with_context(self, query: str, image_description: str, image_base64_list: list = None):
        """
        최적화 병렬 처리된 데이터를 받아 최종 답변만 생성하는 메서드
        """

        # 1. 검색 쿼리 구성
        search_query = query
        if image_description:
            search_query = f"상황: {image_description}\n 질문: {query}"

        # 2. 문서 검색 (I/O 작업 이므로 비동기 래핑)
        # PGVector의 similarity_search는 동기 함수이므로 스레드로 분리하여 논블로킹 처리
        relevant_docs = await asyncio.to_thread(self._get_relevant_docs, search_query)
        context = self._format_docs_for_synthesis(relevant_docs)

        # 3. 최종 프롬프트 구성
        system_prompt = f"""당신은 교통사고 판단 보조 AI입니다.
        [검색된 법률 정보]
        {context}

        위 법률 정보와 사용자의 진술/사진을 종합하여 판단하세요.
        """

        messages = [SystemMessage(content=system_prompt)]
        user_content = [{"type": "text", "text": query}]

        if image_description:
            user_content.append({"type": "text", "text": f"(이미지 분석 요약: {image_description})"})

        # 최종 답변 생성 시에도 여러장의 사진 보여주기
        if image_base64_list:
            for img_b64 in image_base64_list:
                user_content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "auto"}
                })

        messages.append(HumanMessage(content=user_content))

        #  4. 최종 답변 생성 (비동기)
        response = await self.llm.ainvoke(messages)

        return response.content, relevant_docs