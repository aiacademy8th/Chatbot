import os
import base64
import asyncio
from io import BytesIO
from typing import List, Optional
from PIL import Image
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from sqlalchemy import create_engine
from pydantic import BaseModel, Field       # 구조화된 출력을 위한 라이브러리

# --- [구조화된 출력 데이터 모델] ---
class FaultRatio(BaseModel):
    me: int = Field(description="나(사용자)의 과실 비율 (0~100)")
    opponent: int = Field(description="상대방의 과실 비율 (0~100)")

class AccidentAnalysisResult(BaseModel):
    summary: str = Field(description="사고 상황을 3문장 이내로 명확하게 요약")
    fault_ratio: FaultRatio = Field(description="판례와 상황에 기반한 추정 과실 비율")
    legal_basis: List[str] = Field(description="판단의 근거가 된 관련 법규 또는 판례 제목 리스트 (최대 3개)")
    advice: str = Field(description="운전자에 대한 조언")
    reasoning: str = Field(description="과실 비율을 산정한 논리적 근거 상세 설명")

# --- RAG 엔진 클래스
class AccidentRAGEngine:
    def __init__(self):
        load_dotenv()

        # 벡터 스토어 초기화
        self.vector_store = self._get_vector_store()

        # [속도 최적화] 모델 이원화 전략
        # 1. 빠른 모델: 단순 요약, 키워드 추출용
        self.llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 2. 고성능 모델: 비전 분석, 최종 법률 판단용
        self.llm_smart = ChatOpenAI(model="gpt-4o", temperature=0)

    def _get_vector_store(self):
        """PostgreSQL(PGVector) 연결 설정"""
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
    
    def _encode_image(self, image_path: str) -> str:
        """
        이미지 경로를 받아 리사이징 및 압축 후 Base64 문자열 반환
        [최적화] 이미지 크기를 줄여 토큰 비용 절감 및 속도 향상
        """

        try:
            with Image.open(image_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # [속도 최적화] 최대 800px로 리사이징(GPT-4o Vision 인식에 충분)
                img.thumbnail((800, 800))

                buffered = BytesIO()
                # 품질을 70% 로 압축
                img.save(buffered, format="JPEG", quality=70)
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
            
        except Exception as e:
            print(f"⚠️ 이미지 인코딩 실패 ({image_path}): {e}")
            return ""
        
    # --- [Tool 1] 이미지 분석 (Vision Worker용) ---
    async def analyze_images_from_paths(self, image_paths: list) -> str:
        """이미지 파일들을 분석하여 객관적인 상황 묘사 텍스트 반환"""
        if not image_paths:
            return "분석할 이미지가 없습니다."
        
        # [프롬프트 강화] Final Solver가 이미지를 보지 않으므로
        # 여기서 법적 판단에 필요한 모든 팩트를 텍스트 해야 함
        content = [{
            "type": "text",
            "text": """
            교통사고 전문 조사관의 관점에서 사진을 정밀 분석하세요.
            최종 과실 비율 판단을 위해 다음 핵심 요소를 반드시 포함하여 서술하세요:

            1. [차선 정보] 실선/점선 여부, 중앙선 침범 여부, 교차로 내 진입 변경 금지 구간 여부
            2. [신호 상태] 신호등 색상 및 점등 위치 (식별 분가능 하면 '식별 불가' 명시)
            3. [충돌 부위] 내 차와 상대차의 정확한 충돌 지점 (예: 내 차 우측 휀더, 상대 차 좌측 범퍼)
            4. [노면/환경] 빗길 미끄러짐, 포트홀, 공사 구간 표지판 등 특이사항
            """
        }]

        valid_images = 0
        for path in image_paths:
            b64_str = self._encode_image(path)
            if b64_str:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
                })
                valid_images += 1

        if valid_images == 0:
            return "이미지 파일 처리에 실패했습니다."
        
        msg = HumanMessage(content=content)

        # [속도 최적화] max_tokens 제한으로 빠른 답변 유도 (상세 묘사를 위해 700개 지정)
        response = await self.llm_smart.ainvoke([msg], config={"max_tokens": 700})
        return response.content
    
    # --- [Tool 2] 법률 검색 (Search Worker용) ---
    async def search_legal_docs(self, query: str) -> str:
        """쿼리와 유사한 판례 검색 (비동기 처리)"""

        # [최적화] DB I/O 바운드 작업이므로 별도 스레드에서 실행
        try:
            docs_with_scores = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score, query, k=3
            )

            results = []
            
            for i, (doc, score) in enumerate(docs_with_scores):
                # 거리(distance)를 유사도(similarity)로 변환 (Cosine distance 가정)
                # score 가 낮을 수록 유사항 (0에 가까움), 라이브러리 버전에 따라 다를 수 있음
                # 여기서는 안전하게 score 값 자체를 로깅하며 필터링

                src = doc.metadata.get("source", "판례 DB")
                content = doc.page_content.replace("\n", " ").strip()[:400] # 너무 길면 자름
                results.append(f"[{i + 1}] (출처:{src}) {content}...")

            return "\n\n".join(results) if results else "검색된 관련 판례가 없습니다."
        
        except Exception as e:
            return f"판례 검색 중 시스템 오류 발생: {str(e)}"
        
    # --- [Tool 3] 최종 솔루션 생성 (Final Solver 용 - 구조화된 출력) ---
    async def generate_final_solution_structured(self, messages: list, image_paths: list) -> AccidentAnalysisResult:
        """
        대화 기록 (Context)과 텍스트 요약본을 종합하여 최종 답변 생성
        [비용 절감] 이미지는 여기서 사용하지 않고 텍스트만 처리함
        """

        # 1. 구조화된 출력을 위한 LLM 설정
        structured_llm = self.llm_smart.with_structured_output(AccidentAnalysisResult)

        # 2. 시스템 프롬프트 (페르소나 및 제약조건)
        system_prompt_text = """
        당신은 교통사고 전문 AI 변호사입니다.
        제공된 [Vision 분석 결과]와 [법률 판례]를 바탕으로 과실 비율을 판단하세요.
        
        [지침]
        1. 당신의 판단은 '참고용'임을 명심하십시오
        2. Vision AI가 분석한 텍스트 묘사를 사실로 가정하고 판단하십시오.
        3. 판단이 불확실할 경우 과실 비율을 50:50으로 설정하고 이유를 성명하십시오.
        """

        # 3. 프롬프트 구성 (텍스트 전용)
        # [비용 절감] Graph에서 image_paths=[] (빈 리스트)를 보내므로 이미지는 포함되지 않음
        prompt_content = [{"type": "text", "text": system_prompt_text}]

        for path in image_paths:
            b64_str = self._encode_image(path)
            if b64_str:
                prompt_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
                })

        # --- [메시지 다이어트: 토큰 관리] ---
        # (1) 가장 최신의 HumanMessage(현재 질문) 확보
        current_query = messages[-1] if messages else None

        # (2) 과거 히스토리 Trimming
        trimmed_history = trim_messages(
            messages[:-1],                 # 현재 메시지 제외한 과거 기록만
            max_tokens=6000,
            strategy="last",
            token_counter=self.llm_smart,   # 모델의 토크나이저 사용
            include_system=False,
            allow_partial=False
        )

        # (3) 최종 메시지 조립
        final_messages = [HumanMessage(content=prompt_content)] + trimmed_history + ([current_query] if current_query else [])

        # 4. 실행 및 결과 반환
        try:
            response = await structured_llm.ainvoke(final_messages)
            return response
        except Exception as e:
            # 실패 시 기본값 반환 (Fallback)
            print(f"❌ 구조화된 출력 생성 실패: {e}")
            return AccidentAnalysisResult(
                summary="분석 중 오류가 발생했습니다.",
                fault_ratio=FaultRatio(me=0, opponent=0),
                legal_basis=[],
                advice="시스템 오류로 인해 상세 분석을 완료하지 못했습니다. 다시 시도해주세요.",
                reasoning=str(e)
            )
