import os
import base64
import asyncio
import logging
from io import BytesIO
from typing import List, Optional, Literal, Dict, Any
from PIL import Image
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field       # 구조화된 출력을 위한 라이브러리

# 재시도 로직을 위함 tenacity
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- [구조화된 출력 데이터 모델] ---
class FaultRatio(BaseModel):
    me: int = Field(description="나(사용자)의 과실 비율 (0~100)")
    opponent: int = Field(description="상대방의 과실 비율 (0~100)")

class AccidentAnalysisResult(BaseModel):
    summary: str = Field(
        description=(
            "반드시 아래 형식을 따르는 육하원칙 사고 요약문을 작성하세요. "
            "형식: '{사고일시}경 {사고장소}에서 발생한 사고입니다. "
            "A차량({내차량정보})은 {방면}에서 {방향}으로 {차선}에서 {행동}하며 이동 중이었습니다. "
            "B차량({상대차량정보})은 {방면}에서 {방향}으로 {차선}에서 {행동}하며 이동 중이었습니다. "
            "사고는 A차량이 {상세행동} 중 {신호상태}에서 {충돌경위}하여 B차량의 {충돌부위}와 추돌한 것으로 파악됩니다.' "
            "정보가 부족한 항목은 '미상' 또는 '확인 불가'로 표기하되, 전체 서술 흐름을 유지하세요. "
            "과실 비율 분석이나 판례 정보는 이 필드에 포함하지 마세요(별도 필드가 있음)."
        )
    )
    fault_ratio: FaultRatio = Field(description="판례와 상황에 기반한 추정 과실 비율")
    legal_basis: List[str] = Field(description="판단의 근거가 된 관련 법규 또는 판례 제목 리스트 (최대 3개)")
    advice: str = Field(description="운전자에 대한 조언")
    reasoning: str = Field(description="과실 비율을 산정한 논리적 근거 상세 설명")
    # 위험도 등급 대신 '실질적 행동 가이드'로 변경
    action_guide: Literal["개인 합의 유리", "보험 처리 유리", "보험 처리 권장"] = Field(
        description="과실 비율과 사고 상황에 따른 최적의 해결 방안"
    )

# --- [Query Expanson용 데이터 모델] ---
class MetadataFilters(BaseModel):
    accident_type: Optional[str] = Field(None, description="사고 유형")
    has_signal: Optional[bool] = Field(None, description="신호등 유무")
    # 필요한 필드를 미리 정의

class ExpandedQuery(BaseModel):
    """쿼리 확장 결과"""
    legal_terms: List[str] = Field(description="법률 용어로 변환된 키워드 (예: 좌회전 -> 회전교차로 진입)")
    accident_type: str = Field(description="사고 유형 분류 (교차로/차선변경/후방추돌/주차장/기타)")
    key_factors: List[str] = Field(description="과실 판단 핵심 요소 (신호위반/중앙선침범/안전거리미확보 등)")
    expanded_query: str = Field(description="검색 최적화된 재구성 쿼리")
    metadata_filters: MetadataFilters = Field(description="메타데이터 필터링 조건")

# --- RAG 엔진 클래스
class AccidentRAGEngine:
    """
    교통사고 RAG 엔진
    [적용된 사항]
    1. Query Expansion (쿼리 확장)
    2. Metadata Filtering (메타데이터 필터링)
    3. Reranking (재순위화)
    4. 비동기 처리 최적화 (Cohere 블로킹 제거)
    5. 에러 핸들링 강화
    6. 재시도 로직 (tenacity)
    7. 타임아웃 설정 (모든 외부 호출)
    8. Connection Pool 모니터링
    """

    # 클래스 수준에서 리소스 관리 (싱글톤 변수들)
    _engine = None
    _vector_store = None
    _embeddings = None      # 임베딩 인스턴스 추가
    _reranker = None        # Reranker 인스턴스

    def __init__(self):
        load_dotenv()

        # 인스턴스 생성시 싱글톤 초기화 호출
        self.vector_store = self._get_vector_store()

        # [속도 최적화] 모델 이원화 전략
        # 1. 빠른 모델: 단순 요약, 키워드 추출용
        self.llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 2. 고성능 모델: 비전 분석, 최종 법률 판단용
        self.llm_smart = ChatOpenAI(model="gpt-4o", temperature=0)

        # 3. Query Expansion 전용 모델 (구조화된 출력)
        self.llm_expander = self.llm_fast.with_structured_output(
            ExpandedQuery,
            method="function_calling"
        )

    def _get_embeddings(self):
        """OpenAIEmbeddings 싱글톤 인스턴스 반환"""
        if AccidentRAGEngine._embeddings is None:
            AccidentRAGEngine._embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
            logger.info("OpenAIEmbeddings 싱글톤 인스턴스가 생성되었습니다")

        return AccidentRAGEngine._embeddings

    def _get_vector_store(self):
        """PostgreSQL(PGVector) 및 엔진 싱글톤 반환"""
        if AccidentRAGEngine._vector_store is None:
            user = os.getenv("DB_USER")
            password = os.getenv("DB_PASSWORD")
            host = os.getenv("DB_HOST")
            port = os.getenv("DB_PORT", "5432")
            db_name = os.getenv("DB_NAME")

            connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        
            # DB 엔진 싱글톤 초기화, 커넥션 풀 설정 추가
            if AccidentRAGEngine._engine is None:
                AccidentRAGEngine._engine = create_engine(
                    connection_string,
                    pool_size=10,           # 동시 연결 수
                    max_overflow=20,        # 추가 연결 허용
                    pool_pre_ping=True,     # 연결 상태 검증
                    pool_recycle=3600       # 1시간마다 연결 재활용
                )
                logger.info("DB 엔진 및 커넥션 풀이 생성되었습니다.")

            # PGVector 인스턴스 초기화 (재사용되는 엔진과 임베징 전달)
            AccidentRAGEngine._vector_store = PGVector(
                connection=AccidentRAGEngine._engine,
                embeddings=self._get_embeddings(),      # 임베딩 싱글톤 사용
                collection_name="accident_vectors",
                use_jsonb=True
            )
            logger.info("PGVector 싱글톤 인스턴스가 준비되었습니다.")
        
        return AccidentRAGEngine._vector_store
    
    def _get_reranker(self):
        """
        Reranker 초기화 (Cohere 또는 CrossEncoder)
        [옵션 1] Cohere API 사용 (품질 우수)
        [옵션 2] SentenceTransformer (로컬 - 비용 절감)
        """
        if AccidentRAGEngine._reranker is None:
            try:
                # 옵션 1: Cohere 사용 (API 키 필요)
                cohere_api_key = os.getenv("COHERE_API_KEY")
                if cohere_api_key:
                    import cohere
                    AccidentRAGEngine._reranker = cohere.Client(cohere_api_key)
                    logger.info("Cohere Reranker 초기화 완료")
                else:
                    # 옵션 2: CrossEncoder 사용 (로컬)
                    from sentence_transformers import CrossEncoder
                    AccidentRAGEngine._reranker = CrossEncoder(
                        'cross-encoder/ms-marco-MiniLM-L-6-v2',
                        max_length=512
                    )
                    logger.info("CrossEncoder Reranker 초기화 완료 (로컬)")
            except Exception as e:
                logger.warning(f"Reranker 초기화 실패: {e}. Reranking 스킵됩니다.")
                AccidentRAGEngine._reranker = None
        
        return AccidentRAGEngine._reranker
    
    def _encode_image(self, image_path: str) -> str:
        """
        이미지 경로를 받아 리사이징 및 압축 후 Base64 문자열 반환
        [최적화] 이미지 크기를 줄여 토큰 비용 절감 및 속도 향상
        """

        try:
            with Image.open(image_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # [속도 최적화] 최대 768px로 리사이징(GPT-4o Vision 인식에 충분)
                img.thumbnail((768, 768))

                buffered = BytesIO()
                # 품질을 70% 로 압축
                img.save(buffered, format="JPEG", quality=70)
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
            
        except Exception as e:
            logger.warning(f"이미지 인코딩 실패 ({image_path}): {e}")
            return ""
        
    # --- [HIGH PRIORITY 1] Query Expansion ---
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _expand_query(self, original_query: str) -> ExpandedQuery:
        """
        사용자 쿼리를 법률 용어와 메타데이터 필터로 확장
        쿼리 확장 (재시도 + 타임아웃)

        Args: 
            original_query: 원본 사용자 질문
        
        Returns:
            ExpandedQuery: 확장된 쿼리 정보
        """
        
        expansion_prompt = f"""
        당신은 교통사고 법률 전문가입니다. 사용자의 질문을 분석하여 다음을 추출하세요:

        [원본 질문]
        {original_query}

        [작업 지침]
        1. legal_terms: 법률 데이터베이스 검색에 최적화된 용어로 변환
            예: "좌회전하다가" -> ["좌회전 진행", "교차로 진입", "회전 중 충돌"]
        
        2. accident_type: 다음 중 하나로 분류
            - 교차로 (신호/비신호 교차로 사고)
            - 차선변경 (끼어들기, 합류 사고)
            - 후방추돌 (정차/서행 중 추돌)
            - 주차장 (주차 중 접촉 사고)
            - 기타 (위 분류에 해당 없음)

        3. key_factors: 과실 판단 핵심 요소 추출
            예: ["신호위반", "안전거리 미확보", "방향지시등 미점등", "중앙선 침범"]

        4. expanded_query: 검색 최적화 쿼리 (법률 용어 + 핵심 상황 조합)
            예: "신호대기 중인 차량에 우회전 차량이 충돌한 교차로 사고의 과실 비율"

        5. metadata_filters: 검색 필터 조건 (JSON 형태)
            예: {{"accident_type": "교차로", "has_signal": true}}

        [중요] 질문에서 명시되지 않은 정보는 추측하지 말고 빈 값으로 남겨두세요
        """

        try:
            expansion_msg = HumanMessage(content=expansion_prompt)
            expanded = await asyncio.wait_for(
                self.llm_expander.ainvoke([expansion_msg]),
                timeout=10.0
            )

            logger.info(f"[Query Expansion] {original_query[:30]}... -> {expanded.accident_type}")

            return expanded
        except asyncio.TimeoutError:
            logger.error(f"Query Expansion 타임아웃")
            raise
        except Exception as e:
            logger.error(f"Query Expansion 실패: {e}")
            # Fallback: 원본 쿼리 그대로 사용
            return ExpandedQuery(
                legal_terms=[original_query],
                accident_type="기타",
                key_factors=[],
                expanded_query=original_query,
                metadata_filters={}
            )
        
    # --- [HIGH PRIORITY 2] Metadata Filtering ---
    async def _search_with_metadata_filter(
        self,
        query: str,
        metadata_filters: Dict[str, Any],
        k: int = 10     # 1차 검색은 더 많이 (Reranking 대비)
    ) -> List[tuple]:
        """
        메타데이터 필터를 적용한 유사도 검색 (재시도 + 타임아웃)

        Args:
            query: 검색 쿼리
            metadata_filters: 필터 조건 (예: {"accident_type": "교차로"})
            k: 반환한 문서 수
        Returns:
            (Document, score) 튜플 리스트
        """
        try:
            if metadata_filters:
                docs_with_scores = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.vector_store.similarity_search_with_score,
                        query,
                        k=k,
                        filter=metadata_filters
                    ),
                    timeout=15.0
                )

                # 결과가 0이면 필터 끄고 재검색!
                if not docs_with_scores:
                    logger.warning(f"[Metadata Filter] 결과 0개 -> 필터 해제 후 재검색 시도")
                    return await asyncio.wait_for(
                        asyncio.to_thread(
                            self.vector_store.similarity_search_with_score,
                            query,
                            k=k
                        ),
                        timeout=15.0
                    )

                logger.info(f"[Metadata Filter] {metadata_filters}, 결과: {len(docs_with_scores)}개")
            else:
                # 필터 없으면 기본 검색
                docs_with_scores = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.vector_store.similarity_search_with_score,
                        query,
                        k=k
                    ),
                    timeout=15.0
                )
                logger.info(f"[Basic Search] 결과: {len(docs_with_scores)}개")

            return docs_with_scores
        
        except asyncio.TimeoutError:
            logger.error(f"DB 검색 타임아웃")
            raise
        except Exception as e:
            logger.error(f"DB 검색 실패: {e}")
            
            if metadata_filters:
                logger.info(" -> 필터 없이 재시도...")
                return await asyncio.to_thread(
                    self.vector_store.similarity_search_with_score,
                    query,
                    k=k
                )
            
            raise
    
    # --- [HIGH PRIORITY 3] Reranking ---
    async def _rerank_documents(
        self,
        query: str,
        docs_with_scores: List[tuple],
        top_k: int = 3
    ) -> List[tuple]:
        """
        CrossEncoder 또는 Cohere를 사용한 재순위화 (Cohere 비동기 처리 적용)

        Args:
            query: 검색 쿼리
            docs_with_scores: (Document, score) 리스트
            top_k: 최종 반환할 문서 수

        Returns:
            재순위화된 상위 문서 리스트
        """
        reranker = self._get_reranker()

        if not reranker or not docs_with_scores:
            # Reranker 없으면 원본 그대로 반환
            return docs_with_scores[:top_k]
        
        try:
            # Cohere API 사용
            if hasattr(reranker, "rerank"):
                documents = [doc.page_content for doc, _ in docs_with_scores]

                # 비동기 처리 (블로킹 제거)
                rerank_response = await asyncio.wait_for(
                    asyncio.to_thread(
                        reranker.rerank,
                        query=query,
                        documents=documents,
                        top_n=top_k,
                        model="rerank-multilingual-v3.0"
                    ),
                    timeout=10.0
                )
                
                # Cohere 결과를 원본 문서와 매핑
                reranked = []
                for result in rerank_response.results:
                    idx = result.index
                    reranked.append((
                        docs_with_scores[idx][0],       # Document
                        result.relevance_score          # Rerank score
                    ))
                
                logger.info(f"[Cohere Rerank] {len(docs_with_scores)}개 -> {len(reranked)}개")
                return reranked
            # CrossEncoder 사용 (로컬)
            else:
                pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
                scores = await asyncio.wait_for(
                    asyncio.to_thread(reranker.predict, pairs),
                    timeout=5.0
                )
                
                # 점수 기준 재정렬
                reranked = sorted(
                    zip(docs_with_scores, scores),
                    key=lambda x: x[1],
                    reverse=True
                )

                # 상위 top_k개 반환
                result = [(doc, score) for (doc, _), score in reranked[:top_k]]

                logger.info(f"[CrossEncoder Rerank] {len(docs_with_scores)}개 -> {len(result)}개")
                return result

        except asyncio.TimeoutError:
            logger.error(f"Reranking 타임아웃")
            return docs_with_scores[:top_k]
        except Exception as e:
            logger.error(f"Reranking 실패: {e}")
            return docs_with_scores[:top_k]

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
            2. [신호 상태] 신호등 색상 및 점등 위치 (식별 불가능 하면 '식별 불가' 명시)
            3. [충돌 부위] 내 차와 상대차의 정확한 충돌 지점 (예: 내 차 우측 휀더, 상대 차 좌측 범퍼)
            4. [노면/환경] 빗길 미끄러짐, 포트홀, 공사 구간 표지판 등 특이사항
            """
        }]

        # 이미지 인코딩 병렬 처리
        async def encode_async(path):
            return await asyncio.to_thread(self._encode_image, path)
        
        encoded_images = await asyncio.gather(*[encode_async(p) for p in image_paths])

        valid_images = 0
        for b64_str in encoded_images:
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
        try:
            response = await asyncio.wait_for(
                self.llm_smart.ainvoke([msg], config={"max_tokens": 700}),
                timeout=30.0
            )
            
            logger.info(f"[Vision] {valid_images}개 이미지 분석 완료")
            return response.content
        except asyncio.TimeoutError:
            logger.error(f"이미지 분석 타임아웃")
            return "이미지 분석 시간 초과"
        except Exception as e:
            logger.error(f"이미지 분석 API 호출 실패: {e}")
            return f"이미지 분석 중 오류 발생: {str(e)}"
    
    # --- [Tool 2] 법률 검색 (Search Worker용) - [기능 추가: 3줄 요약] ---
    async def search_legal_docs(self, query: str) -> str:
        """
        쿼리와 유사한 판례 검색 (Query Expansion + Metadata Filtering + Reranking 적용)

        Args:
            query: 원본 사용자 질문

        Returns:
            요약된 판례 정보 (3줄) 
        """

        try:
            # STEP 1: Query Expansion
            expanded = await self._expand_query(query)
            search_query = expanded.expanded_query

            logger.info(f"[검색 시작] 원본: {query[:50]}...")
            logger.info(f"[확장 쿼리] {search_query}")

            # STEP 2: Metadata Filtering 검색 (1차: k=10)
            docs_with_scores = await self._search_with_metadata_filter(
                query=search_query,
                metadata_filters=expanded.metadata_filters.model_dump(exclude_none=True),
                k=10        # Reranking을 위해 더 많이 검색
            )

            if not docs_with_scores:
                return "검색된 관련 판례가 없습니다."

            # STEP 3: Reranking (10개 -> 3개)
            reranked_docs  =await self._rerank_documents(
                query=search_query,
                docs_with_scores=docs_with_scores,
                top_k=3
            )

            # STEP 4: 결과 포맷팅
            raw_results = []
            for i, (doc, score) in enumerate(reranked_docs):
                src = doc.metadata.get("source", "판례 DB")
                accident_type = doc.metadata.get("accident_type", "")
                content = doc.page_content.replace("\n", " ").strip()[:500]

                # 메타데이터 정보 추가
                meta_info = f"[{accident_type}]" if accident_type else ""
                raw_results.append(
                    f"사례 {i + 1} {meta_info} (관련도: {score:.2f}, 출처: {src}): {content}"
                )

            # 2. 프롬프트 수정 개별 나열 -> 종합 3줄 요약
            summary_prompt = f"""
            아래 제공된 교통사고 판례들을 종합적으로 분석하세요.
            개별 사례를 단순히 나열하지 말고, **공통된 과실 판단 기준과 법적 근거를 통합하여 총 3줄로 요약**해 주세요.

            [작성 지침]
            1. '사례 1은..., 사례 2는...' 식의 나열을 금지합니다.
            2. "주요 판례에 따르면~" 과 같이 하나의 문맥으로 통합하세요.
            3. 과실 비율이 달라지는 핵심 변수(속도, 신호, 진입 시점 등)가 무엇인지 포함하세요.
            4. 검색된 사례의 사고 유형({expanded.accident_type})을 고려하세요.

            [검색된 판례 목록]:
            {chr(10).join(raw_results)}
            """

            summary_msg = HumanMessage(content=summary_prompt)

            summary_response = await asyncio.wait_for(
                self.llm_fast.ainvoke([summary_msg]),
                timeout=15.0
            )

            logger.info("[검색 완료] 요약 생성 완료")
            return summary_response.content
        
        except asyncio.TimeoutError:
            logger.error("법률 검색 타임아웃")
            return "검색 시간 초과"
        except Exception as e:
            logger.error(f"판례 검색 중 오류: {e}")
            return f"판례 검색 중 시스템 오류 발생: {str(e)}"
        
    # --- [Tool 3] 최종 솔루션 생성 (Final Solver 용 - 구조화된 출력) - [기능 추가: 조언 및 위험도] ---
    async def generate_final_solution_structured(self, messages: list, image_paths: list) -> AccidentAnalysisResult:
        """
        대화 기록 (Context)과 텍스트 요약본을 종합하여 최종 답변 생성
        [비용 절감] 이미지는 여기서 사용하지 않고 텍스트만 처리함
        최종 분석 (타임아웃 + 에러 핸들링)
        """

        # 1. 구조화된 출력을 위한 LLM 설정
        structured_llm = self.llm_smart.with_structured_output(AccidentAnalysisResult)

        # 2. 시스템 프롬프트 (페르소나 및 제약조건), [프롬프트 강화] 경미 사고 조언 및 위험도
        system_prompt_text = """
        당신은 운전자의 이익을 최우선으로 생각하는 '교통사고 대응 전략가'입니다.
        먼저 **예상 과실 비율(Fault Ratio)**을 산정한 뒤, 이를 기준으로 최적의 대응 전략(action_guide)을 결정하세요.
        
        [입력 데이터 구조 안내]
        - 사용자의 사고 설명: HumanMessage로 전달됩니다.
        - 현장 시각 정보 요약 (Vision AI 분석): SystemMessage의 "[상황 분석 리포트] > 1. 현장 시각 정보 요약" 항목에 포함됩니다.
        - 관련 판례 및 법규 (RAG 검색): SystemMessage의 "[상황 분석 리포트] > 2. 관련 판례 및 법규" 항목에 포함됩니다.
        위 세가지 정보를 종합하여 summary, fault_ratio, advice 등을 생성하세요.

        [판단 기준 (action_guide Decision Tree)]

        1. **개인 합의 유리 (Personal Settlement Best)**
            - **[핵심 조건]**: 내 과실이 높음(100% 또는 가해자) AND 피해가 경미함.
            - [상황]: 문콕, 단순 긁힘 등 수리비가 소액(50만원 미만)인 경우
            - [논리]: 내 과실이므로 보험 처리 시 할증/유예 패널티가 발생함. 수리비가 소액이라면 현금 합의가 총비용(할증분 포함) 면에서 유리함.

        2. **보험 처리 유리 (Insurance Best)**
            - **[핵심 조건]**: 내 과실이 없거나 매우 적음 (0% ~ 20%).
            - [상황]: 상대방의 신호위반, 후방 추돌, 혹은 내가 주차 구획 내에 있는 데 당한 경우 등 명확한 피해자 입장.
            - [논리]: 내 과실이 거의 없으므로 보험료 할증 영향이 미미하거나 없음. 상대 보험서로부터 수리비/렌트비/격락손해 등을 최대한 보장받는 것이 이득임.

        3. **보험 처리 권장 (Insurance Recommended)**
            - **[핵심 조건]**: 쌍방 과실 (30% ~ 70%)로 분쟁 예상 OR 고액 사고/인명 피해.
            - [상황]: 신호 없는 교차로 사고, 차선 변경 중 충돌 등 과실 비율 다툼이 있는 경우. 또는 내 과실이 높더라도 수리비가 고액이라 개인이 감당하기 힘든 경우.
            - [논리]: 과실 비율이 확정되지 않았으므로 보험사의 전문 보상팀을 통해 비율을 방어해야 함. 또한 인명 피해나 고액 사고는 개인 합의가 불가능하거나 위험함.

        [조언 작성 지침 (advice 필드)]
        - "판례에 따르면 사용자분의 과실은 약 0%로 예상됩니다. 따라서..." 형태로 시작하세요.
        - 왜 그 선택(합의 vs 보험)이 경제적/법적으로 유리한지 구체적으로 설득하세요.

        [★ 가장 중요: summary 작성 규칙]
        summary 필드는 반드시 아래 육하원칙 형식을 따라야 합니다. 
        사용자의 사고 설명(HumanMessage)과 Vision AI/RAG 분석 결과(SystemMessage)를 결합하여 작성합니다.
        정보가 부족한 항목은 "미상" 또는 "확인 불가"로 표기하되, 전체 서술 흐름을 유지하세요.

        [summary 필수 형식]
        "{사고일시}경 {사고장소}에서 발생한 사고입니다.
        A차량({내차량정보})은 {방면}에서 {방향}으로 {차선}에서 {행동}하며 이동 중이었습니다.
        B차량({상대차량정보})은 {방면}에서 {방향}으로 {차선}에서 {행동}하며 이동 중이었습니다.
        사고는 A차량이 {상세행동} 중 {신호상태}에서 {충돌경위}하여 B차량의 {충돌부위}와 추돌한 것으로 파악됩니다.

        [summary 작성 체크리스트 - 6가지 모두 포함 필수]
        1. 언제: 사고 발생 일시 (정보 없으면 "일시 미상" 표기)
        2. 어디서: 사고 발생 장소, 도로 유형, 차선 정보 포함
        3. 누가: A차량(사용자)과 B차량(상대방) 정보
        4. 무엇을: 각 차량의 행동 (직진, 좌회전, 차선변경 등)
        5. 어떻게: 충돌 경위 (충돌 부위, 각도 등)
        6. 왜: 사고 원인 분석 (과실 비율 근거와 연결)

        [summary에 절대 포함하지 말 것]
        - [과실 비율 분석] 섹션 -> reasoning 필드에 작성
        - [관련 판례] 섹션 -> legal_basis 필드에 작성
        - 조언이나 행동 가이드 -> advice, action_guide 필드에 작성
        """

        # 3. [개선] 메시지 타입별 분리 및 순서 관리
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        # 현재 쿼리 (마지막 메시지)
        current_query = other_msgs[-1] if other_msgs else None
        history_msgs = other_msgs[:-1] if len(other_msgs) > 1 else []

        # 4. [메시지 다이어트: 토큰 관리]
        trimmed_history = []
        if history_msgs:
            try:
                # (2) 과거 히스토리 Trimming
                trimmed_history = trim_messages(
                    history_msgs,
                    max_tokens=5000,                # 6000 -> 5000으로 조정 (시스템 프롬프트 공간 확보)
                    strategy="last",
                    token_counter=self.llm_smart,   # 모델의 토크나이저 사용
                    include_system=False,
                    allow_partial=False
                )
            except Exception as e:
                logger.warning(f"메시지 트리밍 실패, 전체 히스토리 사용: {e}")
                trimmed_history = history_msgs       

        # 5 최종 메시지 조립 (순서: System -> History -> Current)
        system_prompt_msg = SystemMessage(content=system_prompt_text)
        final_messages = [system_prompt_msg] + system_msgs + trimmed_history

        if current_query:
            final_messages.append(current_query)

        # 6. 실행 및 결과 반환
        try:
            response = await asyncio.wait_for(
                structured_llm.ainvoke(final_messages),
                timeout=20.0
            )
            
            logger.info("[Final Solution] 분석 완료")
            return response
        except asyncio.TimeoutError:
            logger.error("최종 분석 타임아웃")
            return AccidentAnalysisResult(
                summary="분석 시간 초과",
                fault_ratio=FaultRatio(me=0, opponent=0),
                legal_basis=[],
                advice="시스템 지연 중입니다. 잠시 후 재시도하세요.",
                reasoning="타임아웃",
                action_guide="보험 처리 권장"
            )
        except Exception as e:
            # 실패 시 기본값 반환 (Fallback)
            logger.error(f"구조화된 출력 생성 실패: {e}")
            return AccidentAnalysisResult(
                summary="분석 중 오류가 발생했습니다.",
                fault_ratio=FaultRatio(me=0, opponent=0),
                legal_basis=[],
                advice="일시적인 오류가 발생했습니다. 잠시후 다시 시도해 주세요.",
                reasoning=str(e),
                action_guide="보험 처리 권장"
            )
        
    # --- [유틸리티] Pool 상태 ---
    def get_pool_status(self) -> Dict[str, Any]:
        """Connection Pool 상태"""
        if AccidentRAGEngine._engine:
            pool = AccidentRAGEngine._engine.pool
            return {
                "size": pool.size(),
                "checked": pool.checkedout(),
                "overflow": pool.overflow()
            }
        
        return {"error": "Engine not initialized"}
