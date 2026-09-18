# 🚦 교통사고 과실 판단 AI 솔루션 (Traffic Accident Fault Assessment AI)

교통사고 현장 **사진**과 **사고 정황(정형 입력)**을 종합 분석하여, 한국의 교통사고 조사 규칙 및 주요 보험사(삼성화재 등) 과실 비율 인정 기준을 근거로 **예상 과실 비율·관련 판례·전략적 대응 가이드**를 제시하는 멀티모달 RAG 기반 AI 챗봇 서비스입니다.

> 단순 응급처치 안내에서 출발해, **판례를 검색·추론하는 LangGraph 멀티 에이전트**와 **GCP 클라우드에 배포된 프로덕션 백엔드**로 발전했습니다.

* **개발 기간:** 2026년 1월 ~ 2026년 2월
* **프로젝트명:** 교통사고 처리 솔루션 챗봇

---

## ✨ 핵심 특징

* **멀티모달 분석:** 사고 현장 사진(다중 업로드) + 상세 UI 정형 데이터 기반 상황 분석
* **하이브리드 LangGraph:** Supervisor + 병렬 처리(Fan-out) + 동기화(Join)를 갖춘 엔터프라이즈급 워크플로우
* **고도화된 RAG:** Multi-Query 확장 · Context Injection · CrossEncoder Re-ranking으로 판례 검색 정확도 극대화
* **전략적 행동 가이드:** 위험도 표시를 넘어 "합의 vs 보험 처리" 등 운전자의 경제적 이익 중심 판단 제공
* **프로덕션 배포:** GCP Cloud Run 서버리스 + PostgreSQL(pgvector) DB로 실서비스 운영 가능

---

## 🛠️ 기술 스택

### Backend & AI
* **Framework:** FastAPI, Uvicorn
* **AI Orchestration:** LangChain, LangGraph (멀티 에이전트 워크플로우)
* **LLM 전략:** `gpt-4o-mini`(요약·정제) / `gpt-4o`(최종 과실 판단) 이원화
* **Vision:** GPT-4o Vision (이미지 → 법적 텍스트 묘사 변환, 768px 리사이징 최적화)
* **RAG:** PostgreSQL + `pgvector`, CrossEncoder(`BAAI/bge-reranker-v2-m3`) 재순위화
* **Embeddings:** OpenAI `text-embedding-3-small`

### Frontend
* **Streamlit:** 상세 사고 정황 입력 폼, 과실 비율 시각화, 카카오톡 스타일 스트리밍 챗봇 UI

### Database & Infrastructure
* **DB:** PostgreSQL + pgvector (Docker `pgvector/pgvector:pg16`), `AsyncPostgresSaver` 세션 영속화
* **Cloud:** Google Cloud Platform — Cloud Run(서버리스 백엔드), Compute Engine(DB), GCS(PDF 스토리지)
* **패키지 관리:** `uv` (Rust 기반 초고속 파이썬 패키지 매니저)
* **언어:** Python 3.12.8

---

## 🏗️ 시스템 아키텍처

**구조:** Hybrid LangGraph (Supervisor + Parallel + Join)

1. **Supervisor** — 입력 데이터(이미지 유무)를 분석해 실행 경로 결정
2. **Parallel Execution (Fan-out)** — Vision(이미지 분석)과 Search(판례 검색) 동시 실행
3. **Join Node (Fan-in)** — 병렬 작업 완료 대기 및 흐름 통합, Final Solver 중복 실행 방지
4. **Final Solver** — 수집 정보를 종합해 텍스트 기반 최종 판결 생성

### 적용 디자인 패턴
MVC · Singleton(엔진 인스턴스 공유) · Facade(RAG/Vision 로직 은닉) · Supervisor · Fan-out/Fan-in

### RAG 고도화 파이프라인
`Query Rewrite → Multi-Query 확장 → Context Injection → 벡터 검색(15건) → CrossEncoder Re-ranking(Top 3) → 판례 3줄 종합 요약`
* 검색 결과 0건 시 **필터 자동 해제 후 재검색**(Self-Healing) — 회복 탄력성 확보
* 병렬 처리로 응답 시간 약 30초 → 20초 단축

---

## 🚀 시작하기

```bash
# 시스템 의존성 (WSL/Linux)
sudo apt update
sudo apt install -y ffmpeg

# uv 설치 (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
# uv 설치 (Windows)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 의존성 설치 및 가상환경 생성 (pyproject.toml 기반)
uv sync

# 백엔드 실행
uv run uvicorn chatbot_backend:app --host 0.0.0.0 --port 8001

# 프론트엔드 실행
uv run streamlit run streamlit_app.py
```

`.env` 파일에 `OPENAI_API_KEY` 및 DB 접속 정보를 설정해야 합니다.

---

## 📂 폴더 구조

```text
Project_Root/
├── chatbot_backend.py        # FastAPI 서버: /analyze, /chat, /chat/stream
├── streamlit_app.py          # Streamlit UI: 입력 폼 + 스트리밍 챗봇
├── RAG/
│   └── AccidentRAGEngine.py  # RAG 엔진, Vision 분석, 판례 검색·재순위화 (Singleton)
├── LangGraphScripts/
│   └── AccidentGraph.py      # LangGraph 워크플로우 (Supervisor/Parallel/Join)
├── STT/
│   └── google_stt_handler.py # (보존, 현재 미사용)
├── build_vector_db.py        # PDF 인입·벡터화 파이프라인
├── sync_gcs_to_db.py         # GCS ↔ DB 파일 동기화
├── .env                      # 환경 변수 (보안 주의)
├── pyproject.toml            # 의존성 명세
└── uv.lock                   # 버전 고정
```

---

## ☁️ 배포 (GCP Cloud Run)

* **프로젝트 ID:** `accident-detection-db-485509`
* **서비스명:** `chatbot-backend`
* **서비스 URL:** `https://chatbot-backend-599050237852.asia-northeast3.run.app`

```bash
gcloud run services update chatbot-backend \
  --port 8001 \
  --min-instances 0 \
  --region asia-northeast3
```

### 주요 배포 트러블슈팅
* **DB 연결 차단:** VPC 방화벽에 `tcp:5432` Ingress 허용 규칙(`allow-postgres-5432`) 추가
* **Port Mismatch:** 컨테이너 포트 `8001`을 명시적으로 매핑 (Cloud Run 기본 8080 헬스체크 실패 해결)
* **비용 최적화:** `--min-instances 0` (Scale-to-zero)

---

## 📊 데이터 파이프라인

* GCS 버킷(`pdf-storage-2026`)의 PDF를 DB에 자동 등록·삭제 동기화
* `is_vectorized` 플래그 + `file_hash` 기반 **증분 벡터화** (신규/변경 문서만 처리)
* 벡터 저장 성공 후에만 상태 갱신하는 트랜잭션 관리로 DB↔벡터 정합성 유지
* 현재 33개 PDF → **580개 벡터 청크** 주입 완료 (HNSW 인덱스)

---

## 📌 개발 이력 요약

| 단계 | 내용 |
| :--- | :--- |
| 1. 데이터 구축 | FAISS 로컬 → PostgreSQL+pgvector 전환, PDF 증분 벡터화 파이프라인 |
| 2. 인프라 | WSL Docker → GCP Compute Engine 마이그레이션, 방화벽·보안 설정 |
| 3. RAG 고도화 | Multi-Query·Context Injection·Re-ranking, Self-Healing 재검색 |
| 4. 에이전트 | LangGraph Supervisor + Fan-out/Join 하이브리드 구조, 비동기 전환 |
| 5. 실용화 | 음성 입력 제거, 행동 가이드(합의/보험) 로직, 스트리밍 챗봇 UI |
| 6. 배포 | GCP Cloud Run 서버리스 배포 완료 |
```
