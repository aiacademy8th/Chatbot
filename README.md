# 🚦 교통사고 과실 판단 AI (Traffic Accident Fault Assessment AI)

교통사고 상황 텍스트와 현장 사진을 입력받아, LangGraph 기반 멀티 에이전트와 판례 RAG(검색 증강 생성)를 통해 **추정 과실 비율·법적 근거·대응 가이드**를 산출하는 AI 백엔드입니다.

> LangGraph 슈퍼바이저 + 병렬 처리 워크플로우와 PostgreSQL(pgvector) 기반 판례 검색을 결합한 구조입니다.

---

## 🛠️ 기술 스택

`pyproject.toml` 기준 실제 의존성입니다.

### Backend & AI
* **Framework:** FastAPI, Uvicorn (`uvicorn[standard]`)
* **AI Orchestration:** LangChain, LangGraph (`langgraph`, `langgraph-checkpoint-postgres`)
* **LLM / 임베딩:** `langchain-openai` — `gpt-4o`(비전·최종 판단), `gpt-4o-mini`(요약·키워드), `text-embedding-3-small`
* **RAG / Vector Store:** `langchain-postgres`(PGVector), 그 외 `chromadb` 의존성 포함
* **Vision:** GPT-4o Vision (이미지 분석, Pillow 리사이징)
* **STT (미사용):** `google-cloud-speech`, `pydub` — 코드에 존재하나 현재 로직에서 비활성화
* **기타 LLM 백엔드 의존성:** `langchain-community`, `langchain-huggingface`, `langchain-ollama`

### Database & Infrastructure
* **DB:** PostgreSQL + pgvector, `psycopg` / `psycopg2-binary` / `psycopg-pool`
* **세션 영속화:** `AsyncPostgresSaver` + `AsyncConnectionPool` (LangGraph 체크포인터)
* **스토리지:** Google Cloud Storage (`google-cloud-storage`)
* **문서 파싱:** PyMuPDF, pypdf, BeautifulSoup4
* **패키지 관리 / 언어:** `uv`, Python 3.12.8

### Frontend
* **Streamlit:** `src/streamlit_app.py` — 사고 정황을 항목별로 입력받는 폼 UI (백엔드 `http://127.0.0.1:8000/analyze` 호출)
* 저장소에 `frontend/` 디렉터리(Vite/React 계열 툴체인 흔적)가 있으나 애플리케이션 소스는 커밋되어 있지 않습니다.

---

## 🏗️ 아키텍처

LangGraph 슈퍼바이저 패턴 + 병렬 처리 기반 상태 그래프(`AgentState`)로 구성됩니다.

* **AgentState** — `messages`(누적, `operator.add`), `image_paths`, `image_summary`, `rag_context`, `required_tools`, `final_result`
* **Supervisor** — 입력(텍스트/이미지)에 따라 실행할 도구 결정
* **Vision Tool** — 사고 현장 이미지 분석
* **Search Tool** — PGVector 기반 판례·법규 검색
* **Final Solver** — 종합하여 구조화된 결과(`AccidentAnalysisResult`) 생성

### 구조화된 출력 (Pydantic)
`AccidentAnalysisResult` 모델로 결과를 강제합니다.
* `summary` — 사고 상황 3문장 요약
* `fault_ratio` — `{ me, opponent }` 과실 비율(0~100)
* `legal_basis` — 근거 법규·판례 제목 (최대 3개)
* `advice`, `reasoning` — 조언 및 산정 논리
* `action_guide` — `"개인 합의 유리" | "보험 처리 유리" | "보험 처리 권장"` 중 하나

### 리소스 관리
`AccidentRAGEngine`은 엔진/벡터스토어/임베딩을 클래스 변수로 공유하는 싱글톤 구조이며, FastAPI `lifespan`에서 `agent_instance.setup()`으로 DB 풀·체크포인터를 초기화합니다.

---

## 🚀 시작하기

### 1) 사전 준비
```bash
# uv 설치 (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
# uv 설치 (Windows)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 의존성 설치 및 가상환경 생성
uv sync
```

### 2) 환경 변수 (.env)
코드에서 참조하는 값 예시입니다.
```env
OPENAI_API_KEY=...
DB_USER=...
DB_PASSWORD=...
# 이하 DB 호스트/포트/이름 등 접속 정보
```

### 3) 백엔드 실행
`src/main.py`는 `app`을 정의하며, 기본 실행 포트는 8000입니다.
```bash
cd src
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4) 프론트엔드(Streamlit) 실행
```bash
uv run streamlit run src/streamlit_app.py
```

---

## 🔌 API

### `POST /analyze`
| 필드 | 타입 | 필수 | 설명 |
| :--- | :--- | :--- | :--- |
| `text_query` | Form(str) | ✅ | 사고 상황 텍스트 설명 |
| `image_files` | File(list) | ❌ | 현장 사진 (다중) |
| `thread_id` | Form(str) | ❌ | 멀티턴 세션 ID (없으면 자동 생성) |

**응답:** `status`, `thread_id`, `transcript`, `rag_context`, `result`(구조화된 분석 결과)

---

## 📂 폴더 구조

```text
Chatbot/
├── data/                          # 원본 데이터
│   ├── P02_01_01_001_20210101.pdf # 보험 약관 PDF
│   └── car_vs_car_full.csv        # 과실 비율 데이터
├── src/
│   ├── main.py                    # FastAPI 엔트리포인트 (/analyze, lifespan)
│   ├── streamlit_app.py           # Streamlit 입력 폼 UI
│   ├── DB/
│   │   └── gcs_to_db_trigger.py   # GCS → DB 연동 트리거
│   ├── RAG/
│   │   ├── AccidentRAGEngine.py   # RAG 엔진·Vision·구조화 출력 모델 (Singleton)
│   │   ├── build_vector_DB.py     # 벡터 DB 구축 파이프라인
│   │   ├── chatbot_rag.py
│   │   ├── rag_process.py
│   │   └── qa_bot_old.py          # (구버전)
│   ├── LangGraphScripts/
│   │   ├── AccidentGraph.py       # LangGraph 상태 그래프·슈퍼바이저 Agent
│   │   ├── accident_engine.py
│   │   └── agreement_process.py
│   ├── STT/                       # 음성 처리 모듈 (현재 미사용)
│   │   ├── STT_RAG_backend.py
│   │   ├── STT_RAG_frontend.py
│   │   ├── STT_backend.py
│   │   ├── STT_frontend.py
│   │   └── google_stt_handler.py
│   └── 학습/                       # RAG·LangGraph 학습 노트북/자료
├── frontend/                      # 프론트엔드 툴체인 (앱 소스 미커밋)
├── vectorDB/                      # 로컬 FAISS 인덱스 산출물
├── pgvector/                      # pgvector 관련 리소스
├── pyproject.toml                 # 의존성 명세
└── uv.lock                        # 버전 고정
```

---

## 📝 참고
* 저장소에 기본 브랜치는 `main`이며, 소스는 `src/` 하위에 모듈별로 분리되어 있습니다.
* 본 문서는 저장소의 `pyproject.toml`과 `src/` 소스 코드를 근거로 작성되었습니다. 배포 URL·인프라 세부 값 등 코드에 포함되지 않은 정보는 기재하지 않았습니다.
