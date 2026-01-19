# 🚑 교통사고 처치 솔루션 챗봇 (Traffic Accident AI Agent)

교통사고 발생 시 사용자의 당황한 상태를 고려하여, LLM(LangChain, LangGraph)과 RAG(Retrieval-Augmented Generation) 기술을 통해 실시간 응급처치 및 사고 대응 가이드를 제공하는 AI 에이전트 서비스입니다.

---

## 기술 스택 및 특징

* **패키지 관리자:** `uv` (Rust 기반의 초고속 파이썬 패키지 매니저)
* **프레임워크:** LangChain, LangGraph, LangServe
* **언어:** Python 3.12.8
* **특징:**
    * `uv` 는 기존 `pip` 대비 최대 10~100배 빠른 패키지 설치 및 동기화.
    * `pyproject.toml`을 통한 엄격한 의존성 관리 및 재현성 보장.
    * 응급처치 매뉴얼 데이터를 활용한 신뢰성 높은 RAG 답변 생성.

---

## 시작하기
    # uv 설치 (macOS/Linux)
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # uv 설치 (Windows)
    powershell -c "ir https://astral.sh/uv/install.ps1 | iex"

    # 필요한 파이썬 모듈 설치
    # 랭체인, 랭그래프 등 필요한 모듈들 한번에 설치
    # 가상환경도 한번에 생성
    # uv 가 pyproject.toml 에 지정한 모듈 정보를 읽고 해당 모듈을 설치
    uv sync

## 폴더 구조

```text
chatbot/
├── .venv/               # uv 가상환경 (Git 추적 제외)
├── data/                # 응급처치 매뉴얼 (PDF, CSV 등 원천 데이터)
├── src/
│   ├── agents/          # LangGraph를 이용한 상태 제어 및 에이전트 로직
│   ├── chains/          # RAG 프로세스 및 프롬프트 체인 구성
│   └── main.py          # LangServe 실행 및 API 엔트리포인트
├── .env                 # API 키 및 환경 변수 설정 (보안 주의)
├── .gitignore           # Git 업로드 제외 목록 설정
├── pyproject.toml       # 프로젝트 메타데이터 및 의존성 명세
├── uv.lock              # 의존성 버전 고정 파일
└── README.md            # 프로젝트 문서
```

