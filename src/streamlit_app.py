import streamlit as st
import requests
import json

# 1. 페이지 설정
st.set_page_config(page_title="교통사고 판단 보조 시스템", page_icon="🚗", layout="wide")

# 2. 백엔드 API 주소 (FastAPI 서버 주소)
API_URL = "http://127.0.0.1:8000/analyze"

st.title("🚗 교통사고 대응 및 판례 분석 시스템")
st.info("사고 상황을 입력하면 AI가 [판례 검색] 후 [종합 판단]을 수행합니다.")

# --- 사이드바: 데이터 입력 ---
with st.sidebar:
    st.header("📋 사고 상황 입력")
    accident_type = st.selectbox("사고 유형", ["정차후출발", "주차중", "차선변경", "후방추돌", "기타", "불명"])
    speed = st.select_slider("주행 속도", options=["저속", "중속", "고속", "불명"], value="저속")
    injury = st.radio("본인 부상 여부", ["없음", "애매", "있음", "불명"], horizontal=True)
    pain_now = st.selectbox("현재 통증 정도", ["없음", "경미", "지속", "악화", "불명"])
    hospital_visit = st.selectbox("병원 방문 상태", ["없음", "예정", "완료", "불명"])
    damage = st.selectbox("차량 파손 정도", ["없음", "스크래치", "찌그러짐", "파손", "불명"])

    with st.expander("➕ 상세 정보 및 상대방 반응"):
        adas_sensor = st.selectbox("ADAS 센서 작동/경고", ["없음", "있음", "불명"])
        v_type = st.selectbox("차종 구분", ["국산", "수입", "전기차", "불명"])
        evidence = st.radio("증거 확보(사진/블박)", ["충분", "일부", "없음", "불명"], horizontal=True)
        opp_attitude = st.selectbox("상대방 태도", ["원만", "애매", "공격적", "불명"])
        opp_hosp = st.radio("상대의 병원/치료 언급", ["아니오", "예", "불명"], horizontal=True)
        opp_ins = st.radio("상대의 보험처리 요구", ["아니오", "예", "불명"], horizontal=True)
        notes = st.text_area("사고 메모", placeholder="상황을 자유롭게 기재하세요.")

    analyze_btn = st.button("🔍 분석 실행", use_container_width=True, type="primary")

# --- 메인 화면: 결과 출력 ---
if analyze_btn:
    # 에러 방지: payload 변수를 여기서 정확히 정의합니다.
    payload = {
        "accident_type": accident_type,
        "speed": speed,
        "injury": injury,
        "pain_now": pain_now,
        "hospital_visit": hospital_visit,
        "vehicle_damage": damage,
        "adas_sensor": adas_sensor,
        "vehicle_type": v_type,
        "evidence": evidence,
        "opponent_attitude": opp_attitude,
        "opponent_mentions_hospital": opp_hosp,
        "opponent_mentions_insurance": opp_ins,
        "notes": notes
    }

    try:
        with st.spinner("단계 1: 관련 판례 검색 중... 단계 2: AI 종합 추론 중..."):
            # 백엔드 호출
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()

        # --- 1. RAG 검색 결과 표시 (상단) ---
        st.subheader("📚 1. 관련 법규 및 유사 판례 (RAG 결과)")
        if result.get("relevant_sources"):
            # 소스 문서를 가로로 배치하거나 리스트로 표시
            for idx, doc in enumerate(result["relevant_sources"][:3]): # 상위 3개만 표시
                with st.expander(f"📍 근거 문헌 {idx+1}: {doc['source']} (유사도: {doc['similarity']:.2f})", expanded=True):
                    st.write(doc['content'])
        else:
            st.info("검색된 직접적인 판례가 없습니다. 일반 법규를 바탕으로 분석을 진행합니다.")

        st.divider()

        # --- 2. LangGraph 분석 결과 표시 (하단) ---
        st.subheader("🧠 2. AI 종합 판단 리포트 (LangGraph)")
        
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            # 리스크 등급 시각화
            bucket = result.get("risk_bucket", "UNKNOWN").upper()
            if "RED" in bucket:
                st.error(f"### 리스크 등급: {bucket}")
            elif "YELLOW" in bucket:
                st.warning(f"### 리스크 등급: {bucket}")
            else:
                st.success(f"### 리스크 등급: {bucket}")
            
            # 위험 요소(Flags) 표시
            if result.get("flags_red"):
                st.markdown("**🚨 고위험 요소**")
                for flag in result["flags_red"]:
                    st.caption(f"• {flag}")
            
            if result.get("flags_yellow"):
                st.markdown("**⚠️ 주의 요소**")
                for flag in result["flags_yellow"]:
                    st.caption(f"• {flag}")

        with res_col2:
            # 최종 분석 텍스트
            st.markdown(result.get("final_answer", "분석 리포트를 불러올 수 없습니다."))

    except requests.exceptions.HTTPError as e:
        st.error(f"❌ 백엔드 서버 에러 (500): 백엔드 터미널의 로그를 확인하세요.")
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")

else:
    st.write("👈 왼쪽 사이드바에서 사고 상황을 입력하고 **[분석 실행]** 버튼을 눌러주세요.")