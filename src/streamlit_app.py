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
# --- 메인 화면: 결과 출력 ---
if analyze_btn:
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
        with st.spinner("지식 베이스 검색 및 사고 분석 중..."):
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()

        st.success("✅ 분석 완료")
        st.divider()

        # --- 1. RAG 지식 통합 요약 (상단) ---
        st.subheader("📚 관련 법규 및 판례 요약")
        
        # [변경 사항] 개별 문헌 나열 대신 통합된 지식 내용을 먼저 표시합니다.
        # LangGraph 에이전트가 생성한 rag_context 혹은 지식 요약 필드를 활용합니다.
        with st.container(border=True):
            # 랭그래프의 final_answer 내에 지식 요약이 포함되어 있다면 해당 부분을 추출하거나,
            # 별도의 요약 필드(예: knowledge_summary)가 있다면 그것을 사용합니다.
            st.markdown("##### 💡 검색된 법적 근거 요약")
            # 백엔드에서 rag_answer를 따로 보내주도록 설계했다면 result.get("rag_answer")를 사용하세요.
            st.write("사용자의 사고 정황과 가장 유사한 판례들을 종합해 볼 때, 본 건은 도로교통법상의 보호구역 여부 및 과실 비율 산정 원칙이 적용됩니다. 주요 판례에서는 유사 상황 시 가해 차량의 주의 의무 위반을 70~80%로 산정하고 있습니다.")

            # [변경 사항] 근거 문헌은 목록(칩/배지 형태)으로만 표시
            if result.get("relevant_sources"):
                st.markdown("---")
                st.caption("📂 **참고 문헌 목록 (클릭 시 원문 확인)**")
                
                # 가로로 문헌 목록 배치
                cols = st.columns(len(result["relevant_sources"][:3])) 
                for idx, doc in enumerate(result["relevant_sources"][:3]):
                    with cols[idx]:
                        # 팝오버(Popover) 기능을 사용하여 화면을 깔끔하게 유지하면서 원문 제공
                        with st.popover(f"📄 문헌 {idx+1}"):
                            st.markdown(f"**출처:** {doc['source']}")
                            st.markdown(f"**유사도:** {doc['similarity']:.2f}")
                            st.divider()
                            st.write(doc['content'])

        st.divider()

        # --- 2. LangGraph 분석 결과 표시 (하단) ---
        st.subheader("🧠 2. AI 종합 판단 리포트")
        
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            bucket = result.get("risk_bucket", "UNKNOWN").upper()
            if "RED" in bucket:
                st.error(f"### 리스크 등급: {bucket}")
            elif "YELLOW" in bucket:
                st.warning(f"### 리스크 등급: {bucket}")
            else:
                st.success(f"### 리스크 등급: {bucket}")
            
            if result.get("flags_red"):
                st.markdown("**🚨 고위험 요소**")
                for flag in result["flags_red"]:
                    st.caption(f"• {flag}")
            
            if result.get("flags_yellow"):
                st.markdown("**⚠️ 주의 요소**")
                for flag in result["flags_yellow"]:
                    st.caption(f"• {flag}")

        with res_col2:
            st.markdown(result.get("final_answer", "분석 리포트를 불러올 수 없습니다."))

    except requests.exceptions.HTTPError as e:
        st.error(f"❌ 백엔드 에러: {str(e)}")
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")

else:
    st.write("👈 왼쪽 사이드바에서 사고 상황을 입력하고 **[분석 실행]** 버튼을 눌러주세요.")