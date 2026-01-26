import streamlit as st
import requests
import json

# 페이지 설정
st.set_page_config(page_title="교통사고 판단 보조 시스템", page_icon="🚗", layout="wide")

# 백엔드 API 주소
API_URL = "http://127.0.0.1:8000/analyze"

st.title("🚗 보험처리 vs 개인합의 판단 보조 도구")
st.info("사고 상황을 입력하시면 리스크 등급을 판정하고 대응 체크리스트를 생성합니다.")

# --- 사이즈바: 사고 데이터 입력 ---
with st.sidebar:
    st.header("📋 사고 상황 입력")

    # 1. 사고 유형
    accident_type = st.selectbox("사고 유형", ["정차후출발", "주차중", "차선변경", "후방추돌", "기타", "불명"])

    # 2. 주행 속도
    speed = st.select_slider("주행 속도", options=["저속", "중속", "고속", "불명"], value="저속")
    
    # 3. 부상 여부
    injury = st.radio("본인 부상 여부", ["없음", "애매", "있음", "불명"], horizontal=True)
    
    # 4. 현재 통증
    pain_now = st.selectbox("현재 통증 정도", ["없음", "경미", "지속", "악화", "불명"])
    
    # 5. 병원 방문 여부 (추가됨)
    hospital_visit = st.selectbox("병원 방문 상태", ["없음", "예정", "완료", "불명"])
    
    # 6. 차량 손상
    damage = st.selectbox("차량 파손 정도", ["없음", "스크래치", "찌그러짐", "파손", "불명"])

    with st.expander("➕ 상세 정보 및 상대방 반응"):
        # 7. ADAS 센서 (추가됨)
        adas_sensor = st.selectbox("ADAS 센서 작동/경고", ["없음", "있음", "불명"])
        
        # 8. 차종
        v_type = st.selectbox("차종 구분", ["국산", "수입", "전기차", "불명"])
        
        # 9. 증거 확보
        evidence = st.radio("증거 확보(사진/블박)", ["충분", "일부", "없음", "불명"], horizontal=True)
        
        # 10. 상대방 태도 (추가됨)
        opp_attitude = st.selectbox("상대방 태도", ["원만", "애매", "공격적", "불명"])
        
        # 11. 상대방 병원 언급 (추가됨)
        opp_hosp = st.radio("상대의 병원/치료 언급", ["아니오", "예", "불명"], horizontal=True)
        
        # 12. 상대방 보험 언급
        opp_ins = st.radio("상대의 보험처리 요구", ["아니오", "예", "불명"], horizontal=True)
        
        # 13. 사고 메모
        notes = st.text_area("사고 메모", placeholder="상황을 자유롭게 기재하세요.")

    analyze_btn = st.button("🔍 분석 실행", use_container_width=True, type="primary")

# --- 메인 화면: 결과 출력 ---
if analyze_btn:
    # 1. 요청 데이터 구성 (백엔드 AnalysisRequest 모델과 1:1 매칭)
    # facts 딕셔너리로 감싸지 않고 평면 구조로 전송하여 model_dump()와 호환성을 높임
    payload = {
        "accident_type": accident_type,
        "speed": speed,
        "injury": injury,
        "pain_now": pain_now,
        "hospital_visit": hospital_visit,
        "vehicle_damage": damage,                   # UI 'damage' -> Backend 'vehicle_damage'
        "adas_sensor": adas_sensor,
        "vehicle_type": v_type,                     # UI 'v_type' -> Backend 'vehicle_type'
        "evidence": evidence,
        "opponent_attitude": opp_attitude,          # UI 'opp_attitude' -> Backend 'opponent_attitude'
        "opponent_mentions_hospital": opp_hosp,     # UI 'opp_hosp' -> Backend 'opponent_mentions_hospital'
        "opponent_mentions_insurance": opp_ins,     # UI 'opp_ins' -> Backend 'opponent_mentions_insurance'
        "notes": notes
    }

    try:
        with st.spinner("백엔드 엔진 분석 중..."):
            # 2. 백엔드 API 호출 (연결 지점)
            response = requests.post(API_URL, json=payload)
            response.raise_for_status() # 에러 발생 시 예외 처리
            result = response.json()
        
        # 3. 결과 렌더링
        st.divider()
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("🚦 리스크 등급")
            # 백엔드 결과의 risk_bucket 값에 따라 색상 분기
            bucket = result.get("risk_bucket", "UNKNOWN").upper()

            if "GREEN" in bucket or "낮음" in bucket:
                st.success(f"### {bucket}")
                st.balloons()
                st.write("상대적으로 리스크가 낮은 사고입니다.")
            elif bucket == "YELLOW":
                st.warning(f"### {bucket}")
                st.write("주의가 필요합니다. 기록을 철저히 하세요.")
            else:
                st.error(f"### {bucket}")
                st.write("분쟁 위험이 높습니다. 보험 처리를 강력 고려하세요.")

            st.metric(label="분석 상태", value="완료")

        with col2:
            st.subheader("📝 분석 리포트")
            st.markdown(result.get("final_answer", "분석 결과가 없습니다."))

            # 위험 신호 (Flags) 표시
            if result.get("flags_red") or result.get("flags_yellow"):
                st.write("")
                with st.expander("🚩 감지된 구체적 위험 요소", expanded=True):
                    if result["flags_red"]:
                        for flag in result["flags_red"]:
                            st.markdown(f"🔴 **고위험**: {flag}")
                            st.write("**🔴 고위험:** " + ", ".join(result["flags_red"]))
                    if result["flags_yellow"]:
                        for flag in result["flags_yellow"]:
                            st.markdown(f"🟡 **주의**: {flag}")
                            st.write("**🟡 주의:** " + ", ".join(result["flags_yellow"]))
    except requests.exceptions.ConnectionError:
        st.error("❌ 백엔드 서버(main.py)가 실행 중이지 않습니다. 서버를 먼저 띄워주세요.")
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")

else:
    # 대기 화면
    st.write("👈 왼쪽 사이드바에서 사고 내용을 입력하고 버튼을 눌러주세요.")
    st.image("https://via.placeholder.com/800x400.png?text=Accident+Analysis+Waiting...", use_container_width=True)