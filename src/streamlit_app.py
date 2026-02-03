import streamlit as st
import requests
from datetime import datetime

BACKEND_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="교통사고 AI 솔루션", page_icon="⚖️", layout="wide")
st.title("⚖️ 교통사고 과실 판단 AI")

# 세션 상태 초기화
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None

# -----------------------------------------
# 1. 기본 정보 (공통)
# -----------------------------------------
st.header("1. 기본 정보")
c1, c2 = st.columns(2)
with c1:
    acc_date = st.date_input("발생 일자", datetime.now())
    acc_time = st.time_input("발생 시각", datetime.now())
with c2:
    acc_loc = st.text_input("발생 장소 (예: 강남역 사거리)")
    bbox = st.radio("블랙박스 유무", ["예", "아니요", "불명"], horizontal=True)

st.divider()

# ------------------------------------------
# 2. 입력 방식 선택
# ------------------------------------------
input_method = st.radio("입력 방식 선택", ["📝 상세 UI 입력", "🎤 음성/자유 입력"], horizontal=True)

final_text_prompt = ""
voice_data = None

# ------------------------------------------
# [CASE A] 상세 UI 입력 
# ------------------------------------------
if input_method == "📝 상세 UI 입력":
    st.info("아래 항목들을 빠짐없이 선택해주세요.")

    # --- 2-2. 사고 장소 유형 ---
    st.subheader("2. 사고 장소 및 신호")
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        road_type = st.radio(
            "사고 발생 도로 유형",
            ["신호등 있는 교차로", "신호등 없는 교차로", "직선 도로", "주차장", "기타"]
        )
    with col2_2:
        traffic_light = st.radio(
            "신호등 상태",
            ["녹색(직진)", "황색", "적색", "좌회전", "기타/신호없음"]
        )

    st.divider()

    # --- 3. 내 차량 상황 ---
    st.subheader("3. 내 차량 상황")
    col3_1, col3_2, col3_3, col3_4 = st.columns(4)
    with col3_1:
        my_action = st.radio(
            "사고 당시 내 행동",
            ["직진 중", "좌회전 중", "우회전 중", "유턴 중", "차선 변경 중", "정차 중", "후진 중"],
            key="my_action"
        )
    with col3_2:
        my_signal = st.radio(
            "내 차 방향지시등",
            ["켜지 않음", "좌측 깜빡이", "우측 깜빡이", "비상등", "기타"],
            help="신호등이 아니라 내 차의 방향지시등 상태입니다.",
            key="my_signal"
        )
    with col3_3:
        my_traffic_light = st.radio(
            "내 주행 신호 (진입 시)",
            ["녹색 (정상 진입)", "황색 (딜레마 존)", "적색 (신호 위반)", "좌회전 신호", "비보호 좌회전", "우회전 전용 신호", "신호 없음"],
            help="교차로 진입 당시 내 차로의 신호등 상태를 선택하세요."
        )
    with col3_4:
        my_speed = st.radio(
            "내 차량 속도",
            ["정지", "서행(20km/h 이하)", "보통(20~50km/h)", "빠름(50km/h 이상)"],
            key="my_speed"
        )

    st.divider()

    # --- 4. 상대 차량 상황 ---
    st.subheader("4. 상대 차량 상황")
    col4_1, col4_2, col4_3, col4_4 = st.columns(4)
    with col4_1:
        opp_action = st.radio(
            "상대 차량 행동",
            ["직진 중", "좌회전 중", "우회전 중", "유턴 중", "차선 변경 중", "정차 중", "후진 중", "기타/모름"]
        )
    with col4_2:
        opp_signal = st.radio(
            "상대 차 방향 지시등",
            ["켜지 않음", "좌측 깜빡이", "우측 깜빡이", "비상등", "기타"],
        )
    with col4_3:
        opp_traffic_light = st.radio(
            "상대 주행 신호 (진입 시)",
            ["녹색 (정상 진입)", "황색 (딜레마 존)", "적색 (신호 위반)", "좌회전 신호", "비보호 좌회전", "우회전 전용 신호", "신호 없음"]
        )
    with col4_4:
        opp_direction = st.radio(
            "상대 차량 진입 방향",
            ["정면 (맞은 편)", "좌측에서 진입", "우측에서 진입", "후방 추돌", "기타"]
        )

    st.divider()

    # --- 5. 충돌 상황 ---
    st.subheader("5. 충돌 부위")
    hit_part = st.radio(
        "내 차량 충돌 부위",
        ["전면", "전면 좌측", "전면 우측", "좌측면", "우측면", "후면", "후면 좌측", "후면 우측"],
        horizontal=True
    )

    st.divider()

    # --- 6. 추가 위반 사항 (다중 선택) ---
    st.subheader("6. 추가 정황 (중복 선택 가능)")
    col6_1, col6_2 = st.columns(2)
    with col6_1:
        my_fault = st.multiselect(
            "내 차량 특이사항",
            ["과속", "신호 위반", "중앙선 침범", "안전거리 미확보", "급정거", "음주/무면허", "해당 없음"],
            default=["해당 없음"]
        )
    with col6_2:
        opp_fault = st.multiselect(
            "상대 차량 특이사항",
            ["과속", "신호 위반", "중앙선 침범", "안전거리 미확보", "급정거", "음주/무면허", "해당 없음"],
            default=["해당 없음"]
        )

    # --- 프롬프트 조합 (백엔드 전송용) ---
    # UI 선택값들을 LLM이 이해하기 쉬운 텍스트로 변환
    final_text_prompt = f"""
    [기본 정보] 일시: {acc_date} {acc_time}, 장소: {acc_loc}, 블랙박스: {bbox}
    [도로 환경] 도로 형태: {road_type}, 신호 시스템 상태: {traffic_light}

    [나의 주행 정보]
    - [행동]: {my_action}
    - [진입 신호]: {my_traffic_light} (중요)
    - [깜빡이]: {my_signal}
    - [속도]: {my_speed}
    - [특이사항]: {', '.join(my_fault)}

    [상대방 주행 정보]
    - [행동]: {opp_action}
    - [진입 신호]: {opp_traffic_light}
    - [깜빡이]: {opp_signal}
    - [진입 방향]: {opp_direction}
    - [특이사항]: {', '.join(opp_fault)}

    [충돌 정보] 내 차 충돌 부위: {hit_part}
    """

# ----------------------------------------------
# [CASE B] 음성 / 자유 입력
# ----------------------------------------------
else:
    st.info("마이크를 켜고 사고 상황을 자유롭게 설명해주세요.")
    voice_data = st.audio_input("녹음 시작")
    if not voice_data:
        st.warning("음성 입력이 없으면 분석을 진행할 수 없습니다.")

st.divider()

# ----------------------------------------------
# 7. 사고 사진 업로드 (공통)
# ----------------------------------------------
st.header("7. 현장 사진 (선택)")
images = st.file_uploader("사진을 업로드하면 분석 정확도가 올라갑니다.", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if images:
    st.markdown("##### 📸 미리보기")
    cols = st.columns(min(len(images), 4))
    for idx, img in enumerate(images):
        with cols[idx % 4]:
            st.image(img, caption=img.name, use_container_width=True)

st.divider()

# ==========================================
# 8. 분석 요청
# ==========================================
if st.button("🚀 AI 분석 시작", type="primary", use_container_width=True):
    # 필수 입력값 검증
    if input_method == "📝 상세 UI 입력" and not acc_loc:
        st.error("기본 정보의 '발생 장소'를 입력해주세요.")
    elif input_method == "🎤 음성/자유 입력" and not voice_data:
        st.error("음성을 녹음해주세요.")
    else:
        with st.spinner("AI가 사고 정황을 분석하고 판례를 검색 중입니다..."):
            try:
                # 1. 파일 페이로드 준비
                files = []
                if images:
                    for img in images:
                        img.seek(0)
                        files.append(("image_files", (img.name, img, img.type)))
                
                # 2. 데이터 페이로드 준비
                data_payload = {}
                
                # 세션 ID (멀티턴)
                if st.session_state["thread_id"]:
                    data_payload["thread_id"] = st.session_state["thread_id"]
                
                if voice_data:
                    # 음성 모드
                    files.append(("voice_file", ("voice.wav", voice_data, "audio/wav")))
                    summary_prompt = f"일시: {acc_date} {acc_time}, 장소: {acc_loc}, 블랙박스: {bbox}"
                    data_payload["text_query"] = summary_prompt
                else:
                    # 텍스트 모드 (UI 조합 프롬프트)
                    data_payload["text_query"] = final_text_prompt
                
                # 3. 백엔드 요청
                res = requests.post(BACKEND_URL, files=files, data=data_payload, timeout=120)
                
                if res.status_code == 200:
                    response_data = res.json()
                    
                    # Thread ID 갱신
                    st.session_state["thread_id"] = response_data["thread_id"]
                    
                    result = response_data["result"]
                    rag_context = response_data["rag_context"]
                    transcript = response_data["transcript"]

                    # 4. 결과 시각화
                    st.success("분석이 완료되었습니다!")
                    
                    # (A) 분석에 사용된 내용 확인
                    with st.expander("📝 분석에 사용된 사고 내용 (프롬프트/STT)", expanded=False):
                        if result.get("transcript"):
                            st.write(result["transcript"])
                        else:
                            st.text(final_text_prompt)

                    # (B) 과실 비율 그래프
                    st.subheader("📊 예상 과실 비율")
                    if "fault_ratio" in result:
                        me = result["fault_ratio"]["me"]
                        opp = result["fault_ratio"]["opponent"]
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("나 (사용자)", f"{me}%")
                            st.progress(me / 100)
                        with c2:
                            st.metric("상대방", f"{opp}%")
                            st.progress(opp / 100)
                    
                    # (C) 리포트
                    st.markdown("### 💡 AI 분석 리포트")
                    st.info(f"**상황 요약:** {result.get('summary', '내용 없음')}")
                    
                    st.markdown("#### 🛡️ 대응 조언")
                    st.write(result.get('advice', ''))
                    
                    # (D) 근거 및 판례
                    tab1, tab2 = st.tabs(["🔍 판단 근거", "📜 참조 판례"])
                    
                    with tab1:
                        st.write(result.get('reasoning', ''))
                    
                    with tab2:
                        if "legal_basis" in result and result["legal_basis"]:
                            for item in result["legal_basis"]:
                                st.markdown(f"- {item}")
                        else:
                            st.write("관련된 명확한 법규를 특정하지 못했습니다.")
                        
                        st.markdown("---")
                        st.caption("참조된 검색 데이터:")
                        st.text(rag_context)

                else:
                    st.error(f"서버 오류: {res.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("백엔드 서버에 연결할 수 없습니다. main.py가 실행 중인지 확인해주세요.")
            except Exception as e:
                st.error(f"오류 발생: {e}")