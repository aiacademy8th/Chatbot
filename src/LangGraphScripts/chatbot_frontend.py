import streamlit as st
import requests
import json
from datetime import datetime

# ------------------------------------------
# 1. 설정 및 상수 정의
# ------------------------------------------
BACKEND_URL = "http://127.0.0.1:8000/analyze"
CHAT_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="교통사고 AI 솔루션", page_icon="⚖️", layout="wide")

# 스타일링 (카카오톡 느낌의 말풍선 + 스크롤바 디자인)
st.markdown("""
<style>
/* 채팅 메시지 스타일 */
.user-msg {
    background-color: #FEE500;
    color: #000000;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px 0;
    text-align: right;
    float: right;
    clear: both;
    max-width: 70%;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
.bot-msg {
    background-color: #f0f2f5;
    color: #000000;
    padding: 15px;
    border-radius: 15px;
    margin: 5px 0;
    text-align: left;
    float: left;
    clear: both;
    max-width: 70%;
    border: 1px solid #ddd;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
.clearfix {
    overflow: auto;
}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ 교통사고 과실 판단 AI + 법률 챗봇")

# ------------------------------------------
# 2. 세션 상태 초기화
# ------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
    
# 챗봇용 컨텍스트 변수
if "accident_context" not in st.session_state:
    st.session_state["accident_context"] = ""
if "vision_summary" not in st.session_state:
    st.session_state["vision_summary"] = ""
if "initial_rag" not in st.session_state:  # 오타 수정 (initual -> initial)
    st.session_state["initial_rag"] = ""
if "fault_analysis" not in st.session_state:
    st.session_state["fault_analysis"] = ""


# -----------------------------------------
# 3. 입력 UI
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

st.info("아래 항목들을 빠짐없이 선택해주세요.")

# 2. 사고 장소 및 신호
st.subheader("2. 사고 장소 및 신호")
col2_1, col2_2 = st.columns(2)
with col2_1:
    road_type = st.radio("사고 발생 도로 유형", ["신호등 있는 교차로", "신호등 없는 교차로", "직선 도로", "주차장", "기타"])
with col2_2:
    traffic_light = st.radio("신호등 상태", ["녹색(직진)", "황색", "적색", "좌회전", "기타/신호없음"])

st.divider()

# 3. 내 차량 상황
st.subheader("3. 내 차량 상황")
col3_1, col3_2, col3_3, col3_4 = st.columns(4)
with col3_1:
    my_action = st.radio("사고 당시 내 행동", ["직진 중", "좌회전 중", "우회전 중", "유턴 중", "차선 변경 중", "정차 중", "후진 중"], key="my_action")
with col3_2:
    my_signal = st.radio("내 차 방향지시등", ["켜지 않음", "좌측 깜빡이", "우측 깜빡이", "비상등", "기타"], key="my_signal")
with col3_3:
    my_traffic_light = st.radio("내 주행 신호 (진입 시)", ["녹색 (정상 진입)", "황색 (딜레마 존)", "적색 (신호 위반)", "좌회전 신호", "비보호 좌회전", "우회전 전용 신호", "신호 없음"])
with col3_4:
    my_speed = st.radio("내 차량 속도", ["정지", "서행(20km/h 이하)", "보통(20~50km/h)", "빠름(50km/h 이상)"], key="my_speed")

st.divider()

# 4. 상대 차량 상황
st.subheader("4. 상대 차량 상황")
col4_1, col4_2, col4_3, col4_4 = st.columns(4)
with col4_1:
    opp_action = st.radio("상대 차량 행동", ["직진 중", "좌회전 중", "우회전 중", "유턴 중", "차선 변경 중", "정차 중", "후진 중", "기타/모름"])
with col4_2:
    opp_signal = st.radio("상대 차 방향 지시등", ["켜지 않음", "좌측 깜빡이", "우측 깜빡이", "비상등", "기타"])
with col4_3:
    opp_traffic_light = st.radio("상대 주행 신호 (진입 시)", ["녹색 (정상 진입)", "황색 (딜레마 존)", "적색 (신호 위반)", "좌회전 신호", "비보호 좌회전", "우회전 전용 신호", "신호 없음"])
with col4_4:
    opp_direction = st.radio("상대 차량 진입 방향", ["정면 (맞은 편)", "좌측에서 진입", "우측에서 진입", "후방 추돌", "기타"])

st.divider()

# 5. 충돌 상황
st.subheader("5. 충돌 부위")
hit_part = st.radio("내 차량 충돌 부위", ["전면", "전면 좌측", "전면 우측", "좌측면", "우측면", "후면", "후면 좌측", "후면 우측"], horizontal=True)

st.divider()

# 6. 추가 위반 사항
st.subheader("6. 추가 정황 (중복 선택 가능)")
col6_1, col6_2 = st.columns(2)
with col6_1:
    my_fault = st.multiselect("내 차량 특이사항", ["과속", "신호 위반", "중앙선 침범", "안전거리 미확보", "급정거", "음주/무면허", "해당 없음"], default=["해당 없음"])
with col6_2:
    opp_fault = st.multiselect("상대 차량 특이사항", ["과속", "신호 위반", "중앙선 침범", "안전거리 미확보", "급정거", "음주/무면허", "해당 없음"], default=["해당 없음"])

# 프롬프트 조합
final_text_prompt = f"""
[기본 정보] 일시: {acc_date} {acc_time}, 장소: {acc_loc}, 블랙박스: {bbox}
[도로 환경] 도로 형태: {road_type}, 신호 시스템 상태: {traffic_light}
[나의 주행 정보] 행동: {my_action}, 진입 신호: {my_traffic_light}, 깜빡이: {my_signal}, 속도: {my_speed}, 특이사항: {', '.join(my_fault)}
[상대방 주행 정보] 행동: {opp_action}, 진입 신호: {opp_traffic_light}, 깜빡이: {opp_signal}, 진입 방향: {opp_direction}, 특이사항: {', '.join(opp_fault)}
[충돌 정보] 내 차 충돌 부위: {hit_part}
"""

st.divider()

# 7. 사진 업로드
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
# 8. 분석 요청 버튼
# ==========================================
if st.button("🚀 AI 분석 시작", type="primary", use_container_width=True):
    if not acc_loc:
        st.error("기본 정보의 '발생 장소'를 입력해주세요.")
    else:
        with st.spinner("AI가 사고 정황을 분석하고 판례를 검색 중입니다..."):
            try:
                files = []
                if images:
                    for img in images:
                        img.seek(0)
                        files.append(("image_files", (img.name, img, img.type)))
                
                data_payload = {"text_query": final_text_prompt}
                if st.session_state["thread_id"]:
                    data_payload["thread_id"] = st.session_state["thread_id"]
                
                res = requests.post(BACKEND_URL, files=files, data=data_payload, timeout=120)
                
                if res.status_code == 200:
                    response_data = res.json()
                    st.session_state["thread_id"] = response_data["thread_id"]
                    st.session_state["analysis_result"] = response_data["result"]
                    
                    # 챗봇용 컨텍스트 저장
                    st.session_state["accident_context"] = final_text_prompt
                    st.session_state["vision_summary"] = response_data.get("image_summary", "")
                    st.session_state["initial_rag"] = response_data.get("rag_context") # 오타 수정
                    
                    result = response_data["result"]
                    fault_analysis_text = f"과실 비율: 나 {result['fault_ratio']['me']}% : 상대방 {result['fault_ratio']['opponent']}%"
                    st.session_state["fault_analysis"] = fault_analysis_text
                    
                    # 분석 완료 후 새로고침 (결과 화면 표시를 위해)
                    st.rerun()

                else:
                    st.error(f"서버 오류: {res.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("백엔드 서버에 연결할 수 없습니다. main.py가 실행 중인지 확인해주세요.")
            except Exception as e:
                st.error(f"오류 발생: {e}")

# ==========================================
# 9. 분석 결과 표시 (분석된 경우에만 표시)
# ==========================================
if st.session_state["analysis_result"]:
    result = st.session_state["analysis_result"]
    
    st.success("분석이 완료되었습니다!")

    # 전략 제안 배너
    guide = result.get("action_guide", "보험 처리 권장")
    if guide == "개인 합의 유리":
        st.success(f"💰 **[전략 제안: {guide}]**\n\n> **\"지금은 현금 합의가 지갑을 지키는 길입니다.\"**\n> 사용자분의 과실이 높지만 피해가 경미합니다. 보험 처리 시 **3년 갱신 유예 및 할증**으로 인한 손해가 수리비보다 클 가능성이 높습니다.")
    elif guide == "보험 처리 유리":
        st.info(f"🛡️ **[전략 제안: {guide}]**\n\n> **\"내 과실이 거의 없습니다. 보험사를 적극 활용하세요.\"**\n> 사용자분은 명확한 피해자입니다. 보험료 할증 걱정 없이 **상대방 보험사로부터 수리비와 렌트비**를 전액 보상받으시면 됩니다.")
    else:
        st.warning(f"⚖️ **[전략 제안: {guide}]**\n\n> **\"분쟁 가능성이 높습니다. 보험사를 방패로 쓰세요.\"**\n> 과실 비율 다툼이 예상되거나 사고 규모가 큽니다. 개인 합의 시 추후 **뺑소니 신고나 과도한 합의금 요구** 등 법적 위험이 있으니 보험 접수가 가장 안전합니다.")

    # 상세 조언
    st.markdown("### 💡 AI 상세 전략 가이드")
    st.info(result.get('advice', 'AI가 상세 조언을 생성하지 못했습니다.'))

    # 과실 비율 그래프
    st.subheader("📊 예상 과실 비율")
    if "fault_ratio" in result:
        me = result["fault_ratio"]["me"]
        opp = result["fault_ratio"]["opponent"]
        c1, c2 = st.columns(2)
        with c1:
            st.metric("나 (사용자)", f"{me}")
            st.progress(me / 100)
        with c2:
            st.metric("상대방", f"{opp}")
            st.progress(opp / 100)

    # 근거 및 판례 탭
    st.markdown("---")
    tab1, tab2 = st.tabs(["🔍 과실 산정 근거", "📜 참조 판례 (요약)"])
    with tab1:
        st.write(result.get("reasoning", ""))
        st.caption(f"상황 요약: {result.get('summary', '')}")
    with tab2:
        st.markdown(st.session_state["initial_rag"] if st.session_state["initial_rag"] else "관련 판례 정보가 없습니다.")
    
    with st.expander("📝 분석에 사용된 입력 데이터 확인", expanded=False):
        st.text(final_text_prompt)

    # ==========================================
    # 10. AI 법률 챗봇 (스크롤 박스 적용)
    # ==========================================
    st.markdown("---")
    st.header("💬 AI 법률 상담 챗봇")
    st.caption("분석 결과를 바탕으로 추가적인 법률 상담을 받아보세요.")

    # [핵심 수정 사항] height=600으로 설정하여 스크롤 가능한 박스 생성
    chat_container = st.container(height=600)

    # 대화 히스토리 표시 (스크롤 박스 내부)
    with chat_container:
        if not st.session_state["chat_history"]:
            st.info("궁금한 점을 물어보세요! (예: 상대방이 합의금을 너무 많이 요구하면 어떡해?)")
        
        for chat in st.session_state["chat_history"]:
            if chat["role"] == "user":
                st.markdown(f"""
                <div class="clearfix">
                    <div class="user-msg">{chat['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="clearfix">
                    <div class="bot-msg">🤖 {chat['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # 입력창 (스크롤 박스 외부 하단에 위치)
    st.markdown("---")
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_question = st.text_input("질문을 입력하세요.", placeholder="메시지 입력...", label_visibility="collapsed")
        with col_btn:
            send_btn = st.form_submit_button("전송", use_container_width=True)

    if send_btn and user_question:
        # 사용자 메시지 표시
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        
        # 즉시 UI 갱신을 위해 컨테이너에 사용자 메시지 추가 (선택 사항이나 리런으로 대체)
        # st.rerun()을 호출하면 어차피 다시 그려지므로 여기서는 데이터만 추가하고 요청 시작
        
        with st.spinner("AI가 답변을 생성하고 있습니다..."):
            try:
                chat_payload = {
                    "thread_id": st.session_state["thread_id"],
                    "user_message": user_question,
                    "accident_context": st.session_state["accident_context"],
                    "vision_summary": st.session_state["vision_summary"],
                    "initial_rag": st.session_state["initial_rag"],
                    "fault_analysis": st.session_state["fault_analysis"]
                }
                
                chat_res = requests.post(CHAT_URL, data=chat_payload, timeout=60)
                
                if chat_res.status_code == 200:
                    chat_data = chat_res.json()
                    ai_response = chat_data["response"]
                    
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": ai_response
                    })
                    st.rerun() # 화면 갱신하여 새 메시지 표시
                else:
                    st.error(f"챗봇 오류: {chat_res.text}")
            except Exception as e:
                st.error(f"통신 오류: {e}")

    # 대화 초기화 버튼
    if st.session_state["chat_history"]:
        if st.button("대화 기록 초기화"):
            st.session_state["chat_history"] = []
            st.rerun()