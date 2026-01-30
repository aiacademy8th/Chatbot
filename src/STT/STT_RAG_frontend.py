import streamlit as st
import requests

# 백엔드 주소
BACKEND_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="교통사고 AI 솔루션", page_icon="⚖️", layout="wide")

st.title("⚖️ 교통사고 과실 판단 AI")
st.markdown("음성으로 상황을 설명하고, 사고 사진을 올려주세요. AI가 과실 비율과 대응법을 알려드립니다.")

col1, col2 = st.columns(2)

with col1:
    st.header("1. 정보 입력")

    # 이미지 업로드
    uploaded_image = st.file_uploader("📸 사고 현장 사진 (선택)", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        # [수정] 오타 수정: 업도르된 -> 업로드된
        st.image(uploaded_image, caption="업로드된 사진", use_container_width=True)

    # 음성 녹음
    st.markdown("#### 🎤 사고 정황 설명")
    st.info("마이크 버튼을 누르고 사고 상황(위치, 진행 방향, 신호 등)을 자세히 말씀해 주세요.")
    audio_value = st.audio_input("녹음 시작")

with col2:
    st.header("2. 분석 결과")

    # 분석 버튼
    if audio_value:
        if st.button("🚀 AI 분석 시작", type="primary", use_container_width=True):
            with st.spinner("음성 분석 및 판례 검색 중입니다... (최대 30초 소요)"):
                try:
                    # 파일 준비
                    files = {
                        "voice_file": ("voice.wav", audio_value, "audio/wav")
                    }
                    
                    if uploaded_image:
                        # 파일명, 데이터, MIME 타입 순서
                        files["image_file"] = (uploaded_image.name, uploaded_image, uploaded_image.type)
                    
                    # 서버 요청
                    # [추가] timeout 설정: GPT 분석이 길어질 수 있으므로 넉넉하게 60초 설정
                    response = requests.post(BACKEND_URL, files=files, timeout=60)

                    if response.status_code == 200:
                        result = response.json()

                        # 결과 출력
                        st.success("분석이 완료되었습니다!")

                        st.markdown("### 🗣️ 사용자 진술 (STT)")
                        st.info(result["transcript"])

                        st.markdown("### 🤖 AI 법률 자문 결과")
                        st.write(result["answer"])

                        if result.get("references"):
                            with st.expander("📚 참고한 판례/법규 확인"):
                                for ref in result["references"]:
                                    st.write(f"- {ref}")
                    else:
                        st.error(f"서버 오류: {response.status_code}")
                        st.write(response.text)

                except requests.exceptions.Timeout:
                    st.error("⏳ 서버 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.")
                except Exception as e:
                    st.error(f"연결 오류: {e}")
    else:
        st.warning("👈 왼쪽에서 음성을 먼저 녹음해 주세요.")