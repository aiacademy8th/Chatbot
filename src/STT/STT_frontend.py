import streamlit as st
import requests

# 벡인드 주소
BACKEND_URL = "https://chatbot-backend-599050237852.asia-northeast3.run.app/stt"

st.set_page_config(page_title="교통사고 AI 음성 입력", page_icon="🚓")
st.title("🚓 교통사고 정황 음성 리포트")
st.markdown("### 1단계: 사고 상황을 구두로 진술해 주세요.")
st.info("아래 마이크 버튼을 클릭하여 녹음하고, 다시 클릭하여 종료하세요.")

# --- 1. 음성 녹음 위젯 ---
# Streamlit 1.40+ 버전의 네이티브 위젝 사용
audio_value = st.audio_input("마이크를 켜고 말씀하세요.")

if audio_value:
    # 녹음된 오디오를 화면에서 재생 확인
    st.audio(audio_value)

    # --- 2. 서버 전송 버튼 ---
    if st.button("🚀 분석 요청 (서버 전송)"):
        with st.spinner("음성을 텍스트로 변환 중입니다..."):
            try:
                # 파일 객체 준비 ("file"은 FastAPI의 매개변수명과 일치해야 함)
                files = {"file": ("input_voice.wav", audio_value, "audio/wav")}
                
                # POST 요청 전송
                response = requests.post(BACKEND_URL, files=files)

                # --- 3. 결과 처리 ---
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ 변환 완료!")

                    # 텍스트 출력 영역
                    st.text_area(
                        "변환된 텍스트 (STT 결과)",
                        value=result.get("text", ""),
                        height=200
                    )
                else:
                    st.error(f"서버 오류 발생: {response.status_code}")
                    st.write(response.text)
            except requests.exceptions.ConnectionError:
                st.error("❌ 서버에 연결할 수 없습니다. backend.py가 실행 중인지 확인하세요.")
            except Exception as e:
                st.error(f"알 수 없는 오류: {e}")