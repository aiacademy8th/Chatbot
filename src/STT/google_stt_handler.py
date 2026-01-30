import os
import io
import threading
import wave
from dotenv import load_dotenv
from google.cloud import speech
from pydub import AudioSegment
import pyaudio                  # 마이크 입력을 위해 추가

load_dotenv()

class GoogleSTTHandler:
    def __init__(self):
        """
        초기화: 환경 변수에서 인증 정보를 자동으로 확인
        """

        # 환경 변수 확인
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not credentials_path:
            raise ValueError(
                "❌ 'GOOGLE_APPLICATION_CREDENTIALS' 환경 변수가 설정되지 않았습니다. "
                ".env 파일을 확인해 주세요."
            )
        
        print(f"🔑 Google STT 인증 키 로드 확인: {credentials_path}")

        # Google Client는 환경 변수 GOOGLE_APPLICATION_CREDENTIALS 가 있으면
        # 자동으로 그것을 사용하여 인증

        try:
            self.client = speech.SpeechClient()
        except Exception as e:
            print(f"❌ 클라이언트 초기화 실패. 키 파일 경로와 권한을 확인하세요.\n에러: {e}")
            self.client = None

    def _convert_to_linear16(self, audio_input):
        """
        다양한 포맷(mp3, ogg, webm 등)을 Google STT용 WAV(Linear16, 16kHz, Mono)로 변환
        """

        try:
            # 입력이 파일 경로인지 바이트인지 확인
            if isinstance(audio_input, str) and os.path.exists(audio_input):
                audio = AudioSegment.from_file(audio_input)
            else:
                # 바이너리 데이터인 경우
                audio = AudioSegment.from_file(io.BytesIO(audio_input))

            # 포맷 변환: Mono(채널 1), 16000Hz
            audio = audio.set_channels(1).set_frame_rate(16000)

            # 바이트 버퍼로 내보내기
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            return buffer.getvalue()
        
        except Exception as e:
            print(f"⚠️ 오디오 변환 중 오류 발생: {e}")
            return None
        
    def transcribe_audio(self, audio_source) -> str:
        """
        오디오 -> 텍스트 변환 실행
        """

        if not self.client:
            return "오류: STT 클라이언트가 초기화되지 않았습니다."
        
        # 1. 포맷 정규화(전처리)
        content = self._convert_to_linear16(audio_source)
        if not content:
            return "오디오 전처리(변환) 실패"
        
        # 2. Google STT 설정
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
            enable_automatic_punctuation=True,      # 구두점 자동
            use_enhanced=True,
            model="default"
        )

        # 3. API 호출
        try:
            response = self.client.recognize(config=config, audio=audio)

            # 결과 결합
            full_transcript = []
            for result in response.results:
                full_transcript.append(result.alternatives[0].transcript)

            final_text = " ".join(full_transcript)

            if not final_text:
                return "(인식된 음성 내용 없음)"
            
            return final_text
        
        except Exception as e:
            return f"STT API 호출 중 오류 발생: {e}"
        
# --- 🎤 실시간 녹음 테스트 로직 ---
def record_micriphone(output_filename="temp_record.wav"):
    """
    사용자가 말을 멈출 때까지 마이크 입력을 녹음하는 함수 (별도 스레드에서 실행됨)
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000        # Google STT에 최적화된 주파스

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
        
# --- 테스트 실행 코드 ---
if __name__ == "__main__":
    # 테스트용 파일 경로
    pass