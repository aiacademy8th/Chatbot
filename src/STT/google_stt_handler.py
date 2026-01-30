import os
import io
import threading
import wave
from dotenv import load_dotenv
from google.cloud import speech
from pydub import AudioSegment
import pyaudio                  # 마이크 입력을 위해 추가

load_dotenv()

# 전역 변수로 녹음 상태 제어
is_recording = False

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
        [디버깅 강화 버전]
        다양한 포맷을 Google STT용 WAV(Linear16, 16kHz, Mono)로 변환
        중간 과정을 파일로 저장하여 확인합니다.
        """
        try:
            print("🔊 [Debug] 오디오 변환 시작...")
            
            # 1. 입력 데이터 로드
            if isinstance(audio_input, str) and os.path.exists(audio_input):
                audio = AudioSegment.from_file(audio_input)
            else:
                # 바이너리 데이터인 경우
                input_bytes = io.BytesIO(audio_input)
                # 디버깅용: 들어온 원본 파일 저장해보기 (프로젝트 폴더에 저장됨)
                with open("debug_received_input.webm", "wb") as f:
                    f.write(audio_input)
                print("💾 [Debug] 원본 파일 저장 완료: debug_received_input.webm")
                
                audio = AudioSegment.from_file(input_bytes)

            print(f"   ℹ️ 원본 오디오 정보: {audio.channels}ch, {audio.frame_rate}Hz, {audio.duration_seconds}초")

            # 2. 포맷 변환: Mono(채널 1), 16000Hz
            audio = audio.set_channels(1).set_frame_rate(16000)

            # 3. 변환된 결과 확인용 저장
            audio.export("debug_converted_output.wav", format="wav")
            print("💾 [Debug] 변환 파일 저장 완료: debug_converted_output.wav")

            # 4. 바이트 버퍼로 내보내기
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            return buffer.getvalue()
        
        except Exception as e:
            print(f"⚠️ 오디오 변환 중 치명적 오류: {e}")
            # ffmpeg 경로 문제일 가능성이 높음
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
def record_microphone(output_filename="temp_record.wav"):
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
    
    print(f"   [System] 녹음 스레드 시작 (저장 경로: {output_filename})")

    frames = []

    # 글로벌 플래그를 확인하며 녹음
    while is_recording:
        try:
            # overflow 예외 무시 (버퍼 꽉 참 방지)
            data = stream.read(CHUNK)
            frames.append(data)
        except Exception as e:
            print(f"녹음 중 에러: {e}")
            break
        
    
    stream.start_stream()
    stream.close()
    p.terminate()

    # WAV 파일로 저장
    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print("   [System] 녹음 파일 저장 완료")
        
# --- 테스트 실행 코드 ---
if __name__ == "__main__":
    handler = GoogleSTTHandler()

    print("\n" + "=" * 40)
    print("🎙️ Google STT 실시간 녹음 테스트")
    print("=" * 40)

    # 1. 녹음 시작 대기
    while True:
        user_input = input("녹음을 시작하려면 'y'를 입력하고 엔터를 누르세요: ").strip().lower()
        if user_input == 'y':
            break
    
    # 2. 녹음 시작 (스레드 활용)
    is_recording = True
    record_thread = threading.Thread(target=record_microphone, args=("test_voice.wav", ))
    record_thread.start()

    print("\n🔴 녹음 중입니다... (말씀하세요)")

    # 3. 녹음 종료 대기
    while True:
        user_input = input("녹음을 종료하려면 'y'를 입력하고 엔터를 누르세요: ").strip().lower()
        if user_input == "y":
            is_recording = False        # 스레드 루프 종료 신호
            record_thread.join()        # 녹은 스레드가 완전히 끝날 때까지 대기
            print("⏹️ 녹음 완료. 변환을 시작합니다...\n")
            break

    # 4. 변환 요청
    if os.path.exists("test_voice.wav"):
        result_text = handler.transcribe_audio("test_voice.wav")
        print("-" * 30)
        print("📝 변환 결과:")
        print(f"👉 {result_text}")
        print("-" * 30)

        try:
            # 테스트 후 파일 삭제
            os.remove("test_voice.wav")
        except:
            print("test_voice.wav 파일 삭제 실패\n")
        
    else:
        print("❌ 녹음 파일 생성 실패")