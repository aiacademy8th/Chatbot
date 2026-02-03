import os
import io
from dotenv import load_dotenv
from google.cloud import speech
from pydub import AudioSegment

load_dotenv()

class GoogleSTTHandler:
    def __init__(self):
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
             print("⚠️ 경고: 구글 인증 키 환경변수가 없습니다.")
        
        try:
            self.client = speech.SpeechClient()
        except Exception:
            self.client = None

    def _convert_to_linear16(self, audio_input):
        try:
            # 입력이 파일 경로면 파일에서, 아니면 바이트에서 로드
            if isinstance(audio_input, str) and os.path.exists(audio_input):
                audio = AudioSegment.from_file(audio_input)
            else:
                audio = AudioSegment.from_file(io.BytesIO(audio_input))

            # Google STT 포맷: Mono, 16kHz
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            return buffer.getvalue()
        
        except Exception as e:
            print(f"오디오 변환 실패: {e}")
            return None
        
    def transcribe_audio(self, audio_source) -> str:
        if not self.client:
            return "STT 클라이언트 초기화 실패"
        
        content = self._convert_to_linear16(audio_source)
        if not content:
            return "오디오 전처리 실패"
        
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
            enable_automatic_punctuation=True,
            use_enhanced=True,
            model="default"
        )

        try:
            response = self.client.recognize(config=config, audio=audio)
            transcripts = [res.alternatives[0].transcript for res in response.results]
            return " ".join(transcripts) if transcripts else "(인식된 음성 내용 없음)"
        except Exception as e:
            return f"STT API 오류: {e}"