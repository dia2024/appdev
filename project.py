import streamlit as st
from openai import OpenAI
import scipy.io.wavfile as wav
import tempfile
import sounddevice as sd
import os
import shutil
import cv2
import av
from keras.models import load_model
from PIL import Image, ImageOps
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import random
from dotenv import load_dotenv 
load_dotenv() 

# -----------------------------
# 설정값
# -----------------------------
sd.default.device = 1
SAMPLERATE = 48000
DURATION = 3

# 이미 선언한 client 사용 (키가 코드에 들어있다고 가정)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or st.secrets['OPENAI_API_KEY']
client = OpenAI(api_key=OPENAI_API_KEY)
# -----------------------------
# 유틸 함수들
# -----------------------------
def change_panel(goto: int):
    if goto == 0:
        st.session_state.panel = "voice_select"
        st.rerun()
    elif goto == 1:
        st.session_state.panel = "memo"
        st.rerun()
    elif goto == 2:
        st.session_state.panel = "hand_lang"
        st.rerun()


def record_audio():
    print("🎙️ 녹음 중...")
    audio = sd.rec(int(SAMPLERATE * DURATION), samplerate=SAMPLERATE, channels=1, dtype="int16")
    sd.wait()
    return audio

def transcribe(audio_data):
    # 녹음 데이터를 임시 wav로 저장 → Whisper(Transcribe) 호출 → 텍스트 반환
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        wav.write(temp_wav.name, SAMPLERATE, audio_data)
        temp_wav_path = temp_wav.name

    try:
        with open(temp_wav_path, "rb") as file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=file,
                language="ko"
            )
        return transcript.text
    finally:
        try:
            os.remove(temp_wav_path)
        except:
            pass

def text_to_speech(text: str, voicec: str = "alloy") -> str:
    """
    text -> mp3 파일 생성 후 파일 경로 반환.
    voice: OpenAI 음성 이름 (예: "alloy", "fable", "echo", "nova", "shimmer" 등)
    반환값: 생성된 mp3 파일의 전체 경로
    """
    # 임시 파일 생성
    out_dir = tempfile.mkdtemp(prefix="tts_out_")
    out_path = os.path.join(out_dir, "speech.mp3")

    try:
        # OpenAI TTS 호출 (SDK의 stream_to_file 이용)
        # response = client.audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text)
        # response.stream_to_file(out_path)

        # 일부 SDK 버전에서는 stream_to_file가 없을 수 있으므로 안전하게 시도
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voicec,
            input=text
        )

        # resp가 stream_to_file 메서드를 제공하면 사용하고, 아니면 바이너리로 저장 시도
        if hasattr(resp, "stream_to_file"):
            resp.stream_to_file(out_path)
        else:
            # resp를 읽어서 파일로 쓰기 (대부분의 최신 SDK에서 필요 없을 수 있음)
            # resp는 response-like 객체라 가정. 아래는 안전한 fallback.
            try:
                data = resp.read()  # 일부 구현에서 .read()로 바이너리 얻기
            except Exception:
                # 마지막 수단: resp를 str로 변환해서 바이트로 저장 (잘 동작하지 않을 수 있음)
                data = bytes(str(resp), "utf-8")
            with open(out_path, "wb") as f:
                f.write(data)

        return out_path
    except Exception as e:
        # 실패 시 임시 디렉토리 정리
        shutil.rmtree(out_dir, ignore_errors=True)
        raise e

# -----------------------------
# Streamlit 초기화
# -----------------------------
if "panel" not in st.session_state:
    st.session_state.panel = "voice_select"

st.sidebar.markdown("메뉴 이동기")

if st.sidebar.button("목소리 선택으로 이동."):
    if st.session_state.panel == "voice_select":
        st.toast("이미 목소리 선택에 있습니다. 휴먼.")
    else:
        change_panel(0)

if st.sidebar.button("음성 메모로 이동."):
    if st.session_state.panel == "memo":
        st.toast("이미 음성 메모에 있습니다. 휴먼.")
    else:
        change_panel(1)

if st.sidebar.button("수어 인식기로 이동."):
    if st.session_state.panel == "hand_lang":
        st.toast("이미 수어 인식기에 있습니다. 휴먼.")
    else:
        change_panel(2)


# 목소리 선택 라디오 (전역 키로 저장)
if st.session_state.panel == "voice_select":
    st.title("원하시는 목소리를 골라주세요.")
    voice = st.radio("", options=["어린이 목소리", "어른이 목소리", "노인이 목소리"])
    if st.button("대화 시작하기"):
        if voice == "어린이 목소리":
            questions = [
                "오늘 기분이 어때?",
                "좋아하는 음식은 뭐야?",
                "오늘 뭐 하고 놀고 싶어?",
                "오늘 학교에서 무슨일이 있었어?",
                "최근에 배운 재밌는 건 뭐야?",
                "오늘 하고 싶은 게임이 있어?",
                "가장 좋아하는 만화는 뭐야?"
            ]
            txt = random.choice(questions)
            voice_name = "fable"
        elif voice == "어른이 목소리":
            questions = [
                "오늘 업무는 잘 되었나요?",
                "최근 읽은 책이 있나요?",
                "주말 계획이 있나요?",
                "오늘 점심 뭐 드셨나요?",
                "최근 관심 있는 뉴스가 있나요?"
            ]
            txt = random.choice(questions)
            voice_name = "alloy"
        elif voice == "노인이 목소리":
            questions = [
                "오늘 하루는 어땠나요?",
                "옛날 이야기를 하나 해주실래요?",
                "좋아하는 취미가 무엇인가요?",
                "젊었을 때 가장 기억에 남는 일은 뭐예요?",
                "건강 관리는 잘 하고 계신가요?"
            ]
            txt = random.choice(questions)
            voice_name = "echo"
        
        try:
            with st.spinner("TTS 생성 중..."):
                mp3_path = text_to_speech(txt, voicec=voice_name)
            # Streamlit에서 재생
            with open(mp3_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")
        except:
            print("error")

# 음성 메모 패널
elif st.session_state.panel == "memo":
    st.title("음성 메모")

    # 버튼 - 녹음(음성→텍스트)
    if st.button("음성 녹음하기 (STT)"):
        audio_data = record_audio()
        user_text = transcribe(audio_data)
        st.write(user_text)

        

# 수어 인식기 패널 (플레이스홀더)
elif st.session_state.panel == "hand_lang":
    st.title("수어 인식기")
    st.title("✋ 사랑의 가위바위보 게임 (Love Machine)")

    # --- 모델과 라벨 불러오기 ---

    @st.cache_resource
    def load_teachable_model():
        model = load_model("keras_model.h5", compile=False)
        return model

    @st.cache_data
    def load_labels():
        with open("labels.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]

    model = load_teachable_model()
    labels = load_labels()

    # --- 이미지 전처리 함수 ---
    def preprocess_for_model(frame):
        """Teachable Machine 입력 형식(224x224)으로 변환"""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        return np.expand_dims(normalized_image_array, axis=0)

    # --- 비디오 프레임 처리 함수 ---
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            # 모델 입력 준비
            data = preprocess_for_model(img)
            # 예측 수행
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = labels[index].strip()
            confidence = prediction[0][index]
            # 결과 화면에 표시
            text = f"{class_name.split()[1]} ({confidence*100:.1f}%)"
            cv2.putText(img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 3, cv2.LINE_AA)

        except Exception as e:
            cv2.putText(img, "Error", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # --- WebRTC 실행 ---
    webrtc_streamer(
        key="hand-detect",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
    )
    st.write("카메라가 켜지면 손 모양을 자동으로 인식하고, Teachable Machine 모델로 분류합니다 ✋")
