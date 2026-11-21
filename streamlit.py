import difflib
import io
import random
import re

import streamlit as st
from gtts import gTTS
from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError

# API 키는 코드에 넣지 않고, 화면 왼쪽 위(사이드바)에서 입력받습니다.
STT_MODEL = "whisper-1"  # 또는 "gpt-4o-mini-transcribe", "gpt-4o-transcribe"

st.set_page_config(page_title="말하며 배우는 영어 단어", page_icon="🎤", layout="centered")

# ---------------------------------------------------------------
# 기본 단어장 (영어, 한글 뜻)
# ---------------------------------------------------------------
DEFAULT_WORDS = [
    ("apple", "사과"), ("book", "책"), ("water", "물"), ("friend", "친구"),
    ("school", "학교"), ("beautiful", "아름다운"), ("important", "중요한"),
    ("difficult", "어려운"), ("remember", "기억하다"), ("understand", "이해하다"),
    ("different", "다른"), ("vegetable", "채소"), ("restaurant", "식당"),
    ("comfortable", "편안한"), ("environment", "환경"), ("experience", "경험"),
    ("necessary", "필요한"), ("language", "언어"), ("knowledge", "지식"),
    ("opportunity", "기회"), ("tomorrow", "내일"), ("choose", "선택하다"),
    ("develop", "발전시키다"), ("journey", "여정"), ("weather", "날씨"),
    ("library", "도서관"), ("breakfast", "아침 식사"), ("traditional", "전통적인"),
    ("improve", "향상시키다"), ("arrive", "도착하다"),
]

PASS_SCORE = 0.8  # 이 유사도 이상이면 정답 처리


# ---------------------------------------------------------------
# 유틸 함수
# ---------------------------------------------------------------
def normalize(text: str) -> str:
    return re.sub(r"[^a-z\s]", "", text.lower()).strip()


def similarity(target: str, transcript: str) -> float:
    """목표 단어와 인식 결과의 최고 유사도 (문장으로 말해도 단어 단위로 비교)."""
    target = normalize(target)
    transcript = normalize(transcript)
    if not transcript:
        return 0.0
    candidates = [transcript] + transcript.split()
    return max(difflib.SequenceMatcher(None, target, c).ratio() for c in candidates)


class NoSpeechError(Exception):
    """음성이 인식되지 않았을 때 발생."""


def recognize(audio_file, api_key: str) -> str:
    """st.audio_input 의 WAV 데이터를 OpenAI 음성 인식 API로 텍스트 변환."""
    client = OpenAI(api_key=api_key)
    audio_file.seek(0)
    result = client.audio.transcriptions.create(
        model=STT_MODEL,
        file=("speech.wav", audio_file.read(), "audio/wav"),
        language="en",  # 영어 단어 학습이므로 영어로 고정
    )
    text = (result.text or "").strip()
    if not normalize(text):
        raise NoSpeechError()
    return text


@st.cache_data(show_spinner=False)
def tts_bytes(word: str) -> bytes | None:
    try:
        buf = io.BytesIO()
        gTTS(word, lang="en").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def parse_custom_words(raw: str):
    words = []
    for line in raw.splitlines():
        if "," in line:
            en, ko = line.split(",", 1)
            if en.strip():
                words.append((en.strip(), ko.strip()))
    return words


# ---------------------------------------------------------------
# 세션 상태 초기화
# ---------------------------------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("words", DEFAULT_WORDS)
    ss.setdefault("current", random.choice(ss.words))
    ss.setdefault("q", 0)               # 문제 번호 (녹음 위젯 key 갱신용)
    ss.setdefault("result", None)       # 현재 문제의 채점 결과
    ss.setdefault("correct", 0)
    ss.setdefault("total", 0)
    ss.setdefault("wrong", {})          # {단어: 뜻}


def next_word():
    ss = st.session_state
    pool = [w for w in ss.words if w != ss.current] or ss.words
    ss.current = random.choice(pool)
    ss.q += 1
    ss.result = None


def reset_all():
    for k in ["current", "q", "result", "correct", "total", "wrong"]:
        st.session_state.pop(k, None)
    init_state()


init_state()
ss = st.session_state

# ---------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------
with st.sidebar:
    st.header("🔑 OpenAI API 키")
    api_key_input = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-...",
        help="입력한 키는 이 브라우저 세션에서만 사용되며 서버에 저장되지 않습니다.",
        label_visibility="collapsed",
    )
    api_key = api_key_input.strip()
    if not api_key:  # 배포 환경의 Secrets에 키가 있으면 대체 사용 (선택 사항)
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = ""
    if api_key:
        st.caption("✅ API 키가 설정되었습니다.")
    else:
        st.caption("⚠️ API 키를 입력해야 음성 인식을 쓸 수 있어요.")

    st.divider()
    st.header("⚙️ 설정")
    mode = st.radio("학습 모드", ["🗣️ 발음 연습", "🧠 뜻 보고 말하기"])

    with st.expander("나만의 단어장 추가"):
        st.caption("한 줄에 `영어,뜻` 형식으로 입력하세요.")
        raw = st.text_area("단어 목록", placeholder="serendipity,뜻밖의 행운\nresilient,회복력 있는", height=120)
        if st.button("적용"):
            custom = parse_custom_words(raw)
            if custom:
                ss.words = custom
                reset_all()
                st.rerun()
            else:
                st.warning("형식이 올바르지 않습니다.")
        if st.button("기본 단어장으로 복원"):
            ss.words = DEFAULT_WORDS
            reset_all()
            st.rerun()

    st.divider()
    acc = (ss.correct / ss.total * 100) if ss.total else 0
    st.metric("정답 / 시도", f"{ss.correct} / {ss.total}")
    st.progress(min(acc / 100, 1.0), text=f"정확도 {acc:.0f}%")
    if st.button("🔄 기록 초기화"):
        reset_all()
        st.rerun()

# ---------------------------------------------------------------
# 메인 화면
# ---------------------------------------------------------------
st.title("🎤 말하며 배우는 영어 단어")

en, ko = ss.current
is_practice = mode.startswith("🗣️")

if is_practice:
    st.caption("단어를 듣고, 마이크로 따라 읽어 보세요.")
    st.markdown(f"## {en}")
    st.write(f"뜻: **{ko}**")
    audio_bytes = tts_bytes(en)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.caption("⚠️ 발음 듣기를 사용할 수 없습니다 (인터넷 연결 확인).")
else:
    st.caption("뜻을 보고 영어 단어를 말해 보세요.")
    st.markdown(f"## {ko}")
    st.write("이 뜻에 해당하는 영어 단어는? 🤔")

if not api_key:
    st.info("👈 왼쪽 위 사이드바에 OpenAI API 키를 입력하면 녹음 기능이 활성화됩니다.\n\n"
            "모바일에서는 왼쪽 위의 **>** 아이콘을 눌러 사이드바를 열어 주세요.")
    st.stop()

recording = st.audio_input("🎙️ 버튼을 눌러 녹음하고, 다시 눌러 종료하세요", key=f"audio_{mode}_{ss.q}")

# 녹음이 들어오고 아직 채점하지 않았다면 채점
if recording is not None and ss.result is None:
    with st.spinner("음성을 인식하는 중..."):
        try:
            transcript = recognize(recording, api_key)
            score = similarity(en, transcript)
            ss.result = {"transcript": transcript, "score": score, "ok": score >= PASS_SCORE}
        except NoSpeechError:
            ss.result = {"transcript": "", "score": 0.0, "ok": False, "error": "음성을 알아듣지 못했어요. 더 또렷하게 다시 말해 보세요."}
        except AuthenticationError:
            ss.result = {"transcript": "", "score": 0.0, "ok": False, "error": "OpenAI API 키가 올바르지 않습니다. 왼쪽 위 사이드바의 키를 다시 확인해 주세요.", "no_count": True}
        except RateLimitError:
            ss.result = {"transcript": "", "score": 0.0, "ok": False, "error": "API 사용 한도를 초과했거나 크레딧이 부족합니다. OpenAI 계정을 확인하세요.", "no_count": True}
        except APIConnectionError:
            ss.result = {"transcript": "", "score": 0.0, "ok": False, "error": "OpenAI 서버에 연결할 수 없습니다. 인터넷 연결을 확인하세요.", "no_count": True}
        except Exception as e:
            ss.result = {"transcript": "", "score": 0.0, "ok": False, "error": f"오류: {e}", "no_count": True}

    r = ss.result
    if not r.get("no_count") and not r.get("error"):
        ss.total += 1
        if r["ok"]:
            ss.correct += 1
            ss.wrong.pop(en, None)
        else:
            ss.wrong[en] = ko
    elif r.get("error") and not r.get("no_count"):
        # 인식 실패도 시도로 기록
        ss.total += 1
        ss.wrong[en] = ko

# 결과 표시
r = ss.result
if r:
    if r.get("error"):
        st.error(r["error"])
    else:
        st.write(f"인식된 음성: **{r['transcript']}**")
        st.progress(min(r["score"], 1.0), text=f"유사도 {r['score'] * 100:.0f}%")
        if r["ok"]:
            st.success("🎉 정답입니다!" if not is_practice else "🎉 발음이 좋아요!")
            st.balloons()
        else:
            st.warning(f"아쉬워요. 정답은 **{en}** 입니다. 다시 도전해 보세요!")
            if not is_practice:
                audio_bytes = tts_bytes(en)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")

col1, col2 = st.columns(2)
with col1:
    if st.button("🔁 다시 녹음", use_container_width=True):
        ss.q += 1
        ss.result = None
        st.rerun()
with col2:
    if st.button("➡️ 다음 단어", type="primary", use_container_width=True):
        next_word()
        st.rerun()

# 복습 목록
if ss.wrong:
    st.divider()
    st.subheader("📝 복습이 필요한 단어")
    for w, m in ss.wrong.items():
        st.write(f"- **{w}** — {m}")