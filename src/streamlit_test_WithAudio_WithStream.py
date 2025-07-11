import librosa
import streamlit as st
import uuid
import json
import numpy as np
import os
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
from appv4_withThread_withStream import invoke, stream_invoke
from typing import Optional, Any
import wave
import tempfile
import whisper
import soundfile as sf
from st_audiorec import st_audiorec
import io

# ------------------------- CONFIG -------------------------
st.set_page_config(page_title="Threaded Chatbot", layout="wide")
st.title("🧵 Threaded Chat - Integration with SNOW (BETA)")

# ------------------------- HELPERS -------------------------
def get_bot_response(user_message: str, full_context: Optional[Any] = None):
    l = invoke(user_message, full_context)
    return l

def get_bot_response_streaming(user_message: str, full_context: Optional[Any] = None):
    print("[DEBUG] Usertext:", user_message)
    print("[DEBUG] full_context:", full_context)
    for chunk in stream_invoke(user_message, full_context):
            print("chunk",chunk)
            yield chunk

# ------------------------- LOAD THREADS -------------------------
if "threads" not in st.session_state:
    if os.path.exists("threads.json"):
        with open("threads.json") as f:
            st.session_state.threads = json.load(f)
    else:
        st.session_state.threads = {}

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None

# ------------------------- SIDEBAR -------------------------
st.sidebar.header("📂 Threads")

if st.sidebar.button("➕ New Thread"):
    new_id = str(uuid.uuid4())
    st.session_state.threads[new_id] = {"name": f"Thread {len(st.session_state.threads)+1}", "messages": []}
    st.session_state.active_thread_id = new_id
    st.rerun()

def serialize_for_json(obj):
    try:
        return obj.__dict__
    except AttributeError:
        return str(obj)

if st.sidebar.button("💾 Save Chats"):
    with open("threads.json", "w") as f:
        json.dump(st.session_state.threads, f, default=serialize_for_json, indent=2)
    st.sidebar.success("✅ Chats saved!")

for tid, thread in st.session_state.threads.items():
    if st.sidebar.button(thread["name"], key=tid):
        st.session_state.active_thread_id = tid
        st.rerun()

if st.session_state.active_thread_id and st.sidebar.button("🗑️ Delete Current Thread"):
    del st.session_state.threads[st.session_state.active_thread_id]
    st.session_state.active_thread_id = None
    st.rerun()

if not st.session_state.threads:
    st.subheader("🧵 No threads yet")
    st.write("Use the sidebar to create your first thread.")
    st.stop()

if st.session_state.active_thread_id not in st.session_state.threads:
    st.session_state.active_thread_id = list(st.session_state.threads.keys())[0]

active_thread = st.session_state.threads[st.session_state.active_thread_id]

# ------------------------- PLACEHOLDER LOGIC -------------------------
if "waiting_for_bot_reply" not in st.session_state:
    st.session_state["waiting_for_bot_reply"] = False

messages = active_thread["messages"]

if st.session_state["waiting_for_bot_reply"]:
    if messages and messages[-1]["role"] == "bot" and messages[-1].get("placeholder"):
        user_input = None
        for msg in reversed(messages[:-1]):
            if msg["role"] == "user":
                user_input = msg["content"]
                break

        full_response = None
        for msg in reversed(messages[:-1]):
            if msg.get("full_response"):
                full_response = msg["full_response"]
                break

        if user_input:
            # Stream the response in real-time
            placeholder = st.empty()
            partial_text = ""

            for item in get_bot_response_streaming(user_input, full_response):
                print("back in streamlit \n")
                if item["type"] == "chunk":
                    partial_text += item["content"]
                    placeholder.markdown(f"**🤖 Bot:** {partial_text}") 
                elif item["type"] == "final":
                    AIMsg = item["message"]
                    print(">>>",AIMsg)
                    active_thread["messages"].append({
                        "role": "bot",
                        "content": partial_text,
                        "full_response": AIMsg
                    })

                print(f"get_bot_response - {item}\n")
                
            # Replace the placeholder message with the final text
          

        st.session_state["waiting_for_bot_reply"] = False
        st.rerun()

# ------------------------- DISPLAY CHAT -------------------------
st.subheader(f"💬 Chat - {active_thread['name']}")
for msg in active_thread["messages"]:
    role = msg["role"]
    text = msg["content"]
    if role == "user":
        st.markdown(f"**🧑‍💻 You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")

# ------------------------- TEXT INPUT -------------------------
st.markdown("### 💬 Type a message")
with st.form("chat_input_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    active_thread["messages"].append({"role": "user", "content": user_input})
    active_thread["messages"].append({"role": "bot", "content": "🤖 Thinking...", "placeholder": True})
    st.session_state["waiting_for_bot_reply"] = True
    st.rerun()

# ------------------------- AUDIO INPUT -------------------------
st.markdown("### 🎙️ Or speak a message")

wav_audio_data = st_audiorec()

if "stored_audio_data" not in st.session_state:
    st.session_state["stored_audio_data"] = None

if "audio_transcribed" not in st.session_state:
    st.session_state["audio_transcribed"] = True

if wav_audio_data is not None and wav_audio_data != st.session_state["stored_audio_data"]:
    st.session_state["stored_audio_data"] = wav_audio_data
    st.session_state["audio_transcribed"] = False

if st.session_state["stored_audio_data"] is not None and not st.session_state["audio_transcribed"]:
    st.write("✅ Entered audio processing block")
    st.audio(st.session_state["stored_audio_data"], format="audio/wav")
    st.write(f"🔎 Audio data length: {len(st.session_state['stored_audio_data'])} bytes")

    try:
        audio_buffer = io.BytesIO(st.session_state["stored_audio_data"])
        data, samplerate = sf.read(audio_buffer)
        st.write(f"✅ Original sample rate: {samplerate}")
        st.write(f"✅ Original shape: {data.shape}")

        if len(data.shape) == 2:
            data = librosa.to_mono(data.T)

        data = data.astype(np.float32)
        data_16k = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
        st.write(f"✅ Resampled shape: {data_16k.shape}")

        if "whisper_model" not in st.session_state:
            st.session_state.whisper_model = whisper.load_model("tiny.en")

        model = st.session_state.whisper_model

        with st.spinner("⏳ Transcribing with Whisper..."):
            result = model.transcribe(data_16k)
            transcript = result["text"]
            st.success(f"📝 Transcript: {transcript}")

            if st.session_state.get("last_audio_input") != transcript:
                st.session_state.last_audio_input = transcript
                st.session_state.audio_input_processed = False

        st.session_state["audio_transcribed"] = True

    except Exception as e:
        st.error(f"❌ Error reading or processing audio: {e}")

audio_input = st.session_state.get("last_audio_input")

if audio_input:
    active_thread["messages"].append({"role": "user", "content": audio_input})
    st.session_state["waiting_for_bot_reply"] = True
    st.rerun()