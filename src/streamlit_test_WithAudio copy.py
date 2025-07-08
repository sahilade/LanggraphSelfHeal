import streamlit as st
import uuid
import json
import numpy as np
import os
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
from appv4_withThread import invoke
from typing import Optional,Any
import wave
import tempfile
import whisper
import soundfile as sf
from st_audiorec import st_audiorec
import io

st.set_page_config(page_title="Threaded Chatbot", layout="wide")
st.title("🧵 Threaded Chat - Integration with SNOW (BETA)")

# --- Load / Initialize threads ---
if "threads" not in st.session_state:
    if os.path.exists("threads.json"):
        with open("threads.json") as f:
            st.session_state.threads = json.load(f)
    else:
        st.session_state.threads = {}

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None

if "last_audio_input" not in st.session_state:
    st.session_state.last_audio_input = None

# --- Sidebar ---
st.sidebar.header("📂 Threads")

if st.sidebar.button("➕ New Thread"):
    new_id = str(uuid.uuid4())
    st.session_state.threads[new_id] = {"name": f"Thread {len(st.session_state.threads)+1}", "messages": []}
    st.session_state.active_thread_id = new_id
    st.rerun()

def serialize_for_json(obj):
    try:
        return obj.__dict__  # try getting object's attributes
    except AttributeError:
        return str(obj)   
    
if st.sidebar.button("💾 Save Chats"):
    with open("threads.json", "w") as f:
        print("@@@@@@@@@@@@@@@@@@@")
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

# --- Pick active thread ---
if not st.session_state.threads:
    st.subheader("🧵 No threads yet")
    st.write("Use the sidebar to create your first thread.")
    st.stop()

if st.session_state.active_thread_id not in st.session_state.threads:
    st.session_state.active_thread_id = list(st.session_state.threads.keys())[0]

active_thread = st.session_state.threads[st.session_state.active_thread_id]

# --- Display Chat ---
st.subheader(f"💬 Chat - {active_thread['name']}")
for msg in active_thread["messages"]:
    role = msg["role"]
    text = msg["content"]
    if role == "user":
        st.markdown(f"**🧑‍💻 You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")

# --- Helper: Call LLM ---
def get_bot_response(user_message: str, full_context: Optional[Any] = None):
    print("[DEBUG] Usertext :",user_message)
    print("[DEBUG] full_context :",full_context)
    l = invoke(user_message,full_context)
    print("invoke() full result:", l)
    return l

# --- Text Input ---
st.markdown("### 💬 Type a message")
with st.form("chat_input_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    active_thread["messages"].append({"role": "user", "content": user_input})
    print("Active Thread ", active_thread)
    # if len(active_thread["messages"]) >= 2:
    #     last_message = active_thread["messages"][-2]
    #     full_response = last_message.get("full_response")
    #     print("FullResponse, ", full_response)
    # else:
    #     last_message = None
    
    full_response = None

    for msg in reversed(active_thread["messages"]):
        if "full_response" in msg:
            full_response = msg["full_response"]
            break


   # if last_message is not None and last_message["role"] == "bot" and full_response is not None:
    if full_response is not None:
        bot_reply = get_bot_response(user_input, full_response)
    else:
        bot_reply = get_bot_response(user_input)
    active_thread["messages"].append({
        "role": "bot",
        "content": bot_reply["messages"][-1].content,
        "full_response": bot_reply
    })
    st.rerun()

# --- Audio Input ---
st.markdown("### 🎙️ Or speak a message")

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format="audio/wav")
    st.info("⏳ Transcribing your speech...")

    pcm_data = np.frombuffer(wav_audio_data, dtype=np.int16)  # or float32
    buffer = io.BytesIO()
    sf.write(buffer, pcm_data, 16000, format='WAV')
    buffer.seek(0)
    audio_array, sample_rate = sf.read(buffer)
    
    # Load Whisper
    if "whisper_model" not in st.session_state:
     st.session_state.whisper_model = whisper.load_model("tiny.en")
    
    model = st.session_state.whisper_model
    result = model.transcribe(audio_array)
    
    transcript = result["text"]

    st.success(f"📝 Transcript: {transcript}")
    st.session_state.last_audio_input = transcript


if "audio_input_processed" not in st.session_state:
    st.session_state.audio_input_processed = False

audio_input = st.session_state.get("last_audio_input")

audio_input = st.session_state.get("last_audio_input")
if audio_input and not st.session_state.audio_input_processed:
    print("Transcript - ", audio_input)
    active_thread["messages"].append({"role": "user", "content": audio_input})
    print("Active Thread ", active_thread)
    # if len(active_thread["messages"]) >= 2:
    #     last_message = active_thread["messages"][-2]
    #     full_response = last_message.get("full_response")
    #     print("FullResponse, ", full_response)
    # else:
    #     last_message = None
    
    full_response = None

    for msg in reversed(active_thread["messages"]):
        if "full_response" in msg:
            full_response = msg["full_response"]
            break


   # if last_message is not None and last_message["role"] == "bot" and full_response is not None:
    if full_response is not None:
        bot_reply = get_bot_response(audio_input, full_response)
    else:
        bot_reply = get_bot_response(audio_input)
    active_thread["messages"].append({
        "role": "bot",
        "content": bot_reply["messages"][-1].content,
        "full_response": bot_reply
    })
    st.session_state.audio_input_processed = True
    st.session_state.last_audio_input = None
    st.rerun()


    # active_thread["messages"].append({"role": "user", "content": audio_input})
    # bot_reply = get_bot_response(audio_input)
    # active_thread["messages"].append({
    #     "role": "bot",
    #     "content": bot_reply["messages"][-1]["content"],
    #     "full_response": bot_reply
    # })
    # st.session_state.last_audio_input = None
    # st.rerun()
