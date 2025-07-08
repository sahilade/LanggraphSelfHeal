import streamlit as st
from st_audiorec import st_audiorec
st.title("Audio Recorder Example")

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    with open("output.wav", "wb") as f:
        f.write(wav_audio_data)
    st.audio(wav_audio_data, format="audio/wav")