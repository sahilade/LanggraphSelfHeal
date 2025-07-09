
import streamlit as st

st.title("💬 Simple Chatbot")

MY_LINK = "https://www.bing.com/search?pglt=161&q=tutorialspoint&cvid=f4bcb8e7f4894d558f619d661453be76&gs_lcrp=EgRlZGdlKgYIABBFGDkyBggAEEUYOTIGCAEQLhhAMgYIAhAAGEAyBggDEAAYQDIGCAQQABhAMgYIBRAAGEAyBggGEAAYQDIGCAcQABhAMgYICBAFGEAyCAgJEOkHGPxV0gEIMjY1OWowajGoAgCwAgA&FORM=ANNAB1&PC=U531"

user_input = st.text_input("You:", "")

if user_input:
    st.markdown(f"**🧑‍💻 You:** {user_input}")

    if user_input.lower() == "hi show me window":
        st.markdown("**🤖 Bot:** Sure! Here's the website view:")
        
        # Embed iframe (may fail if site blocks it)
        st.components.v1.iframe(
            src=MY_LINK,
            height=600,
            scrolling=True
        )
        
        # Always show fallback link
        st.markdown(
            f"⚠️ If the embedded view doesn't load, [click here to open in a new tab]({MY_LINK})"
        )
        
    else:
        st.markdown("**🤖 Bot:** Sorry, I don't understand that. Try saying `hi show me window`.")
