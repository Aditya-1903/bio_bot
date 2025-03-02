import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from utils import qa_chain, chapters

st.set_page_config(page_title="BioBot", page_icon="🧬", initial_sidebar_state="collapsed")

if "chapter_loaded" not in st.session_state or not st.session_state["chapter_loaded"]:
    st.warning("⚠️ Please load a chapter first!")
    st.stop()

st.title(f"💬 QA ChatBot\n")
st.subheader(f"{st.session_state['selected_chapter']} : {chapters[st.session_state['selected_chapter']]}")

if st.button("🏠 Back to Home"):
    st.session_state["chapter_loaded"] = False
    st.session_state["chat_history"] = []
    st.session_state["vector_store"] = None
    st.switch_page("app.py")

for msg in st.session_state["chat_history"]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

prompt = st.chat_input("📝 Enter your question here")
if prompt:
    try:
        with st.spinner("Generating... ✨"):
            output = qa_chain(prompt, st.session_state["chat_history"], st.session_state["vector_store"])
            st.session_state["chat_history"].extend([HumanMessage(content=prompt), AIMessage(content=output['answer'])])
            st.rerun()  

    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")
