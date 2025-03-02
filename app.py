import streamlit as st
import warnings
from utils import load_chapter_data, chapters

warnings.filterwarnings('ignore')
st.set_page_config(page_title="BioBot", page_icon="🧬", initial_sidebar_state="collapsed")

st.title("🧬 BioBot - Class 12 Biology Tutor")
st.markdown('<span style="font-size: 22px;">Your AI-powered study buddy!</span>', unsafe_allow_html=True)

st.markdown('<span style="font-size: 22px;">Due to Resource Limitations, following chapters are currently not supported: 1,2,4,6,8</span>', unsafe_allow_html=True)

st.header("📚 Table of Contents")
st.markdown("""
    <style>
        .toc {
            font-size: 18px;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

toc_text = "<div class='toc'><ul>"
for ch, title in chapters.items():
    toc_text += f"<li><b>{ch}:</b> {title}</li>"
toc_text += "</ul></div>"
st.markdown(toc_text, unsafe_allow_html=True)

st.write("\n")

st.subheader("📌 Select a Chapter")
chapter_name = st.selectbox("Choose a chapter:", list(chapters.keys()))

if st.button("📂 Load Chapter"):
    try:
        with st.spinner("Loading Chapter... ⏳"):
            docs, retriever = load_chapter_data(chapter_name)
            st.session_state["vector_store"] = retriever
            st.session_state["chat_history"] = []
            st.session_state["chapter_loaded"] = True
            st.session_state["selected_chapter"] = chapter_name
            st.success("✅ Chapter Loaded! Go to the QA Page.")

        st.page_link("pages/1_QA_Bot.py", label="Ask Questions!", icon="💬")

    except Exception as e:
        st.error(f"❌ Error loading chapter: {str(e)}")
