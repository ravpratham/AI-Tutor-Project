# frontend/app.py
import streamlit as st
import requests
import os
import html

# -------- Configuration --------
RAG_BACKEND = os.getenv("RAG_BACKEND_URL", "http://localhost:8000/ask")
TIMEOUT = 120

st.set_page_config(page_title="AI Syllabus Tutor", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stForm {
        background-color: #0a0a0a;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 0.75rem;
        border: solid;
        border-radius: 8px;
        border-color: #471fb4;
        text-color: #ffffff;
    }
    h1 {
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 5px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1565c0;
    }
    .answer-box {
        background-color: black;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("📘 AI Syllabus Tutor")
st.markdown("<p style='font-size: 1.1rem; color: #666; margin-bottom: 2rem;'>Ask questions from the syllabus material uploaded to the backend.</p>", unsafe_allow_html=True)

# Question form in a prominent container
with st.container():
    st.markdown("### 💬 Ask Your Question")
    with st.form("ask_form"):
        question = st.text_input(
            "Enter your question:", 
            "", 
            key="question_input",
            placeholder="e.g., What is machine learning?",
            help="Type your question about the syllabus material here"
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            top_k = st.slider(
                "Number of context chunks to retrieve (top_k):", 
                min_value=1, 
                max_value=5, 
                value=3,
                help="More chunks provide more context but may increase response time"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            submit = st.form_submit_button("🔍 Ask Question", use_container_width=True)

# Display results
if submit:
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🤔 Thinking... Please wait while we retrieve the answer."):
            try:
                payload = {"question": question, "top_k": top_k}
                resp = requests.post(RAG_BACKEND, json=payload, timeout=TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer", "No answer returned.")
                retrieved_count = data.get("retrieved_count", 0)
                
                # Display answer in a styled container
                st.markdown("---")
                st.markdown("### ✅ Answer")
                with st.container():
                    # Create a styled answer box with proper HTML escaping
                    st.markdown(answer)
                    #escaped_answer = html.escape(answer).replace('\n', '<br>')
                    #st.markdown(f'<div class="answer-box">{escaped_answer}</div>', unsafe_allow_html=True)
                
                # Metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📚 **Chunks used:** {retrieved_count}")
                with col2:
                    st.info(f"🔢 **Top K:** {top_k}")
                    
            except requests.exceptions.HTTPError as e:
                # Get error details
                error_detail = ""
                is_context_error = False
                try:
                    if e.response:
                        error_data = e.response.json()
                        error_detail = error_data.get("detail", str(e))
                        # Check if it's a context length error (even if status code isn't 413)
                        error_msg_lower = error_detail.lower()
                        is_context_error = (
                            e.response.status_code == 413 or
                            "context length" in error_msg_lower or 
                            "context overflow" in error_msg_lower or
                            "not enough" in error_msg_lower and "context" in error_msg_lower
                        )
                except:
                    error_detail = str(e)
                    error_msg_lower = error_detail.lower()
                    is_context_error = "context length" in error_msg_lower or "context overflow" in error_msg_lower
                
                # Handle context length overflow error
                if is_context_error:
                    st.error(f"⚠️ **Context Length Overflow**")
                    st.warning("""
                    **The input is too long for the model's context window.**
                    
                    **Solutions:**
                    1. **Reduce the number of context chunks (top_k)** - Try setting it to 1 or 2
                    2. **Reload the model in LM Studio** with a larger context length setting
                    3. **Ask a more specific question** that requires less context
                    """)
                    st.info(f"💡 **Tip:** Lower the 'Number of context chunks' slider and try again.")
                else:
                    status_code = e.response.status_code if e.response else "Unknown"
                    st.error(f"❌ **HTTP Error {status_code}:** {error_detail if error_detail else str(e)}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ **Connection Error:** Failed to reach backend server. Make sure the backend is running.\n\nError details: {e}")
            except Exception as e:
                st.error(f"❌ **Error:** {e}")

# Help/Info section
st.markdown("---")
with st.expander("ℹ️ Instructions & Notes", expanded=False):
    st.markdown("""
    **Getting Started:**
    1. Enter your question in the search box above
    2. Adjust the number of context chunks if needed (more chunks = more context)
    3. Click "Ask Question" to get your answer
    
    **Requirements:**
    - Ensure LM Studio is running and your chosen model is loaded
    - Make sure the backend server is running on `http://localhost:8000`
    
    **Tips:**
    - If you get context length errors, reduce the "top_k" value
    - For better answers, ask specific questions about the syllabus material
    - Re-generate your index with smaller chunks if needed
    """)
