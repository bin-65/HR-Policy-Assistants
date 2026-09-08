import os
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from faiss import IndexFlatL2
from sentence_transformers import SentenceTransformer
from groq import Groq

# Page Setup with Male HR Manager Emoji / Icon
st.set_page_config(
    page_title="HR Policy Assistant | Enterprise Portal",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Professional CSS Styling with Animations & Light Theme Palette
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global CSS Variable Theme Palette */
    :root {
        --bg-primary: #f8fafc;
        --card-bg: #ffffff;
        --accent-indigo: #4f46e5;
        --accent-hover: #4338ca;
        --accent-light: #e0e7ff;
        --text-dark: #0f172a;
        --text-muted: #64748b;
        --border-color: #e2e8f0;
        --sidebar-bg: #0f172a;
    }

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Background */
    .stApp {
        background-color: var(--bg-primary);
    }

    /* Professional Hero Header Gradient Banner */
    .hero-banner {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.25);
        animation: fadeInDown 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .hero-banner h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff !important;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-banner p {
        font-size: 1.05rem;
        margin-top: 8px;
        margin-bottom: 0;
        color: #e0e7ff;
        font-weight: 400;
    }

    /* Custom Stylish Section Cards */
    .custom-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
    }

    /* Category Section Header Badges */
    .category-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 30px;
        background-color: var(--accent-light);
        color: var(--accent-indigo);
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 12px;
        letter-spacing: 0.3px;
    }

    /* Example Questions Button Styling with Hover Effects */
    .stButton > button {
        width: 100%;
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
        color: #334155 !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        padding: 0.65rem 1rem !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
        text-align: left !important;
    }

    .stButton > button:hover {
        border-color: var(--accent-indigo) !important;
        color: var(--accent-indigo) !important;
        background: var(--accent-light) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
    }

    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    /* Status Indicator Badge */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .status-active {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }

    .status-inactive {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    /* Custom CSS Keyframe Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Clean Dividers */
    hr {
        border-top: 1px solid #e2e8f0;
        margin: 1.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ System Configuration")

groq_api_key = st.secrets.get("GROQ_API_KEY", "")

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("🔑 Groq API Key", type="password", help="Enter your Groq API key here")

st.sidebar.markdown("---")

# Sidebar - How it works section
st.sidebar.markdown("### 📚 Architecture Flow")
st.sidebar.markdown("""
1. 📄 **Document Ingestion:** PDF Parsing
2. ✂️ **Chunking:** Context preservation
3. 🧠 **Embeddings:** MiniLM-L6-v2 Model
4. ⚡ **Vector Search:** FAISS Indexing
5. 🤖 **Inference:** Groq Llama-3 Output
""")

st.sidebar.markdown("---")
st.sidebar.caption("👨‍💼 HR Policy RAG Assistant v2.0")

# Main Header Banner
st.markdown("""
    <div class="hero-banner">
        <h1>👨‍💼 HR Policy Assistant</h1>
        <p>Enterprise AI Knowledge Assistant powered by Retrieval-Augmented Generation (RAG)</p>
    </div>
""", unsafe_allow_html=True)

# Embeddings Model Setup
@st.cache_resource(show_spinner="⚡ Initializing Semantic Search Engine...")
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# PDF Processing Functions
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages_text.append({"page": page_num + 1, "text": text})
    return pages_text

def chunk_text(pages_data, chunk_size=500, chunk_overlap=100):
    chunks = []
    for item in pages_data:
        text = item["text"]
        page_num = item["page"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_str = text[start:end]
            chunks.append({"page": page_num, "text": chunk_str})
            start += chunk_size - chunk_overlap
    return chunks

@st.cache_resource(show_spinner="🔍 Building FAISS Vector Index...")
def create_faiss_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, chunks

# 📄 Upload Section with Card Layout
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown("### 📄 Document Ingestion")
st.caption("Upload your corporate HR policy handbook in PDF format to activate the assistant.")

uploaded_file = st.file_uploader("Choose a PDF File", type=["pdf"], label_visibility="collapsed")

faiss_index = None
indexed_chunks = None

if uploaded_file:
    pages_data = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(pages_data)
    faiss_index, indexed_chunks = create_faiss_index(chunks)
    st.markdown("""
        <div class="status-pill status-active" style="margin-top: 10px;">
            <span>🟢 Policy Document Successfully Loaded & Indexed</span>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="status-pill status-inactive" style="margin-top: 10px;">
            <span>🔴 No Policy Document Uploaded</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# 💡 Example Questions Section
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown("### 💡 Frequently Asked Questions")
st.caption("Click any query below to automatically search and get instant answers.")

selected_question = None

# Category 1: Leaves & Attendance
st.markdown('<span class="category-badge">🍃 Leaves & Attendance</span>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("How many annual leave days are allowed?", use_container_width=True):
        selected_question = "How many annual leave days are employees entitled to?"
    if st.button("What is the policy for sick leave?", use_container_width=True):
        selected_question = "What is the policy for sick leave?"

with col2:
    if st.button("What are standard working hours?", use_container_width=True):
        selected_question = "What are the standard working hours?"
    if st.button("Maternity & Paternity Leave rules?", use_container_width=True):
        selected_question = "What is the maternity/paternity leave policy?"

with col3:
    if st.button("How to request emergency leave?", use_container_width=True):
        selected_question = "How do I request emergency leave?"
    if st.button("What happens in case of late arrival?", use_container_width=True):
        selected_question = "What happens if I arrive late to work?"

st.markdown("<br>", unsafe_allow_html=True)

# Category 2: Benefits & Work Culture
st.markdown('<span class="category-badge">💻 Workplace & Benefits</span>', unsafe_allow_html=True)
col4, col5, col6 = st.columns(3)

with col4:
    if st.button("Is remote work / WFH permitted?", use_container_width=True):
        selected_question = "Can employees work remotely?"
    if st.button("What is the official dress code?", use_container_width=True):
        selected_question = "What is the employee dress code?"

with col5:
    if st.button("Medical & health insurance details?", use_container_width=True):
        selected_question = "What health insurance benefits are provided?"
    if st.button("Is there a annual bonus scheme?", use_container_width=True):
        selected_question = "Is there a performance bonus policy?"

with col6:
    if st.button("How to claim expense reimbursements?", use_container_width=True):
        selected_question = "What is the policy for expense reimbursement?"
    if st.button("What is the required notice period?", use_container_width=True):
        selected_question = "What are the rules regarding notice period?"

st.markdown('</div>', unsafe_allow_html=True)

# Groq Query Handler Function
def query_groq_rag(user_query, index, chunks, top_k=2):
    query_vector = embed_model.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), top_k)
    
    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    context = "\n\n".join([f"(Page {c['page']}): {c['text']}" for c in retrieved_chunks])
    
    client = Groq(api_key=groq_api_key)
    
    system_prompt = (
        "You are an expert HR Policy Assistant. Use the provided HR Policy document context "
        "to answer the user's question. Your answer MUST be strictly a single sentence (one line only). "
        "Do not add introductory fluff or extra details. Be extremely concise.\n\n"
        f"Context:\n{context}"
    )
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.1
    )
    
    return response.choices[0].message.content, retrieved_chunks

# Chat Interface Input
user_input = st.chat_input("Ask any question regarding your HR policy...")

query_to_process = user_input or selected_question

if query_to_process:
    if not groq_api_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar or Streamlit Secrets.")
    elif not faiss_index:
        st.error("⚠️ Please upload an HR Policy PDF document first.")
    else:
        st.chat_message("user", avatar="👤").write(query_to_process)
        with st.chat_message("assistant", avatar="👨‍💼"):
            with st.spinner("Analyzing document context & generating concise response..."):
                try:
                    answer, ref_chunks = query_groq_rag(query_to_process, faiss_index, indexed_chunks)
                    
                    # Highlighted Single-Line Answer Block
                    st.markdown(f"**Answer:** {answer}")
                    
                    # Concise Reference Clips Display
                    with st.expander("📖 View Verified Reference Clips from Handbook"):
                        for chunk in ref_chunks:
                            st.markdown(f"**📍 Page {chunk['page']}:**")
                            clean_text = chunk['text'].replace("\n", " ").strip()
                            short_snippet = clean_text[:200] + ("..." if len(clean_text) > 200 else "")
                            st.write(f"_{short_snippet}_")
                            st.divider()
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")
