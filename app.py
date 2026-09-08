import os
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from faiss import IndexFlatL2
from sentence_transformers import SentenceTransformer
from groq import Groq

# Page Setup with Male HR Manager Emoji
st.set_page_config(
    page_title="HR Policy Assistant | Enterprise Portal",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Professional CSS Styling with Animations & Light Theme Palette
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

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

    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: var(--bg-primary); }

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
    }

    .hero-banner p {
        font-size: 1.05rem;
        margin-top: 8px;
        margin-bottom: 0;
        color: #e0e7ff;
    }

    .custom-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
    }

    .category-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 30px;
        background-color: var(--accent-light);
        color: var(--accent-indigo);
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
        color: #334155 !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        padding: 0.65rem 1rem !important;
        transition: all 0.25s ease !important;
        text-align: left !important;
    }

    .stButton > button:hover {
        border-color: var(--accent-indigo) !important;
        color: var(--accent-indigo) !important;
        background: var(--accent-light) !important;
        transform: translateY(-2px);
    }

    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
    }

    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

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
</style>
""", unsafe_allow_html=True)

# Sidebar Setup
st.sidebar.markdown("### ⚙️ System Configuration")

groq_api_key = st.secrets.get("GROQ_API_KEY", "")

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("🔑 Groq API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Architecture Flow")
st.sidebar.markdown("""
1. 📄 **Document Extraction:** Paragraph & Line Tracking
2. ✂️ **Chunking:** Chunk-to-Line Indexing
3. 🧠 **Embeddings:** MiniLM-L6-v2 Engine
4. ⚡ **Search:** FAISS Retrieval
5. 🤖 **Inference:** Groq Llama-3 AI
""")

st.sidebar.markdown("---")
st.sidebar.caption("👨‍💼 HR Policy RAG Assistant v2.0")

# Header Banner
st.markdown("""
    <div class="hero-banner">
        <h1>👨‍💼 HR Policy Assistant</h1>
        <p>Enterprise Knowledge Base with Exact Page, Paragraph & Line Citation</p>
    </div>
""", unsafe_allow_html=True)

# Load SentenceTransformer Model
@st.cache_resource(show_spinner="⚡ Loading Embedding Model...")
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# Extract Blocks, Paragraphs, Line Numbers and Text
def extract_structured_pdf_data(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    structured_chunks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Get structured blocks from page (blocks = paragraphs)
        blocks = page.get_text("blocks")
        
        para_counter = 1
        for block in blocks:
            # block[4] contains actual text content
            block_text = block[4].strip()
            if block_text and len(block_text) > 20: # Ignore tiny noise
                # Split paragraph into lines to count lines accurately
                lines = [line.strip() for line in block_text.split("\n") if line.strip()]
                line_range = f"Line 1-{len(lines)}" if len(lines) > 1 else "Line 1"
                
                structured_chunks.append({
                    "page": page_num + 1,
                    "paragraph": para_counter,
                    "line_info": line_range,
                    "text": " ".join(lines)
                })
                para_counter += 1

    return structured_chunks

@st.cache_resource(show_spinner="🔍 Building FAISS Vector Index...")
def create_faiss_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, chunks

# 📄 Upload Section
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown("### 📄 Document Ingestion")
st.caption("Upload your HR policy PDF handbook to analyze page, paragraph, and line metadata.")

uploaded_file = st.file_uploader("Choose a PDF File", type=["pdf"], label_visibility="collapsed")

faiss_index = None
indexed_chunks = None

if uploaded_file:
    chunks = extract_structured_pdf_data(uploaded_file)
    faiss_index, indexed_chunks = create_faiss_index(chunks)
    st.markdown("""
        <div class="status-pill status-active" style="margin-top: 10px;">
            <span>🟢 Document Successfully Processed & Indexed with Paragraph/Line Locations</span>
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

# Category 2: Benefits & Culture
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
    if st.button("Is there an annual bonus scheme?", use_container_width=True):
        selected_question = "Is there a performance bonus policy?"

with col6:
    if st.button("How to claim expense reimbursements?", use_container_width=True):
        selected_question = "What is the policy for expense reimbursement?"
    if st.button("What is the required notice period?", use_container_width=True):
        selected_question = "What are the rules regarding notice period?"

st.markdown('</div>', unsafe_allow_html=True)

# Groq RAG Execution Engine
def query_groq_rag(user_query, index, chunks, top_k=2):
    query_vector = embed_model.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), top_k)
    
    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    
    context = "\n\n".join([
        f"(Page {c['page']}, Paragraph {c['paragraph']}, {c['line_info']}): {c['text']}"
        for c in retrieved_chunks
    ])
    
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

# Chat Prompt Input
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
            with st.spinner("Analyzing document context & generating response..."):
                try:
                    answer, ref_chunks = query_groq_rag(query_to_process, faiss_index, indexed_chunks)
                    
                    # Single-Line Concise Output
                    st.markdown(f"**Answer:** {answer}")
                    
                    # Detailed Page, Paragraph, and Line Number Citations
                    with st.expander("📌 View Page, Paragraph & Line Citations from Book"):
                        for chunk in ref_chunks:
                            st.markdown(
                                f"**📍 Page {chunk['page']} | Paragraph {chunk['paragraph']} ({chunk['line_info']}):**"
                            )
                            snippet = chunk['text'][:220] + ("..." if len(chunk['text']) > 220 else "")
                            st.write(f"_{snippet}_")
                            st.divider()
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")
