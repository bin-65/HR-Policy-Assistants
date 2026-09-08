import os
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from faiss import IndexFlatL2
from sentence_transformers import SentenceTransformer
from groq import Groq

# 1. Page Configuration
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👩‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Styling for Premium UI
st.markdown("""
    <style>
    /* Card Styles */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 18px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 24px;
        color: #1f2937;
    }
    .metric-card p {
        margin: 0;
        font-size: 13px;
        color: #6b7280;
    }
    /* Expander Styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #374151;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Streamlit Secrets (TOML) or Sidebar Fallback
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Get key from console.groq.com")

st.sidebar.markdown("---")

st.sidebar.header("📚 How it works")
st.sidebar.markdown("""
1. **Upload** your HR Policy PDF
2. **Text Extraction** via PyMuPDF
3. **Chunking** text into chunks
4. **Vector Embeddings** via MiniLM-L6
5. **FAISS** in-memory search
6. **Groq LLaMA3** AI response generation
""")

st.sidebar.markdown("---")
st.sidebar.caption("⚡ Powered by FAISS & Groq LLaMA3")

# 4. Main UI Header
st.title("👩‍💼 HR Policy Assistant")
st.caption("Ask questions about your company HR policy using Retrieval-Augmented Generation (RAG).")

st.markdown("---")

# 5. Load Embedding Model
@st.cache_resource(show_spinner="⚡ Initializing Embeddings Engine...")
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# 6. PDF Processing Functions
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

# 7. Document Upload Section
st.header("📄 Upload HR Policy")
uploaded_file = st.file_uploader("Upload your HR Policy PDF", type=["pdf"])

faiss_index = None
indexed_chunks = None

if uploaded_file:
    pages_data = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(pages_data)
    faiss_index, indexed_chunks = create_faiss_index(chunks)
    
    # Dashboard Metrics Display
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><h3>{len(pages_data)}</h3><p>Pages Processed</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><h3>{len(chunks)}</h3><p>Text Chunks Created</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><h3>FAISS Index</h3><p>Status: Ready ✅</p></div>', unsafe_allow_html=True)
        
    st.success("✅ Policy document processed successfully!")

st.markdown("---")

# 8. Interactive Example Questions
st.header("💡 Example Questions")

col1, col2 = st.columns(2)
selected_question = None

with col1:
    if st.button("How many annual leave days are employees entitled to?", use_container_width=True):
        selected_question = "How many annual leave days are employees entitled to?"
    if st.button("What are the standard working hours?", use_container_width=True):
        selected_question = "What are the standard working hours?"

with col2:
    if st.button("Can employees work remotely?", use_container_width=True):
        selected_question = "Can employees work remotely?"
    if st.button("How many sick leave days are available?", use_container_width=True):
        selected_question = "How many sick leave days are available?"

st.markdown("---")

# 9. Groq RAG Execution Function
def query_groq_rag(user_query, index, chunks, top_k=3):
    query_vector = embed_model.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), top_k)
    
    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    context = "\n\n".join([f"(Page {c['page']}): {c['text']}" for c in retrieved_chunks])
    
    client = Groq(api_key=groq_api_key)
    
    system_prompt = (
        "You are an expert HR Policy Assistant. Use the provided HR Policy document context "
        "to answer the user's question accurately, professionally, and concisely. "
        "If the context does not contain enough info, state clearly that the document does not specify it.\n\n"
        f"Context:\n{context}"
    )
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content, retrieved_chunks

# 10. Query Handling Section
user_input = st.chat_input("Ask a question about your HR policy...")
query_to_process = user_input or selected_question

if query_to_process:
    if not groq_api_key:
        st.error("⚠️ Please enter a valid Groq API Key in the sidebar or TOML secrets.")
    elif not faiss_index:
        st.error("⚠️ Please upload an HR Policy PDF document first.")
    else:
        st.chat_message("user").write(query_to_process)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document & generating AI answer..."):
                try:
                    answer, ref_chunks = query_groq_rag(query_to_process, faiss_index, indexed_chunks)
                    st.write(answer)
                    
                    with st.expander("📌 View Reference Policy Clauses"):
                        for chunk in ref_chunks:
                            st.markdown(f"**Page {chunk['page']}:**")
                            st.write(chunk['text'])
                            st.divider()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
