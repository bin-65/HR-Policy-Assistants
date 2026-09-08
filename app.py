import os
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from faiss import IndexFlatL2
from sentence_transformers import SentenceTransformer
from groq import Groq

# Streamlit Page Setup
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👩‍💼",
    layout="wide"
)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Streamlit Secrets (TOML) ya Sidebar se API key read karna
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

st.sidebar.markdown("---")

# Sidebar - How it works section
st.sidebar.header("📚 How it works")
st.sidebar.markdown("""
1. Upload an HR Policy PDF
2. Extract text from the PDF
3. Split text into chunks
4. Generate embeddings
5. Store embeddings in FAISS
6. Retrieve relevant policy sections
7. Generate an answer using Groq
""")

# Main UI Header
st.title("👩‍💼 HR Policy Assistant")
st.caption("Ask questions about your company HR policy using Retrieval-Augmented Generation (RAG).")

st.markdown("---")

# Embeddings Model Setup
@st.cache_resource(show_spinner="Loading Embedding Model...")
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# PDF Functions
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

@st.cache_resource(show_spinner="Indexing HR Document with FAISS...")
def create_faiss_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, chunks

# 📄 Upload Section
st.header("📄 Upload HR Policy")
uploaded_file = st.file_uploader("Upload your HR Policy PDF", type=["pdf"])

faiss_index = None
indexed_chunks = None

if uploaded_file:
    pages_data = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(pages_data)
    faiss_index, indexed_chunks = create_faiss_index(chunks)
    st.success("✅ HR Policy successfully uploaded and indexed!")

st.markdown("---")

# 💡 Example Questions Section
st.header("💡 Example Questions")

selected_question = None

# Category 1: Leaves & Attendance
st.subheader("🍃 Leaves & Attendance")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("How many annual leave days are employees entitled to?", use_container_width=True):
        selected_question = "How many annual leave days are employees entitled to?"
    if st.button("What is the policy for sick leave?", use_container_width=True):
        selected_question = "What is the policy for sick leave?"

with col2:
    if st.button("What are the standard working hours?", use_container_width=True):
        selected_question = "What are the standard working hours?"
    if st.button("What is the maternity/paternity leave policy?", use_container_width=True):
        selected_question = "What is the maternity/paternity leave policy?"

with col3:
    if st.button("How do I request emergency leave?", use_container_width=True):
        selected_question = "How do I request emergency leave?"
    if st.button("What happens if I arrive late to work?", use_container_width=True):
        selected_question = "What happens if I arrive late to work?"

# Category 2: Work Environment & Benefits
st.subheader("💻 Work Environment & Benefits")
col4, col5, col6 = st.columns(3)

with col4:
    if st.button("Can employees work remotely?", use_container_width=True):
        selected_question = "Can employees work remotely?"
    if st.button("What is the employee dress code?", use_container_width=True):
        selected_question = "What is the employee dress code?"

with col5:
    if st.button("What health insurance benefits are provided?", use_container_width=True):
        selected_question = "What health insurance benefits are provided?"
    if st.button("Is there a performance bonus policy?", use_container_width=True):
        selected_question = "Is there a performance bonus policy?"

with col6:
    if st.button("What is the policy for expense reimbursement?", use_container_width=True):
        selected_question = "What is the policy for expense reimbursement?"
    if st.button("What are the rules regarding notice period?", use_container_width=True):
        selected_question = "What are the rules regarding notice period?"

st.markdown("---")

# Groq Query Function
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

# Chat Prompt Input
user_input = st.chat_input("Ask a question about your HR policy...")

query_to_process = user_input or selected_question

if query_to_process:
    if not groq_api_key:
        st.error("Please enter a Groq API Key in the sidebar or TOML Secrets.")
    elif not faiss_index:
        st.error("Please upload an HR Policy PDF first.")
    else:
        st.chat_message("user").write(query_to_process)
        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating response..."):
                try:
                    answer, ref_chunks = query_groq_rag(query_to_process, faiss_index, indexed_chunks)
                    st.write(answer)
                    
                    with st.expander("View Reference Policy Clips"):
                        for chunk in ref_chunks:
                            st.markdown(f"**Page {chunk['page']}:**")
                            clean_text = chunk['text'].replace("\n", " ").strip()
                            short_snippet = clean_text[:200] + ("..." if len(clean_text) > 200 else "")
                            st.write(short_snippet)
                            st.divider()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
