import os
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from faiss import IndexFlatL2
from sentence_transformers import SentenceTransformer
from groq import Groq

# Streamlit Page Setup
st.set_page_config(
    page_title="Built HR Policy Assistant",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Built HR Policy Assistant")
st.caption("Ask questions about your HR policy using FAISS & Groq RAG.")

# Get Groq API Key automatically from Streamlit Secrets (TOML) or Sidebar input
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")

if not groq_api_key:
    st.info("👈 Please configure `GROQ_API_KEY` in Streamlit Secrets or enter it in the sidebar.")
    st.stop()

# Initialize Embeddings Model
@st.cache_resource(show_spinner="Loading Embedding Model...")
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

# Vector Store Indexing with FAISS
@st.cache_resource(show_spinner="Indexing HR Document with FAISS...")
def create_faiss_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, chunks

# File Uploader
uploaded_file = st.file_uploader("Upload HR Policy PDF File", type=["pdf"])

if uploaded_file:
    pages_data = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(pages_data)
    faiss_index, indexed_chunks = create_faiss_index(chunks)
    st.success("✅ HR Policy Document successfully processed and indexed!")
else:
    st.warning("Please upload an HR Policy PDF document to begin.")
    st.stop()

# Groq RAG Query Execution
def query_groq_rag(user_query, index, chunks, top_k=3):
    query_vector = embed_model.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), top_k)
    
    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    context = "\n\n".join([f"(Page {c['page']}): {c['text']}" for c in retrieved_chunks])
    
    client = Groq(api_key=groq_api_key)
    
    system_prompt = (
        "You are an expert HR Policy Assistant. Use the provided HR Policy document context "
        "to answer the user's question accurately and concisely. If the context does not "
        "contain enough info, reply stating that the information isn't available in the document.\n\n"
        f"Context:\n{context}"
    )
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content, retrieved_chunks

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload an HR PDF policy above, then ask me anything about leaves, office hours, or policies."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if user_input := st.chat_input("Ask a question about your HR policy..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            try:
                answer, ref_chunks = query_groq_rag(user_input, faiss_index, indexed_chunks)
                st.write(answer)
                
                with st.expander("View Document Reference Clips"):
                    for chunk in ref_chunks:
                        st.markdown(f"**Page {chunk['page']}:**")
                        st.write(chunk['text'])
                        st.divider()

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
