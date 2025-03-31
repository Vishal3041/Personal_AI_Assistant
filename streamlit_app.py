import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
import re
import asyncio

# ✅ Set Streamlit page config (MUST be first)
st.set_page_config(page_title="Personal AI Assistant", layout="wide")

# ✅ Fix "0 active drivers" issue
torch.set_num_threads(1)

# ✅ Disable Torch Compilation for compatibility
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# ✅ Get Pinecone API key from environment variables or Streamlit secrets
def get_pinecone_credentials():
    try:
        # First try to get from Streamlit secrets (for Streamlit Cloud)
        pinecone_api_key = st.secrets["pinecone"]["api_key"]
    except:
        # Fallback to environment variables (for local development)
        pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not pinecone_api_key:
        st.error("Pinecone API key not found. Please set it in environment variables or Streamlit secrets.")
        return None

    return pinecone_api_key

# ✅ Initialize Pinecone with credentials
pinecone_api_key = get_pinecone_credentials()
if pinecone_api_key:
    pc = Pinecone(api_key=pinecone_api_key)
else:
    st.error("Failed to initialize Pinecone client. Please check your credentials.")
    st.stop()

# ✅ Define Indexes
INDEXES = {
    "Chrome": "chrome-history-index",
    "YouTube": "youtube-data-index"
}

# ✅ Ensure the index exists before querying
selected_app = st.selectbox("Choose the application:", ["Chrome", "YouTube"])
index_name = INDEXES[selected_app]

# ✅ Check if the index exists in Pinecone
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    st.error(f"⚠️ Index '{index_name}' does not exist in Pinecone. Please verify or create it.")
    st.stop()

# ✅ Access the Pinecone index
index = pc.Index(index_name)

# ✅ Hugging Face Model Paths
YOUTUBE_MODEL_PATH = "Vishal3041/falcon_finetuned_llm"
CHROME_MODEL_PATH = "Vishal3041/TransNormerLLM_finetuned"

# ✅ Load Sentence Transformer for correct embedding size (384)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Load Model from Hugging Face
def load_model(model_path):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # ✅ Move model to CPU (or GPU if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        print("✅ Model and tokenizer loaded successfully!")  # Debugging
        return model, tokenizer, device
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None, None

# ✅ Load the Model **Once** to Reduce Latency
model_path = YOUTUBE_MODEL_PATH if selected_app == "YouTube" else CHROME_MODEL_PATH
if "model" not in st.session_state:
    st.session_state["model"], st.session_state["tokenizer"], st.session_state["device"] = load_model(model_path)

model = st.session_state["model"]
tokenizer = st.session_state["tokenizer"]
device = st.session_state["device"]

# ✅ Debugging: Ensure tokenizer is loaded inside Streamlit
if tokenizer is None:
    st.error("❌ Tokenizer not loaded properly. Check the model path and structure!")

st.title("🔍 Personal AI Assistant")
st.subheader("Chat with your YouTube or Chrome history!")

st.markdown("### 💬 Ask a question based on your history")

# ✅ Store Chat History in Streamlit Session
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display Previous Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ User Query Input
user_query = st.text_input("Type your question...")

# ✅ Query and Context Retrieval
def perform_filtered_query(query, filter_conditions=None, query_vector=None):
    """
    Perform a Pinecone query with combined filters and augment the result.
    """
    if filter_conditions is None:
        filter_conditions = {}

    # ✅ Ensure query_vector matches Pinecone index dimension (384)
    if query_vector is None:
        query_vector = [0.0] * 384  # Dummy vector for metadata-only filtering

    # ✅ Query Pinecone with combined filters
    results = index.query(vector=query_vector, top_k=5, include_metadata=True, filter=filter_conditions)

    # ✅ Construct formatted context for UI
    context_list = []
    for res in results.get("matches", []):
        metadata = res.get("metadata", {})
        title = metadata.get("Title", "No Title")
        timestamp = metadata.get("Timestamp", "No Date")

        # ✅ Chrome results formatting
        if selected_app == "Chrome":
            formatted_entry = f"📌 **{title}**\n   🕒 *Visited on: {timestamp}*"
        
        # ✅ YouTube results formatting
        else:
            watched_at = metadata.get("Watched At", "Unknown Date")
            video_link = metadata.get("Video Link", "#")
            formatted_entry = f"🎬 **[{title}]({video_link})**\n   📅 *Watched on: {watched_at}*"

        context_list.append(formatted_entry)

    # ✅ Join context entries neatly
    context = "\n\n".join(context_list) if context_list else "No relevant results found."
    
    # ✅ Limit context length to avoid exceeding model input size
    return context[:500]

# ✅ Detect Query Type and Extract Filters
def perform_rag_query(query):
    filter_conditions = {}
    query_vector = None

    # ✅ Detect date in query
    date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', query, re.IGNORECASE)
    if date_match:
        filter_conditions["Timestamp"] = date_match.group(1)

    # ✅ Detect domain in query (Chrome)
    domain_match = re.search(r'\b(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', query, re.IGNORECASE)
    if domain_match and selected_app == "Chrome":
        filter_conditions["Domain"] = domain_match.group(1)

    # ✅ Detect category in query (YouTube)
    categories = ["Science & Technology", "Education", "News & Politics", "Autos & Vehicles"]
    for category in categories:
        if category.lower() in query.lower() and selected_app == "YouTube":
            filter_conditions["Category"] = category

    # ✅ Use embeddings for title-based search
    if not filter_conditions or "title" in query.lower():
        query_vector = embedding_model.encode(query).tolist()

    return perform_filtered_query(query, filter_conditions=filter_conditions, query_vector=query_vector)

# ✅ Handling User Query
if st.button("Ask"):
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        # ✅ Retrieve context from Pinecone
        try:
            context = perform_rag_query(user_query)
        except Exception as e:
            st.error(f"⚠️ Error fetching context from Pinecone: {e}")
            st.stop()

        # ✅ Generate Response Using Model
        input_text = f"### 🔍 **Search Results for:** \"{user_query}\"\n\n{context}\n\n💡 **Answer:**"

        try:
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            output = model.generate(input_ids, max_new_tokens=512, do_sample=True, top_p=0.9, temperature=0.7)  # ✅ Fix applied here
            response = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            st.error(f"⚠️ Error generating response: {e}")
            st.stop()

        # ✅ Store & Display Response (Formatted)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(f"{input_text} {response}", unsafe_allow_html=True)
