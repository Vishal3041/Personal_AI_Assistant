import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
import re

# ✅ Disable Torch Compilation for compatibility
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# ✅ Initialize Pinecone with the correct API key
pc = Pinecone(api_key="pcsk_6awTRp_rSsr7eom3bSZXZZcnDLDwc87RnpU2Sp9WEzyEFdEj2TtiyRwjEfnaXswVjGqLi")

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

# ✅ Load Sentence Transformer for correct embedding size (384)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Model Paths (Hugging Face Models Instead of Local Paths)
YOUTUBE_MODEL_PATH = "Vishal3041/falcon_finetuned_llm"
CHROME_MODEL_PATH = "Vishal3041/TransNormerLLM_finetuned"

# ✅ Load Model with CPU Support
def load_model(model_path):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            device_map="auto"  # ✅ Automatically assigns the best device (CPU/GPU)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        return model, tokenizer
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None

# ✅ Load the Model **Once** to Reduce Latency
model_path = YOUTUBE_MODEL_PATH if selected_app == "YouTube" else CHROME_MODEL_PATH
if "model" not in st.session_state:
    st.session_state["model"], st.session_state["tokenizer"], st.session_state["device"] = load_model(model_path)

model = st.session_state["model"]
tokenizer = st.session_state["tokenizer"]
device = st.session_state["device"]

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
    if filter_conditions is None:
        filter_conditions = {}

    # ✅ Use embeddings for title-based search
    if query_vector is None:
        query_vector = embedding_model.encode(query).tolist()

    # ✅ Use keyword arguments for Pinecone query
    results = index.query(vector=query_vector, top_k=5, include_metadata=True, filter=filter_conditions)

    # ✅ Construct formatted context for UI
    context_list = []
    for res in results.get("matches", []):
        metadata = res.get("metadata", {})
        title = metadata.get("Title", "No Title")
        timestamp = metadata.get("Timestamp", "No Date")

        if selected_app == "Chrome":
            formatted_entry = f"📌 **{title}**\n   🕒 *Visited on: {timestamp}*"
        else:
            watched_at = metadata.get("Watched At", "Unknown Date")
            video_link = metadata.get("Video Link", "#")
            formatted_entry = f"🎬 **[{title}]({video_link})**\n   📅 *Watched on: {watched_at}*"

        context_list.append(formatted_entry)

    return "\n\n".join(context_list) if context_list else "No relevant results found."

# ✅ Detect Query Type and Extract Filters
def perform_rag_query(query):
    filter_conditions = {}
    query_vector = embedding_model.encode(query).tolist()

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
            output = model.generate(input_ids, max_length=512, do_sample=True, top_p=0.9, temperature=0.7)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            st.error(f"⚠️ Error generating response: {e}")
            st.stop()

        # ✅ Store & Display Response (Formatted)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(f"{input_text} {response}", unsafe_allow_html=True)
