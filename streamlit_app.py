import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pinecone
from pinecone import Pinecone
import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Initialize Pinecone
pc = Pinecone(api_key="85e39b43-9316-4d8b-b684-eb46542c34ef", environment="us-east-1")

# ✅ Use Absolute Paths for Model Checkpoints
YOUTUBE_MODEL_PATH = "s3://298b-models/youtube_llm_model/fine_tuned_model_new"
CHROME_MODEL_PATH = "s3://298b-models/youtube_llm_model/merged_fine_tuned_model"

# ✅ Define Indexes
INDEXES = {
    "YouTube": "youtube-data-index",
    "Chrome": "chrome-history-index"
}

# ✅ Load models and tokenizers properly
def load_model(model_path):
    try:
        # Explicitly specify `trust_remote_code=True`
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# ✅ Streamlit UI
st.set_page_config(page_title="Personal AI Assistant", layout="wide")

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .stChatMessage { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
    .css-1d391kg { background-color: #f8f9fa !important; }
    </style>
    """, unsafe_allow_html=True
)

st.title("🔍 Personal AI Assistant")
st.subheader("Chat with your YouTube or Chrome history!")

# ✅ Dropdown Menu for App Selection
selected_app = st.selectbox("Choose the application:", ["YouTube", "Chrome"])

# ✅ Load the Correct Model
model_path = YOUTUBE_MODEL_PATH if selected_app == "YouTube" else CHROME_MODEL_PATH
model, tokenizer = load_model(model_path)

if model is None or tokenizer is None:
    st.error("⚠️ Failed to load the model. Check if the model folder is correctly structured.")
    st.stop()

# ✅ Load Pinecone Index
index_name = INDEXES[selected_app]
index = pc.Index(index_name)  # 🔥 FIXED

st.markdown("### 💬 Ask a question based on your history")

# ✅ Store chat history in Streamlit session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ User Query Input
user_query = st.text_input("Type your question...")

if st.button("Ask"):
    if user_query:
        # Store user query
        st.session_state.messages.append({"role": "user", "content": user_query})

        # ✅ RAG Pipeline: Fetch relevant context
        try:
            query_embedding = model.get_input_embeddings()(torch.tensor(tokenizer.encode(user_query)).unsqueeze(0))
            results = index.query(query_embedding.tolist(), top_k=5, include_metadata=True)
            context = "\n".join([doc["metadata"]["text"] for doc in results["matches"]])
        except Exception as e:
            st.error(f"Error fetching context from Pinecone: {e}")
            st.stop()

        # ✅ Generate Response
        input_text = f"Context: {context}\nUser Question: {user_query}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        try:
            output = model.generate(input_ids, max_length=512, do_sample=True, top_p=0.9, temperature=0.7)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.stop()

        # ✅ Store & Display Response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
