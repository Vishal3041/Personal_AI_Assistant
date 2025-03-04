import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pinecone
from pinecone import Pinecone
import os

# ✅ Disable Torch Compilation for Compatibility
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# ✅ Initialize Pinecone
pc = Pinecone(api_key="85e39b43-9316-4d8b-b684-eb46542c34ef", environment="us-east-1")

# ✅ Define Hugging Face Model Paths (Replace S3 with HF Model IDs)
HF_MODELS = {
    "YouTube": "Vishal3041/falcon_finetuned_llm",
    "Chrome": "Vishal3041/TransNormerLLM_finetuned"
}

# ✅ Define Pinecone Indexes
INDEXES = {
    "YouTube": "youtube-data-index",
    "Chrome": "chrome-history-index"
}

# ✅ Load models and tokenizers properly from Hugging Face
def load_model(model_id):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None

# ✅ Streamlit UI Setup
st.set_page_config(page_title="Personal AI Assistant", layout="wide")

# ✅ Custom CSS for Better UI
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

# ✅ Load the Correct Model from Hugging Face
model_id = HF_MODELS[selected_app]
model, tokenizer = load_model(model_id)

if model is None or tokenizer is None:
    st.error("⚠️ Failed to load the model from Hugging Face. Check model repo & structure.")
    st.stop()

# ✅ Load Pinecone Index
index_name = INDEXES[selected_app]
index = pc.Index(index_name)

st.markdown("### 💬 Ask a question based on your history")

# ✅ Store chat history in Streamlit session
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display Previous Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ User Query Input
user_query = st.text_input("Type your question...")

if st.button("Ask"):
    if user_query:
        # Store user query
        st.session_state.messages.append({"role": "user", "content": user_query})

        # ✅ RAG Pipeline: Fetch relevant context from Pinecone
        try:
            query_embedding = model.get_input_embeddings()(torch.tensor(tokenizer.encode(user_query)).unsqueeze(0))
            results = index.query(query_embedding.tolist(), top_k=5, include_metadata=True)
            context = "\n".join([doc["metadata"]["text"] for doc in results["matches"]])
        except Exception as e:
            st.error(f"⚠️ Error fetching context from Pinecone: {e}")
            st.stop()

        # ✅ Generate Response from LLM
        input_text = f"Context: {context}\nUser Question: {user_query}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        try:
            output = model.generate(input_ids, max_length=512, do_sample=True, top_p=0.9, temperature=0.7)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            st.error(f"⚠️ Error generating response: {e}")
            st.stop()

        # ✅ Store & Display Response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
