import os
import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex,  StorageContext, Settings, load_index_from_storage
from llama_index.vector_stores.milvus import MilvusVectorStore

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = os.getenv("OPENAI_API_KEY")
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about CRC's Publications!",
        }
    ]

def get_vector_store():
    address = os.getenv("MILVUS_ADDRESS")
    collection = os.getenv("MILVUS_COLLECTION")
    if not address or not collection:
        raise ValueError(
            "Please set milvus_address and milvus_collection to your environment variables"
            " or config them in the .env file"
        )
    store = MilvusVectorStore(
        uri=address,
        user=os.getenv("MILVUS_USERNAME"),
        password=os.getenv("MILVUS_PASSWORD"),
        collection_name=collection,
        dim=int(os.getenv("EMBEDDING_DIM")),
    )
    return store

@st.cache_resource(show_spinner=False)
def load_data():
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )
    
    store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(store)
    return index


index = load_data()

if index is None:
    print("Connection to Milvus Failed")

system_prompt=os.getenv("SYSTEM_PROMPT")
top_k = os.getenv("TOP_K", 3)

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        similarity_top_k=int(top_k),
        system_prompt=system_prompt,
        chat_mode="condense_plus_context", 
        verbose=True, 
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
