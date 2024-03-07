import os
from pathlib import Path

import chainlit as cl
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core import StorageContext, load_index_from_storage, SimpleDirectoryReader, GPTVectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from dotenv import load_dotenv

load_dotenv()

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900
Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])

try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
except:

    documents = SimpleDirectoryReader("./data").load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist()


@cl.on_chat_start
async def factory():
    query_engine = index.as_query_engine(
        streaming=True,
    )

    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    response = await cl.make_async(query_engine.query)(message.content)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    if response.response_txt:
        response_message.content = response.response_txt

    await response_message.send()
