import os
import pandas as pd
import chainlit as cl
from chainlit import user_session
from chainlit.types import LLMSettings
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever


current_agent = os.environ["AGENT"]


def load_dialogues():
    df = pd.read_excel(os.environ["DIALOGUE_SHEET"], header=0, keep_default_na=False)
    df = df[df["Agent"] == current_agent]
    return df.astype(str)


def load_persona():
    df = pd.read_excel(os.environ["PERSONA_SHEET"], header=0, keep_default_na=False)
    df = df[df["Agent"] == current_agent]
    return df.astype(str)


def load_prompt_engineering():
    df = pd.read_excel(
        os.environ["PROMPT_ENGINEERING_SHEET"], header=0, keep_default_na=False
    )
    df = df[df["Agent"] == current_agent]
    return df.astype(str)


def load_documents(df, page_content_column: str):
    return DataFrameLoader(df, page_content_column).load()


def init_embedding_function():
    EMBEDDING_MODEL_FOLDER = ".embedding-model"
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=EMBEDDING_MODEL_FOLDER,
    )


def load_vectordb(init: bool = False):
    vectordb = None
    VECTORDB_FOLDER = ".vectordb"
    if not init:
        vectordb = Chroma(
            embedding_function=init_embedding_function(),
            persist_directory=VECTORDB_FOLDER,
        )
    if init or not vectordb.get()["ids"]:
        vectordb = Chroma.from_documents(
            documents=load_documents(load_dialogues(), page_content_column="Utterance"),
            embedding=init_embedding_function(),
            persist_directory=VECTORDB_FOLDER,
        )
        vectordb.persist()
    return vectordb


def get_retriever(context_state: str, vectordb):
    return VectorStoreRetriever(
        vectorstore=vectordb,
        search_type="similarity",
        search_kwargs={
            "filter": {
                "$or": [{"Context": {"$eq": ""}}, {"Context": {"$eq": context_state}}]
            },
            "k": 1,
        },
    )


vectordb = load_vectordb()


@cl.langchain_factory(use_async=True)
def factory():
    df_prompt_engineering = load_prompt_engineering()
    user_session.set("context_state", "")

    llm_settings = LLMSettings(
        model_name="text-davinci-003",
        temperature=df_prompt_engineering["Temperature"].values[0],
    )
    user_session.set("llm_settings", llm_settings)

    llm = AzureOpenAI(
        deployment_name="davinci003",
        model_name=llm_settings.model_name,
        temperature=llm_settings.temperature,
        streaming=True,
    )

    utterance_prompt = PromptTemplate.from_template(
        df_prompt_engineering["Utterance-Prompt"].values[0]
    )

    chat_memory = ConversationBufferWindowMemory(
        memory_key="History",
        input_key="Utterance",
        k=df_prompt_engineering["History"].values[0],
    )

    utterance_chain = LLMChain(
        prompt=utterance_prompt,
        llm=llm,
        verbose=False,
        memory=chat_memory,
    )

    continuation_prompt = PromptTemplate.from_template(
        df_prompt_engineering["Continuation-Prompt"].values[0]
    )

    continuation_chain = LLMChain(
        prompt=continuation_prompt,
        llm=llm,
        verbose=False,
        memory=chat_memory,
    )

    user_session.set("continuation_chain", continuation_chain)

    return utterance_chain


@cl.langchain_run
async def run(agent, input_str):
    global vectordb
    if input_str == "/reload":
        vectordb = load_vectordb(True)
        await cl.Message(content="Data loaded").send()
    else:
        df_persona = load_persona()

        retriever = get_retriever(user_session.get("context_state"), vectordb)

        document = retriever.get_relevant_documents(query=input_str)

        response = await agent.acall(
            {
                "Persona": df_persona.loc[
                    df_persona["AI"] == document[0].metadata["AI"]
                ]["Persona"].values[0],
                "Utterance": input_str,
                "Response": document[0].metadata["Response"],
            },
            callbacks=[cl.AsyncLangchainCallbackHandler()],
        )
        await cl.Message(
            content=response["text"],
            author=document[0].metadata["AI"],
            llm_settings=user_session.get("llm_settings"),
        ).send()
        user_session.set("context_state", document[0].metadata["Contextualisation"])
        continuation = document[0].metadata["Continuation"]

        while continuation != "":
            document_continuation = vectordb.get(where={"Intent": continuation})
            continuation_chain = user_session.get("continuation_chain")
            response = await continuation_chain.acall(
                {
                    "Persona": df_persona.loc[
                        df_persona["AI"] == document_continuation["metadatas"][0]["AI"]
                    ]["Persona"].values[0],
                    "Utterance": "",
                    "Response": document_continuation["metadatas"][0]["Response"],
                },
                callbacks=[cl.AsyncLangchainCallbackHandler()],
            )
            await cl.Message(
                content=response["text"],
                author=document_continuation["metadatas"][0]["AI"],
                llm_settings=user_session.get("llm_settings"),
            ).send()
            user_session.set(
                "context_state",
                document_continuation["metadatas"][0]["Contextualisation"],
            )
            continuation = document_continuation["metadatas"][0]["Continuation"]
