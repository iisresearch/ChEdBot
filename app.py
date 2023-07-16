import os
import pandas as pd
import chainlit as cl
from chainlit import user_session
from chainlit.types import LLMSettings
from chainlit.logger import logger
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever


current_agent = "Demo"
vectordb = None


def load_agent():
    df = pd.read_excel(os.environ["AGENT_SHEET"], header=0, keep_default_na=False)
    df = df[df["Agent"] == current_agent]
    return df


def load_dialogues():
    df = pd.read_excel(os.environ["DIALOGUE_SHEET"], header=0, keep_default_na=False)
    df = df[df["Agent"] == current_agent]
    return df.astype(str)


def load_persona():
    df = pd.read_excel(os.environ["PERSONA_SHEET"], header=0, keep_default_na=False)
    df = df[df["Agent"] == current_agent]
    return df


def load_prompts():
    df = pd.read_excel(os.environ["PROMPT_SHEET"], header=0, keep_default_na=False)
    df = df[df["Agent"] == current_agent]
    return df


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
    global vectordb
    VECTORDB_FOLDER = ".vectordb"
    if not init and vectordb is None:
        vectordb = Chroma(
            embedding_function=init_embedding_function(),
            persist_directory=VECTORDB_FOLDER,
        )
        if not vectordb.get()["ids"]:
            init = True
        else:
            logger.info(f"Vector DB loaded")
    if init:
        vectordb = Chroma.from_documents(
            documents=load_documents(load_dialogues(), page_content_column="Utterance"),
            embedding=init_embedding_function(),
            persist_directory=VECTORDB_FOLDER,
        )
        vectordb.persist()
        logger.info(f"Vector DB initialised")
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


@cl.langchain_factory(use_async=True)
def factory():
    df_agent = load_agent()
    load_vectordb()
    user_session.set("context_state", "")
    user_session.set("df_prompts", load_prompts())
    user_session.set("df_persona", load_persona())

    llm_settings = LLMSettings(
        model_name="text-davinci-003",
        temperature=0.7,
    )
    user_session.set("llm_settings", llm_settings)

    chat_memory = ConversationBufferWindowMemory(
        memory_key="History",
        input_key="Utterance",
        k=df_agent["History"].values[0],
    )
    user_session.set("chat_memory", chat_memory)

    llm = AzureOpenAI(
        deployment_name="davinci003",
        model_name=llm_settings.model_name,
        temperature=llm_settings.temperature,
        streaming=True,
    )

    default_prompt = """{History}
    ##
    System: {Persona}
    ##
    Human: {Utterance}
    Response: {Response}
    ##
    AI:"""

    return LLMChain(
        prompt=PromptTemplate.from_template(default_prompt),
        llm=llm,
        verbose=True,
        memory=chat_memory,
    )


@cl.langchain_run
async def run(agent, input_str):
    global vectordb
    if input_str == "/reload":
        vectordb = load_vectordb(True)
        return await cl.Message(content="Data loaded").send()

    df_prompts = user_session.get("df_prompts")
    df_persona = user_session.get("df_persona")
    llm_settings = user_session.get("llm_settings")

    retriever = get_retriever(user_session.get("context_state"), vectordb)
    document = retriever.get_relevant_documents(query=input_str)

    prompt = document[0].metadata["Prompt"]
    if not prompt:
        await cl.Message(
            content=document[0].metadata["Response"],
            author=document[0].metadata["Role"],
        ).send()
    else:
        agent.prompt = PromptTemplate.from_template(
            df_prompts.loc[df_prompts["Prompt"] == prompt]["Template"].values[0]
        )
        llm_settings.temperature = df_prompts.loc[df_prompts["Prompt"] == prompt][
            "Temperature"
        ].values[0]
        agent.llm.temperature = llm_settings.temperature

        response = await agent.acall(
            {
                "Persona": df_persona.loc[
                    df_persona["Role"] == document[0].metadata["Role"]
                ]["Persona"].values[0],
                "Utterance": input_str,
                "Response": document[0].metadata["Response"],
            },
            callbacks=[cl.AsyncLangchainCallbackHandler()],
        )
        await cl.Message(
            content=response["text"],
            author=document[0].metadata["Role"],
            llm_settings=llm_settings,
        ).send()

    user_session.set("context_state", document[0].metadata["Contextualisation"])
    continuation = document[0].metadata["Continuation"]

    while continuation != "":
        document_continuation = vectordb.get(where={"Intent": continuation})

        prompt = document_continuation["metadatas"][0]["Prompt"]
        if not prompt:
            await cl.Message(
                content=document_continuation["metadatas"][0]["Response"],
                author=document_continuation["metadatas"][0]["Role"],
            ).send()
        else:
            agent.prompt = PromptTemplate.from_template(
                df_prompts.loc[df_prompts["Prompt"] == prompt]["Template"].values[0]
            )
            llm_settings.temperature = df_prompts.loc[df_prompts["Prompt"] == prompt][
                "Temperature"
            ].values[0]
            agent.llm.temperature = llm_settings.temperature

            response = await agent.acall(
                {
                    "Persona": df_persona.loc[
                        df_persona["Role"]
                        == document_continuation["metadatas"][0]["Role"]
                    ]["Persona"].values[0],
                    "Utterance": "",
                    "Response": document_continuation["metadatas"][0]["Response"],
                },
                callbacks=[cl.AsyncLangchainCallbackHandler()],
            )
            await cl.Message(
                content=response["text"],
                author=document_continuation["metadatas"][0]["Role"],
                llm_settings=llm_settings,
            ).send()
        user_session.set(
            "context_state",
            document_continuation["metadatas"][0]["Contextualisation"],
        )
        continuation = document_continuation["metadatas"][0]["Continuation"]
