import os
import time
import pandas as pd
import chainlit as cl
from chainlit import user_session
from chainlit.logger import logger
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from chromadb.config import Settings


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
    return OpenAIEmbeddings(deployment="text-embedding-ada-002")


def load_vectordb(init: bool = False):
    global vectordb
    VECTORDB_FOLDER = ".vectordb"
    if not init and vectordb is None:
        vectordb = Chroma(
            embedding_function=init_embedding_function(),
            persist_directory=VECTORDB_FOLDER,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
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
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        vectordb.persist()
        logger.info(f"Vector DB initialised")
    return vectordb


def get_retriever(context_state: str, score_threshold: str, vectordb):
    return VectorStoreRetriever(
        vectorstore=vectordb,
        search_type="similarity_score_threshold",
        search_kwargs={
            "filter": {
                "$or": [{"Context": {"$eq": ""}}, {"Context": {"$eq": context_state}}]
            },
            "k": 1,
            "score_threshold": score_threshold,
        },
    )


async def sendMessageNoLLM(content: str, author: str):
    msg = cl.Message(
        content="",
        author=author,
    )
    tokens = content.split()
    for i, token in enumerate(tokens):
        time.sleep(0.1)
        await msg.stream_token(token)
        if i < len(tokens) - 1:
            await msg.stream_token(" ")
    await msg.send()


@cl.on_chat_start
def factory():
    df_agent = load_agent()
    load_vectordb()
    user_session.set("context_state", df_agent["Context"].values[0])
    user_session.set("score_threshold", df_agent["Threshold"].values[0])
    user_session.set("df_prompts", load_prompts())
    user_session.set("df_persona", load_persona())

    chat_memory = ConversationBufferWindowMemory(
        memory_key="History",
        input_key="Utterance",
        k=df_agent["History"].values[0],
    )
    user_session.set("chat_memory", chat_memory)

    llm = AzureOpenAI(
        deployment_name="davinci003",
        model_name="text-davinci-003",
        temperature=0.7,
        streaming=True,
        openai_api_key=cl.user_session.get("env").get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" not in os.environ
        else None,
    )

    default_prompt = """{History}
    ##
    System: {Persona}
    ##
    Human: {Utterance}
    Response: {Response}
    ##
    AI:"""

    user_session.set(
        "llm_chain",
        LLMChain(
            prompt=PromptTemplate.from_template(default_prompt),
            llm=llm,
            verbose=True,
            memory=chat_memory,
        ),
    )


@cl.on_message
async def run(message: str):
    global vectordb
    if message == "/reload":
        vectordb = load_vectordb(True)
        return await cl.Message(content="Data loaded").send()

    df_prompts = user_session.get("df_prompts")
    df_persona = user_session.get("df_persona")
    agent = user_session.get("llm_chain")

    retriever = get_retriever(
        user_session.get("context_state"), user_session.get("score_threshold"), vectordb
    )
    document = retriever.get_relevant_documents(query=message)

    if len(document) == 1:
        user_session.set("fallback_intent", document[0].metadata["Fallback"])
        user_session.set("context_state", document[0].metadata["Contextualisation"])
        continuation = document[0].metadata["Continuation"]
        prompt = document[0].metadata["Prompt"]
        if not prompt:
            await sendMessageNoLLM(
                document[0].metadata["Response"], document[0].metadata["Role"]
            )
        else:
            agent.prompt = PromptTemplate.from_template(
                df_prompts.loc[df_prompts["Prompt"] == prompt]["Template"].values[0]
            )
            agent.llm.temperature = df_prompts.loc[df_prompts["Prompt"] == prompt][
                "Temperature"
            ].values[0]

            response = await agent.acall(
                {
                    "Persona": df_persona.loc[
                        df_persona["Role"] == document[0].metadata["Role"]
                    ]["Persona"].values[0],
                    "Utterance": message,
                    "Response": document[0].metadata["Response"],
                },
                callbacks=[cl.AsyncLangchainCallbackHandler()],
            )
            await cl.Message(
                content=response["text"],
                author=document[0].metadata["Role"],
            ).send()
    else:
        continuation = user_session.get("fallback_intent")

    while continuation != "":
        document_continuation = vectordb.get(where={"Intent": continuation})

        prompt = document_continuation["metadatas"][0]["Prompt"]
        if not prompt:
            await sendMessageNoLLM(
                document_continuation["metadatas"][0]["Response"],
                document_continuation["metadatas"][0]["Role"],
            )
        else:
            agent.prompt = PromptTemplate.from_template(
                df_prompts.loc[df_prompts["Prompt"] == prompt]["Template"].values[0]
            )
            agent.llm.temperature = df_prompts.loc[df_prompts["Prompt"] == prompt][
                "Temperature"
            ].values[0]

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
            ).send()
        user_session.set(
            "context_state",
            document_continuation["metadatas"][0]["Contextualisation"],
        )
        user_session.set(
            "fallback_intent", document_continuation["metadatas"][0]["Fallback"]
        )
        continuation = document_continuation["metadatas"][0]["Continuation"]
