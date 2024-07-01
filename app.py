import os
import shutil
import time
import pandas as pd
import chainlit as cl
import database as db
from chainlit import user_session
from chainlit.logger import logger
from chainlit.input_widget import TextInput
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAI
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_chroma.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
# from chainlit.playground.config import add_llm_provider
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()
vectordb = None


def load_documents(df, page_content_column: str):
    return DataFrameLoader(df, page_content_column).load()


def init_embedding_function():
    return AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002")


# Initialize a Vector DB using Chroma. Store and retrieve embeddings of dialogue utterances for efficient similarity
# search. It provides ability to match user input to the most relevant response based on contextual similarity.
def load_vectordb(init: bool = False):
    global vectordb
    VECTORDB_FOLDER = ".vectordb"
    df_character = user_session.get("current_character")
    if not init and vectordb is None:
        if os.path.exists(VECTORDB_FOLDER):
            logger.info(f"Deleting existing Vector DB")
            shutil.rmtree(VECTORDB_FOLDER)
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
        vectordb = Chroma.from_documents(  # Create a new Vector DB from the loaded documents
            documents=load_documents(db.load_dialogues(df_character.id.iloc[0]), page_content_column="utterance"),  # Load dialogue utterances
            embedding=init_embedding_function(),  # Initialize embedding function
            persist_directory=VECTORDB_FOLDER,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        logger.info(f"Vector DB initialised")
    return vectordb


# Create and return an instance of VectoreStoreRetriever for the given context state and score threshold.
def get_retriever(context_state: str, score_threshold: str, vectordb):
    return VectorStoreRetriever(
        vectorstore=vectordb,  # Vector DB
        search_type="similarity_score_threshold",  # Search type
        search_kwargs={  # Search parameters
            "filter": {
                "$and": [
                    {
                        "$or": [
                            {"Context": {"$eq": ""}},  # Empty context state
                            {"Context": {"$eq": context_state}}  # Context state
                        ]
                    },
                    {"Agent": {"$eq": user_session.get("current_character")}}  # Current agent
                ]
            },
            "k": 1,  # Number of results to return
            "score_threshold": score_threshold,  # Minimum similarity score
        },
    )


# Send a message without using LLM, directly sending the content to the user.
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


# Chainlit function that sets up the initial chat and user session, including the current agent, chat settings, and memory for the conversation.
@cl.on_chat_start
async def start():
    # Get the agent ID from the URL query parameter
    agent_id = await cl.CopilotFunction(name="url_query_parameter", args={"msg": "agent_id"}).acall()

    if not agent_id or agent_id == "":
        logger.error("No agent ID found in the URL query parameter.")
        await cl.Message(content="No agent_id provided. Please provide a valid agent_id.").send()
        return
    # Load the agent's data from Postgres DB, see database.py
    df_character = db.load_agent(agent_id)
    print("Available agents:" + str(df_character))

    await cl.Message(content=f"Welcome to the \n{df_character.name.iloc[0]}\n chatbot! You can now start your conversation.").send()

    if df_character is not None and not df_character.empty:
        settings = await cl.ChatSettings(
            [   # Set the chat settings for the user if needed
                # TextInput(id="Agent", label="Agent", initial=available_agents[0]),
                # cl.input_widget.Tags(id="StopSequence", label="OpenAI - StopSequence", initial=["Answer:"]),
                # cl.input_widget.Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            ]
        ).send()
        user_session.set("current_character", df_character)
        logger.info(f"Agent set to {user_session.get('current_character')}")
    else:
        logger.error("No available characters found in df_character. Please check the Postgres DB for the 'Character' table.")
        return
    load_vectordb()
    await set_agent()


@cl.step(name="set_agent", type="llm", show_input=True)
async def set_agent():
    df_character = user_session.get("current_character")
    # user_session.set("context_state", df_character.loc[df_character["Agent"] == user_session.get("current_character"), "Context"].iloc[0])
    user_session.set("context_state", db.load_contexts(df_character.id.iloc[0]).iloc[0])
    #print(f"Context state: {user_session.get("context_state")}")
    # user_session.set("score_threshold", df_character.loc[df_character["Agent"] == user_session.get("current_character"), "Threshold"].iloc[0])
    user_session.set("score_threshold", 0.8)  # Set the similarity score threshold for the user
    user_session.set("df_prompts", db.load_prompts())
    user_session.set("df_persona", db.load_persona(df_character))
    user_session.set("variable_storage", VariableStorage())
    user_session.set("variable_request", "")
    user_session.set("variable_request_continuation", "")
    # Chat memory is managed using ConversationBufferWindowMemory, helping in maintaining context throughout the chat.
    chat_memory = ConversationBufferWindowMemory(
        memory_key="History",
        input_key="Utterance",
        k=df_character["history"],
    )
    user_session.set("chat_memory", chat_memory)

    # Check user environment
    user_env = cl.user_session.get("env")
    print(f"User env: {user_env}")
    print(f"OpenAI API key: {os.getenv('OPENAI_API_KEY')}")
    openai_api_key = user_env.get("OPENAI_API_KEY") if os.getenv(
        "OPENAI_API_KEY") not in os.environ else os.getenv("OPENAI_API_KEY")

    # Sets the LLM and env vars
    llm = AzureOpenAI(
        deployment_name="davinci003",
        model_name="text-davinci-003",
        temperature=0.7,
        streaming=True,
        openai_api_key=openai_api_key,
    )

    default_prompt = """{History}
    ##
    System: {Persona}
    ##
    Human: {Utterance}
    Response: {Response}
    ##
    AI:"""
    # Initialize the LLMChain with the default prompt, LLM, and chat memory.
    user_session.set(
        "llm_chain",
        LLMChain(
            prompt=PromptTemplate.from_template(default_prompt),
            llm=llm,
            verbose=True,
            memory=chat_memory,
        ),
    )
    # Send a welcome message to the user
    await cl.Message(
        content=f"Welcome to the {user_session.get('current_character')} chatbot! You can now start your conversation.").send()
    # add_llm_provider(AzureOpenAIProvider)


# Core logic happens in run() for handling incoming messages and generating responses. Vector DB finds the best match
# for the user's message based on similarity. If found, it's formatted and sent back to the user. /reload command
# refreshes data.
@cl.on_message
async def run(message: cl.Message):
    message_content = message.content
    global vectordb
    if message_content == "/reload":
        vectordb = load_vectordb(True)
        return await cl.Message(content="Data loaded").send()

    df_prompts = user_session.get("df_prompts")
    df_persona = user_session.get("df_persona")
    agent = user_session.get("llm_chain")
    # Conversation continuation is handled based on predefined intents and fallback mechanisms.
    # It dynamically adjusts the conv. context, intents, and response prompts based on user interations and the result of similarity searches. 
    if (user_session.get("variable_request")) != "":
        continuation = user_session.get("variable_request_continuation")
        user_session.get("variable_storage").add(
            user_session.get("variable_request"), message_content
        )
    else:
        retriever = get_retriever(
            user_session.get("context_state"),
            user_session.get("score_threshold"),
            vectordb,
        )
        document = retriever.get_relevant_documents(query=message_content)

        if len(document) == 1:
            user_session.set("context_state", document[0].metadata["Contextualisation"])
            user_session.set("fallback_intent", document[0].metadata["Fallback"])
            user_session.set("variable_request", document[0].metadata["Variable"])
            if (user_session.get("variable_request")) == "":
                continuation = document[0].metadata["Continuation"]
            else:
                continuation = ""
                user_session.set(
                    "variable_request_continuation",
                    document[0].metadata["Continuation"],
                )
            prompt = document[0].metadata["Prompt"]
            if not prompt:
                await sendMessageNoLLM(
                    user_session.get("variable_storage").replace(
                        document[0].metadata["Response"]
                    ),
                    document[0].metadata["Role"],
                )
            else:
                agent.prompt = PromptTemplate.from_template(
                    df_prompts.loc[df_prompts["Prompt"] == prompt]["Template"].values[0]
                )
                agent.llm.temperature = df_prompts.loc[df_prompts["Prompt"] == prompt][
                    "Temperature"
                ].values[0]

                response = await agent.ainvoke(
                    {
                        "Persona": df_persona.loc[
                            df_persona["Role"] == document[0].metadata["Role"]
                            ]["Persona"].values[0],
                        "Utterance": message_content,
                        "Response": user_session.get("variable_storage").replace(
                            document[0].metadata["Response"]
                        ),
                    },
                    callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)],
                )
                await cl.Message(
                    content=response["text"],
                    author=document[0].metadata["Role"],
                ).send()
        else:
            continuation = user_session.get("fallback_intent")

    while continuation != "":
        print(f"Continuation value: {continuation}")
        document_continuation = vectordb.get(where={"Intent": continuation})

        prompt = document_continuation["metadatas"][0]["Prompt"]
        if not prompt:
            await sendMessageNoLLM(
                user_session.get("variable_storage").replace(
                    document_continuation["metadatas"][0]["Response"]
                ),
                document_continuation["metadatas"][0]["Role"],
            )
        else:
            agent.prompt = PromptTemplate.from_template(
                df_prompts.loc[df_prompts["Prompt"] == prompt]["Template"].values[0]
            )
            agent.llm.temperature = df_prompts.loc[df_prompts["Prompt"] == prompt][
                "Temperature"
            ].values[0]

            response = await agent.ainvoke(
                {
                    "Persona": df_persona.loc[
                        df_persona["Role"]
                        == document_continuation["metadatas"][0]["Role"]
                        ]["Persona"].values[0],
                    "Utterance": "",
                    "Response": user_session.get("variable_storage").replace(
                        document_continuation["metadatas"][0]["Response"]
                    ),
                },
                callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)],
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
        user_session.set(
            "variable_request", document_continuation["metadatas"][0]["Variable"]
        )
        if (user_session.get("variable_request")) == "":
            continuation = document_continuation["metadatas"][0]["Continuation"]
        else:
            continuation = ""
            user_session.set(
                "variable_request_continuation",
                document_continuation["metadatas"][0]["Continuation"],
            )


# Responds to chat settings updates
@cl.on_settings_update
async def setup_agent(settings):
    user_session.set("current_character", settings["Agent"])
    logger.info(f"Agent changed to {settings['Agent']}")
    await set_agent()


# VariableStorage manages and utilizes variables within the chatbot's conversations and responses
class VariableStorage:
    def __init__(self):
        self.variables = {}

    def add(self, name, value):
        self.variables[name] = value

    def replace(self, text):
        return text.format(**self.variables)

    def iterate(self):
        for name, value in self.variables.items():
            yield name, value


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
