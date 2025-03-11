import logging
import os
import time
from urllib.parse import urlparse, parse_qs

import chainlit as cl
from chainlit import user_session
from chainlit.logger import logger
# from chainlit.playground.config import add_llm_provider
from chromadb.config import Settings
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureChatOpenAI
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

import database as db

logging.basicConfig(level=logging.INFO)
vectordb = None


def load_documents(df, page_content_column: str):
    documents = DataFrameLoader(df, page_content_column).load()
    # Iterate over documents to add page_content_column ("response") to metadata
    for doc in documents:
        # Assuming each document has a unique identifier to match with the DataFrame row
        if not doc.metadata.get(page_content_column):
            # Add or update the page_content_column in metadata
            doc.metadata[page_content_column] = doc.page_content
    return documents


def init_embedding_function():
    # return HuggingFaceEndpointEmbeddings(model="all-miniLM-L6-v2", huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    return AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002")


# Initialize a Vector DB using Chroma. Store and retrieve embeddings of dialogue utterances for efficient similarity
# search. It provides ability to match user input to the most relevant response based on contextual similarity.
def load_vectordb(init: bool = False):
    global vectordb
    VECTORDB_FOLDER = ".vectordb"
    df_character = user_session.get("current_character")
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
        if os.path.exists(VECTORDB_FOLDER):
            vectordb = Chroma(
                embedding_function=init_embedding_function(),
                persist_directory=VECTORDB_FOLDER,
                client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
            )
            character_docs = vectordb.get(where={"character_id": int(df_character.id.iloc[0])})
            logger.info(f"Character docs: {character_docs}")
            if character_docs.get("ids"):
                vectordb.delete(character_docs.get("ids"))
                logger.info(f"Deleted existing Vector DB for character {df_character.id.iloc[0]}")
            else:
                logger.info(f"No existing Vector DB found for character {df_character.id.iloc[0]}")
        # Load latest data from Postgres DB for the current character
        docs = load_documents(db.load_dialogues(int(df_character.id.iloc[0])), page_content_column="utterance")
        filtered_docs = filter_complex_metadata(docs)
        vectordb = Chroma.from_documents(  # Create a new Vector DB from the loaded documents
            documents=filtered_docs,  # Load dialogue utterances
            embedding=init_embedding_function(),  # Initialize embedding function
            persist_directory=VECTORDB_FOLDER,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        logger.info(f"Vector DB initialised")
    return vectordb


# Create and return an instance of VectorStoreRetriever for the given context state and score threshold.
def get_retriever(context_state: str, score_threshold: str, vectordb):
    character_id = int(user_session.get("current_character")['id'].iloc[0])
    return VectorStoreRetriever(
        vectorstore=vectordb,  # Vector DB
        search_type="similarity_score_threshold",  # Search type
        search_kwargs={  # Search parameters
            "filter": {
                "$and": [
                    {
                        "$or": [
                            {"context_name": {"$eq": ""}},  # Empty context state
                            {"context_name": {"$eq": context_state}}  # Context state
                        ]
                    },
                    {"character_id": {"$eq": character_id}}  # Current character id
                ]
            },
            "k": 1,  # Number of results to return
            "score_threshold": score_threshold,  # Minimum similarity score
        },
    )


# Send a message without using LLM, directly sending the content to the user.
async def send_message_no_llm(content: str, author: str, user_message: cl.Message):
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

    chat_memory = cl.user_session.get("chat_memory")
    chat_memory.save_context({"utterance": user_message.content}, {"response": msg.content})

    logger.info(chat_memory.load_memory_variables(inputs={"utterance": user_message.content}))

    await msg.send()


# Chainlit function that sets up the initial chat and user session, including the current character, chat settings, and memory for the conversation.
@cl.on_chat_start
async def start():
    # Get the character ID from the URL query parameter
    # character_id = await cl.CopilotFunction(name="url_query_parameter", args={"msg": "character_id"}).acall()
    character_id = cl.user_session.get("character_id")
    character_id = parse_qs(urlparse(cl.user_session.get("http_referer")).query).get("character_id", [character_id])[0]
    cl.user_session.set("character_id", character_id)
    logging.info(f"Character ID: {character_id}")
    if not character_id or character_id == "":
        logger.error("No character ID found in the URL query parameter.")
        await cl.Message(
            content="No character_id provided. Please provide a valid character_id as a URI query parameter.").send()
        return
    # Load the character's data from Postgres DB, see database.py
    df_character = db.load_character(character_id)
    logger.info(f"Character data: {df_character}")

    if df_character is not None and not df_character.empty:
        settings = await cl.ChatSettings(
            [  # Set the chat settings for the user if needed
                cl.input_widget.Switch(
                    id="Model",
                    label="Use LLM",
                    initial=False,
                ),
                # TextInput(id="Agent", label="Agent", initial=available_characters[0]),
                # cl.input_widget.Tags(id="StopSequence", label="OpenAI - StopSequence", initial=["Answer:"]),
                # cl.input_widget.Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            ]
        ).send()
        user_session.set("settings", settings)
        user_session.set("current_character", df_character)
        logger.info(f"Agent set to {user_session.get('current_character')}")
    else:
        logger.error(
            "No available characters found in df_character. Please check the Postgres DB for the 'Character' table.")
        return
    load_vectordb(True)

    await set_character()


@cl.step
async def set_character():
    df_character = user_session.get("current_character")
    # user_session.set("context_state", df_character.loc[df_character["Agent"] == user_session.get("current_character"), "Context"].iloc[0])
    user_session.set("context_state", db.load_contexts(int(df_character.id.iloc[0]))['name'].iloc[0])
    print(f"Context state: {user_session.get("context_state")}")
    # user_session.set("score_threshold", df_character.loc[df_character["Agent"] == user_session.get("current_character"), "Threshold"].iloc[0])
    user_session.set("score_threshold", 0.7)  # Set the similarity score threshold for the user
    # user_session.set("df_persona", df_character)  # db.load_persona(df_character)
    user_session.set("variable_storage", VariableStorage())
    user_session.set("variable_request", "")
    user_session.set("variable_request_continuation", "")
    character = user_session.get('current_character')

    # Chat memory is managed using ConversationBufferWindowMemory, helping in maintaining context throughout the chat.
    chat_memory = ConversationBufferWindowMemory(
        memory_key="history",
        input_key="utterance",
        k=df_character["history"],
    )
    user_session.set("chat_memory", chat_memory)

    # Check user environment
    print(f"OpenAI API key: {os.getenv('OPENAI_API_KEY')}")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    # Sets the LLM and env vars

    llm = AzureChatOpenAI(
        deployment_name="gpt-4o",
        model_name="gpt-4o",
        temperature=0.2,
        streaming=True,
        openai_api_key=openai_api_key,
        verbose=True,
    )

    # Initialize the LLMChain with the default prompt, LLM, and chat memory.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Passe die folgende 'Response' an 'human' an. Erfinde keine Informationen, die nicht in der 'Response', 'Persona' oder 'history' enthalten sind. \n"
            "Response: {response} \n"
            "Persona: {persona} \n "
            "{history} \n"),
            ("human", "{utterance}"),
            ("ai", ""),
        ]
    )
    user_session.set("df_prompts", prompt)

    chain = prompt | llm | StrOutputParser()

    user_session.set(
        "llm_chain",
        chain,
    )

    # Send a welcome message to the user
    # await cl.Message(
    #     content=f"Welcome to {character['name'].iloc[0]}! {character['name'].iloc[0]} is a {character['title'].iloc[0]}. "
    #             f"You can now start your conversation."
    # ).send()


# Core logic happens in run() for handling incoming messages and generating responses. Vector DB finds the best match
# for the user's message based on similarity. If found, it's formatted and sent back to the user. /reload command
# refreshes data.
@cl.on_message
async def run(message: cl.Message):
    message_content = message.content
    global vectordb
    if message_content == "/reload":
        load_vectordb(True)
        return await cl.Message(content="Data loaded").send()

    prompt = user_session.get("df_prompts")
    # df_persona = user_session.get("df_persona")
    character = user_session.get("llm_chain")
    df_character = user_session.get("current_character")
    chat_memory = user_session.get("chat_memory")
    settings = user_session.get("settings")
    model_settings = settings["Model"]
    print(f"Model settings: {model_settings}")
    # Conversation continuation is handled based on predefined intents and fallback mechanisms.
    # It dynamically adjusts the conv. context, intents, and response prompts based on user interations and the result of similarity searches.
    if (user_session.get("variable_request")) != "":
        continuation = user_session.get("variable_request_continuation")
        user_session.get("variable_storage").add(
            user_session.get("variable_request"), message_content
        )
    else:
        context_state = user_session.get("context_state")
        retriever = get_retriever(
            user_session.get("context_state"),
            user_session.get("score_threshold"),
            vectordb,
        )
        document = retriever.invoke(input=message_content)
        logging.info(f"Retrieved documents: {document}")

        if len(document) == 1:
            user_session.set("context_state", document[0].metadata["contextualisation"])
            user_session.set("fallback_intent", document[0].metadata["fallback"])
            user_session.set("variable_request", document[0].metadata["variable"])
            if (user_session.get("variable_request")) == "":
                continuation = document[0].metadata["continuation"]
            else:
                continuation = ""
                user_session.set(
                    "variable_request_continuation",
                    document[0].metadata["continuation"],
                )
            # prompt = document[0].metadata["prompt"] # TODO: Check if prompts are needed
            # prompt = None

            if not model_settings and document[0].metadata.get("response"):
                await send_message_no_llm(
                    user_session.get("variable_storage").replace(
                        document[0].metadata["response"]
                    ),
                    document[0].metadata["character_title"],
                    message
                )

            else:
                logger.info(f"After retrieval: \n Character: {character}\n\n")

                response = character.astream(
                    {
                        "persona": df_character.loc[
                            df_character["title"] == document[0].metadata["character_title"]
                            ]["description"].values[0],
                        "utterance": message_content,
                        "response": user_session.get("variable_storage").replace(
                            document[0].metadata["response"]
                        ),
                        "history": chat_memory.load_memory_variables(inputs={"utterance": message.content}),
                    },
                    config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)]),
                )
                msg = cl.Message(
                    content="",  # response["text"],
                    author=document[0].metadata["character_title"])
                async for token in response:
                    await msg.stream_token(token)
                # Set History
                chat_memory.save_context({"utterance": message.content}, {"response": msg.content})
                logger.info(chat_memory.load_memory_variables(inputs={"utterance": message.content}))
                await msg.send()
        else:
            continuation = user_session.get("fallback_intent")

    while continuation != "":
        logger.info(f"Continuation value: {continuation}")

        document_continuation = vectordb.get(where={"intent": int(continuation)})
        logger.info(f"Continuation document: \n{document_continuation}")
        # document_continuation["metadatas"][0]["prompt"]
        if not model_settings and document_continuation.get('metadatas')[0].get("response"):
            await send_message_no_llm(
                user_session.get("variable_storage").replace(
                    document_continuation["metadatas"][0]["response"]
                ),
                document_continuation["metadatas"][0]["character_title"],
                message
            )
        else:
            logger.info(f"\n\nCharacter: {character}\n\n")
            response = character.astream(
                {
                    "persona": df_character.loc[
                        df_character["title"] == document_continuation["metadatas"][0]["character_title"]
                        ]["description"].values[0],
                    "utterance": "",
                    "response": user_session.get("variable_storage").replace(
                        document_continuation["metadatas"][0]["response"]
                    ),
                    "history": chat_memory.load_memory_variables(inputs={"utterance": message.content}),
                },
                config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)]),
            )
            msg = cl.Message(
                content="",  # response["text"],
                author=document_continuation["metadatas"][0]["character_title"],
            )
            async for token in response:
                await msg.stream_token(token)
            chat_memory.save_context({"utterance": message.content}, {"response": msg.content})
            logger.info(chat_memory.load_memory_variables(inputs={"utterance": message.content}))
            await msg.send()
        user_session.set(
            "context_state",
            document_continuation["metadatas"][0]["contextualisation"],
        )
        user_session.set(
            "fallback_intent", document_continuation["metadatas"][0]["fallback"]
        )
        user_session.set(
            "variable_request", document_continuation["metadatas"][0]["variable"]
        )
        if (user_session.get("variable_request")) == "":
            continuation = document_continuation["metadatas"][0]["continuation"]
        else:
            continuation = ""
            user_session.set(
                "variable_request_continuation",
                document_continuation["metadatas"][0]["continuation"],
            )


# Responds to chat settings updates
# @cl.on_settings_update
# async def setup_character(settings):
#     user_session.set("current_character", settings["character"])
#     logger.info(f"Character changed to {settings['character']}")
#     await set_character()

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
