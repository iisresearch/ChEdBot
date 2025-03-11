# Database interactions for the application
import logging
import os

import pandas as pd
import sqlalchemy


def connect_to_postgres():
    try:
        database_uri = (
            f"postgresql://{os.getenv('POSTGRES_USER')}:"
            f"{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}/"
            f"{os.getenv('POSTGRES_DB')}"
        )
        engine = sqlalchemy.create_engine(database_uri)
        connection = engine.connect()
    except (sqlalchemy.DatabaseError, Exception) as error:
        logging.error(f"Postgres DB error:  {error}")
        raise error
    return connection


# Retrieve data from Postgres DB, store information about agents, dialogues, personas, and prompts.
def load_character(character_id):
    connection = connect_to_postgres()
    query = f"""
    SELECT ch.*
    FROM character ch
    WHERE ch.id = {character_id} ;"""
    df_character = pd.read_sql_query(query, connection)
    connection.close()
    return df_character
    # return pd.read_excel(os.environ["CHEDBOT_SHEET"], header=0, keep_default_na=False, sheet_name="Agents")


def load_contexts(character_id):
    connection = connect_to_postgres()
    query = f"""
        SELECT * FROM context 
        WHERE "characterId" = {character_id} 
        ORDER BY id ;"""
    df_contexts = pd.read_sql_query(query, connection)
    connection.close()
    return df_contexts
    # return pd.read_excel(os.environ["CHEDBOT_SHEET"], header=0, keep_default_na=False, sheet_name="Contexts")


def load_dialogues(character_id):
    connection = connect_to_postgres()
    query = f"""
        SELECT message.*, context.name as context_name, character.id as character_id, character.name as character_name, 
        character.description as character_description, character.title as character_title, 
        character."chatbotUrl" as "character_chatbotUrl", character."gameId" as "character_gameId", character.history as "character_history",
        character.title as character_title, character.description as character_description
        FROM character
        INNER JOIN context ON character.id = context."characterId"
        INNER JOIN message ON context.id = message."contextId"
        WHERE character.id = {character_id}
        ORDER BY message.intent
        """
    df_messages = pd.read_sql_query(query, connection)
    connection.close()
    return df_messages
    # return pd.read_excel(os.environ["CHEDBOT_SHEET"], header=0, keep_default_na=False, sheet_name="Dialogues").astype(str)


def load_persona(df_character):
    return df_character
    # return pd.read_excel(os.environ["CHEDBOT_SHEET"], header=0, keep_default_na=False, sheet_name="Persona")

