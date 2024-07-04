# Database interactions for the application
import logging

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import os

from sqlalchemy import create_engine, URL


def connect_to_postgres():
    try:
        connection = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
    except (psycopg2.DatabaseError, Exception) as error:
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
        ORDER BY id ASC ;"""
    df_contexts = pd.read_sql_query(query, connection)
    connection.close()
    return df_contexts
    # return pd.read_excel(os.environ["CHEDBOT_SHEET"], header=0, keep_default_na=False, sheet_name="Contexts")


def load_dialogues(character_id):
    connection = connect_to_postgres()
    query = f"""
        SELECT message.*, context.name as context_name, character.id as character_id, character.name as character_name, character.description as character_description, character.title as character_title, character."chatbotUrl" as "character_chatbotUrl", character."gameId" as "character_gameId", character.history as "character_history",
        character.title as character_title, character.description as character_description
        FROM character
        INNER JOIN context ON character.id = context."characterId"
        INNER JOIN message ON context.id = message."contextId"
        WHERE character.id = {character_id}
        ORDER BY message.intent ASC
        """
    df_messages = pd.read_sql_query(query, connection)
    connection.close()
    return df_messages
    # return pd.read_excel(os.environ["CHEDBOT_SHEET"], header=0, keep_default_na=False, sheet_name="Dialogues").astype(str)


def load_persona(df_character):
    return df_character.description
    # return pd.read_excel(os.environ["CHEDBOT_SHEET"], header=0, keep_default_na=False, sheet_name="Persona")


def load_prompts():
    default_prompt = '''
    {History}
    ##
    System: {Persona}
    ##
    Human: {Utterance}
    Response: {Response}
    ##
    Passe die Message ""Response"" an ""Human"" an.
    ##
    AI:
    '''
    return default_prompt
    # return pd.read_excel(os.environ["CHEDBOT_SHEET"], header=0, keep_default_na=False, sheet_name="Prompts")
