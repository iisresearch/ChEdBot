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
def load_agent(agent_id):
    connection = connect_to_postgres()
    query = f"""
    SELECT ch.*
    FROM character ch
    WHERE ch.id = {agent_id} ;"""
    df_agent = pd.read_sql_query(query, connection)
    connection.close()
    return df_agent
    # return pd.read_excel(os.environ["CHEDBOT_SHEET"], header=0, keep_default_na=False, sheet_name="Agents")


def load_contexts(agent_id):
    connection = connect_to_postgres()
    query = f"""
        SELECT * FROM context 
        WHERE "characterId" = {agent_id} 
        ORDER BY id ASC ;
        """
    df_contexts = pd.read_sql_query(query, connection)
    connection.close()
    return df_contexts
    # return pd.read_excel(os.environ["CHEDBOT_SHEET"], header=0, keep_default_na=False, sheet_name="Contexts")


def load_dialogues(agent_id):
    connection = connect_to_postgres()
    query = f"""
        SELECT * FROM character, message, context
        WHERE context.id = message."contextId" AND character.id = {agent_id} 
        ORDER BY message.intent ASC
        """
    df_messages = pd.read_sql_query(query, connection)
    connection.close()
    return df_messages
    # return pd.read_excel(os.environ["CHEDBOT_SHEET"], header=0, keep_default_na=False, sheet_name="Dialogues").astype(str)


def load_persona(df_agent):
    return df_agent.description
    # return pd.read_excel(os.environ["CHEDBOT_SHEET"], header=0, keep_default_na=False, sheet_name="Persona")


def load_prompts():
    return pd.read_excel(os.environ["CHEDBOT_SHEET"], header=0, keep_default_na=False, sheet_name="Prompts")
