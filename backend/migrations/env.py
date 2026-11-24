"""Alembic environment: hand-written SQL migrations, no ORM models."""

from logging.config import fileConfig

import psycopg
from alembic import context
from sqlalchemy import create_engine

from plumbline.db import database_url

if context.config.config_file_name:
    fileConfig(context.config.config_file_name, disable_existing_loggers=False)

url = database_url()

# psycopg parses DATABASE_URL itself, so both URL and key=value forms work.
engine = create_engine("postgresql+psycopg://", creator=lambda: psycopg.connect(url))
with engine.connect() as connection:
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()
