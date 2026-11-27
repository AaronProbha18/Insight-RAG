"""conversations

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE conversations (
          id          bigserial PRIMARY KEY,
          public_id   text UNIQUE NOT NULL,
          title       text NOT NULL CHECK (char_length(title) BETWEEN 1 AND 200),
          created_at  timestamptz NOT NULL DEFAULT now(),
          updated_at  timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX conversations_updated_at ON conversations (updated_at)")
    op.execute(
        "ALTER TABLE answers ADD COLUMN conversation_id bigint"
        " REFERENCES conversations(id) ON DELETE SET NULL"
    )
    op.execute("CREATE INDEX answers_conversation_id ON answers (conversation_id)")


def downgrade() -> None:
    op.execute("ALTER TABLE answers DROP COLUMN conversation_id")
    op.execute("DROP TABLE conversations")
