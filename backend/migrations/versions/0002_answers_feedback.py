"""answers and feedback

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE answers (
          id                bigserial PRIMARY KEY,
          public_id         text UNIQUE NOT NULL,
          question          text NOT NULL,
          answer            text NOT NULL,
          abstained         boolean NOT NULL,
          citations         jsonb NOT NULL,
          chat_model        text NOT NULL,
          embedding_model   text NOT NULL,
          abstain_threshold double precision NOT NULL,
          created_at        timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE feedback (
          id          bigserial PRIMARY KEY,
          answer_id   bigint UNIQUE NOT NULL REFERENCES answers(id) ON DELETE CASCADE,
          rating      text NOT NULL CHECK (rating IN ('up', 'down')),
          reason      text CHECK (reason IN
            ('wrong', 'incomplete', 'wrong_source', 'should_have_answered', 'other')),
          comment     text CHECK (char_length(comment) <= 2000),
          updated_at  timestamptz NOT NULL DEFAULT now(),
          CHECK (reason IS NULL OR rating = 'down')
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE feedback")
    op.execute("DROP TABLE answers")
