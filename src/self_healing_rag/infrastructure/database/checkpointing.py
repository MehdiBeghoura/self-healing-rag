from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from self_healing_rag.config import settings


@contextmanager
def postgres_checkpointer() -> Iterator[PostgresSaver]:
    """Create and manage a PostgreSQL-backed LangGraph checkpointer."""

    # SQLAlchemy uses the "postgresql+psycopg" driver name.
    # Psycopg expects the standard "postgresql" connection URI scheme.
    conninfo = (
        settings.database_url
        .set(drivername="postgresql")
        .render_as_string(hide_password=False)
    )

    with ConnectionPool(
        conninfo=conninfo,
        min_size=1,
        max_size=5,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=False,
        name="self-healing-rag-checkpointer",
    ) as pool:
        # Wait until the initial pool connection is ready.
        pool.wait()

        # Restrict checkpoint deserialization to explicitly
        # permitted message-pack types and safe built-in types.
        serializer = JsonPlusSerializer(
            allowed_msgpack_modules=[],
        )

        checkpointer = PostgresSaver(
            pool,
            serde=serializer,
        )

        # Create or update the required checkpoint tables.
        checkpointer.setup()

        yield checkpointer