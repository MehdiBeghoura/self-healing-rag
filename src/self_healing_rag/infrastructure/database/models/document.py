from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from self_healing_rag.infrastructure.database.base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    source: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )