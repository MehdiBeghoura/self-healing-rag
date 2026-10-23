from functools import lru_cache

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from self_healing_rag.config import settings


@lru_cache(maxsize=1)
def get_tracer_provider() -> TracerProvider:
    """Configure and return the application's shared tracer provider."""
    provider = TracerProvider(
        resource=Resource.create(
            {
                SERVICE_NAME: settings.app_name,
            }
        )
    )

    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)

    return provider
