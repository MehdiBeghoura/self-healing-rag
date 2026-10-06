from dataclasses import dataclass
from typing import Literal


QueryType = Literal[
    "answerable",
    "hard_negative",
    "unrelated",
]


@dataclass(frozen=True)
class BenchmarkDocument:
    title: str
    source: str
    content: str


@dataclass(frozen=True)
class BenchmarkQuery:
    query: str
    relevant_titles: tuple[str, ...]
    query_type: QueryType


BENCHMARK_DOCUMENTS = [
    BenchmarkDocument(
        title="PostgreSQL Vector Guide",
        source="benchmark-postgresql-vector",
        content=(
            "PostgreSQL can use the pgvector extension to store vector "
            "embeddings and perform similarity search over those embeddings."
        ),
    ),
    BenchmarkDocument(
        title="Employee Data Guide",
        source="benchmark-employee-data",
        content=(
            "PostgreSQL can store employee information such as names, "
            "departments, job titles, and salary values in relational tables. "
            "Salary data can be represented using suitable numeric columns."
        ),
    ),
    BenchmarkDocument(
        title="Text Chunking Guide",
        source="benchmark-chunking",
        content=(
            "Text chunking splits long documents into smaller pieces. "
            "Chunk overlap preserves some surrounding context between "
            "consecutive chunks."
        ),
    ),
    BenchmarkDocument(
        title="LangGraph Guide",
        source="benchmark-langgraph",
        content=(
            "LangGraph represents workflows using state, nodes, and edges. "
            "Conditional edges can route execution to different nodes based "
            "on the current state."
        ),
    ),
    BenchmarkDocument(
        title="OpenTelemetry Guide",
        source="benchmark-opentelemetry",
        content=(
            "OpenTelemetry provides APIs and SDKs for collecting telemetry "
            "such as traces, spans, and metrics from applications."
        ),
    ),
]


BENCHMARK_QUERIES = [
    # ------------------------------------------------------------------
    # ANSWERABLE QUERIES
    # ------------------------------------------------------------------

    BenchmarkQuery(
        query="What is pgvector used for?",
        relevant_titles=("PostgreSQL Vector Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="How does PostgreSQL perform similarity search over embeddings?",
        relevant_titles=("PostgreSQL Vector Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="How are vector embeddings stored in PostgreSQL?",
        relevant_titles=("PostgreSQL Vector Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="How can PostgreSQL store employee salary information?",
        relevant_titles=("Employee Data Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="What kind of employee information can PostgreSQL store?",
        relevant_titles=("Employee Data Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="How can salary data be represented in a PostgreSQL table?",
        relevant_titles=("Employee Data Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="What is chunk overlap used for?",
        relevant_titles=("Text Chunking Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="Why are documents split into chunks for retrieval?",
        relevant_titles=("Text Chunking Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="How does chunk overlap help preserve context?",
        relevant_titles=("Text Chunking Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="What are nodes in LangGraph?",
        relevant_titles=("LangGraph Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="What are conditional edges used for in LangGraph?",
        relevant_titles=("LangGraph Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="How does LangGraph control workflow execution?",
        relevant_titles=("LangGraph Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="What does OpenTelemetry collect from an application?",
        relevant_titles=("OpenTelemetry Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="What types of telemetry can OpenTelemetry collect?",
        relevant_titles=("OpenTelemetry Guide",),
        query_type="answerable",
    ),
    BenchmarkQuery(
        query="What is OpenTelemetry used for when observing an application?",
        relevant_titles=("OpenTelemetry Guide",),
        query_type="answerable",
    ),

    # ------------------------------------------------------------------
    # HARD NEGATIVES
    # ------------------------------------------------------------------

    BenchmarkQuery(
        query="How does PostgreSQL store employee information using vector embeddings?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="How can PostgreSQL use pgvector to manage employee salaries?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="Can pgvector store employee salary values as relational data?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="How does LangGraph store vector embeddings in PostgreSQL?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="Does LangGraph use pgvector to represent workflow nodes?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="How does PostgreSQL similarity search control LangGraph workflow edges?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="What does OpenTelemetry use pgvector for?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="Does OpenTelemetry store application telemetry as pgvector embeddings?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="How does OpenTelemetry use LangGraph conditional edges?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="How does text chunking improve LangGraph workflow state?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="Does chunk overlap control conditional edges in LangGraph?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="How can text chunking improve PostgreSQL vector similarity search?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="Can employee salary data be retrieved using OpenTelemetry?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="Does PostgreSQL use OpenTelemetry to perform vector similarity search?",
        relevant_titles=(),
        query_type="hard_negative",
    ),
    BenchmarkQuery(
        query="How does employee data determine LangGraph conditional edges?",
        relevant_titles=(),
        query_type="hard_negative",
    ),

    # ------------------------------------------------------------------
    # UNRELATED QUERIES
    # ------------------------------------------------------------------

    BenchmarkQuery(
        query="What is the capital city of France?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="How do I bake a chocolate cake?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="How are football matches scored?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="What is the weather like in Paris today?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="How do airplanes generate lift?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="What causes earthquakes?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="How do plants perform photosynthesis?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="What is the largest ocean on Earth?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="How does a combustion engine work?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="What are the health benefits of oranges?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="How do I learn to play the guitar?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="What is the population of Japan?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="How are rainbows formed?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="What is the fastest animal in the world?",
        relevant_titles=(),
        query_type="unrelated",
    ),
    BenchmarkQuery(
        query="How does a refrigerator keep food cold?",
        relevant_titles=(),
        query_type="unrelated",
    ),
]