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
    expected_answer: str | None
    required_answer_concepts: tuple[tuple[str, ...], ...] = ()


BENCHMARK_DOCUMENTS = [
    BenchmarkDocument(
        title="PostgreSQL Vector Guide",
        source="benchmark-postgresql-vector",
        content=(
            "PostgreSQL can use the pgvector extension to store vector "
            "embeddings alongside ordinary PostgreSQL data. pgvector "
            "supports exact nearest-neighbor search by default, which "
            "compares the query vector against stored vectors and returns "
            "the nearest results. PostgreSQL can also use approximate "
            "nearest-neighbor search with HNSW or IVFFlat indexes when "
            "faster search is preferred. HNSW builds a multilayer graph "
            "over the vectors, while IVFFlat divides vectors into lists "
            "and searches selected lists. pgvector supports several "
            "distance functions, including cosine distance. The cosine "
            "distance operator is <=>, and cosine similarity can be "
            "computed as 1 minus cosine distance."
        ),
    ),
    BenchmarkDocument(
        title="Employee Data Guide",
        source="benchmark-employee-data",
        content=(
            "PostgreSQL can store employee information in relational "
            "tables using ordinary columns. An employee record can "
            "contain fields such as a name, department, job title, and "
            "salary value. Salary data can be represented using a "
            "suitable numeric column. Employee records can therefore "
            "store structured business data without requiring vector "
            "embeddings."
        ),
    ),
    BenchmarkDocument(
        title="Text Chunking Guide",
        source="benchmark-chunking",
        content=(
            "Text chunking divides a long document into smaller pieces "
            "called chunks. Smaller chunks make individual passages "
            "easier to retrieve and provide more focused context to a "
            "retrieval system. Chunk overlap keeps some text from the "
            "end of one chunk in the following chunk. This preserved "
            "overlapping text helps maintain context when an important "
            "sentence or idea crosses a chunk boundary."
        ),
    ),
    BenchmarkDocument(
        title="LangGraph Guide",
        source="benchmark-langgraph",
        content=(
            "LangGraph represents workflows as a graph with shared state, "
            "nodes, and edges. State contains the information passed "
            "through the workflow. A node represents a discrete step of "
            "work and can read the current state and produce state "
            "updates. Edges define transitions between nodes. Conditional "
            "edges choose the next node by evaluating the current state, "
            "which allows workflow execution to follow different paths."
        ),
    ),
    BenchmarkDocument(
        title="OpenTelemetry Guide",
        source="benchmark-opentelemetry",
        content=(
            "OpenTelemetry is a vendor-neutral observability framework "
            "for instrumenting applications and for generating, "
            "collecting, and exporting telemetry data. Important "
            "telemetry types include traces, metrics, and logs. "
            "Applications can use OpenTelemetry APIs and SDKs to produce "
            "telemetry, which can then be exported for monitoring and "
            "observability."
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
        expected_answer=(
            "pgvector is used in PostgreSQL to store vector embeddings "
            "and perform vector similarity or nearest-neighbor search."
        ),
        required_answer_concepts=(
            ("embeddings", "vector embedding"),
            (
                "similarity search",
                "nearest-neighbor search",
                "nearest neighbor search",
                "vector search",
            ),
        ),
    ),
    BenchmarkQuery(
        query="How does PostgreSQL find the nearest vectors?",
        relevant_titles=("PostgreSQL Vector Guide",),
        query_type="answerable",
        expected_answer=(
            "pgvector compares a query vector against stored vectors "
            "and returns the nearest results using a distance function."
        ),
        required_answer_concepts=(
            ("query vector", "query embedding"),
            ("stored vectors", "stored embeddings", "existing vectors"),
            ("nearest", "closest"),
        ),
    ),
    BenchmarkQuery(
        query="What is the default type of nearest-neighbor search in pgvector?",
        relevant_titles=("PostgreSQL Vector Guide",),
        query_type="answerable",
        expected_answer=(
            "pgvector performs exact nearest-neighbor search by default."
        ),
        required_answer_concepts=(
            ("exact",),
            ("nearest-neighbor", "nearest neighbor"),
        ),
    ),
    BenchmarkQuery(
        query="What approximate nearest-neighbor indexes does pgvector support?",
        relevant_titles=("PostgreSQL Vector Guide",),
        query_type="answerable",
        expected_answer=(
            "pgvector supports HNSW and IVFFlat indexes for approximate "
            "nearest-neighbor search."
        ),
        required_answer_concepts=(
            ("hnsw",),
            ("ivfflat",),
        ),
    ),
    BenchmarkQuery(
        query="What does the pgvector <=> operator represent?",
        relevant_titles=("PostgreSQL Vector Guide",),
        query_type="answerable",
        expected_answer=(
            "The pgvector <=> operator represents cosine distance."
        ),
        required_answer_concepts=(
            ("cosine distance",),
            ("<=>",),
        ),
    ),
    BenchmarkQuery(
        query="How can cosine similarity be calculated from cosine distance?",
        relevant_titles=("PostgreSQL Vector Guide",),
        query_type="answerable",
        expected_answer=(
            "Cosine similarity can be calculated as 1 minus cosine distance."
        ),
        required_answer_concepts=(
            ("1 minus", "one minus", "1 -"),
            ("cosine distance",),
        ),
    ),
    BenchmarkQuery(
        query="How can PostgreSQL store employee salary information?",
        relevant_titles=("Employee Data Guide",),
        query_type="answerable",
        expected_answer=(
            "PostgreSQL can store salary information in an employee "
            "relational table using a suitable numeric column."
        ),
        required_answer_concepts=(
            ("salary",),
            (
                "numeric column",
                "numeric field",
                "number column",
                "numeric data type",
            ),
        ),
    ),
    BenchmarkQuery(
        query="What information can an employee record contain?",
        relevant_titles=("Employee Data Guide",),
        query_type="answerable",
        expected_answer=(
            "An employee record can contain a name, department, job title, "
            "and salary value."
        ),
        required_answer_concepts=(
            ("name",),
            ("department",),
            ("job title",),
            ("salary",),
        ),
    ),
    BenchmarkQuery(
        query="Why are long documents divided into chunks?",
        relevant_titles=("Text Chunking Guide",),
        query_type="answerable",
        expected_answer=(
            "Chunking divides long documents into smaller passages that "
            "are easier to retrieve and provide more focused context."
        ),
        required_answer_concepts=(
            ("smaller chunks", "smaller passages", "smaller pieces"),
            ("retrieve", "retrieval"),
            ("context",),
        ),
    ),
    BenchmarkQuery(
        query="What is the purpose of chunk overlap?",
        relevant_titles=("Text Chunking Guide",),
        query_type="answerable",
        expected_answer=(
            "Chunk overlap preserves surrounding context between "
            "consecutive chunks, especially when an idea crosses a "
            "chunk boundary."
        ),
        required_answer_concepts=(
            ("overlap", "overlapping text"),
            ("context",),
            (
                "chunk boundary",
                "boundary",
                "next chunk",
                "consecutive chunks",
            ),
        ),
    ),
    BenchmarkQuery(
        query="What is a node in LangGraph?",
        relevant_titles=("LangGraph Guide",),
        query_type="answerable",
        expected_answer=(
            "A LangGraph node is a discrete step of work that can read "
            "the current state and produce state updates."
        ),
        required_answer_concepts=(
            ("discrete step", "individual step", "step of work"),
            ("state",),
        ),
    ),
    BenchmarkQuery(
        query="What are conditional edges used for in LangGraph?",
        relevant_titles=("LangGraph Guide",),
        query_type="answerable",
        expected_answer=(
            "Conditional edges evaluate the current state and choose "
            "which node should execute next."
        ),
        required_answer_concepts=(
            ("conditional edges", "conditional edge"),
            ("next node", "which node", "next step"),
            ("state",),
        ),
    ),
    BenchmarkQuery(
        query="What does OpenTelemetry collect?",
        relevant_titles=("OpenTelemetry Guide",),
        query_type="answerable",
        expected_answer=(
            "OpenTelemetry collects telemetry such as traces, metrics, "
            "and logs."
        ),
        required_answer_concepts=(
            ("traces",),
            ("metrics",),
            ("logs",),
        ),
    ),
    BenchmarkQuery(
        query="What is OpenTelemetry used for?",
        relevant_titles=("OpenTelemetry Guide",),
        query_type="answerable",
        expected_answer=(
            "OpenTelemetry is used to instrument applications and "
            "generate, collect, and export telemetry for observability."
        ),
        required_answer_concepts=(
            ("instrument", "instrumentation"),
            ("application",),
            ("telemetry", "observability"),
        ),
    ),
    BenchmarkQuery(
        query="Which telemetry types are mentioned in the OpenTelemetry guide?",
        relevant_titles=("OpenTelemetry Guide",),
        query_type="answerable",
        expected_answer=(
            "The guide mentions traces, metrics, and logs."
        ),
        required_answer_concepts=(
            ("traces",),
            ("metrics",),
            ("logs",),
        ),
    ),

    # ------------------------------------------------------------------
    # HARD-NEGATIVE QUERIES
    #
    # These combine concepts in the corpus but assert relationships
    # that the corpus does not establish. The system should abstain.
    # ------------------------------------------------------------------
    BenchmarkQuery(
        query="Does pgvector store employee salary values as vector embeddings?",
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="Can pgvector replace relational salary columns for employee data?",
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="How does pgvector calculate employee salaries?",
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="Does HNSW store employee records in a multilayer graph?",
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="Does IVFFlat determine which employee department should be stored?",
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="Does cosine similarity calculate employee salary values?",
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="Does chunk overlap create vector embeddings for PostgreSQL?",
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="Can chunk overlap control which LangGraph node executes next?",
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="Does LangGraph use pgvector to store workflow state?",
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="Do LangGraph nodes represent PostgreSQL salary columns?",
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="Does OpenTelemetry use HNSW to collect application traces?",
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="Does OpenTelemetry use pgvector to store application logs?",
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="Can OpenTelemetry conditional edges route LangGraph workflows?",
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query=(
            "Does PostgreSQL use OpenTelemetry metrics to calculate "
            "cosine similarity?"
        ),
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query=(
            "Does employee data determine which OpenTelemetry node "
            "runs next?"
        ),
        relevant_titles=(),
        query_type="hard_negative",
        expected_answer=None,
    ),

    # ------------------------------------------------------------------
    # UNRELATED QUERIES
    # ------------------------------------------------------------------
    BenchmarkQuery(
        query="What is the capital city of France?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="How do I bake a chocolate cake?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="How are football matches scored?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="What is the weather like in Paris today?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="How do airplanes generate lift?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="What causes earthquakes?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="How do plants perform photosynthesis?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="What is the largest ocean on Earth?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="How does a combustion engine work?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="What are the health benefits of oranges?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="How do I learn to play the guitar?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="What is the population of Japan?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="How are rainbows formed?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="What is the fastest animal in the world?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
    BenchmarkQuery(
        query="How does a refrigerator keep food cold?",
        relevant_titles=(),
        query_type="unrelated",
        expected_answer=None,
    ),
]