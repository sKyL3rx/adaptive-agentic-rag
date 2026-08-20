from __future__ import annotations

from dataclasses import dataclass
import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


SUPPORTED_MODEL_PROVIDERS = frozenset({"openai", "runpod"})
DEFAULT_OPENAI_TEST_MODEL = "gpt-4o-mini"


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "y", "on"}:
        return True

    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    return default


class BaseAgentSettings(BaseModel):
    """
    Configuration for one model.
    """

    prefix: str = ""
    role: str = ""

    # Provider configuration.
    model_provider: str = Field(
        default_factory=lambda: _env_str(
            "MODEL_PROVIDER",
            "runpod",
        )
    )
    runpod_api_key: str = Field(
        default_factory=lambda: _env_str(
            "RUNPOD_API_KEY",
            "",
        )
    )
    openai_api_key: str = Field(
        default_factory=lambda: _env_str(
            "OPENAI_API_KEY",
            "",
        )
    )
    openai_base_url: str = Field(
        default_factory=lambda: _env_str(
            "OPENAI_BASE_URL",
            "",
        )
    )
    
    openai_base_id: str = ""
    openai_model: str = ""

    # Generation defaults.
    openai_temperature: float = Field(
        default_factory=lambda: _env_float(
            "OPENAI_TEMPERATURE",
            0.0,
        )
    )
    openai_max_tokens: int = Field(
        default_factory=lambda: _env_int(
            "OPENAI_MAX_TOKENS",
            1024,
        )
    )
    openai_top_p: float = Field(
        default_factory=lambda: _env_float(
            "OPENAI_TOP_P",
            1.0,
        )
    )
    openai_presence_penalty: float = Field(
        default_factory=lambda: _env_float(
            "OPENAI_PRESENCE_PENALTY",
            0.0,
        )
    )
    openai_frequency_penalty: float = Field(
        default_factory=lambda: _env_float(
            "OPENAI_FREQUENCY_PENALTY",
            0.0,
        )
    )

    # NeMo Guardrails.
    nemo_url: str = Field(
        default_factory=lambda: _env_str(
            "NEMO_URL",
            "",
        )
    )
    nemo_config_in: str = Field(
        default_factory=lambda: _env_str(
            "NEMO_GUARD_CONFIG_IN",
            "",
        )
    )
    nemo_config_out: str = Field(
        default_factory=lambda: _env_str(
            "NEMO_GUARD_CONFIG_OUT",
            "",
        )
    )
    nemo_config_inline: str = Field(
        default_factory=lambda: _env_str(
            "NEMO_GUARD_CONFIG_INLINE",
            "",
        )
    )

    handbook_dir: str = Field(
        default_factory=lambda: _env_str(
            "HANDBOOK_DIR",
            "./data/handbook",
        )
    )
    qdrant_url: str = Field(
        default_factory=lambda: _env_str(
            "QDRANT_URL",
            "http://localhost:6333",
        )
    )
    qdrant_api_key: str = Field(
        default_factory=lambda: _env_str(
            "QDRANT_API_KEY",
            "",
        )
    )
    qdrant_collection: str = Field(
        default_factory=lambda: _env_str(
            "QDRANT_COLLECTION",
            "dmv-handbook",
        )
    )
    embedding_model: str = Field(
        default_factory=lambda: _env_str(
            "EMBEDDING_MODEL",
            "text-embedding-3-small",
        )
    )

    enable_output_guard: bool = False
    guard_fail_close: bool = False

    default_timeout: int = Field(
        default_factory=lambda: _env_int(
            "DEFAULT_TIMEOUT",
            30,
        )
    )
    default_max_tries: int = Field(
        default_factory=lambda: _env_int(
            "DEFAULT_MAX_TRIES",
            1,
        )
    )

    def model_post_init(self, __context: object) -> None:
        if not self.prefix:
            self.model_provider = self.model_provider.strip().lower()
            return

        self.model_provider = _env_str(
            f"{self.prefix}_MODEL_PROVIDER",
            self.model_provider,
        ).strip().lower()

        if self.model_provider not in SUPPORTED_MODEL_PROVIDERS:
            raise ValueError(
                f"Unsupported model provider '{self.model_provider}' "
                f"for role '{self.prefix}'. Supported providers: "
                f"{', '.join(sorted(SUPPORTED_MODEL_PROVIDERS))}"
            )

        self.openai_base_id = _env_str(
            f"{self.prefix}_MODEL_ID",
            "",
        ).strip()

        self.openai_model = _env_str(
            f"{self.prefix}_MODEL",
            DEFAULT_OPENAI_TEST_MODEL,
        ).strip()

        self.openai_temperature = _env_float(
            f"{self.prefix}_OPENAI_TEMPERATURE",
            self.openai_temperature,
        )
        self.openai_max_tokens = _env_int(
            f"{self.prefix}_OPENAI_MAX_TOKENS",
            self.openai_max_tokens,
        )
        self.openai_top_p = _env_float(
            f"{self.prefix}_OPENAI_TOP_P",
            self.openai_top_p,
        )
        self.openai_presence_penalty = _env_float(
            f"{self.prefix}_OPENAI_PRESENCE_PENALTY",
            self.openai_presence_penalty,
        )
        self.openai_frequency_penalty = _env_float(
            f"{self.prefix}_OPENAI_FREQUENCY_PENALTY",
            self.openai_frequency_penalty,
        )

        self.enable_output_guard = _env_bool(
            f"ENABLE_OUTPUT_GUARD_{self.prefix}",
            False,
        )
        self.guard_fail_close = _env_bool(
            f"GUARD_FAIL_CLOSE_{self.prefix}",
            False,
        )

        self.default_timeout = _env_int(
            f"{self.prefix}_TIMEOUT_SECONDS",
            self.default_timeout,
        )
        self.default_max_tries = _env_int(
            f"{self.prefix}_MAX_RETRIES",
            self.default_max_tries,
        )

        self.role = self.prefix.lower()


@dataclass(frozen=True)
class ModelRoleSettings:
    compiler: BaseAgentSettings
    context: BaseAgentSettings
    planner: BaseAgentSettings
    answer: BaseAgentSettings
    reviewer: BaseAgentSettings


def load_model_role_settings() -> ModelRoleSettings:
    """Load fresh role settings from the current environment."""

    return ModelRoleSettings(
        compiler=BaseAgentSettings(prefix="COMPILER"),
        context=BaseAgentSettings(prefix="CONTEXT"),
        planner=BaseAgentSettings(prefix="PLANNER"),
        answer=BaseAgentSettings(prefix="ANSWER"),
        reviewer=BaseAgentSettings(prefix="REVIEWER"),
    )


@dataclass(frozen=True)
class WorkflowSettings:
    # Entire request/workflow.
    workflow_deadline_ms: int
    workflow_max_iterations: int

    # Central budgets.
    max_model_calls: int
    max_retrieval_rounds: int
    max_sub_questions: int

    # Stage timeouts.
    compile_timeout_seconds: float
    context_timeout_seconds: float
    plan_timeout_seconds: float
    retrieval_timeout_seconds: float
    answer_timeout_seconds: float
    verify_timeout_seconds: float

    # History limits.
    compile_history_max_turns: int
    compile_history_max_chars: int
    context_history_max_turns: int
    context_history_max_chars: int

    # Retrieval policies.
    single_top_k: int
    broad_top_k: int
    complex_top_k: int
    recovery_top_k: int
    retrieval_candidate_multiplier: int

    # Coverage policy.
    broad_min_evidence_items: int

    # Prompt payload limits.
    answer_evidence_chars_per_item: int
    review_evidence_chars_per_item: int

    # MCP connection.
    mcp_url: str
    mcp_tool_name: str

    retrieval_corpus_version: str

    mcp_request_timeout_seconds: int
    mcp_sse_read_timeout_seconds: float

    

    def __post_init__(self) -> None:
        self._validate_positive_int(
            "workflow_deadline_ms",
            self.workflow_deadline_ms,
        )
        self._validate_positive_int(
            "workflow_max_iterations",
            self.workflow_max_iterations,
        )
        self._validate_positive_int(
            "max_model_calls",
            self.max_model_calls,
        )
        self._validate_positive_int(
            "max_retrieval_rounds",
            self.max_retrieval_rounds,
        )
        self._validate_positive_int(
            "max_sub_questions",
            self.max_sub_questions,
        )

        for name, value in (
            ("compile_timeout_seconds", self.compile_timeout_seconds),
            ("context_timeout_seconds", self.context_timeout_seconds),
            ("plan_timeout_seconds", self.plan_timeout_seconds),
            ("retrieval_timeout_seconds", self.retrieval_timeout_seconds),
            ("answer_timeout_seconds", self.answer_timeout_seconds),
            ("verify_timeout_seconds", self.verify_timeout_seconds),
            ("mcp_sse_read_timeout_seconds", self.mcp_sse_read_timeout_seconds),
        ):
            self._validate_positive_float(name, value)

        for name, value in (
            ("compile_history_max_turns", self.compile_history_max_turns),
            ("compile_history_max_chars", self.compile_history_max_chars),
            ("context_history_max_turns", self.context_history_max_turns),
            ("context_history_max_chars", self.context_history_max_chars),
            ("single_top_k", self.single_top_k),
            ("broad_top_k", self.broad_top_k),
            ("complex_top_k", self.complex_top_k),
            ("recovery_top_k", self.recovery_top_k),
            (
                "retrieval_candidate_multiplier",
                self.retrieval_candidate_multiplier,
            ),
            ("broad_min_evidence_items", self.broad_min_evidence_items),
            (
                "answer_evidence_chars_per_item",
                self.answer_evidence_chars_per_item,
            ),
            (
                "review_evidence_chars_per_item",
                self.review_evidence_chars_per_item,
            ),
            ("mcp_request_timeout_seconds", self.mcp_request_timeout_seconds),
        ):
            self._validate_positive_int(name, value)

        for name, value in (
            ("single_top_k", self.single_top_k),
            ("broad_top_k", self.broad_top_k),
            ("complex_top_k", self.complex_top_k),
            ("recovery_top_k", self.recovery_top_k),
        ):
            if value > 20:
                raise ValueError(
                    f"{name} must be <= 20; got {value}"
                )

        normalized_url = self.mcp_url.strip().rstrip("/")
        parsed = urlparse(normalized_url)

        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                "MCP_URL must start with http:// or https://"
            )

        if not parsed.netloc:
            raise ValueError(
                "MCP_URL must include a host and port/domain"
            )

        if not parsed.path:
            raise ValueError(
                "MCP_URL must include the MCP endpoint path, "
                "for example /sse or /mcp"
            )

        tool_name = self.mcp_tool_name.strip()
        if not tool_name:
            raise ValueError(
                "MCP_TOOL_NAME must not be empty"
            )

        corpus_version = (
            self.retrieval_corpus_version
            .strip()
        )

        if not corpus_version:
            raise ValueError(
                "RETRIEVAL_CORPUS_VERSION "
                "must not be empty"
        )

        object.__setattr__(self, "mcp_url", normalized_url)
        object.__setattr__(self, "mcp_tool_name", tool_name)
        object.__setattr__(self, "retrieval_corpus_version", corpus_version)

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(
                f"{name} must be > 0; got {value}"
            )

    @staticmethod
    def _validate_positive_float(name: str, value: float) -> None:
        if value <= 0:
            raise ValueError(
                f"{name} must be > 0; got {value}"
            )


def load_workflow_settings() -> WorkflowSettings:
    return WorkflowSettings(
        workflow_deadline_ms=_env_int(
            "WORKFLOW_DEADLINE_MS",
            30_000,
        ),
        workflow_max_iterations=_env_int(
            "WORKFLOW_MAX_ITERATIONS",
            18,
        ),
        max_model_calls=_env_int(
            "MAX_MODEL_CALLS",
            3,
        ),
        max_retrieval_rounds=_env_int(
            "MAX_RETRIEVAL_ROUNDS",
            2,
        ),
        max_sub_questions=_env_int(
            "MAX_SUB_QUESTIONS",
            4,
        ),
        compile_timeout_seconds=_env_float(
            "COMPILE_TIMEOUT_SECONDS",
            3.0,
        ),
        context_timeout_seconds=_env_float(
            "CONTEXT_TIMEOUT_SECONDS",
            4.0,
        ),
        plan_timeout_seconds=_env_float(
            "PLAN_TIMEOUT_SECONDS",
            5.0,
        ),
        retrieval_timeout_seconds=_env_float(
            "RETRIEVAL_TIMEOUT_SECONDS",
            8.0,
        ),
        answer_timeout_seconds=_env_float(
            "ANSWER_TIMEOUT_SECONDS",
            12.0,
        ),
        verify_timeout_seconds=_env_float(
            "VERIFY_TIMEOUT_SECONDS",
            5.0,
        ),
        compile_history_max_turns=_env_int(
            "COMPILE_HISTORY_MAX_TURNS",
            4,
        ),
        compile_history_max_chars=_env_int(
            "COMPILE_HISTORY_MAX_CHARS",
            1_600,
        ),
        context_history_max_turns=_env_int(
            "CONTEXT_HISTORY_MAX_TURNS",
            4,
        ),
        context_history_max_chars=_env_int(
            "CONTEXT_HISTORY_MAX_CHARS",
            1_600,
        ),
        single_top_k=_env_int(
            "SINGLE_TOP_K",
            5,
        ),
        broad_top_k=_env_int(
            "BROAD_TOP_K",
            8,
        ),
        complex_top_k=_env_int(
            "COMPLEX_TOP_K",
            4,
        ),
        recovery_top_k=_env_int(
            "RECOVERY_TOP_K",
            4,
        ),
        retrieval_candidate_multiplier=_env_int(
            "RETRIEVAL_CANDIDATE_MULTIPLIER",
            3,
        ),
        broad_min_evidence_items=_env_int(
            "BROAD_MIN_EVIDENCE_ITEMS",
            2,
        ),
        answer_evidence_chars_per_item=_env_int(
            "ANSWER_EVIDENCE_CHARS_PER_ITEM",
            1_200,
        ),
        review_evidence_chars_per_item=_env_int(
            "REVIEW_EVIDENCE_CHARS_PER_ITEM",
            1_200,
        ),
        mcp_url=_env_str(
            "MCP_URL",
            "http://localhost:8002/mcp",
        ),
        mcp_tool_name=_env_str(
            "MCP_TOOL_NAME",
            "batch_semantic_search",
        ),

        retrieval_corpus_version=_env_str(
            "RETRIEVAL_CORPUS_VERSION",
            (
                "dmv_ca_2025_"
                "qwen3_06b_c450_o80_v1"
            ),
        ),
        
        mcp_request_timeout_seconds=_env_int(
            "MCP_REQUEST_TIMEOUT_SECONDS",
            15,
        ),
        mcp_sse_read_timeout_seconds=_env_float(
            "MCP_SSE_READ_TIMEOUT_SECONDS",
            300.0,
        ),
    )