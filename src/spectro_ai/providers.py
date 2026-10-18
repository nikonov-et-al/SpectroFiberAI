"""LLM provider adapters and grounded assessment orchestration."""

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spectro_ai.evidence import EvidenceRetriever, canonical_context_block
from spectro_ai.evidence_assessment import evidence_fingerprint, validate_assessment
from spectro_ai.guardrails import (
    abstention_report,
    missing_fields,
    prioritized_missing_fields,
    prioritized_questions,
)
from spectro_ai.models import (
    CUSTOMER_FACT_FIELDS,
    ApplicationSupportAssessment,
    AssessmentReport,
    CustomerUseCase,
    Evidence,
    EvidenceAssessment,
    EvidenceAssessmentDiagnostics,
    EvidenceClaim,
    EvidenceConflict,
    EvidenceCoverage,
    EvidenceObservation,
    EvidenceQueryPlan,
    EvidenceQuestion,
    FeasibilityPlan,
    QueryDiagnostics,
    RetrievalDiagnostic,
    RetrievedEvidence,
    TechniqueCandidate,
    ValidationDiagnostic,
)

_OLLAMA_LOCK = Lock()


class ProviderError(ValueError):
    """A provider is unknown, incomplete, or deliberately unavailable."""


class RetrievalDeadlineExceeded(ProviderError):
    """Optional multi-query retrieval exhausted its request deadline."""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ProviderError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class RagSettings:
    provider: str = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4:12b-it-q4_K_M"
    embedding_model: str = "qwen3-embedding:0.6b"
    context: int = 4096
    evidence_count: int = 8
    index: str = ".cache/rag/index.sqlite3"
    reranker: str = "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_revision: str = "main"
    reranker_candidates: int = 30
    multi_query: bool = False
    multi_query_max_queries: int = 4
    multi_query_timeout: int = 30
    multi_query_search_timeout: int = 30

    def __post_init__(self) -> None:
        if self.reranker not in {"none", "cross_encoder"}:
            raise ValueError("reranker must be none or cross_encoder")
        if not self.reranker_model.strip() or not self.reranker_revision.strip():
            raise ValueError("reranker model and revision must be nonempty")
        if self.reranker_candidates < 1:
            raise ValueError("reranker_candidates must be at least 1")
        if not 2 <= self.multi_query_max_queries <= 4:
            raise ValueError("multi_query_max_queries must be between 2 and 4")
        if self.multi_query_timeout < 1 or self.multi_query_search_timeout < 1:
            raise ValueError("multi-query timeouts must be positive")

    @classmethod
    def from_env(cls) -> RagSettings:
        try:
            context = int(os.getenv("SPECTRO_RAG_CONTEXT", "4096"))
            evidence_count = int(os.getenv("SPECTRO_RAG_EVIDENCE_COUNT", "8"))
        except ValueError:
            raise ProviderError(
                "SPECTRO_RAG_CONTEXT and SPECTRO_RAG_EVIDENCE_COUNT must be integers"
            ) from None
        if context < 1024:
            raise ProviderError("SPECTRO_RAG_CONTEXT must be at least 1024")
        if evidence_count < 1:
            raise ProviderError("SPECTRO_RAG_EVIDENCE_COUNT must be at least 1")
        try:
            reranker_candidates = int(os.getenv("SPECTRO_RAG_RERANK_CANDIDATES", "30"))
            multi_query_max_queries = int(os.getenv("SPECTRO_RAG_MULTI_QUERY_MAX_QUERIES", "4"))
            multi_query_timeout = int(os.getenv("SPECTRO_RAG_MULTI_QUERY_TIMEOUT", "30"))
            multi_query_search_timeout = int(
                os.getenv("SPECTRO_RAG_MULTI_QUERY_SEARCH_TIMEOUT", "30")
            )
        except ValueError:
            raise ProviderError("SPECTRO_RAG_RERANK_CANDIDATES must be an integer") from None
        reranker = os.getenv("SPECTRO_RAG_RERANKER", "none").strip().lower()
        if reranker not in {"none", "cross_encoder"}:
            raise ProviderError("SPECTRO_RAG_RERANKER must be none or cross_encoder")
        if reranker_candidates < 1:
            raise ProviderError("SPECTRO_RAG_RERANK_CANDIDATES must be at least 1")
        provider = os.getenv("SPECTRO_RAG_PROVIDER", "ollama").strip().lower()
        if provider not in {"ollama", "openai"}:
            raise ProviderError("SPECTRO_RAG_PROVIDER must be ollama or openai")
        return cls(
            provider=provider,
            ollama_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_url).rstrip("/"),
            model=os.getenv("OLLAMA_MODEL", cls.model),
            embedding_model=os.getenv("OLLAMA_EMBED_MODEL", cls.embedding_model),
            context=context,
            evidence_count=evidence_count,
            index=os.getenv("SPECTRO_RAG_INDEX", cls.index),
            reranker=reranker,
            reranker_model=os.getenv("SPECTRO_RAG_RERANKER_MODEL", cls.reranker_model),
            reranker_revision=os.getenv("SPECTRO_RAG_RERANKER_REVISION", cls.reranker_revision),
            reranker_candidates=reranker_candidates,
            multi_query=_env_bool("SPECTRO_RAG_MULTI_QUERY", False),
            multi_query_max_queries=multi_query_max_queries,
            multi_query_timeout=multi_query_timeout,
            multi_query_search_timeout=multi_query_search_timeout,
        )


_RERANKERS: dict[tuple[str, str], object] = {}
_RERANKER_LOCK = Lock()


def _cross_encoder(settings: RagSettings, *, local_files_only: bool = True):
    key = (settings.reranker_model, settings.reranker_revision)
    with _RERANKER_LOCK:
        if key not in _RERANKERS:
            try:
                from sentence_transformers import CrossEncoder

                _RERANKERS[key] = CrossEncoder(
                    settings.reranker_model,
                    revision=settings.reranker_revision,
                    device="cpu",
                    max_length=512,
                    local_files_only=local_files_only,
                    trust_remote_code=False,
                    cache_folder=str(Path(__file__).resolve().parents[2] / ".cache" / "reranker"),
                )
            except Exception as exc:
                raise ProviderError(f"Could not load local reranker: {exc}") from exc
        return _RERANKERS[key]


def _resolved_reranker_revision(settings: RagSettings, model: object) -> str:
    config = getattr(getattr(model, "model", None), "config", None)
    revision = getattr(config, "_commit_hash", None)
    if revision and revision != settings.reranker_revision:
        return revision
    reference = (
        Path(__file__).resolve().parents[2]
        / ".cache"
        / "reranker"
        / f"models--{settings.reranker_model.replace('/', '--')}"
        / "refs"
        / settings.reranker_revision
    )
    try:
        return reference.read_text(encoding="utf-8").strip() or settings.reranker_revision
    except OSError:
        return settings.reranker_revision


def rerank_scores(
    settings: RagSettings, query: str, passages: list[str]
) -> tuple[list[float], str | None]:
    """Score query/passage pairs on CPU; only setup_reranker permits downloads."""
    if not passages:
        return [], None
    try:
        model = _cross_encoder(settings)
        tokenizer = model.tokenizer
        max_length = 512
        query_ids = tokenizer.encode(query, add_special_tokens=False)[:128]
        query_text = tokenizer.decode(query_ids)
        while len(tokenizer.encode(query_text, add_special_tokens=False)) > 128:
            query_ids = query_ids[:-1]
            query_text = tokenizer.decode(query_ids)
        query_ids = tokenizer.encode(query_text, add_special_tokens=False)
        special = tokenizer.num_special_tokens_to_add(pair=True)
        budget = max_length - len(query_ids) - special
        pairs = []
        for passage in passages:
            passage_ids = tokenizer.encode(passage, add_special_tokens=False)[:budget]
            passage_text = tokenizer.decode(passage_ids)
            while len(tokenizer.encode(passage_text, add_special_tokens=False)) > budget:
                passage_ids = passage_ids[:-1]
                passage_text = tokenizer.decode(passage_ids)
            pairs.append((query_text, passage_text))
        values = model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        import math

        import numpy as np

        array = np.asarray(values)
        if array.ndim != 1 or len(array) != len(passages):
            raise ValueError("reranker must return one scalar score per passage")
        if any(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError("reranker returned boolean scores")
        scores = [float(value) for value in array]
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("reranker returned invalid score count or non-finite values")
        return scores, _resolved_reranker_revision(settings, model)
    except Exception as exc:
        raise ProviderError(f"Reranker inference failed: {exc}") from exc


def setup_reranker(settings: RagSettings | None = None) -> str:
    """Explicitly download and cache configured model; return resolved revision."""
    settings = settings or RagSettings.from_env()
    model = _cross_encoder(settings, local_files_only=False)
    return _resolved_reranker_revision(settings, model)


@dataclass(frozen=True)
class ExtractionSettings:
    provider: str = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4:12b-it-q4_K_M"
    openai_base_url: str = "https://api.openai.com/v1"
    context: int = 16384
    timeout: int = 240
    output_tokens: int = 4096

    @classmethod
    def from_env(cls) -> ExtractionSettings:
        try:
            provider = os.getenv("SPECTRO_AI_PROVIDER", "ollama").strip().lower()
            context = int(os.getenv("SPECTRO_AI_CONTEXT", "16384"))
            timeout = int(os.getenv("SPECTRO_AI_TIMEOUT", "240"))
            output_tokens = int(os.getenv("SPECTRO_AI_OUTPUT_TOKENS", "4096"))
        except ValueError:
            raise ProviderError(
                "SPECTRO_AI_CONTEXT, SPECTRO_AI_TIMEOUT and SPECTRO_AI_OUTPUT_TOKENS "
                "must be integers"
            ) from None
        if context < 4096 or timeout < 1 or output_tokens < 128:
            raise ProviderError(
                "SPECTRO_AI_CONTEXT must be at least 4096, timeout positive, "
                "and SPECTRO_AI_OUTPUT_TOKENS at least 128"
            )
        if provider not in {"ollama", "openai"}:
            raise ProviderError(
                f"Unsupported SPECTRO_AI_PROVIDER {provider!r}; use ollama or openai."
            )
        model = (
            os.getenv("OPENAI_MODEL", "")
            if provider == "openai"
            else os.getenv("OLLAMA_MODEL", cls.model)
        )
        if provider == "openai" and (not model or not os.getenv("OPENAI_API_KEY")):
            raise ProviderError("Set OPENAI_MODEL and OPENAI_API_KEY for OpenAI assessment.")
        return cls(
            provider=provider,
            ollama_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_url).rstrip("/"),
            model=model,
            openai_base_url=os.getenv("OPENAI_BASE_URL", cls.openai_base_url).rstrip("/"),
            context=context,
            timeout=timeout,
            output_tokens=output_tokens,
        )


def ollama_request(
    settings: RagSettings | ExtractionSettings,
    route: str,
    payload: dict | None = None,
    *,
    timeout: int = 120,
    lock_timeout: float | None = None,
) -> dict:
    deadline = None
    if lock_timeout is not None:
        from time import monotonic

        deadline = monotonic() + max(0.0, lock_timeout)
    try:
        request = Request(
            settings.ollama_url + route,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        if lock_timeout is None:
            _OLLAMA_LOCK.acquire()
        elif not _OLLAMA_LOCK.acquire(timeout=max(0.0, lock_timeout)):
            raise RetrievalDeadlineExceeded("Ollama request lock exceeded retrieval deadline")
        try:
            request_timeout = timeout
            if deadline is not None:
                request_timeout = min(timeout, max(0.001, deadline - monotonic()))
            with urlopen(request, timeout=request_timeout) as response:
                result = json.load(response)
        finally:
            _OLLAMA_LOCK.release()
        if deadline is not None and monotonic() >= deadline:
            raise RetrievalDeadlineExceeded("Ollama request exceeded retrieval deadline")
        if not isinstance(result, dict) or result.get("error"):
            raise ValueError
        return result
    except RetrievalDeadlineExceeded:
        raise
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        if lock_timeout is not None and (
            isinstance(exc, TimeoutError)
            or isinstance(exc, URLError)
            and isinstance(exc.reason, TimeoutError)
        ):
            raise RetrievalDeadlineExceeded("Ollama request exceeded retrieval deadline") from None
        raise ProviderError(
            "Ollama request failed. Check endpoint, installed model and context; "
            f"timeout is {timeout}s."
        ) from None


def _extraction_prompt() -> str:
    checkout = Path(__file__).resolve().parents[2] / "prompts" / "extraction.md"
    packaged = Path(__file__).parent / "data" / "prompts" / "extraction.md"
    path = checkout if checkout.is_file() else packaged
    try:
        prompt = path.read_text(encoding="utf-8")
        fields = ", ".join(CUSTOMER_FACT_FIELDS)
        prompt = prompt.replace("{{CUSTOMER_FACT_FIELDS}}", fields)
        return (
            prompt
            + "\nAuthoritative output JSON schema:\n"
            + json.dumps(_extraction_schema(), separators=(",", ":"))
        )
    except OSError:
        raise ProviderError("Extraction prompt is missing from the installation") from None


def _extraction_schema() -> dict:
    """Small transport schema; service-owned metadata stays out of model output."""
    descriptions = {
        "physical_state": (
            "Material state such as solid, liquid, gas, powder, slurry, or paste; "
            "not opacity or turbidity."
        ),
        "concentration_range": (
            "Customer-stated target concentration/range with its quantities and units; "
            "not a qualitative matrix descriptor."
        ),
        "turbidity": "Customer-stated turbidity or optical cloudiness, not physical state.",
    }
    properties = {
        field: {
            "type": ["string", "null"],
            **({"description": descriptions[field]} if field in descriptions else {}),
        }
        for field in CUSTOMER_FACT_FIELDS
    }
    properties.update(
        {
            "customer_preferences": {"type": "array", "items": {"type": "string"}},
            "strict_requirements": {"type": "array", "items": {"type": "string"}},
            "ambiguities": {"type": "array", "items": {"type": "string"}},
            "contradictions": {"type": "array", "items": {"type": "string"}},
            "field_quotes": {
                "type": "object",
                "properties": {
                    **{field: {"type": "string"} for field in CUSTOMER_FACT_FIELDS},
                    **{
                        f"{field}_{index}": {"type": "string"}
                        for field in (
                            "customer_preferences",
                            "strict_requirements",
                            "ambiguity",
                            "contradiction",
                        )
                        for index in range(4)
                    },
                },
                "additionalProperties": False,
            },
        }
    )
    return _strict_schema({"type": "object", "properties": properties})


def _query_plan_schema(use_case: CustomerUseCase | None = None) -> dict:
    schema = EvidenceQueryPlan.model_json_schema()
    if use_case is not None:
        available = [field for field in CUSTOMER_FACT_FIELDS if getattr(use_case, field)]
        schema["$defs"]["EvidenceQuestion"]["properties"]["anchor_fields"]["items"]["enum"] = (
            available
        )
    return _strict_schema(schema)


def validate_evidence_query_plan(
    plan: EvidenceQueryPlan,
    use_case: CustomerUseCase,
    *,
    max_questions: int = 3,
) -> tuple[list[EvidenceQuestion], list[str]]:
    """Keep only short questions anchored to populated, validated customer facts."""
    from spectro_ai.evidence import _tokens

    accepted: list[EvidenceQuestion] = []
    reasons: list[str] = []
    seen: list[set[str]] = []
    facts = " ".join(
        str(getattr(use_case, field)) for field in CUSTOMER_FACT_FIELDS if getattr(use_case, field)
    )
    facts += " " + " ".join(use_case.strict_requirements)
    numeric = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?\s*(?:%|[A-Za-zµμ°]+(?:/[A-Za-zµμ]+)?)?")
    unit = re.compile(
        r"(?<!\w)(?:°\s?[CFK]|%|ppm|ppb|ppt|mg|µg|μg|g|kg|ml|mL|µL|μL|L|mol|mmol|µmol|μmol|Pa|kPa|MPa|bar|Hz|kHz|MHz|s|ms|min|h|nm|µm|μm)(?!\w)",
        re.IGNORECASE,
    )
    fact_quantities = {value.casefold() for value in numeric.findall(facts)}
    fact_units = {value.casefold().replace(" ", "") for value in unit.findall(facts)}

    for item in plan.questions:
        fields = list(dict.fromkeys(item.anchor_fields))
        if not fields or any(
            field not in CUSTOMER_FACT_FIELDS or not getattr(use_case, field) for field in fields
        ):
            reasons.append("invalid_anchor")
            continue
        question = " ".join(item.question.split()).strip()
        if not question or len(question) > 400:
            reasons.append("invalid_length")
            continue
        question_tokens = set(_tokens(question))
        anchor_tokens = {
            token for field in fields for token in _tokens(str(getattr(use_case, field)))
        }
        if not question_tokens.intersection(anchor_tokens):
            reasons.append("anchor_not_present")
            continue
        if any(value.casefold() not in fact_quantities for value in numeric.findall(question)):
            reasons.append("unsupported_quantity")
            continue
        if any(
            value.casefold().replace(" ", "") not in fact_units for value in unit.findall(question)
        ):
            reasons.append("unsupported_unit")
            continue
        normalized = set(question_tokens)
        if any(len(normalized & prior) / max(len(normalized | prior), 1) >= 0.8 for prior in seen):
            reasons.append("duplicate_question")
            continue
        seen.append(normalized)
        accepted.append(EvidenceQuestion(question=question, anchor_fields=fields))
        if len(accepted) >= max_questions:
            break
    return accepted, reasons


def generate_evidence_query_plan(
    use_case: CustomerUseCase,
    settings: ExtractionSettings | None = None,
    *,
    timeout: int | None = None,
    usage: dict[str, int] | None = None,
) -> EvidenceQueryPlan:
    """Generate complementary search questions in one selected-provider call."""
    settings = settings or ExtractionSettings.from_env()
    timeout = min(settings.timeout, timeout or settings.timeout)
    checkout = Path(__file__).resolve().parents[2] / "prompts" / "retrieval_queries.md"
    packaged = Path(__file__).parent / "data" / "prompts" / "retrieval_queries.md"
    prompt_path = checkout if checkout.is_file() else packaged
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except OSError:
        raise ProviderError("Retrieval query prompt is missing from the installation") from None
    facts = {
        field: value
        for field in CUSTOMER_FACT_FIELDS
        if (value := getattr(use_case, field)) is not None
    }
    request_data = {
        "validated_customer_facts": facts,
        "field_provenance": use_case.field_provenance,
        "unknown_fields": missing_fields(use_case),
        "strict_requirements": use_case.strict_requirements,
        "available_anchor_fields": [
            field for field in CUSTOMER_FACT_FIELDS if getattr(use_case, field)
        ],
        "source_text_untrusted": use_case.source_text,
    }
    query_schema = _query_plan_schema(use_case)
    messages = [
        {
            "role": "system",
            "content": (
                prompt
                + "\nAllowed anchor fields: "
                + ", ".join(request_data["available_anchor_fields"])
                + "\nOutput schema:\n"
                + json.dumps(query_schema)
            ),
        },
        {"role": "user", "content": json.dumps(request_data, ensure_ascii=False)},
    ]
    if len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 1024 > settings.context:
        raise ProviderError("Retrieval query input exceeds configured assessment context budget")
    if usage is not None:
        usage["call_count"] = 1
    if settings.provider == "openai":
        from dataclasses import replace

        content = _openai_json(
            replace(settings, timeout=timeout, output_tokens=768),
            messages,
            query_schema,
            usage=usage,
            temperature=0,
        )
    else:
        payload = {
            "model": settings.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": query_schema,
            "options": {"num_ctx": settings.context, "num_predict": 768, "temperature": 0},
        }
        try:
            result = ollama_request(
                settings, "/api/chat", payload, timeout=timeout, lock_timeout=timeout
            )
        except TimeoutError as exc:
            raise RetrievalDeadlineExceeded(str(exc)) from None
        message = result.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if usage is not None:
            for field, key in (
                ("prompt_eval_count", "prompt_tokens"),
                ("eval_count", "completion_tokens"),
            ):
                if isinstance(result.get(field), int):
                    usage[key] = result[field]
            usage["call_count"] = 1
    if not isinstance(content, str):
        raise ProviderError("Provider returned no retrieval query JSON")
    return EvidenceQueryPlan.model_validate_json(content)


def _strict_schema(schema: dict) -> dict:
    """OpenAI strict JSON schema requires every object property to be required."""

    def visit(value: object) -> object:
        if isinstance(value, dict):
            result = {key: visit(item) for key, item in value.items()}
            if result.get("type") == "object":
                result["additionalProperties"] = False
                result["required"] = list(result.get("properties", {}))
            return result
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return visit(schema)  # type: ignore[return-value]


def _clean_extraction(
    payload: str, customer_text: str, provider: str = "ollama"
) -> CustomerUseCase:
    parsed = CustomerUseCase.model_validate_json(payload)
    values = parsed.model_dump()
    values["source_text"] = customer_text or None
    warnings: list[str] = []
    valid_quotes: dict[str, str] = {}
    values["request_id"] = "req_" + sha256(customer_text.encode("utf-8")).hexdigest()[:16]

    def supported(value: str, quote: str | None) -> bool:
        if not quote or quote not in customer_text:
            return False
        if re.search(r"(?<![\w])\d+(?:[.,]\d+)?", value):
            return value in quote
        return True

    def field_semantics_supported(field: str, value: str) -> bool:
        if field == "physical_state":
            states = (
                r"\b(solid|liquid|gas|powder|slurry|paste|gel|granular|pellet|suspension|melt)\b"
            )
            return re.search(states, value, flags=re.IGNORECASE) is not None
        if field == "concentration_range":
            return re.search(r"(?<![\w])\d+(?:[.,]\d+)?", value) is not None
        return True

    for field in CUSTOMER_FACT_FIELDS:
        value = getattr(parsed, field)
        if value is None or value == []:
            values[field] = value
            continue
        quote = parsed.field_quotes.get(field)
        if supported(value, quote) and field_semantics_supported(field, value):
            valid_quotes[field] = quote
            continue
        values[field] = [] if isinstance(value, list) else None
        warnings.append(
            f"Ignored {field}: supporting quote was missing, nonliteral, "
            "omitted a stated number, or did not support the field meaning."
        )

    for field in ("customer_preferences", "strict_requirements"):
        supported_items = []
        for index, item in enumerate(getattr(parsed, field)):
            quote = parsed.field_quotes.get(f"{field}_{index}")
            if supported(item, quote):
                supported_items.append(item)
                valid_quotes[f"{field}_{index}"] = quote
            else:
                warnings.append(f"Ignored unsupported {field[:-1]} item {index + 1}.")
        values[field] = supported_items

    for group, prefix in (("ambiguities", "ambiguity"), ("contradictions", "contradiction")):
        supported_items = []
        for index, item in enumerate(getattr(parsed, group)):
            quote = parsed.field_quotes.get(f"{prefix}_{index}")
            if quote and quote in customer_text:
                supported_items.append(item)
                valid_quotes[f"{prefix}_{index}"] = quote
            else:
                warnings.append(f"Ignored unsupported {group[:-1]} statement.")
        values[group] = supported_items

    values["field_quotes"] = valid_quotes
    values["field_provenance"] = {
        field: "customer quote"
        for field in (*CUSTOMER_FACT_FIELDS, "customer_preferences", "strict_requirements")
        if field in valid_quotes or any(key.startswith(f"{field}_") for key in valid_quotes)
    }
    values["unknown_or_missing_information"] = []
    values["extraction_mode"] = provider
    values["extraction_warnings"] = warnings
    cleaned = CustomerUseCase.model_validate(values)
    cleaned.unknown_or_missing_information = missing_fields(cleaned)
    return cleaned


def _openai_json(
    settings: ExtractionSettings,
    messages: list[dict],
    schema: dict,
    *,
    usage: dict[str, int] | None = None,
    temperature: float | None = None,
) -> str:
    if not settings.model or not os.getenv("OPENAI_API_KEY"):
        raise ProviderError("Set OPENAI_MODEL and OPENAI_API_KEY for OpenAI assessment.")
    try:
        from openai import OpenAI
    except ImportError:
        raise ProviderError("Install OpenAI support with `uv sync --extra llm`.") from None
    try:
        with OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=settings.openai_base_url,
            timeout=settings.timeout,
            max_retries=0,
        ) as client:
            options = {
                "model": settings.model,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "spectro_ai_output",
                        "strict": True,
                        "schema": _strict_schema(schema),
                    },
                },
                "max_completion_tokens": settings.output_tokens,
            }
            if temperature is not None:
                options["temperature"] = temperature
            response = client.chat.completions.create(**options)
            response_usage = getattr(response, "usage", None)
            if usage is not None and response_usage is not None:
                usage["prompt_tokens"] = response_usage.prompt_tokens
                usage["completion_tokens"] = response_usage.completion_tokens
                usage["call_count"] = 1
    except Exception:
        raise ProviderError(
            "OpenAI request failed. Check endpoint, model, credentials, timeout, "
            "and structured-output support."
        ) from None
    choice = response.choices[0] if response.choices else None
    if choice is None or choice.message.refusal:
        raise ProviderError("OpenAI refused the request; no output was produced.")
    if choice.finish_reason != "stop":
        raise ProviderError("OpenAI output was incomplete; increase output budget or reduce input.")
    return choice.message.content or ""


class OllamaUseCaseExtractor:
    """Structured local extraction; only quoted customer facts survive validation."""

    def __init__(self, settings: ExtractionSettings | None = None) -> None:
        self.settings = settings or ExtractionSettings.from_env()

    def extract(self, customer_text: str) -> CustomerUseCase:
        if self.settings.provider == "openai":
            return OpenAIUseCaseExtractor(self.settings).extract(customer_text)
        system = _extraction_prompt()
        reserved_context = self.settings.output_tokens + 256
        if len((system + customer_text).encode("utf-8")) + reserved_context > self.settings.context:
            raise ProviderError(
                "Extraction input exceeds the configured context budget. Increase "
                "SPECTRO_AI_CONTEXT or shorten the complete customer text."
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": customer_text},
        ]
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": _extraction_schema(),
            "options": {
                "num_ctx": self.settings.context,
                "num_predict": self.settings.output_tokens,
                "temperature": 0,
            },
        }
        for attempt in range(2):
            result = ollama_request(
                self.settings,
                "/api/chat",
                payload,
                timeout=self.settings.timeout,
            )
            if result.get("done_reason") == "length":
                raise ProviderError(
                    "Extraction exceeded output budget. Increase SPECTRO_AI_CONTEXT "
                    "or shorten input."
                )
            message = result.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                if attempt == 0:
                    continue
                raise ProviderError("Ollama returned invalid extraction output")
            try:
                return _clean_extraction(message["content"], customer_text, "ollama")
            except ValueError, TypeError:
                if attempt == 1:
                    raise ProviderError("Ollama returned invalid customer-use-case JSON") from None
        raise ProviderError("Ollama extraction failed after one structural retry")


class OpenAIUseCaseExtractor:
    """OpenAI structured extraction; output passes through shared literal-quote checks."""

    def __init__(self, settings: ExtractionSettings | None = None) -> None:
        self.settings = settings or ExtractionSettings.from_env()

    def extract(self, customer_text: str) -> CustomerUseCase:
        settings = self.settings
        system = _extraction_prompt()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": customer_text},
        ]
        for attempt in range(2):
            if (
                len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
                + settings.output_tokens
                + 256
                > settings.context
            ):
                kind = "retry" if attempt else "input"
                raise ProviderError(
                    f"Extraction {kind} exceeds the configured context budget. "
                    "Increase SPECTRO_AI_CONTEXT or shorten the complete customer text."
                )
            raw = _openai_json(settings, messages, _extraction_schema())
            try:
                return _clean_extraction(raw, customer_text, "openai")
            except ValidationError, ValueError, TypeError:
                if attempt:
                    raise ProviderError("OpenAI returned invalid customer-use-case JSON.") from None
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Return one JSON object matching schema."},
                ]
        raise ProviderError("OpenAI extraction failed after one structural retry.")


def embedding_identity(
    settings: RagSettings, *, timeout: int = 120, lock_timeout: float | None = None
) -> str:
    result = ollama_request(settings, "/api/tags", timeout=timeout, lock_timeout=lock_timeout)
    models = result.get("models")
    if not isinstance(models, list) or any(not isinstance(m, dict) for m in models):
        raise ProviderError("Ollama returned invalid model metadata")
    for model in models:
        if model.get("name") == settings.embedding_model and model.get("digest"):
            return str(model["digest"])
    raise ProviderError("Embedding model unavailable. Run ollama pull " + settings.embedding_model)


def embed_texts(
    settings: RagSettings,
    texts: list[str],
    *,
    timeout: int = 120,
    lock_timeout: float | None = None,
) -> list[list[float]]:
    result = ollama_request(
        settings,
        "/api/embed",
        {
            "model": settings.embedding_model,
            "input": texts,
            "truncate": False,
            "keep_alive": 0,
        },
        timeout=timeout,
        lock_timeout=lock_timeout,
    )
    vectors = result.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ProviderError("Ollama returned an invalid embedding count")
    return vectors


class _CandidateDraft(TechniqueCandidate):
    spectroscopy_method: str = Field(
        description="Name each specific method; combined methods list every component."
    )
    supporting_evidence: list[Evidence] = Field(default_factory=list, max_length=0)
    evidence_claims: list[EvidenceClaim] = Field(
        default_factory=list,
        description=(
            "Optional source quotations; each must be an exact excerpt substring "
            "and carry its supplied ID."
        ),
    )


class _TechnicalDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["provisional", "insufficient_information", "outside_supported_scope"] = Field(
        description="Provisional only for a source-supported study; otherwise abstain."
    )
    suggested_technical_questions: list[str] = Field(
        default_factory=list,
        description="Prioritize questions; state which study decision each answer changes.",
    )
    candidate_techniques: list[_CandidateDraft] = Field(
        default_factory=list,
        description="Describe source applications only; do not rank or claim customer suitability.",
    )
    recommendation_rationale: str | None = Field(
        default=None,
        description=(
            "Attribute source claims; do not predict customer outcomes or a winning method."
        ),
    )
    proposed_feasibility_experiment: FeasibilityPlan = Field(
        default_factory=FeasibilityPlan,
        description=(
            "Use independent references, held-out samples, customer criteria, and expert review."
        ),
    )
    assumptions_and_uncertainties: list[str] = Field(
        default_factory=list,
        description="State unknowns; do not infer facts.",
    )


class TechnicalAssessor(Protocol):
    def assess(
        self,
        use_case: CustomerUseCase,
        evidence: list[RetrievedEvidence],
        *,
        evidence_assessment: EvidenceAssessment,
    ) -> AssessmentReport: ...


class EvidenceAssessmentDraft(BaseModel):
    """Provider output; provenance and assessment identity are host supplied."""

    model_config = ConfigDict(extra="forbid")

    observations: list[EvidenceObservation] = Field(default_factory=list)
    support_assessments: list[ApplicationSupportAssessment] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    coverage: list[EvidenceCoverage] = Field(default_factory=list)


class EvidenceAssessor(Protocol):
    def __call__(
        self,
        use_case: CustomerUseCase,
        evidence: list[RetrievedEvidence],
        *,
        allow_retry: bool = True,
    ) -> EvidenceAssessment: ...


class UseCaseExtractor(Protocol):
    def extract(self, customer_text: str) -> CustomerUseCase: ...


def _technical_schema() -> dict:
    """Compact transport schema; Pydantic still validates omitted empty source fields."""
    schema = _TechnicalDraft.model_json_schema()

    def strip_metadata(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: strip_metadata(item)
                for key, item in value.items()
                if key not in {"title", "default"}
            }
        if isinstance(value, list):
            return [strip_metadata(item) for item in value]
        return value

    compact = strip_metadata(schema)
    assert isinstance(compact, dict)
    definitions = compact.get("$defs", {})
    candidate_schema = definitions.get("_CandidateDraft", {})
    candidate_schema.get("properties", {}).pop("supporting_evidence", None)
    definitions.pop("Evidence", None)
    return _strict_schema(compact)


def _assessment_prompt() -> str:
    checkout = Path(__file__).resolve().parents[2] / "prompts" / "assessment.md"
    packaged = Path(__file__).parent / "data" / "prompts" / "assessment.md"
    path = checkout if checkout.is_file() else packaged
    try:
        prompt = path.read_text(encoding="utf-8")
    except OSError:
        raise ProviderError("Assessment prompt is missing from the installation") from None
    return prompt + "\nOutput JSON schema is supplied separately in the structured response format."


class OllamaEvidenceAssessor:
    """Assess source observations before synthesis using configured provider."""

    VERSION = "application-evidence-assessment-v3"

    def __init__(self, settings: ExtractionSettings | None = None) -> None:
        self.settings = settings or ExtractionSettings.from_env()

    @staticmethod
    def _schema(evidence: list[RetrievedEvidence] | None = None) -> dict:
        def compact(value):
            if isinstance(value, dict):
                return {
                    key: compact(item)
                    for key, item in value.items()
                    if key not in {"title", "default"}
                }
            if isinstance(value, list):
                return [compact(item) for item in value]
            return value

        schema = _strict_schema(compact(EvidenceAssessmentDraft.model_json_schema()))
        if evidence is not None:
            definitions = schema["$defs"]
            definitions["CustomerFactField"] = {
                "type": "string",
                "enum": list(CUSTOMER_FACT_FIELDS),
            }
            evidence_ids = sorted(item.evidence_id for item in evidence)
            block_ids = sorted(block.block_id for item in evidence for block in item.context_blocks)
            definitions["EvidenceReference"]["properties"]["evidence_id"]["enum"] = evidence_ids
            context_schema = definitions["EvidenceReference"]["properties"]["context_block_id"]
            context_schema["anyOf"] = (
                [{"type": "string", "enum": block_ids}, {"type": "null"}]
                if block_ids
                else [{"type": "null"}]
            )
            definitions["EvidenceCoverage"]["properties"]["evidence_id"]["enum"] = evidence_ids
            for name in ("EvidenceConditionComparison", "ExploratoryBridge"):
                definitions[name]["properties"]["customer_fields"]["items"] = {
                    "$ref": "#/$defs/CustomerFactField"
                }
            definitions["EvidenceObservation"]["properties"].pop("context_availability", None)
            definitions["EvidenceObservation"]["required"].remove("context_availability")
        return schema

    @staticmethod
    def _messages(use_case: CustomerUseCase, evidence: list[RetrievedEvidence]) -> list[dict]:
        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "evidence_assessment.md"
        if not prompt_path.is_file():
            prompt_path = Path(__file__).parent / "data" / "prompts" / "evidence_assessment.md"
        try:
            prompt = prompt_path.read_text(encoding="utf-8")
        except OSError:
            raise ProviderError(
                "Evidence assessment prompt is missing from the installation"
            ) from None
        customer = {
            "validated_customer_facts": {
                field: value
                for field in (*CUSTOMER_FACT_FIELDS, "customer_preferences", "strict_requirements")
                if (value := getattr(use_case, field)) not in (None, [])
            },
            "unknown_fields": missing_fields(use_case),
            "contradictions": list(use_case.contradictions),
        }
        sources = []
        for item in evidence:
            sources.append(
                {
                    "evidence_id": item.evidence_id,
                    "source_id": item.source_id,
                    "document_version": item.document_version,
                    "page_or_section": item.page_or_section,
                    "source_locations": [
                        loc.model_dump(mode="json") for loc in item.source_locations
                    ],
                    "excerpt": item.excerpt,
                    "review_status": item.review_status,
                    "source_scope": item.source_scope,
                    "context_availability": "attached_context"
                    if item.context_blocks
                    else "child_only",
                    "context_omissions": item.context_omissions,
                    "context_blocks": [
                        block.model_dump(mode="json") for block in item.context_blocks
                    ],
                }
            )
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"customer": customer, "evidence_bundles": sources},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]

    def fit_evidence(
        self, use_case: CustomerUseCase, evidence: list[RetrievedEvidence]
    ) -> tuple[list[RetrievedEvidence], list[str]]:
        """Fit complete evidence bundles; keep each excerpt with all attached context."""
        retained = list(evidence)
        dropped: list[str] = []
        schema_bytes = len(
            json.dumps(self._schema(evidence), separators=(",", ":")).encode("utf-8")
        )
        while retained:
            messages = self._messages(use_case, retained)
            request_bytes = (
                len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                + schema_bytes
            )
            if request_bytes + self.settings.output_tokens + 256 <= self.settings.context:
                return retained, dropped
            dropped.append(retained.pop().evidence_id)
        return [], dropped

    def __call__(
        self,
        use_case: CustomerUseCase,
        evidence: list[RetrievedEvidence],
        *,
        allow_retry: bool = True,
    ) -> EvidenceAssessment:
        from time import perf_counter

        messages = self._messages(use_case, evidence)
        schema = self._schema(evidence)
        serialized_bytes = len(
            json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ) + len(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
        if serialized_bytes + self.settings.output_tokens + 256 > self.settings.context:
            raise ProviderError("Evidence assessment exceeds configured context budget")
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {
                "num_ctx": self.settings.context,
                "num_predict": self.settings.output_tokens,
                "temperature": 0,
            },
        }
        started = perf_counter()
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "call_count": 0}
        for attempt in range(2 if allow_retry else 1):
            try:
                if attempt:
                    messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": "Return one valid JSON object matching the schema.",
                        },
                    ]
                    payload["messages"] = messages
                    request_bytes = len(
                        json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode(
                            "utf-8"
                        )
                    ) + len(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
                    if request_bytes + self.settings.output_tokens + 256 > self.settings.context:
                        raise ProviderError(
                            "Evidence assessment structural retry exceeds context budget"
                        )
                if self.settings.provider == "openai":
                    usage = {}
                    content = _openai_json(
                        self.settings, messages, schema, usage=usage, temperature=0
                    )
                    for key in usage_total:
                        usage_total[key] += usage.get(key, 1 if key == "call_count" else 0)
                else:
                    result = ollama_request(
                        self.settings, "/api/chat", payload, timeout=self.settings.timeout
                    )
                    usage_total["call_count"] += 1
                    usage_total["prompt_tokens"] += result.get("prompt_eval_count", 0)
                    usage_total["completion_tokens"] += result.get("eval_count", 0)
                    if result.get("done_reason") == "length":
                        raise ProviderError("Evidence assessment exceeded output-token limit")
                    message = result.get("message")
                    content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str):
                    raise ValueError("missing JSON response")
                draft = EvidenceAssessmentDraft.model_validate_json(content)
                raw = EvidenceAssessment(
                    observations=draft.observations,
                    support_assessments=draft.support_assessments,
                    conflicts=draft.conflicts,
                    coverage=draft.coverage,
                    input_fingerprint="provider-output",
                    assessor_version=self.VERSION,
                    diagnostics=EvidenceAssessmentDiagnostics(
                        call_count=usage_total["call_count"],
                        elapsed_ms=(perf_counter() - started) * 1000,
                        serialized_bytes=serialized_bytes,
                        prompt_tokens=usage_total["prompt_tokens"] or None,
                        completion_tokens=usage_total["completion_tokens"] or None,
                    ),
                )
                fingerprint = evidence_fingerprint(use_case, evidence, self.VERSION)
                return validate_assessment(
                    raw,
                    use_case,
                    {item.evidence_id: item for item in evidence},
                    fingerprint,
                    self.VERSION,
                    f"{self.settings.provider}:{self.settings.model}",
                )
            except ProviderError:
                raise
            except ValidationError, ValueError, TypeError:
                if attempt + 1 == (2 if allow_retry else 1):
                    raise ProviderError(
                        "Evidence assessment returned invalid structured output"
                    ) from None
        raise ProviderError("Evidence assessment failed")


class OllamaTechnicalAssessor:
    """Generate a provisional, evidence-ID-bound technical assessment."""

    def __init__(self, settings: ExtractionSettings | None = None) -> None:
        self.settings = settings or ExtractionSettings.from_env()

    def fit_evidence(
        self, use_case: CustomerUseCase, evidence: list[RetrievedEvidence]
    ) -> tuple[list[RetrievedEvidence], list[str]]:
        """Fit complete excerpt/context bundles, preserving attached qualifications."""
        retained = list(evidence)
        dropped: list[str] = []
        while retained:
            messages = self._messages(use_case, retained)
            if (
                len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
                + (self.settings.output_tokens + 256)
                <= self.settings.context
            ):
                return retained, dropped
            dropped.append(retained.pop().evidence_id)
        return [], dropped

    @staticmethod
    def _messages(
        use_case: CustomerUseCase,
        evidence: list[RetrievedEvidence],
        *,
        evidence_assessment: EvidenceAssessment | None = None,
    ) -> list[dict[str, str]]:
        customer = {
            "source_text_untrusted": use_case.source_text,
            "validated_customer_facts": {
                field: value
                for field in (*CUSTOMER_FACT_FIELDS, "customer_preferences", "strict_requirements")
                if (value := getattr(use_case, field)) not in (None, [])
            },
            "unknown_fields": missing_fields(use_case),
        }
        sources = []
        for item in evidence:
            source = {
                "evidence_id": item.evidence_id,
                "source_id": item.source_id,
                "document_version": item.document_version,
                "page_or_section": item.page_or_section,
                "source_locations": [loc.model_dump(mode="json") for loc in item.source_locations],
                "excerpt": item.excerpt,
                "review_status": item.review_status,
                "source_scope": item.source_scope,
                "context_availability": "attached_context" if item.context_blocks else "child_only",
                "context_omissions": item.context_omissions,
            }
            if item.context_blocks:
                source["context_relation"] = "same source context; not independent evidence"
                source["context_blocks"] = [
                    block.model_dump(mode="json") for block in item.context_blocks
                ]
            sources.append(source)
        if evidence_assessment is not None:
            sources = [
                {
                    "evidence_id": item.evidence_id,
                    "source_id": item.source_id,
                    "document_version": item.document_version,
                    "page_or_section": item.page_or_section,
                    "source_locations": [
                        loc.model_dump(mode="json") for loc in item.source_locations
                    ],
                    "review_status": item.review_status,
                    "source_scope": item.source_scope,
                    "context_blocks": [
                        {
                            "block_id": block.block_id,
                            "source_locations": [
                                loc.model_dump(mode="json") for loc in block.source_locations
                            ],
                            "review_status": block.review_status,
                        }
                        for block in item.context_blocks
                    ],
                }
                for item in evidence
            ]
            user = {
                "customer_case_untrusted_data": customer,
                "verified_source_metadata": sources,
                "assessed_evidence_untrusted_data": evidence_assessment.model_dump(mode="json"),
            }
        else:
            user = {
                "customer_case_untrusted_data": customer,
                "retrieved_sources_untrusted_data": sources,
            }
        return [
            {"role": "system", "content": _assessment_prompt()},
            {
                "role": "user",
                "content": json.dumps(user, ensure_ascii=False, separators=(",", ":")),
            },
        ]

    def assess(
        self,
        use_case: CustomerUseCase,
        evidence: list[RetrievedEvidence],
        *,
        evidence_assessment: EvidenceAssessment,
    ) -> AssessmentReport:
        messages = self._messages(use_case, evidence, evidence_assessment=evidence_assessment)
        request_bytes = len(
            json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        schema_bytes = len(json.dumps(_technical_schema(), separators=(",", ":")).encode("utf-8"))
        if request_bytes + schema_bytes + self.settings.output_tokens + 256 > self.settings.context:
            raise ProviderError(
                "Assessment synthesis input exceeds configured context budget; "
                "preserve qualifications by reducing complete evidence bundles."
            )
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": _technical_schema(),
            "options": {
                "num_ctx": self.settings.context,
                "num_predict": self.settings.output_tokens,
                "temperature": 0,
            },
        }
        for attempt in range(2):
            if attempt:
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": "Return one valid JSON object matching the supplied schema.",
                    },
                ]
                payload["messages"] = messages
                if (
                    len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
                    + (self.settings.output_tokens + 256)
                    > self.settings.context
                ):
                    raise ProviderError(
                        "Assessment retry exceeds the configured context budget. Increase "
                        "SPECTRO_AI_CONTEXT or reduce retrieved evidence."
                    )
            if self.settings.provider == "openai":
                content = _openai_json(self.settings, messages, _technical_schema())
            else:
                result = ollama_request(
                    self.settings, "/api/chat", payload, timeout=self.settings.timeout
                )
                if result.get("done_reason") == "length":
                    raise ProviderError(
                        "Assessment exceeded SPECTRO_AI_OUTPUT_TOKENS. Increase that setting "
                        "or request a shorter assessment."
                    )
                message = result.get("message")
                content = message.get("content") if isinstance(message, dict) else None
            try:
                if not isinstance(content, str):
                    raise ValueError("missing JSON response")
                parsed = _TechnicalDraft.model_validate_json(content)
                return AssessmentReport(
                    extracted_requirements=use_case,
                    missing_information=missing_fields(use_case),
                    suggested_technical_questions=(
                        parsed.suggested_technical_questions or prioritized_questions(use_case)
                    ),
                    candidate_techniques=[
                        TechniqueCandidate(**candidate.model_dump())
                        for candidate in parsed.candidate_techniques
                    ],
                    recommendation_rationale=parsed.recommendation_rationale,
                    proposed_feasibility_experiment=parsed.proposed_feasibility_experiment,
                    assumptions_and_uncertainties=parsed.assumptions_and_uncertainties,
                    status=parsed.status,
                    assessment_method=(
                        "openai_grounded"
                        if self.settings.provider == "openai"
                        else "ollama_grounded"
                    ),
                    assessment_provider=self.settings.provider,
                    report_version=f"grounded-{self.settings.provider}-v2",
                )
            except ValidationError, ValueError, TypeError:
                if attempt:
                    raise ProviderError("Ollama returned invalid assessment JSON") from None
        raise ProviderError("Ollama assessment failed after one structural retry")


def _verified_retrieval(
    retriever: EvidenceRetriever, evidence: list[RetrievedEvidence]
) -> tuple[list[RetrievedEvidence], str]:
    index = retriever.index_metadata()
    by_id: dict[str, tuple[str, dict]] = {}
    for chunk in index["chunks"]:
        for provenance in chunk["provenance"]:
            by_id[provenance["evidence_id"]] = (chunk["text"], provenance)
    sections = {item["section_id"]: item for item in index.get("sections", [])}
    verified = []
    seen = set()
    seen_context_blocks: set[str] = set()
    for item in evidence:
        actual = by_id.get(item.evidence_id)
        if not actual or item.evidence_id in seen:
            raise ProviderError(
                "Retrieved evidence ID is absent or duplicated in the validated index"
            )
        text, provenance = actual
        expected = (
            provenance["source_id"],
            provenance["document_version"],
            provenance["page_or_section"],
            provenance["review_status"],
            provenance["source_scope"],
            text,
        )
        supplied = (
            item.source_id,
            item.document_version,
            item.page_or_section,
            item.review_status,
            item.source_scope,
            item.excerpt,
        )
        if expected != supplied:
            raise ProviderError("Retrieved evidence metadata does not match the validated index")
        provenance_locations = provenance.get("source_locations", [])
        supplied_locations = [
            location.model_dump(mode="json") for location in item.source_locations
        ]
        if supplied_locations and supplied_locations != provenance_locations:
            raise ProviderError("Retrieved evidence locations do not match the validated index")
        child_section = provenance.get("section_id")
        child_document = provenance.get("document_id") or (sections.get(child_section) or {}).get(
            "document_id"
        )
        for context in item.context_blocks:
            try:
                canonical = canonical_context_block(index, context.block_id)
            except ValueError as exc:
                raise ProviderError(
                    "Retrieved context block is absent from the validated index"
                ) from exc
            if (
                context.block_id in seen_context_blocks
                or context.document_id != child_document
                or context.section_id != child_section
                or context.source_id != provenance["source_id"]
                or context.document_version != provenance["document_version"]
                or context.model_dump(mode="json") != canonical.model_dump(mode="json")
            ):
                raise ProviderError(
                    "Retrieved context block does not match the validated child source section"
                )
            seen_context_blocks.add(context.block_id)
        seen.add(item.evidence_id)
        if "technical" in item.source_scope.casefold() and not any(
            marker in item.source_scope.casefold()
            for marker in ("business_context", "pointer_only", "secondary_transcription")
        ):
            verified.append(item)
    return verified, _evidence_set_version(verified, str(index["manifest_version"]))


def _evidence_set_version(evidence: list[RetrievedEvidence], manifest_version: str) -> str:
    version_material = {
        "manifest": manifest_version,
        "evidence": [
            {
                "id": item.evidence_id,
                "source": item.source_id,
                "document_version": item.document_version,
                "page": item.page_or_section,
                "excerpt": sha256(item.excerpt.encode("utf-8")).hexdigest(),
                "review_status": item.review_status,
                "scope": item.source_scope,
                **(
                    {"context": [block.model_dump(mode="json") for block in item.context_blocks]}
                    if item.context_blocks
                    else {}
                ),
            }
            for item in sorted(evidence, key=lambda row: row.evidence_id)
        ],
    }
    return sha256(json.dumps(version_material, sort_keys=True).encode()).hexdigest()


def _budget_drop_warning(dropped: list[str]) -> str:
    contexts = [item for item in dropped if item.startswith("context:")]
    children = [item for item in dropped if not item.startswith("context:")]
    parts = []
    if contexts:
        parts.append(f"{len(contexts)} whole optional context blocks ({', '.join(contexts)})")
    if children:
        parts.append(f"{len(children)} lower-ranked whole excerpts ({', '.join(children)})")
    return "Dropped " + " and ".join(parts) + " to preserve the complete request context budget."


def _supports_candidate(
    use_case: CustomerUseCase, candidate: TechniqueCandidate, cited: list[RetrievedEvidence]
) -> bool:
    if not cited or not use_case.target_analyte_or_property or not use_case.sample_matrix:
        return False
    generic = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "concentration",
        "composition",
        "property",
        "content",
        "sample",
        "changes",
        "monitoring",
        "measurement",
        "analysis",
        "signal",
    }
    target_terms = {
        term.casefold()
        for term in re.findall(r"[a-zA-Z]{3,}", use_case.target_analyte_or_property)
        if term.casefold() not in generic
    }
    matrix_terms = {
        term.casefold()
        for term in re.findall(r"[a-zA-Z]{3,}", use_case.sample_matrix)
        if term.casefold() not in generic
    }
    method = candidate.spectroscopy_method.casefold()
    components: list[list[str]] = []
    if re.search(r"\bnir\b|near[ -]infrared", method):
        components.append(["nir", "near infrared", "near-infrared"])
    if "raman" in method:
        components.append(["raman"])
    if "mir" in method or "ftir" in method or "mid infrared" in method:
        components.append(["mir", "ftir", "mid infrared", "mid-infrared"])
    if "fluorescence" in method:
        components.append(["fluorescence"])
    if not components:
        components.append([method])

    def has_phrase(text: str, phrase: str) -> bool:
        return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None

    excerpt = " ".join(
        text
        for item in cited
        for text in (item.excerpt, *(block.text for block in item.context_blocks))
    ).casefold()
    has_all_method_components = all(
        any(has_phrase(excerpt, alias) for alias in component) for component in components
    )
    has_case_match = any(has_phrase(excerpt, term) for term in target_terms | matrix_terms)
    return has_all_method_components and has_case_match


def _candidate_method_named(candidate: TechniqueCandidate, text: str) -> bool:
    method = candidate.spectroscopy_method.casefold()
    aliases = []
    if re.search(r"\bnir\b|near[ -]infrared", method):
        aliases.append(("nir", "near infrared", "near-infrared"))
    if "raman" in method:
        aliases.append(("raman",))
    if "mir" in method or "ftir" in method or "mid infrared" in method:
        aliases.append(("mir", "ftir", "mid infrared", "mid-infrared"))
    if "fluorescence" in method:
        aliases.append(("fluorescence",))
    if not aliases:
        aliases.append((method,))
    return all(
        any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.I) for alias in group)
        for group in aliases
    )


def _abstention_report(
    use_case: CustomerUseCase,
    reason: str,
    *,
    status: Literal["insufficient_information", "outside_supported_scope"],
    method: Literal["evidence_abstention"] = "evidence_abstention",
    provider: str | None = None,
    evidence: list[RetrievedEvidence] | None = None,
    evidence_version: str | None = None,
    knowledge_index_revision: str | None = None,
    retrieval_diagnostics: list[dict] | None = None,
    query_diagnostics: QueryDiagnostics | None = None,
) -> AssessmentReport:
    report = abstention_report(
        use_case,
        reason,
        status=status,
        provider=provider,
        evidence=evidence,
        evidence_version=evidence_version,
    )
    report.assessment_method = method
    report.knowledge_index_revision = knowledge_index_revision
    report.retrieval_diagnostics = [
        RetrievalDiagnostic.model_validate(item) for item in (retrieval_diagnostics or [])
    ]
    report.query_diagnostics = query_diagnostics
    return report


def _apply_feasibility_safeguards(
    use_case: CustomerUseCase,
    plan: FeasibilityPlan,
    candidates: list[TechniqueCandidate] | None = None,
) -> tuple[FeasibilityPlan, list[str]]:
    """Keep qualitative LLM planning while replacing risky thresholds and configuration."""
    safe = plan.model_copy(deep=True)
    safe.objective = "Explore whether a source-supported candidate merits a controlled study."
    safe.hypothesis = (
        "Test whether and which retained candidate distinguishes the target on representative, "
        "independent held-out samples; no result is assumed."
    )
    stated_criteria = [
        value
        for value in (use_case.required_accuracy, use_case.detection_limit)
        if value and value.strip()
    ] + [value for value in use_case.strict_requirements if value.strip()]
    safe.success_criteria = (
        stated_criteria
        if stated_criteria
        else ["Customer-defined acceptance criteria are TBD and must be agreed before testing."]
    )
    safe.evaluation_metrics = [
        "Compare held-out results with independent references using customer-agreed measures."
    ]
    safe.sample_preparation = None
    unsafe_configuration = bool(safe.proposed_configuration)
    safe.proposed_configuration = None
    if safe.proposed_measurement_technique and re.search(
        r"\d", safe.proposed_measurement_technique
    ):
        safe.proposed_measurement_technique = None
    stated_values = [
        getattr(use_case, field) for field in CUSTOMER_FACT_FIELDS
    ] + use_case.strict_requirements
    numeric_spec = re.compile(
        r"(?<![\w])[-+]?\d+(?:[.,]\d+)?\s*(?:%|(?:°?[A-Za-zµμ]+(?:/[A-Za-zµμ]+)?))?"
    )

    def is_customer_quantity(quantity: str) -> bool:
        return any(
            re.search(rf"(?<![\w]){re.escape(quantity.strip())}(?![\w])", value)
            for value in stated_values
            if isinstance(value, str)
        )

    rejected_numeric_steps = 0
    filtered_steps = []
    for step in safe.experimental_steps:
        quantities = numeric_spec.findall(step)
        if any(not is_customer_quantity(quantity) for quantity in quantities):
            rejected_numeric_steps += 1
            continue
        filtered_steps.append(step)
    safe.experimental_steps = filtered_steps
    candidates = candidates or []
    methods = [candidate.spectroscopy_method for candidate in candidates]
    if candidates:
        components = [
            method
            for method in methods
            if not re.search(r"\b(?:and|plus|combined|fusion|multimodal)\b|\+", method, re.I)
        ]
        safe.proposed_measurement_technique = "; ".join(components or methods)
        safe.experimental_steps = [
            "Collect representative samples spanning customer-agreed batches, runs, matrix "
            "variation, and relevant operating conditions.",
            "Measure each candidate using the same samples; compare combined methods only "
            "as optional additions to component methods.",
            "Agree and qualify an independent reference method, including uncertainty "
            "and alignment to the same samples or process time points.",
            "Split training and held-out evaluation by independent batch or process run "
            "before any preprocessing or model fitting.",
            "Evaluate held-out results against customer-agreed measures and acceptance "
            "criteria; agree criteria before testing.",
        ]
    else:
        if not any("independent reference" in step.casefold() for step in safe.experimental_steps):
            safe.experimental_steps.append(
                "Agree independent reference measurements before testing."
            )
    safe.reference_measurement = use_case.reference_measurement or (
        "Independent reference measurement to be agreed before testing."
    )
    safe.success_actions = [
        "Review study results with customer and domain experts before deciding next steps."
    ]
    safe.failure_actions = [
        "Investigate study limitations, then revise or stop the feasibility study."
    ]
    caveat = (
        "Source demonstrations do not establish customer or instrument performance; "
        "hardware selection and production integration require separate expert review."
    )
    safe.expected_limitations = [caveat]
    if candidates and any(
        re.search(r"\b(?:and|plus|combined|fusion|multimodal)\b|\+", method, re.I)
        for method in methods
    ):
        safe.expected_limitations.append(
            "Combined measurement is an optional comparison; its added value is unknown."
        )
    warnings = [
        "Deterministic safeguards replaced acceptance thresholds and constrained study planning; "
        "they are not scientific validation."
    ]
    if not stated_criteria:
        warnings.append(
            "No customer-defined acceptance criteria were supplied; results remain "
            "exploratory until criteria are agreed."
        )
    if rejected_numeric_steps:
        warnings.append(
            f"Removed {rejected_numeric_steps} experimental step(s) containing numeric quantities "
            "that did not exactly match a structured customer fact."
        )
    if unsafe_configuration:
        warnings.append(
            "Removed unverified proposed configuration; compatibility remains unchecked."
        )
    return safe, warnings


def assess_use_case(
    use_case: CustomerUseCase,
    *,
    assessor: TechnicalAssessor | None = None,
    evidence_assessor: EvidenceAssessor | None = None,
    retriever: EvidenceRetriever | None = None,
    query_generator: Callable[[CustomerUseCase], EvidenceQueryPlan] | None = None,
) -> AssessmentReport:
    """Assess an existing extracted/corrected case through current provider policy."""
    use_case = use_case.model_copy(deep=True)
    use_case.unknown_or_missing_information = missing_fields(use_case)
    provider = os.getenv("SPECTRO_AI_PROVIDER", "ollama").strip().lower()
    if provider == "baseline":
        raise ProviderError("The baseline provider was removed. Use ollama or openai.")
    if provider != "ollama":
        if provider != "openai":
            raise ProviderError(f"Unsupported provider {provider!r}. Use ollama or openai.")
    if not (
        use_case.measurement_objective
        and use_case.target_analyte_or_property
        and use_case.sample_matrix
    ):
        return _abstention_report(
            use_case,
            "Objective, target, or sample matrix is missing; candidate selection "
            "is insufficiently specified.",
            status="insufficient_information",
            provider=provider,
        )
    try:
        source = retriever or EvidenceRetriever()
        retrieval_limit = getattr(getattr(source, "settings", None), "evidence_count", 8)
        snapshot = getattr(source, "snapshot", None)
        if snapshot is not None:
            source = snapshot()
        initial_metadata = source.index_metadata()
        index_revision = str(
            initial_metadata.get("revision") or initial_metadata["manifest_version"]
        )
        query_diagnostics = None
        selected_questions: list[EvidenceQuestion] = []
        if getattr(getattr(source, "settings", None), "multi_query", False):
            from time import perf_counter

            query_started = perf_counter()
            generation_usage: dict[str, int] = {}
            try:
                query_settings = (
                    None if query_generator is not None else ExtractionSettings.from_env()
                )
                max_total = getattr(source.settings, "multi_query_max_queries", 4)
                query_timeout = getattr(source.settings, "multi_query_timeout", 30)
                if query_generator is None:
                    plan = generate_evidence_query_plan(
                        use_case,
                        query_settings,
                        timeout=query_timeout,
                        usage=generation_usage,
                    )
                else:
                    plan = query_generator(use_case)
                    if not isinstance(plan, EvidenceQueryPlan):
                        plan = EvidenceQueryPlan.model_validate(plan)
                selected_questions, rejected = validate_evidence_query_plan(
                    plan, use_case, max_questions=min(3, max_total - 1)
                )
                query_diagnostics = QueryDiagnostics(
                    status="fallback" if not selected_questions else "applied",
                    accepted_questions=selected_questions,
                    rejected_count=len(rejected),
                    reason_codes=rejected
                    if selected_questions
                    else [*rejected, "no_useful_questions"],
                    model_identity=(
                        f"{query_settings.provider}:{query_settings.model}"
                        if query_settings is not None
                        else f"{provider}:injected"
                    ),
                    prompt_tokens=(
                        generation_usage.get("prompt_tokens") if query_generator is None else None
                    ),
                    completion_tokens=(
                        generation_usage.get("completion_tokens")
                        if query_generator is None
                        else None
                    ),
                    planner_call_count=(
                        generation_usage.get("call_count", 1 if query_generator else 0)
                    ),
                    elapsed_ms=(perf_counter() - query_started) * 1000,
                )
            except (ProviderError, ValidationError, ValueError, TypeError) as exc:
                query_diagnostics = QueryDiagnostics(
                    status="fallback",
                    reason_codes=["generation_failed"],
                    model_identity=provider,
                    planner_call_count=generation_usage.get(
                        "call_count", 1 if query_generator else 0
                    ),
                    elapsed_ms=(perf_counter() - query_started) * 1000,
                )
                query_diagnostics.reason_codes.append(type(exc).__name__)

        if selected_questions and query_diagnostics is not None:
            try:
                question_search = getattr(source, "search_questions", None)
                if question_search is None:
                    raise ValueError("Retriever does not support explicit evidence questions")
                retrieved_raw = question_search(
                    use_case,
                    [item.question for item in selected_questions],
                    limit=retrieval_limit,
                    timeout=getattr(source.settings, "multi_query_search_timeout", 30),
                )
                last_diagnostics = getattr(source, "last_query_diagnostics", {})
                if not last_diagnostics and hasattr(source, "retriever"):
                    last_diagnostics = getattr(source.retriever, "last_query_diagnostics", {})
                query_diagnostics.question_evidence = last_diagnostics.get("question_evidence", {})
                query_diagnostics.candidate_count = last_diagnostics.get("candidate_count", 0)
                query_diagnostics.unique_document_count = last_diagnostics.get(
                    "unique_document_count", 0
                )
                query_diagnostics.selected_count = len(retrieved_raw)
                if not retrieved_raw:
                    query_diagnostics.status = "fallback"
                    query_diagnostics.reason_codes.append("no_candidates")
                    retrieved_raw = source.search(use_case, limit=retrieval_limit)
                retrieved, version = _verified_retrieval(source, retrieved_raw)
            except RetrievalDeadlineExceeded:
                query_diagnostics.status = "fallback"
                query_diagnostics.reason_codes.append("search_deadline")
                retrieved, version = _verified_retrieval(
                    source, source.search(use_case, limit=retrieval_limit)
                )
        else:
            if query_diagnostics is None and getattr(
                getattr(source, "settings", None), "multi_query", False
            ):
                query_diagnostics = QueryDiagnostics(
                    status="fallback", reason_codes=["generation_failed"]
                )
            retrieved, version = _verified_retrieval(
                source, source.search(use_case, limit=retrieval_limit)
            )
        if not retrieved:
            retrieval_diagnostics = getattr(source, "retrieval_diagnostics", [])
            return _abstention_report(
                use_case,
                "No technical source evidence was retrieved; method selection is "
                "outside currently supported evidence.",
                status="outside_supported_scope",
                method="evidence_abstention",
                provider=provider,
                evidence_version=version,
                knowledge_index_revision=index_revision,
                retrieval_diagnostics=retrieval_diagnostics,
                query_diagnostics=query_diagnostics,
            )
        selected_assessor = assessor or OllamaTechnicalAssessor()
        selected_evidence_assessor = evidence_assessor or OllamaEvidenceAssessor()
        dropped_ids: list[str] = []
        fit_evidence = getattr(selected_evidence_assessor, "fit_evidence", None)
        if fit_evidence is None:
            fit_evidence = getattr(selected_assessor, "fit_evidence", None)
        if fit_evidence is not None:
            retrieved, dropped_ids = fit_evidence(use_case, retrieved)
            version = _evidence_set_version(
                retrieved, str(source.index_metadata()["manifest_version"])
            )
        if not retrieved:
            retrieval_diagnostics = getattr(source, "retrieval_diagnostics", [])
            report = _abstention_report(
                use_case,
                "No retrieved excerpt fits the complete request within the configured "
                "context budget; assessment abstained.",
                status="insufficient_information",
                method="evidence_abstention",
                provider=provider,
                evidence_version=version,
                knowledge_index_revision=index_revision,
                retrieval_diagnostics=retrieval_diagnostics,
                query_diagnostics=query_diagnostics,
            )
            report.assessment_warnings.append(_budget_drop_warning(dropped_ids))
            return report
        evidence_assessment = EvidenceAssessment.model_validate(
            selected_evidence_assessor(use_case, retrieved)
        )
        evidence_assessment = validate_assessment(
            evidence_assessment,
            use_case,
            {item.evidence_id: item for item in retrieved},
            evidence_fingerprint(
                use_case,
                retrieved,
                getattr(
                    selected_evidence_assessor, "VERSION", "application-evidence-assessment-v1"
                ),
            ),
            getattr(selected_evidence_assessor, "VERSION", "injected-evidence-assessor"),
            getattr(evidence_assessment, "model_identity", None),
        )
        evidence_assessment.diagnostics.dropped_evidence_ids.extend(
            item
            for item in dropped_ids
            if item not in evidence_assessment.diagnostics.dropped_evidence_ids
        )
        conflict_ids = {
            observation_id
            for conflict in evidence_assessment.conflicts
            for observation_id in conflict.observation_ids
        }
        if not any(
            support.basis != "unsupported"
            and not conflict_ids.intersection(support.observation_ids)
            for support in evidence_assessment.support_assessments
        ):
            report = _abstention_report(
                use_case,
                "No nonconflicting supported or exploratory assessment was produced.",
                status="insufficient_information",
                provider=provider,
                evidence=retrieved,
                evidence_version=version,
                knowledge_index_revision=index_revision,
                retrieval_diagnostics=getattr(source, "retrieval_diagnostics", []),
                query_diagnostics=query_diagnostics,
            )
            report.evidence_assessment = evidence_assessment
            report.report_version = f"grounded-{provider}-v2"
            report.assessment_warnings.extend(evidence_assessment.diagnostics.warnings)
            for item in retrieved:
                if item.review_status not in {"reviewed", "reviewed_for_phase2", "source_reviewed"}:
                    report.assessment_warnings.append(
                        f"Source {item.source_id} {item.page_or_section} has review status "
                        f"{item.review_status}; review does not determine applicability."
                    )
            return report
        try:
            draft = selected_assessor.assess(
                use_case, retrieved, evidence_assessment=evidence_assessment
            )
        except ProviderError as exc:
            if "context budget" not in str(exc).casefold():
                raise
            report = _abstention_report(
                use_case,
                "Assessed evidence and qualifications exceed synthesis context; "
                "no recommendation generated.",
                status="insufficient_information",
                provider=provider,
                evidence=retrieved,
                evidence_version=version,
                knowledge_index_revision=index_revision,
                retrieval_diagnostics=getattr(source, "retrieval_diagnostics", []),
                query_diagnostics=query_diagnostics,
            )
            report.evidence_assessment = evidence_assessment
            report.report_version = f"grounded-{provider}-v2"
            report.assessment_warnings.extend(evidence_assessment.diagnostics.warnings)
            report.assessment_warnings.append(
                "Synthesis abstained to preserve all assessed limitations and qualifications."
            )
            for item in retrieved:
                if item.review_status not in {"reviewed", "reviewed_for_phase2", "source_reviewed"}:
                    report.assessment_warnings.append(
                        f"Source {item.source_id} {item.page_or_section} has review status "
                        f"{item.review_status}; review does not determine applicability."
                    )
            return report
        repair_additions: dict[str, RetrievedEvidence] = {}
        searched_claims: set[str] = set()
        initial_known = {item.evidence_id: item for item in retrieved}
        existing_context_ids = {
            block.block_id for item in retrieved for block in item.context_blocks
        }
        for proposed in draft.candidate_techniques:
            for claim in proposed.evidence_claims:
                current = initial_known.get(claim.evidence_id)
                if claim.context_block_id is not None or (
                    current is not None and claim.quote in current.excerpt
                ):
                    continue
                query = f'{claim.claim} "{claim.quote}"'
                if query.casefold() in searched_claims or len(searched_claims) >= 4:
                    continue
                searched_claims.add(query.casefold())
                search_claim = getattr(source, "search_claim", None)
                if search_claim is None:
                    continue
                try:
                    extra = search_claim(use_case, query, limit=4)
                    match = next((item for item in extra if claim.quote in item.excerpt), None)
                    if match is not None:
                        verified, _ = _verified_retrieval(source, [match])
                        if verified and verified[0].evidence_id not in initial_known:
                            addition = verified[0]
                            repeated_blocks = [
                                block
                                for block in addition.context_blocks
                                if block.block_id in existing_context_ids
                            ]
                            if repeated_blocks:
                                repeated_ids = [block.block_id for block in repeated_blocks]
                                addition = addition.model_copy(
                                    update={
                                        "context_blocks": [
                                            block
                                            for block in addition.context_blocks
                                            if block.block_id not in existing_context_ids
                                        ],
                                        "context_omissions": [
                                            *addition.context_omissions,
                                            "Repeated context blocks already attached to prior "
                                            f"evidence: {', '.join(repeated_ids)}",
                                        ],
                                    }
                                )
                            repair_additions[addition.evidence_id] = addition
                            existing_context_ids.update(
                                block.block_id for block in addition.context_blocks
                            )
                except ProviderError, ValueError:
                    continue
        repair_warning = None
        if repair_additions:
            old_evidence = list(retrieved)
            candidate_set = [*old_evidence, *repair_additions.values()]
            # Recheck exact pinned provenance after deduplicating only newly added context.
            candidate_set, _ = _verified_retrieval(source, candidate_set)
            fitted, newly_dropped = (
                fit_evidence(use_case, candidate_set) if fit_evidence else (candidate_set, [])
            )
            preserves_old = len(fitted) >= len(old_evidence) and all(
                old.model_dump(mode="json") == new.model_dump(mode="json")
                for old, new in zip(old_evidence, fitted, strict=False)
            )
            if preserves_old and len(fitted) > len(old_evidence):
                try:
                    augmented = EvidenceAssessment.model_validate(
                        selected_evidence_assessor(use_case, fitted, allow_retry=False)
                    )
                    augmented = validate_assessment(
                        augmented,
                        use_case,
                        {item.evidence_id: item for item in fitted},
                        evidence_fingerprint(
                            use_case,
                            fitted,
                            getattr(
                                selected_evidence_assessor,
                                "VERSION",
                                "application-evidence-assessment-v1",
                            ),
                        ),
                        getattr(
                            selected_evidence_assessor, "VERSION", "injected-evidence-assessor"
                        ),
                        getattr(augmented, "model_identity", None),
                    )
                    prior_diagnostics = evidence_assessment.diagnostics
                    next_diagnostics = augmented.diagnostics
                    augmented.diagnostics = next_diagnostics.model_copy(
                        update={
                            "status": "partial"
                            if "partial" in {prior_diagnostics.status, next_diagnostics.status}
                            else next_diagnostics.status,
                            "warnings": list(
                                dict.fromkeys(
                                    prior_diagnostics.warnings + next_diagnostics.warnings
                                )
                            ),
                            "dropped_evidence_ids": list(
                                dict.fromkeys(
                                    prior_diagnostics.dropped_evidence_ids
                                    + next_diagnostics.dropped_evidence_ids
                                    + newly_dropped
                                )
                            ),
                            "call_count": prior_diagnostics.call_count
                            + next_diagnostics.call_count,
                            "elapsed_ms": (prior_diagnostics.elapsed_ms or 0)
                            + (next_diagnostics.elapsed_ms or 0),
                            "serialized_bytes": (prior_diagnostics.serialized_bytes or 0)
                            + (next_diagnostics.serialized_bytes or 0),
                            "prompt_tokens": (
                                prior_diagnostics.prompt_tokens + next_diagnostics.prompt_tokens
                                if prior_diagnostics.prompt_tokens is not None
                                and next_diagnostics.prompt_tokens is not None
                                else None
                            ),
                            "completion_tokens": (
                                prior_diagnostics.completion_tokens
                                + next_diagnostics.completion_tokens
                                if prior_diagnostics.completion_tokens is not None
                                and next_diagnostics.completion_tokens is not None
                                else None
                            ),
                        }
                    )
                    augmented_draft = selected_assessor.assess(
                        use_case, fitted, evidence_assessment=augmented
                    )
                    retrieved = fitted
                    evidence_assessment = augmented
                    version = _evidence_set_version(
                        retrieved, str(initial_metadata["manifest_version"])
                    )
                    draft = augmented_draft
                    dropped_ids.extend(newly_dropped)
                except ProviderError, ValidationError, ValueError, TypeError:
                    repair_warning = (
                        "Claim-specific additions rejected: augmented assessment failed."
                    )
            else:
                repair_warning = (
                    "Claim-specific additions rejected: earlier evidence could not "
                    "all be preserved."
                )
        known = {item.evidence_id: item for item in retrieved}
        candidates: list[TechniqueCandidate] = []
        warnings: list[str] = []
        if dropped_ids:
            warnings.append(_budget_drop_warning(dropped_ids))
        if repair_warning:
            warnings.append(repair_warning)
        conflicts = {
            oid for conflict in evidence_assessment.conflicts for oid in conflict.observation_ids
        }
        observations = {item.observation_id: item for item in evidence_assessment.observations}
        for support in evidence_assessment.support_assessments:
            if support.basis == "unsupported":
                warnings.append(
                    f"Unsupported candidate {support.method} withheld: "
                    f"{support.rejection_reason or support.bounded_conclusion}"
                )
                continue
            if any(oid in conflicts for oid in support.observation_ids):
                warnings.append(f"Withheld {support.method}: unresolved evidence conflict.")
                continue
            rows = [observations[oid] for oid in support.observation_ids if oid in observations]
            refs = [reference for row in rows for reference in row.references]
            bridge = support.exploratory_bridge
            if bridge is not None:
                refs.extend(bridge.source_premises)
            unique_refs = []
            seen_refs = set()
            for reference in refs:
                key = (reference.evidence_id, reference.context_block_id, reference.quote)
                if key not in seen_refs and reference.evidence_id in known:
                    seen_refs.add(key)
                    unique_refs.append(reference)
            if not unique_refs:
                warnings.append(f"Withheld {support.method}: no validated source references.")
                continue
            claims = [
                EvidenceClaim(
                    claim=f"Source states: “{ref.quote}”",
                    evidence_id=ref.evidence_id,
                    quote=ref.quote,
                    context_block_id=ref.context_block_id,
                    observation_id=next(
                        (
                            row.observation_id
                            for row in rows
                            if any(
                                item.evidence_id == ref.evidence_id
                                and item.context_block_id == ref.context_block_id
                                and ref.quote in item.quote
                                for item in row.references
                            )
                        ),
                        None,
                    ),
                    support_assessment_id=support.support_assessment_id,
                    claim_use="source_observation",
                )
                for ref in unique_refs
            ]
            conclusion = support.bounded_conclusion.strip()
            if support.basis == "reasoned_exploratory":
                conclusion = (
                    f"Engineering reasoning (exploratory): {conclusion} "
                    f"Hypothesis: {bridge.hypothesis} Inference: {bridge.inference}"
                )
            limitations = list(
                dict.fromkeys(
                    [text for row in rows for text in row.transfer_limitations]
                    + (bridge.application_differences if bridge else [])
                    + (bridge.unknown_gaps if bridge else [])
                    + (bridge.possible_failure_mechanisms if bridge else [])
                )
            )
            candidates.append(
                TechniqueCandidate(
                    spectroscopy_method=support.method,
                    measurement_geometry=next((row.geometry for row in rows if row.geometry), None),
                    advantages=[conclusion],
                    limitations=limitations,
                    required_assumptions=[
                        "Exploratory bridge needs expert review and a falsifiable study."
                    ]
                    if bridge
                    else ["Customer-specific performance and compatibility remain unestablished."],
                    supporting_evidence_ids=list(
                        dict.fromkeys(ref.evidence_id for ref in unique_refs)
                    ),
                    evidence_claims=claims,
                    support_assessment_id=support.support_assessment_id,
                    consideration_basis=support.basis,
                    rule_rationale=conclusion,
                )
            )
        for item in retrieved:
            if item.review_status not in {"reviewed", "reviewed_for_phase2", "source_reviewed"}:
                warnings.append(
                    f"Source {item.source_id} {item.page_or_section} has review status "
                    f"{item.review_status}; review does not determine applicability."
                )
        for warning in evidence_assessment.diagnostics.warnings:
            warnings.append(f"Evidence assessment: {warning}")
        if not candidates:
            status = (
                "outside_supported_scope"
                if draft.status == "outside_supported_scope"
                else "insufficient_information"
            )
            if evidence_assessment.support_assessments:
                warnings.append("No candidate passed structured evidence assessment checks.")
            rationale = (
                "No candidate passed structured evidence assessment; consideration withheld."
            )
            plan = _abstention_report(
                use_case,
                rationale,
                status=status,
                method="evidence_abstention",
                evidence=retrieved,
                evidence_version=version,
            ).proposed_feasibility_experiment
        else:
            status = "provisional"
            rationale = (
                "Provisional feasibility candidates retained from source quotations. "
                "Transferability and customer-specific performance remain unestablished; "
                "production readiness is not assessed."
            )
            plan = draft.proposed_feasibility_experiment
        application_supported = [
            candidate
            for candidate in candidates
            if candidate.consideration_basis == "application_supported"
        ]
        plan, plan_warnings = _apply_feasibility_safeguards(use_case, plan, application_supported)
        if candidates and not application_supported:
            plan = FeasibilityPlan(
                objective="Resolve evidence gaps before deciding on an application study.",
                hypothesis="No application-specific performance outcome is established.",
                experimental_steps=[
                    "Resolve application differences and unknown gaps with the customer.",
                    "Request expert review of exploratory reasoning before study selection.",
                ],
                expected_limitations=[
                    "Exploratory consideration establishes neither suitability nor priority."
                ],
            )
        warnings.extend(plan_warnings)
        diagnostics = [
            ValidationDiagnostic(
                level="not_checked",
                check="scientific_entailment",
                message=(
                    "Entailment, scenario binding, transfer, and qualification completeness "
                    "are model-assessed "
                    "and require expert review; quote membership alone cannot prove them."
                ),
                evidence_ids=[item.evidence_id for item in retrieved],
            ),
            ValidationDiagnostic(
                level="not_checked",
                check="units_and_compatibility",
                message=(
                    "Customer units were preserved without conversion. Instrument, fiber, "
                    "temperature, pressure, and material compatibility were not checked."
                ),
            ),
        ]
        if use_case.contradictions:
            diagnostics.append(
                ValidationDiagnostic(
                    level="warning",
                    check="customer_constraints",
                    message="Contradictory customer statements require expert resolution.",
                )
            )
        if warnings:
            diagnostics.append(
                ValidationDiagnostic(
                    level="warning",
                    check="assessment_guardrails",
                    message="; ".join(warnings),
                )
            )
        return AssessmentReport(
            extracted_requirements=use_case,
            missing_information=prioritized_missing_fields(use_case),
            suggested_technical_questions=prioritized_questions(use_case),
            candidate_techniques=candidates,
            recommendation_rationale=rationale,
            proposed_feasibility_experiment=plan,
            assumptions_and_uncertainties=[
                "Source results may not transfer to customer samples or operating conditions.",
                *(
                    ["Customer-defined acceptance criteria are missing."]
                    if not (
                        use_case.required_accuracy
                        or use_case.detection_limit
                        or use_case.strict_requirements
                    )
                    else []
                ),
            ],
            status=status,
            assessment_method=f"{provider}_grounded",
            assessment_provider=provider,
            evidence_set_version=version,
            knowledge_index_revision=index_revision,
            retrieved_evidence=retrieved,
            report_version=f"grounded-{provider}-v2",
            assessment_warnings=warnings,
            validation_diagnostics=diagnostics,
            retrieval_diagnostics=[
                RetrievalDiagnostic.model_validate(item)
                for item in getattr(source, "retrieval_diagnostics", [])
            ],
            query_diagnostics=query_diagnostics,
            evidence_assessment=evidence_assessment,
        )
    except (ProviderError, ValueError) as exc:
        raise ProviderError(f"Grounded assessment failed: {exc}") from None


def assess_problem(
    text: str,
    extractor: UseCaseExtractor | None = None,
    *,
    retriever: EvidenceRetriever | None = None,
) -> AssessmentReport:
    provider = os.getenv("SPECTRO_AI_PROVIDER", "ollama").strip().lower()
    if provider == "baseline":
        raise ProviderError("The baseline provider was removed. Use ollama or openai.")
    if provider not in {"ollama", "openai"}:
        raise ProviderError(f"Unsupported provider {provider!r}. Use ollama or openai.")
    return assess_use_case(
        (extractor or OllamaUseCaseExtractor()).extract(text), retriever=retriever
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spectro-AI provider utilities")
    parser.add_argument("command", choices=["setup-reranker"])
    parser.parse_args()
    print(f"Reranker cached; resolved revision: {setup_reranker()}")
