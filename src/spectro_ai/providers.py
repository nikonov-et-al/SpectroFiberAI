"""LLM provider adapters and grounded assessment orchestration."""

import json
import os
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spectro_ai.evidence import EvidenceRetriever
from spectro_ai.guardrails import abstention_report, missing_fields, prioritized_questions
from spectro_ai.models import (
    CUSTOMER_FACT_FIELDS,
    AssessmentReport,
    CustomerUseCase,
    Evidence,
    EvidenceClaim,
    FeasibilityPlan,
    RetrievedEvidence,
    TechniqueCandidate,
    ValidationDiagnostic,
)

_OLLAMA_LOCK = Lock()


@dataclass(frozen=True)
class RagSettings:
    provider: str = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4:12b-it-q4_K_M"
    embedding_model: str = "qwen3-embedding:0.6b"
    context: int = 4096
    evidence_count: int = 8
    index: str = ".cache/rag/index.sqlite3"

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
        )


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
) -> dict:
    try:
        request = Request(
            settings.ollama_url + route,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with _OLLAMA_LOCK, urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict) or result.get("error"):
            raise ValueError
        return result
    except HTTPError, URLError, TimeoutError, OSError, ValueError:
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


def _openai_json(settings: ExtractionSettings, messages: list[dict], schema: dict) -> str:
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
            response = client.chat.completions.create(
                model=settings.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "spectro_ai_output",
                        "strict": True,
                        "schema": _strict_schema(schema),
                    },
                },
                max_completion_tokens=settings.output_tokens,
            )
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


def embedding_identity(settings: RagSettings) -> str:
    result = ollama_request(settings, "/api/tags")
    models = result.get("models")
    if not isinstance(models, list) or any(not isinstance(m, dict) for m in models):
        raise ProviderError("Ollama returned invalid model metadata")
    for model in models:
        if model.get("name") == settings.embedding_model and model.get("digest"):
            return str(model["digest"])
    raise ProviderError("Embedding model unavailable. Run ollama pull " + settings.embedding_model)


def embed_texts(settings: RagSettings, texts: list[str]) -> list[list[float]]:
    result = ollama_request(
        settings,
        "/api/embed",
        {
            "model": settings.embedding_model,
            "input": texts,
            "truncate": False,
            "keep_alive": 0,
        },
    )
    vectors = result.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ProviderError("Ollama returned an invalid embedding count")
    return vectors


class ProviderError(ValueError):
    """A provider is unknown, incomplete, or deliberately unavailable."""


class _CandidateDraft(TechniqueCandidate):
    spectroscopy_method: str = Field(
        description="Name each specific method; combined methods list every component."
    )
    supporting_evidence: list[Evidence] = Field(default_factory=list, max_length=0)
    evidence_claims: list[EvidenceClaim] = Field(
        default_factory=list,
        description="Optional source quotations; each must be an exact excerpt substring and carry its supplied ID.",
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
        self, use_case: CustomerUseCase, evidence: list[RetrievedEvidence]
    ) -> AssessmentReport: ...


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
    return (
        prompt
        + "\nAuthoritative output JSON schema:\n"
        + json.dumps(_technical_schema(), separators=(",", ":"))
    )


class OllamaTechnicalAssessor:
    """Generate a provisional, evidence-ID-bound technical assessment."""

    def __init__(self, settings: ExtractionSettings | None = None) -> None:
        self.settings = settings or ExtractionSettings.from_env()

    def fit_evidence(
        self, use_case: CustomerUseCase, evidence: list[RetrievedEvidence]
    ) -> tuple[list[RetrievedEvidence], list[str]]:
        """Drop lowest-ranked whole excerpts until complete request fits the budget."""
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
        use_case: CustomerUseCase, evidence: list[RetrievedEvidence]
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
        sources = [
            {
                "evidence_id": item.evidence_id,
                "source_id": item.source_id,
                "document_version": item.document_version,
                "page_or_section": item.page_or_section,
                "excerpt": item.excerpt,
                "review_status": item.review_status,
                "source_scope": item.source_scope,
            }
            for item in evidence
        ]
        return [
            {"role": "system", "content": _assessment_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "customer_case_untrusted_data": customer,
                        "retrieved_sources_untrusted_data": sources,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    def assess(
        self, use_case: CustomerUseCase, evidence: list[RetrievedEvidence]
    ) -> AssessmentReport:
        messages = self._messages(use_case, evidence)
        if (
            len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
            + (self.settings.output_tokens + 256)
            > self.settings.context
        ):
            raise ProviderError(
                "Assessment input exceeds the configured context budget. Reduce "
                "the complete customer text or retrieved excerpt set, or increase "
                "SPECTRO_AI_CONTEXT."
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
                    report_version=f"grounded-{self.settings.provider}-v1",
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
    verified = []
    seen = set()
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
            }
            for item in sorted(evidence, key=lambda row: row.evidence_id)
        ],
    }
    return sha256(json.dumps(version_material, sort_keys=True).encode()).hexdigest()


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

    excerpt = " ".join(item.excerpt for item in cited).casefold()
    has_all_method_components = all(
        any(has_phrase(excerpt, alias) for alias in component) for component in components
    )
    has_case_match = any(has_phrase(excerpt, term) for term in target_terms | matrix_terms)
    return has_all_method_components and has_case_match


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
    unsafe_configuration = bool(
        safe.proposed_configuration
        and re.search(
            r"\d|customer.{0,20}(?:has|uses|existing|installed|configured)",
            safe.proposed_configuration,
            re.I,
        )
    )
    if unsafe_configuration:
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
        combined = [method for method in methods if method not in components]
        safe.proposed_measurement_technique = "; ".join(components or methods)
        safe.experimental_steps = [
            "Collect representative samples spanning customer-agreed batches, runs, matrix variation, and relevant operating conditions.",
            "Measure each candidate using the same samples and compare any combined method only as an optional addition to its component methods.",
            "Obtain independent reference results for the same samples before interpreting spectroscopy measurements.",
            "Split training and held-out evaluation by independent batch or process run before any preprocessing or model fitting.",
            "Evaluate held-out results against customer-agreed measures and acceptance criteria; agree criteria before testing.",
        ]
        if combined:
            safe.expected_limitations.append(
                "Combined measurement is an optional comparison; its added value is unknown."
            )
    else:
        safe.experimental_steps = [
            "Confirm the decision, target, and representative operating conditions before selecting a study.",
            "Agree suitable independent reference measurements and customer-defined acceptance criteria.",
        ]
    if not safe.reference_measurement:
        safe.reference_measurement = (
            use_case.reference_measurement
            or "Independent reference measurement to be agreed before testing."
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
    if caveat not in safe.expected_limitations:
        safe.expected_limitations.append(caveat)
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
            "Removed numerical or asserted customer configuration; compatibility remains unchecked."
        )
    return safe, warnings


def assess_use_case(
    use_case: CustomerUseCase,
    *,
    assessor: TechnicalAssessor | None = None,
    retriever: EvidenceRetriever | None = None,
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
        retrieved, version = _verified_retrieval(
            source, source.search(use_case, limit=retrieval_limit)
        )
        if not retrieved:
            return _abstention_report(
                use_case,
                "No technical source evidence was retrieved; method selection is "
                "outside currently supported evidence.",
                status="outside_supported_scope",
                method="evidence_abstention",
                provider=provider,
                evidence_version=version,
                knowledge_index_revision=index_revision,
            )
        selected_assessor = assessor or OllamaTechnicalAssessor()
        dropped_ids: list[str] = []
        fit_evidence = getattr(selected_assessor, "fit_evidence", None)
        if fit_evidence is not None:
            retrieved, dropped_ids = fit_evidence(use_case, retrieved)
            version = _evidence_set_version(
                retrieved, str(source.index_metadata()["manifest_version"])
            )
        if not retrieved:
            report = _abstention_report(
                use_case,
                "No retrieved excerpt fits the complete request within the configured "
                "context budget; assessment abstained.",
                status="insufficient_information",
                method="evidence_abstention",
                provider=provider,
                evidence_version=version,
                knowledge_index_revision=index_revision,
            )
            report.assessment_warnings.append(
                f"Dropped {len(dropped_ids)} whole excerpts to preserve complete "
                f"source text: {', '.join(dropped_ids)}."
            )
            return report
        draft = selected_assessor.assess(use_case, retrieved)
        known = {item.evidence_id: item for item in retrieved}
        candidates: list[TechniqueCandidate] = []
        warnings = []
        if dropped_ids:
            warnings.append(
                f"Dropped {len(dropped_ids)} lower-ranked whole excerpts for context "
                f"budget: {', '.join(dropped_ids)}."
            )
        warnings.append(
            "Candidate relevance check is lexical screening only; it cannot validate "
            "entailment or scientific suitability. Expert review required."
        )
        rejected = 0
        relevance_rejected = False
        for proposed in draft.candidate_techniques:
            if proposed.supporting_evidence:
                warnings.append(
                    f"Ignored model-generated source objects for {proposed.spectroscopy_method}."
                )
            citations = list(dict.fromkeys(proposed.supporting_evidence_ids))
            valid_claims: list[EvidenceClaim] = []
            valid_ids: list[str] = []
            for claim in proposed.evidence_claims:
                excerpt = known.get(claim.evidence_id)
                if (
                    claim.evidence_id in citations
                    and excerpt is not None
                    and claim.quote
                    and claim.quote in excerpt.excerpt
                ):
                    valid_claims.append(
                        EvidenceClaim(
                            claim=f'Source states: “{claim.quote}”',
                            evidence_id=claim.evidence_id,
                            quote=claim.quote,
                        )
                    )
                    if claim.evidence_id not in valid_ids:
                        valid_ids.append(claim.evidence_id)
            if len(valid_claims) != len(proposed.evidence_claims):
                warnings.append(
                    f"Removed unsupported or invalidly cited claims from {proposed.spectroscopy_method}."
                )
            negative_claim = re.compile(
                r"\b(no|not|cannot|can't|unable|limited|limitation|unsuitable|"
                r"poor|failed|failure|doesn't|does not|without)\b",
                re.I,
            )
            positive = [
                item
                for item in valid_claims
                if not negative_claim.search(item.quote)
                if _supports_candidate(
                    use_case,
                    proposed,
                    [known[item.evidence_id]],
                )
            ]
            if not positive:
                warnings.append(
                    f"Removed {proposed.spectroscopy_method}: no valid positive source quote "
                    "supports both a named method and the requested case."
                )
                rejected += 1
                relevance_rejected = True
                continue
            supporting = [known[evidence_id] for evidence_id in valid_ids]
            if not _supports_candidate(use_case, proposed, supporting):
                warnings.append(
                    f"Removed {proposed.spectroscopy_method}: retrieved excerpts did "
                    "not lexically support both method and case; expert review required."
                )
                rejected += 1
                relevance_rejected = True
                continue
            candidate_data = proposed.model_dump()
            candidate_data["supporting_evidence"] = []
            candidate_data["advantages"] = [item.claim for item in positive]
            candidate_data["limitations"] = [
                item.claim for item in valid_claims if item not in positive
            ]
            candidate_data["evidence_claims"] = [*positive] + [
                item for item in valid_claims if item not in positive
            ]
            candidate_data["supporting_evidence_ids"] = valid_ids
            candidate_data["rule_rationale"] = None
            source_text = " ".join(item.excerpt.casefold() for item in supporting)
            for field in ("measurement_geometry", "proposed_spectral_region"):
                value = candidate_data[field]
                if value and value.casefold() not in source_text:
                    candidate_data[field] = None
                    warnings.append(
                        f"Removed unsupported {field.replace('_', ' ')} from "
                        f"{proposed.spectroscopy_method}."
                    )
            config = candidate_data["potential_probe_configuration"]
            if config and re.search(
                r"\d|customer.{0,20}(?:has|uses|existing|installed|configured)|"
                r"is installed|is configured",
                config,
                re.IGNORECASE,
            ):
                candidate_data["potential_probe_configuration"] = None
                warnings.append(f"Removed unsupported customer configuration for {proposed.spectroscopy_method}.")
            candidates.append(TechniqueCandidate(**candidate_data))
        if not candidates:
            status = (
                "outside_supported_scope"
                if draft.status == "outside_supported_scope" or relevance_rejected
                else "insufficient_information"
            )
            if draft.candidate_techniques:
                warnings.append("No candidate passed evidence membership and relevance checks.")
            rationale = (
                "No candidate passed source membership and relevance checks; "
                "method selection is withheld."
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
        for item in retrieved:
            if item.review_status not in {"reviewed", "reviewed_for_phase2", "source_reviewed"}:
                warnings.append(
                    f"Source {item.source_id} {item.page_or_section} review status "
                    f"is {item.review_status}."
                )
        plan, plan_warnings = _apply_feasibility_safeguards(use_case, plan, candidates)
        warnings.extend(plan_warnings)
        diagnostics = [
            ValidationDiagnostic(
                level="not_checked",
                check="scientific_entailment",
                message=(
                    "Quote membership and lexical screening do not validate scientific "
                    "entailment, customer suitability, or compatibility; expert review required."
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
            missing_information=missing_fields(use_case),
            suggested_technical_questions=(
                draft.suggested_technical_questions or prioritized_questions(use_case)
            ),
            candidate_techniques=candidates,
            recommendation_rationale=rationale,
            proposed_feasibility_experiment=plan,
            assumptions_and_uncertainties=draft.assumptions_and_uncertainties,
            status=status,
            assessment_method=f"{provider}_grounded",
            assessment_provider=provider,
            evidence_set_version=version,
            knowledge_index_revision=index_revision,
            retrieved_evidence=retrieved,
            report_version=f"grounded-{provider}-v1",
            assessment_warnings=warnings,
            validation_diagnostics=diagnostics,
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
