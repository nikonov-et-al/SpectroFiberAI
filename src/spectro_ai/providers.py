"""Explicit provider selection; the baseline never imports an LLM SDK."""

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

from spectro_ai.baseline import (
    OfflineUseCaseExtractor,
    UseCaseExtractor,
    assess,
    missing_fields,
    prioritized_questions,
)
from spectro_ai.evidence import EvidenceRetriever
from spectro_ai.models import (
    CUSTOMER_FACT_FIELDS,
    AssessmentReport,
    CustomerUseCase,
    Evidence,
    FeasibilityPlan,
    RetrievedEvidence,
    TechniqueCandidate,
)
from spectro_ai.rag_models import DocumentAnswer

_OLLAMA_LOCK = Lock()


@dataclass(frozen=True)
class RagSettings:
    provider: str = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4:12b-it-q4_K_M"
    embedding_model: str = "qwen3-embedding:0.6b"
    context: int = 4096
    index: str = ".cache/rag/index.sqlite3"

    @classmethod
    def from_env(cls) -> RagSettings:
        try:
            context = int(os.getenv("SPECTRO_RAG_CONTEXT", "4096"))
        except ValueError:
            raise ProviderError("SPECTRO_RAG_CONTEXT must be an integer") from None
        if context < 1024:
            raise ProviderError("SPECTRO_RAG_CONTEXT must be at least 1024")
        provider = os.getenv("SPECTRO_RAG_PROVIDER", "ollama").strip().lower()
        if provider not in {"ollama", "openai"}:
            raise ProviderError("SPECTRO_RAG_PROVIDER must be ollama or openai")
        return cls(
            provider=provider,
            ollama_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_url).rstrip("/"),
            model=os.getenv("OLLAMA_MODEL", cls.model),
            embedding_model=os.getenv("OLLAMA_EMBED_MODEL", cls.embedding_model),
            context=context,
            index=os.getenv("SPECTRO_RAG_INDEX", cls.index),
        )


@dataclass(frozen=True)
class ExtractionSettings:
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4:12b-it-q4_K_M"
    context: int = 16384
    timeout: int = 120
    output_tokens: int = 2048

    @classmethod
    def from_env(cls) -> ExtractionSettings:
        try:
            context = int(os.getenv("SPECTRO_AI_CONTEXT", "16384"))
            timeout = int(os.getenv("SPECTRO_AI_TIMEOUT", "120"))
            output_tokens = int(os.getenv("SPECTRO_AI_OUTPUT_TOKENS", "2048"))
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
        return cls(
            ollama_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_url).rstrip("/"),
            model=os.getenv("OLLAMA_MODEL", cls.model),
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
    return {"type": "object", "properties": properties, "additionalProperties": False}


def _clean_extraction(payload: str, customer_text: str) -> CustomerUseCase:
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
    values["extraction_mode"] = "ollama"
    values["extraction_warnings"] = warnings
    cleaned = CustomerUseCase.model_validate(values)
    cleaned.unknown_or_missing_information = missing_fields(cleaned)
    return cleaned


class OllamaUseCaseExtractor:
    """Structured local extraction; only quoted customer facts survive validation."""

    def __init__(self, settings: ExtractionSettings | None = None) -> None:
        self.settings = settings or ExtractionSettings.from_env()

    def extract(self, customer_text: str) -> CustomerUseCase:
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
                return _clean_extraction(message["content"], customer_text)
            except ValueError, TypeError:
                if attempt == 1:
                    raise ProviderError("Ollama returned invalid customer-use-case JSON") from None
        raise ProviderError("Ollama extraction failed after one structural retry")


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


def generate_document_answer(settings: RagSettings, system: str, user: str) -> DocumentAnswer:
    if settings.provider not in {"ollama", "openai"}:
        raise ProviderError("Unsupported document provider; use ollama or openai")
    # UTF-8 bytes conservatively bound text tokens; leave room for framing and output.
    if len((system + user).encode("utf-8")) + 768 > settings.context:
        raise ProviderError(
            "Context budget exceeded. Increase SPECTRO_RAG_CONTEXT or shorten input."
        )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if settings.provider == "ollama":
        result = ollama_request(
            settings,
            "/api/chat",
            {
                "model": settings.model,
                "messages": messages,
                "stream": False,
                "format": DocumentAnswer.model_json_schema(),
                "options": {"num_ctx": settings.context, "num_predict": 512},
            },
        )
        if result.get("done_reason") == "length":
            raise ProviderError("Model output exceeded token budget; shorten the question/context")
        message = result.get("message")
        if not isinstance(message, dict):
            raise ProviderError("Ollama returned invalid chat output")
        content = message.get("content", "")
    else:
        model, key = os.getenv("OPENAI_MODEL"), os.getenv("OPENAI_API_KEY")
        if not model or not key:
            raise ProviderError("Set OPENAI_MODEL and OPENAI_API_KEY for document Q&A")
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("Install API support: uv sync --extra rag --extra llm") from None
        try:
            with OpenAI(
                api_key=key,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                timeout=120,
                max_retries=0,
            ) as client:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=512,
                )
            if response.choices[0].finish_reason != "stop":
                raise ValueError
            content = response.choices[0].message.content or ""
        except Exception:
            raise ProviderError(
                "API generation failed. Check endpoint, credentials, model and JSON-mode support."
            ) from None
    try:
        return DocumentAnswer.model_validate_json(content)
    except ValidationError, ValueError, TypeError:
        raise ProviderError("Model returned invalid document-answer JSON") from None


class ProviderError(ValueError):
    """A provider is unknown, incomplete, or deliberately unavailable."""


class _CandidateDraft(TechniqueCandidate):
    spectroscopy_method: str = Field(
        description="Name each specific method; combined methods list every component."
    )
    supporting_evidence: list[Evidence] = Field(default_factory=list, max_length=0)


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
    return compact


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
                    assessment_method="ollama_grounded",
                    assessment_provider="ollama",
                    report_version="grounded-ollama-v1",
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
    method: Literal["offline_fallback", "evidence_abstention"],
    evidence: list[RetrievedEvidence] | None = None,
    evidence_version: str | None = None,
) -> AssessmentReport:
    report = assess(use_case)
    report.candidate_techniques = []
    report.status = status
    report.recommendation_rationale = reason
    report.proposed_feasibility_experiment = FeasibilityPlan(
        objective="Resolve requirements and source support before choosing a technique.",
        experimental_steps=[
            "Confirm the customer target, matrix, operating conditions, and decision need.",
            "Ask a spectroscopy expert to review relevant primary sources and select "
            "a study only if supported.",
        ],
        expected_limitations=[
            "No method or customer-specific performance is established by this assessment."
        ],
        success_criteria=["A source-supported study and customer-defined criteria are agreed."],
        success_actions=["Proceed to a controlled feasibility study after expert review."],
        failure_actions=["Gather relevant evidence or retain the abstention."],
    )
    report.assessment_method = method
    report.assessment_provider = "baseline" if method == "offline_fallback" else "ollama"
    report.report_version = (
        "offline-fallback-v1" if method == "offline_fallback" else "grounded-abstention-v1"
    )
    report.retrieved_evidence = evidence or []
    report.evidence_set_version = evidence_version
    report.assessment_warnings = [reason]
    return report


def _apply_feasibility_safeguards(
    use_case: CustomerUseCase, plan: FeasibilityPlan
) -> tuple[FeasibilityPlan, list[str]]:
    """Keep generated study plans exploratory and customer-criteria bound."""
    safe = plan.model_copy(deep=True)
    stated_criteria = [
        value
        for value in (use_case.required_accuracy, use_case.detection_limit)
        if value and value.strip()
    ] + [value for value in use_case.strict_requirements if value.strip()]
    safe.hypothesis = (
        "Test whether and which source-supported candidate meets customer-defined "
        "needs on independent held-out samples; no method is assumed to be best."
    )
    safe.success_criteria = (
        stated_criteria
        if stated_criteria
        else ["Customer-defined acceptance criteria are TBD and must be agreed before testing."]
    )
    safe.evaluation_metrics = [
        "Compare held-out results with independent reference measurements using "
        "customer-agreed analysis measures; no threshold is assumed."
    ]
    if not safe.reference_measurement:
        safe.reference_measurement = (
            use_case.reference_measurement
            or "Independent reference measurement to be agreed with the customer before testing."
        )
    if not any(
        "held-out" in step.lower() or "holdout" in step.lower() for step in safe.experimental_steps
    ):
        safe.experimental_steps.append(
            "Reserve independent sample batches or process runs for held-out validation."
        )
    if not any("reference" in step.lower() for step in safe.experimental_steps):
        safe.experimental_steps.append(
            "Compare each candidate with independent reference measurements on the same samples."
        )
    safe.success_actions = [
        "Review held-out feasibility results with the customer and domain experts "
        "before deciding next steps."
    ]
    safe.failure_actions = [
        "If customer-defined criteria are not met, investigate sample and process "
        "effects, then revise or stop the feasibility study."
    ]
    caveat = (
        "Source demonstrations do not establish customer or instrument performance; "
        "hardware selection and production integration require separate expert review."
    )
    if caveat not in safe.expected_limitations:
        safe.expected_limitations.append(caveat)
    warnings = [
        "Deterministic engineering-planning safeguards set the hypothesis, metrics, "
        "criteria, and next-step actions; these process guardrails are not scientific validation."
    ]
    if not stated_criteria:
        warnings.append(
            "No customer-defined acceptance criteria were supplied; results remain "
            "exploratory until criteria are agreed."
        )
    return safe, warnings


def assess_use_case(
    use_case: CustomerUseCase,
    *,
    assessor: TechnicalAssessor | None = None,
    retriever: EvidenceRetriever | None = None,
    allow_offline_fallback: bool = True,
) -> AssessmentReport:
    """Assess an existing extracted/corrected case through current provider policy."""
    use_case = use_case.model_copy(deep=True)
    use_case.unknown_or_missing_information = missing_fields(use_case)
    provider = os.getenv("SPECTRO_AI_PROVIDER", "baseline").strip().lower()
    if provider == "baseline":
        return assess(use_case)
    if provider == "openai":
        raise ProviderError("OpenAI assessment is not implemented. No request was sent.")
    if provider != "ollama":
        raise ProviderError(f"Unsupported provider {provider!r}. Use baseline or ollama.")
    try:
        source = retriever or EvidenceRetriever()
        retrieved, version = _verified_retrieval(source, source.search(use_case, limit=8))
        if not retrieved:
            return _abstention_report(
                use_case,
                "No technical source evidence was retrieved; method selection is "
                "outside currently supported evidence.",
                status="outside_supported_scope",
                method="evidence_abstention",
                evidence_version=version,
            )
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
                method="evidence_abstention",
                evidence=retrieved,
                evidence_version=version,
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
                evidence_version=version,
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
            citations = list(dict.fromkeys(proposed.supporting_evidence_ids))
            if not citations or any(evidence_id not in known for evidence_id in citations):
                warnings.append(
                    f"Removed {proposed.spectroscopy_method}: cited evidence ID was "
                    "absent from retrieved sources."
                )
                rejected += 1
                continue
            supporting = [known[evidence_id] for evidence_id in citations]
            if proposed.supporting_evidence:
                warnings.append(
                    f"Removed {proposed.spectroscopy_method}: model-generated source "
                    "claims are not accepted."
                )
                rejected += 1
                continue
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
            candidate_data["supporting_evidence_ids"] = citations
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
            rationale = draft.recommendation_rationale
            plan = draft.proposed_feasibility_experiment
            if rejected:
                rationale = (
                    "Some generated candidates were removed by source checks; only "
                    "remaining listed candidates passed necessary lexical screening. "
                    "Expert review remains required."
                )
                plan = _abstention_report(
                    use_case,
                    rationale,
                    status="insufficient_information",
                    method="evidence_abstention",
                ).proposed_feasibility_experiment
        for item in retrieved:
            if item.review_status not in {"reviewed", "reviewed_for_phase2", "source_reviewed"}:
                warnings.append(
                    f"Source {item.source_id} {item.page_or_section} review status "
                    f"is {item.review_status}."
                )
        plan, plan_warnings = _apply_feasibility_safeguards(use_case, plan)
        warnings.extend(plan_warnings)
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
            assessment_method="ollama_grounded",
            assessment_provider="ollama",
            evidence_set_version=version,
            retrieved_evidence=retrieved,
            report_version="grounded-ollama-v1",
            assessment_warnings=warnings,
        )
    except (ProviderError, ValueError) as exc:
        if not allow_offline_fallback:
            raise ProviderError(f"Grounded assessment failed: {exc}") from None
        report = assess(use_case)
        report.assessment_method = "offline_fallback"
        report.assessment_provider = "baseline"
        report.report_version = "offline-fallback-v1"
        report.assessment_warnings.append(
            f"Grounded assessment unavailable; deterministic baseline used ({exc})."
        )
        return report


def assess_problem(
    text: str,
    extractor: UseCaseExtractor | None = None,
    *,
    allow_offline_fallback: bool = True,
) -> AssessmentReport:
    provider = os.getenv("SPECTRO_AI_PROVIDER", "baseline").strip().lower()
    if provider == "baseline":
        selected = extractor or OfflineUseCaseExtractor()
        return assess(selected.extract(text))
    if provider == "ollama":
        try:
            selected = extractor or OllamaUseCaseExtractor()
            return assess_use_case(
                selected.extract(text), allow_offline_fallback=allow_offline_fallback
            )
        except ProviderError as exc:
            if extractor is not None or not allow_offline_fallback:
                raise
            case = OfflineUseCaseExtractor().extract(text)
            case.extraction_mode = "offline"
            case.extraction_warnings.append(
                f"Ollama extraction unavailable; conservative offline extraction used ({exc})."
            )
            report = assess(case)
            report.assessment_method = "offline_fallback"
            report.assessment_provider = "baseline"
            report.report_version = "offline-fallback-v1"
            report.assessment_warnings.append(
                "Grounded technical assessment was not run for offline-extracted facts."
            )
            return report
    if provider == "openai":
        configured = bool(os.getenv("OPENAI_MODEL")) and bool(os.getenv("OPENAI_API_KEY"))
        detail = "Configuration present." if configured else "OPENAI_MODEL/API key not configured."
        raise ProviderError(
            f"OpenAI adapter is not implemented. {detail} "
            "No request was sent. Set SPECTRO_AI_PROVIDER=baseline for the offline demo."
        )
    raise ProviderError(
        f"Unsupported provider {provider!r}. Use SPECTRO_AI_PROVIDER=baseline or ollama."
    )
