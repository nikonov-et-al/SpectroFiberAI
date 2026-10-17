"""Validated expert corrections; no provider or UI dependency."""

from spectro_ai.models import CUSTOMER_FACT_FIELDS, CustomerUseCase

EDITABLE_FIELDS = tuple(
    field
    for field in CUSTOMER_FACT_FIELDS
    if CustomerUseCase.model_fields[field].annotation == str | None
)


def correct_requirements(case: CustomerUseCase, edits: dict[str, str | None]) -> CustomerUseCase:
    """Apply only changed fields, retaining raw input and untouched provenance."""
    if edits.keys() - set(EDITABLE_FIELDS):
        raise ValueError("Only requirement fields can be corrected")
    values = case.model_dump()
    values.update(edits)
    corrected = CustomerUseCase.model_validate(values)
    for field in edits:
        if getattr(corrected, field) != getattr(case, field):
            corrected.field_provenance[field] = "expert edit"
            corrected.field_quotes.pop(field, None)
    return corrected
