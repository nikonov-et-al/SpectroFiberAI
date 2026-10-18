"""Download local Docling CPU layout and table models.

Run with ``uv run --extra rag python scripts/setup_docling.py``. OCR assets and
remote conversion services are not used by this setup command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / ".cache" / "docling" / "models"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_model_dir(),
        help="Local model directory (default: .cache/docling/models).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download existing model files.",
    )
    return parser.parse_args(argv)


def _docling_api() -> tuple[Any, Any, Any, Any, Any]:
    """Load optional Docling APIs only when setup runs."""
    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.models.stages.table_structure.table_structure_model import (
            TableStructureModel,
        )
        from docling.models.utils.hf_model_download import download_hf_model
    except ImportError as exc:
        raise RuntimeError(
            "Docling setup requires the RAG extra; run "
            "`uv run --extra rag python scripts/setup_docling.py`."
        ) from exc
    return (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
        TableStructureModel,
        download_hf_model,
    )


def _download_assets(output_dir: Path, *, force: bool = False) -> None:
    (
        accelerator_device,
        accelerator_options,
        pdf_pipeline_options,
        table_model,
        download_hf_model,
    ) = _docling_api()

    options = pdf_pipeline_options(
        accelerator_options=accelerator_options(device=accelerator_device.CPU)
    )
    layout = options.layout_options
    model = layout.model_spec
    if getattr(layout.engine_options.engine_type, "value", layout.engine_options.engine_type) != (
        "transformers"
    ):
        raise RuntimeError("Expected Docling's Transformers layout model for CPU setup.")

    output_dir.mkdir(parents=True, exist_ok=True)
    layout_dir = output_dir / model.repo_id.replace("/", "--")
    table_dir = output_dir / table_model._model_repo_folder
    download_hf_model(
        repo_id=model.repo_id,
        revision=model.revision,
        local_dir=layout_dir,
        force=force,
        progress=True,
    )
    table_model.download_models(
        local_dir=table_dir,
        force=force,
        progress=True,
    )
    _require_files("CPU layout", layout_dir, ("config.json", "model.safetensors"))
    _require_files(
        "TableFormer accurate",
        table_dir,
        (
            "model_artifacts/tableformer/accurate/tm_config.json",
            "model_artifacts/tableformer/accurate/tableformer_accurate.safetensors",
        ),
    )


def _require_files(label: str, directory: Path, relative_paths: tuple[str, ...]) -> None:
    missing = [relative for relative in relative_paths if not (directory / relative).is_file()]
    if missing:
        raise RuntimeError(
            f"{label} model incomplete in {directory}; missing {', '.join(missing)}. "
            "Check download access, then rerun with --force."
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.resolve()
    try:
        _download_assets(output_dir, force=args.force)
    except RuntimeError as exc:
        print(
            f"Docling CPU model setup failed for {output_dir}: {exc}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            f"Docling model download failed for {output_dir}: {exc}\n"
            "Check network access to Hugging Face and rerun the command.",
            file=sys.stderr,
        )
        return 1

    print(f"Docling CPU layout and table models ready in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
