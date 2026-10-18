import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spectro_ai_setup_docling", ROOT / "scripts" / "setup_docling.py"
)
assert SPEC and SPEC.loader
setup_docling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup_docling)


def test_setup_defaults_to_repo_local_model_directory():
    args = setup_docling._parse_args([])
    assert args.output_dir == setup_docling._default_model_dir()
    assert args.force is False


def test_setup_cli_accepts_custom_path_and_force(tmp_path):
    output_dir = tmp_path / "models"
    args = setup_docling._parse_args(["--output-dir", str(output_dir), "--force"])
    assert args.output_dir == output_dir
    assert args.force is True


def test_download_selects_only_cpu_transformer_layout_and_table_assets(monkeypatch, tmp_path):
    calls = []

    class Device:
        CPU = "cpu"

    class Accelerator:
        def __init__(self, *, device):
            assert device == Device.CPU

    class PipelineOptions:
        def __init__(self, *, accelerator_options):
            self.layout_options = SimpleNamespace(
                engine_options=SimpleNamespace(engine_type=SimpleNamespace(value="transformers")),
                model_spec=SimpleNamespace(
                    repo_id="docling-project/docling-layout-heron", revision="resolved-revision"
                ),
            )

    class TableModel:
        _model_repo_folder = "docling-project--docling-models"

        @staticmethod
        def download_models(*, local_dir, force, progress):
            calls.append(("table", local_dir, force, progress))
            accurate = local_dir / "model_artifacts" / "tableformer" / "accurate"
            accurate.mkdir(parents=True)
            (accurate / "tm_config.json").touch()
            (accurate / "tableformer_accurate.safetensors").touch()

    def download_layout(**kwargs):
        calls.append(("layout", kwargs))
        kwargs["local_dir"].mkdir(parents=True)
        (kwargs["local_dir"] / "config.json").touch()
        (kwargs["local_dir"] / "model.safetensors").touch()

    monkeypatch.setattr(
        setup_docling,
        "_docling_api",
        lambda: (Device, Accelerator, PipelineOptions, TableModel, download_layout),
    )

    setup_docling._download_assets(tmp_path / "models", force=True)

    assert calls == [
        (
            "layout",
            {
                "repo_id": "docling-project/docling-layout-heron",
                "revision": "resolved-revision",
                "local_dir": tmp_path / "models" / "docling-project--docling-layout-heron",
                "force": True,
                "progress": True,
            },
        ),
        (
            "table",
            tmp_path / "models" / "docling-project--docling-models",
            True,
            True,
        ),
    ]
    assert not any("ocr" in str(call).lower() for call in calls)


def test_setup_reports_missing_docling_with_install_command(monkeypatch, capsys, tmp_path):
    def missing_api():
        raise RuntimeError("Docling setup requires the RAG extra")

    monkeypatch.setattr(setup_docling, "_docling_api", missing_api)
    assert setup_docling.main(["--output-dir", str(tmp_path / "models")]) == 1
    assert "Docling setup requires the RAG extra" in capsys.readouterr().err


def test_download_refuses_non_transformers_layout_engine(monkeypatch, tmp_path):
    class Device:
        CPU = "cpu"

    class Accelerator:
        def __init__(self, *, device):
            pass

    class PipelineOptions:
        def __init__(self, *, accelerator_options):
            self.layout_options = SimpleNamespace(
                engine_options=SimpleNamespace(engine_type="onnxruntime"),
                model_spec=SimpleNamespace(repo_id="model", revision="main"),
            )

    class UnusedTableModel:
        _model_repo_folder = "table"

    monkeypatch.setattr(
        setup_docling,
        "_docling_api",
        lambda: (Device, Accelerator, PipelineOptions, UnusedTableModel, lambda **_: None),
    )
    with pytest.raises(RuntimeError, match="Transformers layout model"):
        setup_docling._download_assets(tmp_path)


def test_download_rejects_incomplete_local_checkpoints(monkeypatch, tmp_path):
    class Device:
        CPU = "cpu"

    class Accelerator:
        def __init__(self, *, device):
            pass

    class PipelineOptions:
        def __init__(self, *, accelerator_options):
            self.layout_options = SimpleNamespace(
                engine_options=SimpleNamespace(engine_type="transformers"),
                model_spec=SimpleNamespace(repo_id="layout/repo", revision="main"),
            )

    class TableModel:
        _model_repo_folder = "table/repo"

        @staticmethod
        def download_models(*, local_dir, force, progress):
            local_dir.mkdir(parents=True)

    monkeypatch.setattr(
        setup_docling,
        "_docling_api",
        lambda: (Device, Accelerator, PipelineOptions, TableModel, lambda **_: None),
    )
    with pytest.raises(RuntimeError, match="CPU layout model incomplete"):
        setup_docling._download_assets(tmp_path)
