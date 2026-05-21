from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


def test_schaefer_dry_run_reports_unavailable_source(capsys) -> None:
    module = _load_script()

    result = module.run(
        Namespace(
            atlas="schaefer100",
            tribev2_package_path="./tribev2",
            output=None,
            allow_fetch=False,
            dry_run=True,
            bilateral=False,
        )
    )
    output = capsys.readouterr().out

    assert result == 2
    assert "No verified direct Schaefer fsaverage5 mapping" in output


def test_hcp_generation_requires_explicit_fetch() -> None:
    module = _load_script()

    try:
        module.create_hcp_mmp_rows(
            tribev2_package_path="./tribev2",
            allow_fetch=False,
            bilateral=False,
        )
    except RuntimeError as exc:
        assert "--allow-fetch" in str(exc)
    else:
        raise AssertionError("HCP generation should require --allow-fetch")


def test_default_output_path_uses_atlas_name() -> None:
    module = _load_script()

    assert module.default_output_path("hcp_mmp") == Path("data/parcellations/fsaverage5_hcp_mmp.csv")


def _load_script():
    path = Path("scripts/create_fsaverage5_parcellation.py")
    spec = importlib.util.spec_from_file_location("create_fsaverage5_parcellation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["create_fsaverage5_parcellation"] = module
    spec.loader.exec_module(module)
    return module
