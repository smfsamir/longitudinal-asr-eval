import json

import pandas as pd
import pytest

from longitudinal_asr.demo import demo
from longitudinal_asr.runner import run


def test_demo_resume_and_stale_audio(tmp_path):
    output = tmp_path / "demo"
    demo(output)
    summary = pd.read_csv(output / "analysis/summary.csv").iloc[0]
    assert summary.minority_md == pytest.approx(0.375)
    assert summary.standard_md == pytest.approx(0.25)
    config = json.loads((output / "config.json").read_text())
    spec = config["models"][0]
    stats = run(output / "samples.jsonl", spec, output / spec["id"])
    assert stats["new"] == 0
    assert stats["completed"] == 8
    # Same path, different audio must not be treated as a completed run.
    with (output / "tone.wav").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        run(output / "samples.jsonl", spec, output / spec["id"])


def test_import_does_not_load_models():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import longitudinal_asr.models; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
