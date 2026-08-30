import runpy
from pathlib import Path


def test_quickstart_example_runs(capsys):
    root = Path(__file__).resolve().parents[1]

    runpy.run_path(root / "examples" / "quickstart.py", run_name="__main__")

    output = capsys.readouterr().out
    assert "RSI R815/R704: 4.000" in output
    assert "Wynne CI:" in output
    assert "AlOH 2200 depth:" in output
    assert "Wildfire SAM target label: burned; angle: 0.000000 rad" in output
