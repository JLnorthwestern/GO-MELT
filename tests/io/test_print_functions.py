import pytest
import numpy as np
import sys
from io import StringIO
from contextlib import redirect_stdout

from go_melt.io.print_functions import printLevelMaxMin


@pytest.fixture(autouse=True)
def patch_sys_exit(monkeypatch):
    # Replace sys.exit with one that raises SystemExit
    monkeypatch.setattr(
        sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code))
    )


def test_valid_values():
    Ls = [
        {},  # dummy first element
        {"T0": np.array([10, 20, 30])},
        {"T0": np.array([5, 15, 25])},
    ]
    Lnames = ["Level1", "Level2"]

    buf = StringIO()
    with redirect_stdout(buf):
        printLevelMaxMin(Ls, Lnames)
    output = buf.getvalue()
    assert "Level1: [10.00, 30.00]" in output
    assert "Level2: [5.00, 25.00]" in output


def test_invalid_values_nan():
    Ls = [
        {},
        {"T0": np.array([np.nan, 20, 30])},
    ]
    Lnames = ["Level1"]

    buf = StringIO()
    with pytest.raises(SystemExit):
        with redirect_stdout(buf):
            printLevelMaxMin(Ls, Lnames)
    output = buf.getvalue()
    assert "Terminating program" in output


def test_invalid_values_too_low():
    Ls = [
        {},
        {"T0": np.array([-5, 20, 30])},
    ]
    Lnames = ["Level1"]

    buf = StringIO()
    with pytest.raises(SystemExit):
        with redirect_stdout(buf):
            printLevelMaxMin(Ls, Lnames)
    output = buf.getvalue()
    assert "Terminating program" in output


def test_invalid_values_too_high():
    Ls = [
        {},
        {"T0": np.array([10, 2e6])},
    ]
    Lnames = ["Level1"]

    buf = StringIO()
    with pytest.raises(SystemExit):
        with redirect_stdout(buf):
            printLevelMaxMin(Ls, Lnames)
    output = buf.getvalue()
    assert "Terminating program" in output
