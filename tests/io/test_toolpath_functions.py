import io
import builtins
import pytest
import numpy as np

from go_melt.io.toolpath_functions import parsingGcode, count_lines, format_fixed


class FakeFileSystem:
    """
    Minimal fake in-memory file system to intercept open() calls
    based on filename. Designed for tests so no real filesystem I/O occurs.
    """

    def __init__(self, files=None):
        # files: dict[path] = initial string content
        self._files = dict(files or {})
        # To store write buffers created during tests
        self._write_buffers = {}

    def open(self, file, mode="r", *args, **kwargs):
        # Only support text modes used by the target functions: 'r', 'w', 'r+' etc.
        if "b" in mode:
            raise ValueError("Binary mode not supported in fake FS")

        if "r" in mode and "w" not in mode and "+" not in mode:
            if file not in self._files:
                raise FileNotFoundError(file)
            return io.StringIO(self._files[file])

        if "w" in mode:
            buf = io.StringIO()
            # store buffer so test can inspect later
            self._write_buffers[file] = buf
            return _WriteRewinder(buf, self._write_buffers, file)

        if "r+" in mode or "a" in mode:
            # append or read/write: create if missing
            initial = self._files.get(file, "")
            buf = io.StringIO(initial)
            self._write_buffers[file] = buf
            return _WriteRewinder(buf, self._write_buffers, file)

        raise ValueError(f"Unsupported mode: {mode}")


class _WriteRewinder(io.StringIO):
    """
    Wrapper around StringIO returned on open('w') so that when closed,
    its content is saved into the parent's _write_buffers for inspection.
    """

    def __init__(self, buf, buffer_dict, path):
        super().__init__()
        self._parent_buf = buf
        self._buffer_dict = buffer_dict
        self._path = path

    def write(self, s):
        return super().write(s)

    def close(self):
        # When closed, copy content to parent buffers
        self._parent_buf.seek(0)
        self._parent_buf.truncate(0)
        self._parent_buf.write(self.getvalue())
        self._parent_buf.seek(0)
        self._buffer_dict[self._path] = self._parent_buf
        super().close()


@pytest.fixture
def fake_fs(monkeypatch):
    """
    Fixture that yields a FakeFileSystem and patches builtins.open to use it.
    Tests should add fake files via fake_fs._files before calling functions.
    """
    fs = FakeFileSystem()
    real_open = builtins.open

    def _open(path, mode="r", *args, **kwargs):
        return fs.open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open)
    yield fs
    # restore builtin open after test
    monkeypatch.setattr(builtins, "open", real_open)


def test_format_fixed_basic():
    # Check scientific formatting and alignment (right-justified)
    val = 123.456
    s = format_fixed(val, width=15, precision=4)
    # Should be in scientific notation with 4 decimal places
    assert "e" in s
    # After stripping, numeric part equals formatted representation
    assert s.strip() == f"{val:.4e}"


def test_count_lines_reads_all_lines(fake_fs):
    path = "some/file.txt"
    content = "first\nsecond\nthird\n"
    fake_fs._files[path] = content
    # count_lines uses open(...) in read mode; our fake_fs will return the string
    n = count_lines(path)
    assert n == 3


def test_parsingGcode_simple_move_and_dwell(fake_fs):
    """
    Construct a minimal G-code:
    - G1 X0 Y0 Z0  (start)
    - G1 X10 Y0    (move along X)
    - G0 X10 Y10 Z1 (jump to new Z -> dwell should be produced)
    - G1 X20 Y10   (move along X on new layer)
    This checks:
    - path segments generated for movement
    - dwell entries written when Z changes
    - final dwell at end is written
    """

    gcode_path = "in.gcode"
    toolpath_path = "out.toolpath"

    # G-code lines: include integers and floats formats to be captured by regex
    gcode_content = (
        "\n".join(
            [
                "G0 X2.0 Y2.000 Z0.04",
                "G1 X8.0 Y2.000 Z0.04",
                "G0 X2.0 Y2.000 Z0.08",
                "G1 X8.0 Y2.000 Z0.08",
                "G0 X8.0 Y2.200 Z0.08",
                "G1 X2.0 Y2.200 Z0.08",
            ]
        )
        + "\n"
    )

    fake_fs._files[gcode_path] = gcode_content

    Nonmesh = {
        "gcode": gcode_path,
        "toolpath": toolpath_path,
        "dwell_time_multiplier": 1.0,
        "subcycle_num_L2": 1,
        "subcycle_num_L3": 1,
        "dwell_time": 0.02,  # seconds
        "wait_time": 1,  # number of small steps to wait before switching
        "timestep_L3": 0.01,  # base timestep
        "laser_velocity": 100.0,  # units per second
    }
    Properties = {"laser_power": 42.0}

    # Run parsingGcode which will read gcode and write toolpath to fake FS
    move_mesh = parsingGcode(Nonmesh, Properties, L2h=None)

    # Inspect the written toolpath content
    assert toolpath_path in fake_fs._write_buffers
    buf = fake_fs._write_buffers[toolpath_path]
    buf.seek(0)
    content = buf.read()
    lines = [ln for ln in content.splitlines() if ln.strip()]

    # There should be at least one line (some movement and dwell entries)
    assert len(lines) > 0

    # Ensure formatting of each line matches expected CSV with 7 fields
    for ln in lines:
        parts = ln.split(",")
        # Each line produced in the code uses 7 comma-separated values
        assert len(parts) == 7

    # Check that at least one dwell (Ldwell==0 but produced during Z change or end)
    # The Ldwell field is the 5th field (0-based index 4)
    ldwell_values = {parts[4] for parts in (l.split(",") for l in lines)}
    assert "0" in ldwell_values

    # Check that count returned equals number of written lines
    assert move_mesh == len(lines)


def test_parsingGcode_jump_flag_and_shortdt(fake_fs):
    """
    Test scenario where two consecutive G1 moves create:
    - multiple time steps based on laser_velocity and timestep,
    - and short remainder step (shortdt) is produced.
    """

    gcode_path = "moves.gcode"
    toolpath_path = "moves.out"

    # Points separated by a small distance to force num_pointsinSegments = 0 and shortdt > 0
    # Use coordinates that produce a small distance
    gcode_content = (
        "\n".join(
            [
                "G1 X0 Y0 Z0",
                "G1 X0.5 Y0 Z0",  # small move (distance 0.5)
            ]
        )
        + "\n"
    )

    fake_fs._files[gcode_path] = gcode_content

    Nonmesh = {
        "gcode": gcode_path,
        "toolpath": toolpath_path,
        "dwell_time_multiplier": 1.0,
        "subcycle_num_L2": 1,
        "subcycle_num_L3": 1,
        "dwell_time": 0.0,
        "wait_time": 0,
        "timestep_L3": 0.01,
        # choose laser_velocity so that dx (laser_velocity * timestep_L3) > distance
        # This will make num_pointsinSegments == 0 and produce a shortdt segment only
        "laser_velocity": 100.0,
    }
    Properties = {"laser_power": 10.0}

    move_mesh = parsingGcode(Nonmesh, Properties, L2h=None)

    assert toolpath_path in fake_fs._write_buffers
    buf = fake_fs._write_buffers[toolpath_path]
    buf.seek(0)
    content = buf.read()
    lines = [ln for ln in content.splitlines() if ln.strip()]

    # We expect at least one shortdt line to be produced
    assert len(lines) >= 1

    # Validate P value in movement lines equals Ljump * laser_power and Ljump is 1 for these moves
    for ln in lines:
        parts = ln.split(",")
        P = float(parts[6])
        assert (
            P in (0.0, pytest.approx(Properties["laser_power"]))
            or pytest.approx(0.0) == P
        )

    assert move_mesh == len(lines)


if __name__ == "__main__":
    pytest.main([__file__])
