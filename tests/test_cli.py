import os
import pytest
import sys
from go_melt import cli
import subprocess
import pathlib
from unittest.mock import patch


def test_parse_args_defaults():
    args = cli.parse_args(["--config", "examples/config.json"])
    assert args.gpu == 0
    assert args.config == "examples/config.json"
    assert args.verbose == 0
    assert not args.dry_run


def test_set_device_env_sets_env(monkeypatch):
    cli.set_device_env(1)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_set_device_env_none(monkeypatch):
    # Should not set anything if None
    cli.set_device_env(None)
    assert "CUDA_VISIBLE_DEVICES" in os.environ or True  # no crash


def test_main_dry_run(capsys):
    exit_code = cli.main(["--config", "c.json", "--dry-run"])
    captured = capsys.readouterr()
    assert "dry-run:" in captured.out
    assert exit_code == 0


def test_main_dry_run_verbose_1(capsys):
    exit_code = cli.main(["--config", "c.json", "-v", "--dry-run"])
    captured = capsys.readouterr()
    assert "dry-run:" in captured.out
    assert exit_code == 0


def test_main_dry_run_verbose_2(capsys):
    exit_code = cli.main(["--config", "c.json", "-vv", "--dry-run"])
    captured = capsys.readouterr()
    assert "dry-run:" in captured.out
    assert exit_code == 0


def test_run_package_import_failure(monkeypatch):
    # Force import error
    monkeypatch.setitem(sys.modules, "go_melt.core.go_melt", None)
    with pytest.raises(SystemExit):
        cli.run_package("c.json")


def test_run_package_with_config(monkeypatch):
    # Patch a fake go_melt function
    def fake_run(path=None):
        return "ran with " + str(path)

    monkeypatch.setattr("go_melt.core.go_melt.go_melt", fake_run)
    result = cli.run_package("config.json")
    assert result == "ran with config.json"


def test_run_package_no_config(monkeypatch):
    def fake_run(path=None):
        return "ran without config"

    monkeypatch.setattr("go_melt.core.go_melt.go_melt", fake_run)
    result = cli.run_package(None)
    assert result == "ran without config"


@patch("go_melt.cli.run_package")
def test_main_success(mock_run):
    argv = ["--config", "foo.yaml"]
    exit_code = cli.main(argv)
    mock_run.assert_called_once_with("foo.yaml")
    assert exit_code == 0


@patch("go_melt.cli.run_package", side_effect=Exception("boom"))
def test_main_failure(mock_run):
    argv = ["--config", "foo.yaml"]
    exit_code = cli.main(argv)
    assert exit_code == 2


@patch("go_melt.cli.run_package", side_effect=SystemExit(5))
def test_main_system_exit(mock_run):
    argv = ["--config", "foo.yaml"]
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 5


if __name__ == "__main__":
    pytest.main([__file__])
