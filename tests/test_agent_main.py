"""The agent's command line: what an operator runs on the station by hand."""

from __future__ import annotations

import random

from test_agent_ionozond import laptop, systemd

from services.agent import __main__ as cli
from services.agent import ionozond
from services.agent.qsettings import Ini
from services.agent.sounderconfig import build


def station(tmp_path, monkeypatch):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    path = config.console_settings / "config.ini"
    path.write_bytes(path.read_bytes().replace(
        b"whiten_n=20000", b"whiten_n=%d" % random.randint(20001, 90000)))
    monkeypatch.setattr(cli, "_config", lambda args: config)
    return config


def generated(config) -> str:
    console = config.console_settings
    return build(Ini.read(console / "schedule.ini"), Ini.read(console / "config.ini"))


def test_regenerate_writes_the_file_once_the_operator_confirms(tmp_path, monkeypatch):
    config = station(tmp_path, monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    cli.main(["ionozond", "regenerate"])
    assert (config.live_dir / "chirp_config.py").read_text(encoding="utf-8") == (
        generated(config)), "the command line did not regenerate a confirmed change"


def test_regenerate_writes_nothing_when_the_operator_declines(tmp_path, monkeypatch):
    config = station(tmp_path, monkeypatch)
    before = (config.live_dir / "chirp_config.py").read_bytes()
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    cli.main(["ionozond", "regenerate"])
    assert (config.live_dir / "chirp_config.py").read_bytes() == before, (
        "the command line regenerated a change the operator declined")


def test_regenerate_with_yes_asks_nothing(tmp_path, monkeypatch):
    station(tmp_path, monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt: (_ for _ in ()).throw(
        AssertionError("asked")))
    assert cli.main(["ionozond", "regenerate", "--yes"]) == 0, (
        "the command line with --yes did not regenerate without asking")


def test_regenerate_shows_what_the_sounder_would_read_differently(
        tmp_path, monkeypatch, capsys):
    station(tmp_path, monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    cli.main(["ionozond", "regenerate"])
    assert "whiten_n" in capsys.readouterr().out, (
        "the command line did not show what the sounder would read differently")


def test_regenerate_on_an_undrifted_station_says_there_is_nothing_to_do(
        tmp_path, monkeypatch, capsys):
    systemd(monkeypatch)
    config = laptop(tmp_path)
    monkeypatch.setattr(cli, "_config", lambda args: config)
    cli.main(["ionozond", "regenerate", "--yes"])
    assert "nothing to regenerate" in capsys.readouterr().out, (
        "the command line did not say an undrifted station has nothing to regenerate")


def test_a_refused_regeneration_exits_with_failure(tmp_path, monkeypatch):
    station(tmp_path, monkeypatch)
    monkeypatch.setattr(ionozond, "regenerate", lambda config, params: [
        ionozond.control.CommandResult(ionozond.REGENERATE, False, "refused")])
    assert cli.main(["ionozond", "regenerate", "--yes"]) == 1, (
        "a refused regeneration did not exit with failure")
