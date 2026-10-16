"""chirp_config.py: generated as the ionozond console does, read as its sounder does.

The sounder (``rx_dechirp.py``) reads only this file, once, at start. The Qt
console regenerates it from ``schedule.ini`` and ``config.ini`` every time one
of its dialogs is closed, so a web edit has to go into those two files and
then regenerate this one exactly as the console would -- otherwise the next
dialog closed on the laptop undoes it. :func:`build` is a port of ionozond's
``buildChirpConfig`` (``src/configwriter.cpp``), held to the original binary's
own output byte for byte. :func:`parsed` applies ``rx_dechirp.load_config``'s
rules, so a file the sounder could not read is refused before it is written.
"""

from __future__ import annotations

import ast

from .qsettings import Ini

TRAILER = ("\ndef get_all_sounders():\n"
           "    sounder_list = []\n"
           "    for sounder_thread in sounders:\n"
           "        for sounder in sounder_thread:\n"
           "            sounder_list.append(sounder)\n"
           "    return(sounder_list)")


class UnreadableConfig(ValueError):
    """A chirp_config.py the sounder would refuse at start."""


def build(schedule: Ini, config: Ini) -> str:
    """The text ``buildChirpConfig`` writes from the console's two files."""
    receiver = ""
    entries = []
    for group in schedule.groups():
        if _text(schedule, group, "rx").lower() == "true":
            receiver = "rx_station = {'name':'%s','lat':%s,'lon':%s}" % (
                group, _text(schedule, group, "lat"), _text(schedule, group, "lon"))
        if _true(_text(schedule, group, "active")):
            entries.append("[{'name':'%s'%s}]" % (group, "".join(
                ",'%s':%s" % (key, _python(schedule.value(group, key))
                              + ("e3" if key in ("cf", "rate") else ""))
                for key in schedule.keys(group) if key != "active")))
    lines = ["sounders = [\n"]
    if entries:
        lines += [",\n".join(entries), "\n"]
    lines += ["]\n", "\n", receiver + "\n", "\n"]
    for key in config.keys(""):
        if key in ("config_file", "sound_app", "if_rate"):
            continue
        value = _python(config.value("", key))
        if key in ("data_dir", "nas_dest"):
            value = '"%s"' % value
        elif key == "sample_rate":
            value += "e3"
        lines.append("%s = %s\n" % (key, value))
        if key == "sample_rate":
            lines.append("if_rate = sample_rate/dec\n")
    lines.append(TRAILER)
    return "".join(lines)


def parsed(text: str) -> dict:
    """What ``rx_dechirp.load_config`` reads from ``text``, or the reason it would not.

    Literal assignments only, never executed. An empty ``sounders`` is refused
    too: ``rx_dechirp.main`` exits on it, and systemd would restart the sounder
    into the same exit every minute.
    """
    if not text.strip():
        raise UnreadableConfig("chirp_config.py would be empty")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise UnreadableConfig("chirp_config.py would not be valid Python: %s at "
                               "line %s" % (exc.msg, exc.lineno)) from None
    config, opaque = {}, []
    for node in tree.body:
        if (not isinstance(node, ast.Assign) or len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name)):
            continue
        try:
            config[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            opaque.append(node.targets[0].id)
    if "sample_rate" in config and "dec" in config:
        config["if_rate"] = config["sample_rate"] / config["dec"]
    if "sounders" not in config:
        raise UnreadableConfig(
            "chirp_config.py would define 'sounders' as something other than a "
            "literal list" if "sounders" in opaque
            else "chirp_config.py would have no 'sounders'")
    if not [s for thread in config["sounders"] for s in thread]:
        raise UnreadableConfig("chirp_config.py would have no active transmitter, "
                               "and the sounder exits at start without one")
    return config


def _text(ini: Ini, group: str, key: str) -> str:
    """``QSettings::value(key).toString()``, which is empty for a missing key."""
    try:
        return ini.value(group, key)
    except KeyError:
        return ""


def _true(text: str) -> bool:
    """``QVariant(QString).toBool()``."""
    return text.lower() not in ("", "0", "false")


def _python(text: str) -> str:
    """configwriter's ``pythonValue``: booleans capitalised, the rest as written."""
    if text.lower() == "true":
        return "True"
    if text.lower() == "false":
        return "False"
    return text
