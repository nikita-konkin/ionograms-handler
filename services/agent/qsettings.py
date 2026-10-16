"""The ionozond console's settings files, in Qt's own INI dialect.

The Qt console keeps its schedule and parameters in ``schedule.ini`` and
``config.ini`` under ``~/.config/dsChirp``, written by ``QSettings::IniFormat``.
That dialect is not configparser's: values may be quoted and escaped, a bare
comma makes a list, a leading ``@`` makes a typed value, ``[General]`` is the
root group and ``[%General]`` a group of that name. A configparser round trip
would change what the console reads back.

So this reads the bytes the way Qt 5's ``readIniLine``,
``iniUnescapedKey`` and ``iniUnescapedStringList`` do, and edits by replacing
the bytes of one value and nothing else. It writes only values ``QSettings``
itself would write without quoting or escaping -- numbers, ``true`` and
``false``, and names -- which is everything the web console sends. Anything it
would have to guess at, it refuses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: What ``QSettings`` writes as it is, unquoted and unescaped.
PLAIN = re.compile(r"^[A-Za-z0-9_.+-]+$")

#: A group or key name ``QSettings`` writes without percent escapes.
NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

SPACE = b" \t\r\n"
SPECIAL = b'\n\r";=\\'
ESCAPES = {ord("a"): "\a", ord("b"): "\b", ord("f"): "\f", ord("n"): "\n",
           ord("r"): "\r", ord("t"): "\t", ord("v"): "\v", ord('"'): '"',
           ord("?"): "?", ord("'"): "'", ord("\\"): "\\"}
HEX = b"0123456789abcdefABCDEF"


class IniError(ValueError):
    """A value this module will not guess at, or will not write."""


@dataclass(frozen=True)
class Entry:
    """One ``key=value`` line: where it sits, and where its value starts."""

    group: str
    key: str
    start: int
    value: int
    end: int


@dataclass(frozen=True)
class Ini:
    """A QSettings INI file, held as the bytes the console wrote."""

    data: bytes

    @classmethod
    def read(cls, path: str | Path) -> Ini:
        return cls(Path(path).read_bytes())

    def groups(self) -> list[str]:
        """The top-level groups, sorted as ``QSettings::childGroups`` lists them."""
        return sorted({e.group for e in self._sections() if e.group})

    def keys(self, group: str) -> list[str]:
        """The keys of one group, ``""`` for the root, sorted as ``childKeys``."""
        return sorted({e.key for e in self._entries() if e.group == group})

    def value(self, group: str, key: str) -> str:
        """The value ``QSettings::value(...).toString()`` would give.

        The last of two equal keys wins, as in ``QSettings``. A list or a typed
        ``@`` value is refused: the console would write either into
        ``chirp_config.py`` as something the sounder cannot read.
        """
        found = [e for e in self._entries() if e.group == group and e.key == key]
        if not found:
            raise KeyError("%s/%s" % (group, key) if group else key)
        text, listed = _unescaped(self.data[found[-1].value:found[-1].end])
        if listed:
            raise IniError("%s is a list in the console's settings; a single "
                           "value was expected" % key)
        if text.startswith("@@"):
            return text[1:]
        if text.startswith("@"):
            raise IniError("%s holds a typed QSettings value (%s), not text"
                           % (key, text[:24]))
        return text

    def with_value(self, group: str, key: str, text: str) -> Ini:
        """The same file with one value replaced, or added where it is absent.

        Only the bytes of that value change. A new key goes after the last line
        of its group, a new group at the end of the file, and a root key into
        ``[General]`` -- which is put first, where ``QSettings`` puts it, when
        the file has none.
        """
        if not PLAIN.match(text):
            raise IniError("%r would need quoting or escaping in the console's "
                           "settings, which this agent does not write" % text)
        if not NAME.match(key) or not (group == "" or NAME.match(group)):
            raise IniError("%r/%r is not a name the console writes as it is"
                           % (group, key))
        if group.lower() == "general":
            raise IniError("a group named General is written as %General; "
                           "this agent does not write one")
        found = [e for e in self._entries() if e.group == group and e.key == key]
        if len(found) > 1:
            raise IniError("%s appears %d times in its group; which one the "
                           "console reads is not worth guessing" % (key, len(found)))
        if found:
            e = found[0]
            return Ini(self.data[:e.value] + text.encode("ascii") + self.data[e.end:])
        return self._added(group, key, text)

    def _added(self, group: str, key: str, text: str) -> Ini:
        newline = b"\r\n" if b"\r\n" in self.data else b"\n"
        line = key.encode("ascii") + b"=" + text.encode("ascii") + newline
        lines = [e for e in self._sections() if e.group == group]
        if lines:
            at = _after(self.data, max(e.end for e in lines))
            if at < 0:
                return Ini(self.data + newline + line)
            return Ini(self.data[:at] + line + self.data[at:])
        if group == "":
            return Ini(b"[General]" + newline + line + newline + self.data)
        head = b"" if not self.data or self.data.endswith(newline) else newline
        gap = newline if self.data else b""
        return Ini(self.data + head + gap + b"[" + group.encode("ascii") + b"]"
                   + newline + line)

    def _entries(self) -> list[Entry]:
        return [e for e in self._sections() if e.key]

    def _sections(self) -> list[Entry]:
        """Every logical line as an Entry; a section header has an empty key."""
        out = []
        group = ""
        for start, end, equals in _lines(self.data):
            if self.data[start:start + 1] == b"[":
                close = self.data.find(b"]", start)
                raw = self.data[start + 1:close if -1 < close < end else end].strip()
                if raw.lower() == b"general":
                    group = ""
                elif raw.lower() == b"%general":
                    group = raw[1:].decode("latin-1")
                else:
                    group = _unescaped_key(raw)
                out.append(Entry(group, "", start, end, end))
                continue
            if equals < 0:
                continue
            stop = equals
            while stop > start and self.data[stop - 1:stop] in (b" ", b"\t"):
                stop -= 1
            key = _unescaped_key(self.data[start:stop])
            if key:
                out.append(Entry(group, key, start, equals + 1, end))
        return out


def _after(data: bytes, end: int) -> int:
    """Where the line holding ``end`` stops, past its line break; -1 if none."""
    at = data.find(b"\n", end)
    return at + 1 if at >= 0 else -1


def _lines(data: bytes):
    """Qt 5's ``readIniLine``: ``(start, end, equals)`` for each logical line."""
    pos, n = 0, len(data)
    while True:
        start = pos
        while start < n and data[start] in SPACE:
            start += 1
        i, quoted, equals = start, False, -1
        while i < n:
            ch = data[i]
            if ch not in SPECIAL:
                i += 1
                continue
            i += 1
            if ch == ord("="):
                if not quoted and equals == -1:
                    equals = i - 1
            elif ch in b"\n\r":
                if i == start + 1:
                    start += 1
                elif not quoted:
                    i -= 1
                    break
            elif ch == ord("\\"):
                if i < n:
                    first = data[i]
                    i += 1
                    if i < n and {first, data[i]} == {10, 13}:
                        i += 1
            elif ch == ord('"'):
                quoted = not quoted
            elif i == start + 1:
                while i < n and data[i] not in b"\n\r":
                    i += 1
                while i < n and data[i] in SPACE:
                    i += 1
                start = i
            elif not quoted:
                i -= 1
                break
        pos = i
        if i - start <= 0:
            return
        yield start, i, equals


def _unescaped_key(raw: bytes) -> str:
    """Qt 5's ``iniUnescapedKey``: ``%XX`` and ``%UXXXX``, and ``\\`` as ``/``."""
    out, i, n = [], 0, len(raw)
    while i < n:
        ch = raw[i]
        if ch == ord("\\"):
            out.append("/")
            i += 1
            continue
        if ch != ord("%") or i == n - 1:
            out.append(chr(ch))
            i += 1
            continue
        first = i + 2 if raw[i + 1] == ord("U") else i + 1
        digits = 4 if raw[i + 1] == ord("U") else 2
        code = raw[first:first + digits]
        if first + digits > n or not all(c in HEX for c in code):
            out.append("%")
            i += 1
            continue
        out.append(chr(int(code, 16)))
        i = first + digits
    return "".join(out)


def _unescaped(raw: bytes) -> tuple[str, bool]:
    """Qt 5's ``iniUnescapedStringList``: the text, and whether it is a list."""
    out: list[str] = []
    listed = quoted = was_quoted = False
    i, n = _skipped(raw, 0), len(raw)
    chop = 0
    while i < n:
        ch = raw[i]
        if ch == ord("\\"):
            i += 1
            if i >= n:
                return "".join(out), listed
            code = raw[i]
            i += 1
            if code in ESCAPES:
                out.append(ESCAPES[code])
            elif code == ord("x") or code in b"01234567":
                base, digits = (16, HEX) if code == ord("x") else (8, b"01234567")
                begin = i if code == ord("x") else i - 1
                if code == ord("x") and (i >= n or raw[i] not in HEX):
                    if i >= n:
                        return "".join(out), listed
                    chop = len(out)
                    continue
                stop = begin
                while stop < n and raw[stop] in digits:
                    stop += 1
                out.append(chr(int(raw[begin:stop], base) & 0xFFFF))
                i = stop
                if i >= n:
                    return "".join(out), listed
            elif code in b"\n\r" and i < n and raw[i] in b"\n\r" and raw[i] != code:
                i += 1
            chop = len(out)
            continue
        if ch == ord('"'):
            i += 1
            was_quoted, quoted = True, not quoted
            if not quoted:
                i = _skipped(raw, i)
                chop = len(out)
            continue
        if ch == ord(",") and not quoted:
            listed, out, was_quoted, chop = True, [], False, 0
            i = _skipped(raw, i + 1)
            continue
        stop = i + 1
        while stop < n and raw[stop] not in b'\\",':
            stop += 1
        out.extend(raw[i:stop].decode("latin-1"))
        i = stop
    if not was_quoted:
        while len(out) > chop and out[-1] in (" ", "\t"):
            out.pop()
    return "".join(out), listed


def _skipped(raw: bytes, i: int) -> int:
    while i < len(raw) and raw[i] in b" \t":
        i += 1
    return i
