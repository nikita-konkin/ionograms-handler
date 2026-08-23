"""Message catalogs, one module per language.

Plain dicts rather than gettext catalogs -- see `services.api.i18n` for why.
Both modules must carry identical key sets; `tests/test_i18n.py` enforces it,
because a Russian catalog that silently falls behind an edited English one is
the failure mode this arrangement is most exposed to.
"""

from __future__ import annotations

from . import en, ru

__all__ = ["en", "ru"]
