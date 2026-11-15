"""The public page's visit counter (#155).

GoatCounter's count.js ships as a local copy, so the page still runs no code
from elsewhere; the vhost's policy lets it send its beacon to the counter.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.publish import publisher

COUNTER = "https://volgatech-itrc.goatcounter.com/count"


def test_the_page_counts_visits_with_its_own_copy():
    html = (publisher.SITE_SRC / "index.html").read_text(encoding="utf-8")
    assert (f'data-goatcounter="{COUNTER}"' in html
            and 'src="vendor/count.js"' in html), \
        "the page does not count visits with its own copy of the counter"


def test_the_page_fetches_no_counter_from_elsewhere():
    html = (publisher.SITE_SRC / "index.html").read_text(encoding="utf-8")
    assert "gc.zgo.at" not in html, "the page fetches the counter from elsewhere"


def test_the_counter_is_shipped_with_its_licence():
    text = (publisher.SITE_SRC / "vendor" / "count.js").read_text(encoding="utf-8")
    assert "ISC license" in text, "the counter is shipped without its licence"


def test_the_vhost_lets_the_counter_report():
    vhost = (Path(__file__).resolve().parents[1] / "deploy"
             / "public-site.nginx.conf").read_text(encoding="utf-8")
    policies = [line for line in vhost.splitlines()
                if "Content-Security-Policy" in line and "add_header" in line]
    assert policies and all(
        "connect-src 'self' https://volgatech-itrc.goatcounter.com;" in line
        for line in policies), "the vhost's policy stops the counter's beacon"
