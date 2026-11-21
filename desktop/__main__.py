"""桌面入口：python -m desktop"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import webview

from desktop.bridge import Bridge, UI_DIR

log = logging.getLogger("homepilot.desktop")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    bridge = Bridge()
    index_html = (UI_DIR / "index.html").resolve()
    window = webview.create_window(
        "HomePilot",
        url=index_html.as_uri(),
        js_api=bridge,
        width=960,
        height=640,
        min_size=(800, 520),
    )
    try:
        webview.start(debug=False)
    finally:
        bridge.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())