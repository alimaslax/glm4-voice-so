"""Copy docs/media/chapters.json into the <script id="chapter-data"> block of
docs/index.html, so the page works from file:// too (no fetch involved).

Run by docs/anim/render.sh; safe to run again at any time.
"""

import json
import os
import re

here = os.path.dirname(os.path.abspath(__file__))
page = os.path.join(here, "..", "index.html")
data = json.load(open(os.path.join(here, "..", "media", "chapters.json")))

html = open(page).read()
block = json.dumps(data, separators=(",", ":"))
new, n = re.subn(
    r'(<script type="application/json" id="chapter-data">).*?(</script>)',
    lambda m: m.group(1) + block + m.group(2),
    html,
    flags=re.S,
)
if n != 1:
    raise SystemExit("chapter-data block not found in docs/index.html")
open(page, "w").write(new)
print("==> chapters injected:", " · ".join(c["title"] for c in data["chapters"]))
