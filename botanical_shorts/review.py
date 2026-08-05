"""The batch review page: a self-contained grid for accepting or rejecting plates.

Every automated gate in this pipeline is a proxy for a judgement someone would
make by looking. The gates handle what is mechanically wrong -- blank plates,
black scan frames, letterpress, scanner rulers -- and they are cheap and
deterministic. What is left over is genuinely ambiguous: whether a photograph
of a turtle belongs on a channel of engravings, whether a skull diagram reads
as natural history illustration, whether a lionfish tagged under Reptiles is a
problem. Those are judgements, and a minute of a person's attention settles
them better than any prompt.

So this renders the residue as one page: tap to reject, copy a short code,
hand it back. It embeds every thumbnail as a data URI because the page must
work as a single file with no network access -- it is published to a static
host with a strict content-security policy, and a remote <img> would simply
not load.
"""

from __future__ import annotations

import base64
import html
import io
from typing import Any, Sequence

from PIL import Image

# Wide enough to judge a plate, small enough that a hundred of them stay well
# inside the page-size limit. At quality 72 a thumbnail lands around 25-35kB.
THUMB_WIDTH = 300
THUMB_QUALITY = 72


def _thumb_data_uri(img: Image.Image) -> str:
    work = img.convert("RGB")
    height = max(1, round(work.height * THUMB_WIDTH / work.width))
    work = work.resize((THUMB_WIDTH, height), Image.LANCZOS)
    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=THUMB_QUALITY, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render(entries: Sequence[dict[str, Any]], images: Sequence[Image.Image]) -> str:
    """Build the review page for one batch.

    ``entries`` are the batch manifest records; ``images`` the framed plates in
    the same order. Rejection is by index into that order, which is why the
    manifest is written before the page and never reordered afterwards.
    """
    if len(entries) != len(images):
        raise ValueError("entries and images must be the same length")

    cards = []
    for i, (entry, img) in enumerate(zip(entries, images)):
        title = html.escape(str(entry.get("title", "")))
        citation = html.escape(str(entry.get("citation", ""))[:160])
        cards.append(
            f'<figure class="card" data-i="{i}" onclick="t({i})">'
            f'<img src="{_thumb_data_uri(img)}" alt="{title}" loading="lazy">'
            f'<span class="mark">✕</span>'
            f"<figcaption><b>{i + 1}.</b> {title}<small>{citation}</small></figcaption>"
            f"</figure>"
        )

    return f"""<title>Batch review — {len(entries)} plates</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 1rem; background: #fbf9f4; color: #241f16; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #16130f; color: #ece5d8; }}
    .card figcaption {{ color: #b9ae9b; }}
    .bar {{ background: #241f16; border-color: #3a3227; }}
    textarea {{ background: #0f0d0a; color: #ece5d8; border-color: #3a3227; }}
  }}
  :root[data-theme="dark"] body {{ background: #16130f; color: #ece5d8; }}
  :root[data-theme="light"] body {{ background: #fbf9f4; color: #241f16; }}
  h1 {{ font-size: 1.1rem; font-weight: 600; margin: 0 0 .25rem; }}
  p.lead {{ margin: 0 0 1rem; opacity: .75; }}
  .grid {{ display: grid; gap: .75rem;
           grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }}
  .card {{ margin: 0; cursor: pointer; position: relative; border-radius: 8px;
           overflow: hidden; background: #fff2; transition: opacity .12s; }}
  .card img {{ display: block; width: 100%; height: auto; }}
  .card .mark {{ position: absolute; inset: 0; display: none; align-items: center;
                 justify-content: center; font-size: 3rem; color: #fff;
                 background: #b0303099; font-weight: 700; }}
  .card.out {{ opacity: .45; }}
  .card.out .mark {{ display: flex; }}
  .card figcaption {{ font-size: .72rem; padding: .35rem .45rem .5rem;
                      color: #6b6252; }}
  .card figcaption small {{ display: block; opacity: .8; margin-top: .15rem; }}
  .bar {{ position: sticky; bottom: 0; margin-top: 1rem; padding: .75rem;
          background: #fbf9f4; border-top: 1px solid #e2dbcc; }}
  textarea {{ width: 100%; box-sizing: border-box; font-family: ui-monospace, monospace;
              font-size: .9rem; padding: .5rem; border: 1px solid #d6cdb9;
              border-radius: 6px; background: #fff; }}
  button {{ font: inherit; padding: .5rem .9rem; border-radius: 6px;
            border: 1px solid #8a7f68; background: #8a7f68; color: #fff;
            cursor: pointer; margin-top: .5rem; }}
</style>

<h1>Batch review — {len(entries)} plates</h1>
<p class="lead">Tap any plate to reject it. Then copy the line at the bottom and send it back.
Rejected plates are never offered again.</p>

<div class="grid">{"".join(cards)}</div>

<div class="bar">
  <textarea id="out" rows="2" readonly></textarea>
  <button onclick="c()">Copy</button>
  <span id="msg" style="margin-left:.5rem;opacity:.7"></span>
</div>

<script>
  const rejected = new Set();
  function t(i) {{
    const el = document.querySelector('[data-i="' + i + '"]');
    if (rejected.has(i)) {{ rejected.delete(i); el.classList.remove('out'); }}
    else {{ rejected.add(i); el.classList.add('out'); }}
    render();
  }}
  function render() {{
    const list = [...rejected].sort((a, b) => a - b).map(i => i + 1);
    // 1-based to match the caption numbers, so the code is checkable by eye.
    document.getElementById('out').value =
      list.length ? 'reject ' + list.join(',') : 'approve all';
  }}
  function c() {{
    const out = document.getElementById('out');
    navigator.clipboard.writeText(out.value).then(
      () => {{ document.getElementById('msg').textContent = 'copied'; }},
      () => {{ out.removeAttribute('readonly'); out.select(); }}
    );
  }}
  render();
</script>
"""
