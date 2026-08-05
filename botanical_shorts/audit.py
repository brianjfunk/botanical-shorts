"""The pool audit page: every candidate, and which gate judged it.

The selector is lazy -- it stops at the first plate that clears everything, so
the pool it walked past is invisible. All it leaves behind is a count, and a
count cannot tell you whether a gate is saving you from black scan frames or
throwing away good plates. Both look like "98 rejected at download".

So this lays the whole slice out at once, grouped by verdict, with the gate's
own reason under each thumbnail. Reading it is the same act as reviewing a
batch: you look, and you can see immediately whether you agree.

Self-contained like the review page -- thumbnails are data URIs, because it is
published to a static host with a strict content-security policy where a remote
<img> would not load.
"""

from __future__ import annotations

import html
from typing import Any, Sequence

from .review import _thumb_data_uri

# Ordered by what you most need to see. Passes first as the reference point --
# you cannot judge a rejection without knowing what acceptance looks like --
# then the judgement calls, then the mechanical gates, which are the least
# arguable.
GROUPS: list[tuple[str, str]] = [
    ("passed", "Passed every gate"),
    ("vision", "Rejected by the model"),
    ("not inspected", "Cleared the local gates, not sent to the model"),
    ("border", "Rejected: dark border"),
    ("ink", "Rejected: too little ink"),
    ("aspect", "Rejected: too landscape"),
    ("resolution", "Rejected: scan too small"),
    ("download", "Rejected: could not be fetched or decoded"),
    ("vision error", "Vision call failed"),
    ("split", "Spread that could not be cut cleanly"),
    ("licence", "Rejected on licence (never downloaded)"),
]


def _card(i: int, rec: Any) -> str:
    title = html.escape(str(rec.title)[:70])
    reason = html.escape(str(rec.reason)[:200])
    img = (
        f'<img src="{_thumb_data_uri(rec.thumb)}" alt="{title}" loading="lazy">'
        if rec.thumb is not None
        else '<div class="noimg">no image fetched</div>'
    )
    return (
        f'<figure class="card">{img}'
        f"<figcaption><b>{title}</b><small>{reason}</small>"
        f'<code>{html.escape(rec.page_id)}</code></figcaption></figure>'
    )


def render(
    records: Sequence[Any],
    *,
    settings: dict[str, Any],
    groups: list[tuple[str, str]] | None = None,
    lead: str = "",
) -> str:
    """Build the audit page from :class:`pipeline.Audited` records.

    ``groups`` overrides the gate ordering, for pages whose stages are not
    gates at all -- the aspect audit sorts by how wide a plate is.
    """
    GROUPS_IN_USE = groups if groups is not None else GROUPS
    by_stage: dict[str, list[Any]] = {}
    for rec in records:
        by_stage.setdefault(rec.stage, []).append(rec)

    total = len(records) or 1
    sections = []
    # Anything with a stage not in GROUPS still gets shown, at the end, rather
    # than silently vanishing from a page whose whole purpose is completeness.
    seen = {name for name, _ in GROUPS_IN_USE}
    ordered = GROUPS_IN_USE + [(s, s) for s in by_stage if s not in seen]

    for stage, heading in ordered:
        group = by_stage.get(stage)
        if not group:
            continue
        share = 100 * len(group) / total
        cards = "".join(_card(i, r) for i, r in enumerate(group))
        sections.append(
            f'<section><h2>{html.escape(heading)}'
            f'<span class="n">{len(group)} &middot; {share:.0f}%</span></h2>'
            f'<div class="grid">{cards}</div></section>'
        )

    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in settings.items()
    )

    return f"""<title>Pool audit — {len(records)} candidates</title>
<style>
  :root {{ color-scheme: light dark; --bg: #fbf9f4; --fg: #241f16; --dim: #6b6252;
           --line: #e2dbcc; --panel: #fff; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #16130f; --fg: #ece5d8; --dim: #b9ae9b; --line: #3a3227; --panel: #1e1a15; }}
  }}
  :root[data-theme="dark"] {{ --bg: #16130f; --fg: #ece5d8; --dim: #b9ae9b;
                              --line: #3a3227; --panel: #1e1a15; }}
  :root[data-theme="light"] {{ --bg: #fbf9f4; --fg: #241f16; --dim: #6b6252;
                               --line: #e2dbcc; --panel: #fff; }}
  body {{ font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 0; padding: 1rem; background: var(--bg); color: var(--fg); }}
  h1 {{ font-size: 1.25rem; margin: 0 0 .3rem; }}
  p.lead {{ margin: 0 0 1rem; color: var(--dim); max-width: 62ch; }}
  h2 {{ font-size: .95rem; font-weight: 600; margin: 1.75rem 0 .6rem;
        padding-bottom: .35rem; border-bottom: 1px solid var(--line);
        display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }}
  h2 .n {{ font-weight: 400; color: var(--dim); font-variant-numeric: tabular-nums;
           white-space: nowrap; }}
  .grid {{ display: grid; gap: .8rem;
           grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }}
  .card {{ margin: 0; border-radius: 8px; overflow: hidden; background: var(--panel);
           border: 1px solid var(--line); }}
  .card img {{ display: block; width: 100%; height: auto; }}
  .card .noimg {{ padding: 2rem .5rem; text-align: center; color: var(--dim);
                  font-size: .7rem; background: var(--bg); }}
  figcaption {{ font-size: .68rem; padding: .4rem .45rem .5rem; line-height: 1.35; }}
  figcaption small {{ display: block; color: var(--dim); margin: .2rem 0; }}
  figcaption code {{ color: var(--dim); opacity: .7; font-size: .9em; }}
  table {{ border-collapse: collapse; font-size: .8rem; margin-bottom: .5rem; }}
  th {{ text-align: left; font-weight: 500; color: var(--dim); padding: .1rem 1rem .1rem 0; }}
  td {{ font-variant-numeric: tabular-nums; }}
  details {{ margin-bottom: 1rem; }}
  summary {{ cursor: pointer; color: var(--dim); font-size: .85rem; }}
</style>

<h1>Pool audit — {len(records)} candidates</h1>
<p class="lead">{lead or "Every candidate the walk met, in the order it met them, grouped by the gate that judged it. The daily run stops at the first plate in the top group; everything below it is what you never see."}</p>

<details><summary>Thresholds in force for this audit</summary>
<table>{rows}</table></details>

{"".join(sections)}
"""
