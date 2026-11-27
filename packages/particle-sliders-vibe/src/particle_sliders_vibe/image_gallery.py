"""A scrollable record of each native starting image and its three treatments."""
from html import escape
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote

from . import read_json


def teacher_gallery(root, vibe, attempts, kept):
    root = Path(root)
    audit = root / "teacher-structure.json"
    additional = read_json(audit) if audit.exists() else {}
    page = [
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Teacher selections</title><style>'
        'body{font:16px/1.4 system-ui,sans-serif;margin:1rem auto;padding:0 1rem;max-width:86rem}'
        'section{margin:2rem 0}h2{font-size:1.2rem;margin:0}.grid{display:grid;'
        'grid-template-columns:repeat(4,minmax(0,1fr));gap:.7rem}figure{margin:0}'
        'img{width:100%;height:auto;display:block}figcaption{margin:.3rem 0;font-size:.9rem}'
        'strong,small{display:block}.kept{color:#17602e}.broken{color:#9a2723}'
        '@media(max-width:700px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}'
        '</style><main>',
        f'<h1>{escape(vibe)}</h1><p>{len(kept)} scenes qualified.</p>',
    ]
    for attempt in attempts:
        scene, number = attempt["scene"], attempt["attempt"]
        stem = f"{scene}-{number}.json"
        bank = read_json(root / "candidates" / stem)
        statuses = {**additional, **attempt.get("structure", {})}
        page.append(f'<section><h2>{escape(scene.replace("-", " ").title())}</h2>'
                    f'<p>Round {number + 1} · Seed {attempt["seed"]}</p><div class="grid">')
        off = bank["neutral"]
        tiles = [("Starting image", off, "", False)]
        for index, option in enumerate(bank["options"], 1):
            clause = option["positive"].split(", alone, ", 1)[-1]
            clause = clause.partition(", standing in ")[0].partition(", sitting on a stool in ")[0]
            tiles.append((f'Candidate {index}', option["image"], clause,
                          option["positive"] == attempt["chosen"]))
        for label, ref, prompt, selected in tiles:
            url = quote(os.path.relpath(ref["path"], root), safe="/")
            state = statuses.get(ref["sha256"])
            status = {"fail":"Broken image", "unmeasured":"Structure uncertain"}.get(state)
            if selected:
                status = "Selected"
            if label != "Starting image" and status is None:
                status = "Skipped"
            css = "kept" if selected else "broken" if state == "fail" else ""
            caption = f"{label} · {status}" if status else label
            page.append(f'<figure><a href="{url}"><img loading="lazy" alt="{escape(label)}" '
                        f'src="{url}"></a><figcaption><strong class="{css}">{escape(caption)}</strong>'
                        f'{escape(prompt)}</figcaption></figure>')
        page.append('</div></section>')
    page.append("</main></html>")
    with NamedTemporaryFile("w", dir=root, encoding="utf-8", delete=False) as handle:
        handle.write("\n".join(page))
    result = root / "teachers.html"
    os.replace(handle.name, result)
    return result
