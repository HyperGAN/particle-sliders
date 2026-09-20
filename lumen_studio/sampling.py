"""Sampling schedules and gates, using immutable EMA snapshots through Studio."""
from .api import Generation, generation_payloads
from .dataset import compile_manifest


def selection_rows(variation):
    rows = [r for r in compile_manifest("dev")["rows"] if r["variation"] == variation]
    # Six training lighting definitions plus both unseen paraphrases as bare
    # prompts. All four held-out development characters remain represented.
    return rows[:6] + rows[8:10]


def enqueue_cadence(store, identity, variation, checkpoint, step):
    # Keep pilot qualification and all selection grids. Intermediate full-run
    # checkpoints still get saved/probed, without interrupting training to render.
    if step not in (20, 100, 200) and (step <= 0 or step % 400):
        return []
    manifest = compile_manifest("dev")
    rows = [r for r in manifest["rows"] if r["variation"] == variation]
    if step == 20:
        selected, energies, size = [rows[0], rows[9]], [0., 1.], 512
    elif step % 400 == 0:
        selected, energies, size = selection_rows(variation), [0., .25, .5, 1.], 768
    else:
        # Two fixed cases, two rotating cases, including bare prompts.
        offset = 2 + ((step // 100 - 1) * 2) % (len(rows) - 2)
        selected = rows[:2] + [rows[offset], rows[2 + (offset - 1) % (len(rows) - 2)]]
        energies, size = [0., .25, .5, 1.], 512
    payloads = []
    for row in selected:
        request = Generation(prompt=row["neutral"], seed=row["seeds"][0], width=size, height=size,
            mix={variation: 1.}, checkpoints={variation: checkpoint})
        for value in generation_payloads(request, store, identity, energies):
            value.update(case=row["id"], definition=row["definition"], character=row["character"],
                         sampling_step=step, purpose="development", bare=row["bare"],
                         manifest_sha256=manifest["sha256"])
            payloads.append(value)
    return store.enqueue(payloads)


def reference_payloads(identity, split="train"):
    manifest = compile_manifest(split)
    rows = manifest["rows"]
    # Inspect all definitions on more than one identity before approving a manifest.
    result = []
    for row in rows:
        for side in ("neutral", "positive"):
            result.append(dict(model="anima-turbo-v1.1", prompt=row[side], seed=row["seeds"][0], width=512, height=512,
                steps=10, cfg=1, energy=0., mix={}, strengths=dict(candlelit=0., moonlit=0., theatrical=0.),
                checkpoints={}, model_identity=identity, case=row["id"], character=row["character"],
                definition=row["definition"], reference_side=side, purpose="definition_qualification"))
            result[-1]["manifest_sha256"] = manifest["sha256"]
    return result


def select_checkpoint(candidates):
    valid = [c for c in candidates if c["step"] % 400 == 0 and c["cases"] > 0
             and c["atmosphere_wins"] / c["cases"] >= .75
             and c["major_regressions"] / c["cases"] <= .1]
    if not valid:
        return None
    return min(valid, key=lambda c: (-c["atmosphere_wins"] / c["cases"],
                                    c["major_regressions"] / c["cases"], c["residual"]))
