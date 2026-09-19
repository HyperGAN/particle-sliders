"""Freeze fixtures and checkpoint hashes, then import the existing blind shortlist."""
from __future__ import annotations

import json
from pathlib import Path
import secrets
import sys
import time

import yaml

from study import DATA, ROOT, WORK, add_stage, config_id, digest, read, save, sha, write


# Every lyric sheet is new. Musical contexts vary while the requested treatment
# stays soft transients, a relaxed pocket and light tape warmth.
SPECS = [
    ("Muted guitar · 88 BPM", 88, "A low male solo lead with precise, conversational singing.",
     "Muted electric guitar chords, picked bass and a compact acoustic drum kit.",
     "The guitar opens into sustained chords in the chorus; bass and kit stay distinct.",
     """I found the screw beneath the table
And set it by the wooden leg
You held the lamp above my shoulder
I turned it till the chair was steady
Now pull it closer to the counter
There is a bowl for each of us
The kettle clicks against the silence
We have enough to talk about"""),
    ("Fingerpicked guitar · 72 BPM", 72, "A female solo alto, gently sung with clear consonants.",
     "Fingerpicked steel-string guitar, upright bass and light brushes.",
     "A second guitar adds a high countermelody in the chorus; the lead stays solo.",
     """A little rain got through the canvas
We moved the boxes from the wall
The labels curled around the corners
You read the smallest ones aloud
Keep all the cups inside the kitchen
Leave both the coats beside the door
The room will take a while to settle
We have not lived here long before"""),
    ("Piano trio · 104 BPM", 104, "A female solo soprano with brisk, melodic phrasing.",
     "Bright upright piano, walking electric bass and a crisp snare-led drum kit.",
     "The piano moves to wider voicings in the chorus and the drums add a restrained fill.",
     """The morning shift has left the building
A yellow glove is on the rail
I turn the last light by the doorway
And hear the trolley on the tiles
You meet me underneath the awning
With half a sandwich in a bag
We split it on the empty staircase
And watch the first bus coming back"""),
    ("Synth bass · 116 BPM", 116, "A male solo tenor, clean melodic singing with long vowels.",
     "Short synth chords, a moving synth bass and straight electronic kick and snare.",
     "A higher synth countermelody enters at the chorus without replacing the bass pattern.",
     """We marked the route across the paper
And left the map beside the phone
The bridge was closed before the crossing
We took the longer road back home
I know the turn beside the orchard
I know the shop with faded green
You keep a finger on the paper
For all the places in between"""),
    ("Organ and rim clicks · 68 BPM", 68, "A female solo contralto, measured singing with gentle breaths.",
     "Sustained organ chords, fretless electric bass and sparse rim-click percussion.",
     "The organ adds an upper harmony in the chorus; the percussion remains sparse.",
     """Your reading glasses face the ceiling
The open book is on your knee
A piece of thread has caught the binding
You ask me what the time might be
I leave the curtains where you put them
And move the cup beyond your sleeve
The chapter ends beside a drawing
You turn the page and start to read"""),
    ("Bass and hand percussion · 98 BPM", 98, "A male solo baritone with syncopated melodic phrasing.",
     "A syncopated electric bass riff, clipped keyboard chords, shaker and hand drums.",
     "A clean guitar response enters in the chorus; retain the syncopated bass riff.",
     """The flour settles on the radio
You brush a hand across the dial
The dough is rising near the window
We leave it covered for a while
Make room beside the little oven
Put all the trays along the wall
The first one out is for the neighbors
We keep the pieces that are small"""),
    ("Clean guitar band · 132 BPM", 132, "A female solo lead with a firm, energetic melodic delivery.",
     "Interlocking clean electric guitars, driving picked bass and a straight live drum beat.",
     "The guitars play a short repeated hook in the chorus with stronger snare accents.",
     """The lift is stuck above the landing
We take the stairs up two at once
You carry half the folded railing
I keep the screws inside a cup
We put it down beside the doorway
And test the latch against the frame
The work will last until the evening
Tomorrow we can paint again"""),
    ("Vibraphone trio · 80 BPM", 80, "A male solo tenor, light singing with clear words and short phrases.",
     "Vibraphone melody, upright bass and a small brush-and-kick drum setup.",
     "The vibraphone adds a repeated answer between vocal lines in the chorus.",
     """The meter needs another quarter
I search the lining of my coat
You hold the door against the weather
And balance change upon a note
We have enough to wash the blankets
We have enough to dry them through
I take the chair beside the dryer
And leave the other one for you"""),
    ("Acoustic guitar and pad · 92 BPM", 92, "A female solo mezzo, intimate singing with a clear natural tone.",
     "Strummed acoustic guitar, a quiet sustained synth pad, electric bass and a simple kick and hat pattern.",
     "The guitar shifts to wider strums in the chorus; the pad remains behind the solo voice.",
     """The garden tap has started dripping
We put a bucket underneath
The little dog is in the doorway
And watches everything we reach
You pass the wrench across the paving
I turn the fitting half around
We wait a minute by the bucket
And listen for another sound"""),
    ("Electric piano pocket · 124 BPM", 124, "A male solo lead, rhythmic melodic singing without rap.",
     "Rhythmic electric piano, melodic electric bass and a precise four-on-the-floor drum pattern.",
     "A short octave piano hook enters in the chorus while the pulse stays steady.",
     """A strip of tape divides the hallway
The painter leaves a narrow space
We step across it with the groceries
And put the bags beside the sink
The wall is almost dry by supper
The tape comes off in crooked strips
You hold a corner near the cupboard
I catch the pieces as they slip"""),
    ("Mandolin and upright bass · 108 BPM", 108, "A female solo alto with direct, unadorned melodic phrasing.",
     "Picked mandolin, acoustic guitar, upright bass and a light snare and tambourine pattern.",
     "The mandolin moves to a higher register in the chorus; guitar and bass keep the same pulse.",
     """The footpath bends behind the houses
A fallen branch is in the way
We lift it over by the fence post
And carry on beside the stream
Your boots are wet above the laces
My coat is folded on my arm
The houses show between the hedges
We hear a gate before the farm"""),
    ("Sparse piano ballad · 76 BPM", 76, "A male solo baritone with sustained, expressive singing and distinct words.",
     "Sparse grand piano, long electric bass notes and occasional soft kick and snare.",
     "Piano arpeggios fill the chorus while the drums remain spacious and the singer stays solo.",
     """The picture leans against the dresser
We try a nail above the bed
You stand beside the open cupboard
And tell me when the frame is straight
I take a step toward the doorway
You move the lamp a little more
The picture fits above the blanket
We leave the hammer on the floor"""),
]

LONG = [
    ("Full song · brushed guitar quartet", 84, "A female solo alto with warm but articulate singing.",
     "Fingerpicked guitar, electric piano, upright bass and brushed drums.",
     "The chorus opens into strummed guitar and wider piano chords. A brief bridge thins to voice and bass, then the last chorus resolves into a short instrumental ending.",
     """We took the small room by the courtyard
The lock was stiff against the key
A folded chair stood in the corner
You pushed it closer with your knee
Set down the bag beside the window
Leave room enough to cross the floor
We have a table for the evening
And morning waiting at the door
The courtyard filled with empty bottles
A neighbor called down from the stairs
We passed a box across the railing
And borrowed two more wooden chairs
The light above the stove was broken
We ate beneath the window glass
You drew a square upon the paper
For where a longer shelf could pass"""),
    ("Full song · moving bass and keys", 112, "A male solo tenor with rhythmic and intelligible singing.",
     "Rhythmic electric piano, moving electric bass, clean guitar responses and a live drum kit.",
     "A repeated guitar hook lifts the chorus. The bridge breaks down to bass and voice; the final chorus returns, then the instruments end together after a short outro.",
     """The delivery came before the opening
We rolled the crates across the tiles
You checked the list beside the doorway
I stacked the paper in a pile
Turn up the light above the counter
Put out the board beside the street
There is a place behind the window
For everyone we hope to meet
The first order was a pot of coffee
The second came with soaking shoes
You found a towel beneath the counter
And asked which table they would choose
At closing time we counted teaspoons
And pulled the board in from the rain
The chalk had run below the letters
You wiped it clean to write again"""),
    ("Full song · restrained synth arrangement", 96, "A female solo mezzo with precise consonants and long melodic lines.",
     "Plucked synth chords, rounded synth bass, an airy sustained pad and a restrained electronic drum pattern.",
     "The chorus adds a higher synth melody. The bridge leaves space around the voice, then the final chorus resolves with a clear cadence and a quiet instrumental tail.",
     """We drove until the market opened
And parked behind a loading bay
You had a list inside your wallet
I had the change from yesterday
Keep one hand free to hold the railing
The stairs are narrow at the turn
We bring the empty bottles with us
And take the longer way to return
The seller wrapped the bowl in paper
You held it steady on your knees
I drove around the road repair work
And stopped beside the row of trees
At home we spread the paper open
And put the bowl upon the shelf
The empty space beside the kettle
Had found a purpose for itself"""),
]


def make_row(spec, long=False):
    title, bpm, voice, arrangement, chorus, text = spec
    lines = text.splitlines()
    if long:
        lyrics = "[verse]\n" + "\n".join(lines[:4]) + "\n[chorus]\n" + "\n".join(lines[4:8])
        lyrics += "\n[verse]\n" + "\n".join(lines[8:12]) + "\n[bridge]\n" + "\n".join(lines[12:])
        lyrics += "\n[chorus]\n" + "\n".join(lines[4:8]) + "\n[outro]"
    else:
        lyrics = "[verse]\n" + "\n".join(lines[:4]) + "\n[chorus]\n" + "\n".join(lines[4:])
    common = f"BPM {bpm}. Meter: 4/4. Keep the stated tempo, solo singer and core instrumentation."
    neutral = f"Global Metadata:\nA contemporary song. {common} A balanced studio mix with clear transients and defined bass.\n\nVocal Details:\n{voice}\n\nArrangement:\nVerse: {arrangement}\nChorus: {chorus}"
    positive = f"Global Metadata:\nA lo-fi treatment of this song. {common} An intimate mellow mix, gently softened transients and light tape saturation. Keep the lead intelligible, the bass rounded and the drum pocket relaxed with a gentle swing.\n\nVocal Details:\n{voice}\n\nArrangement:\nVerse: {arrangement}\nChorus: {chorus}"
    return dict(title=title, brief=f"{bpm} BPM. {voice} {arrangement} {chorus}",
                row=dict(neutral=neutral, positive=positive, lyrics=lyrics))


def main():
    if (DATA / "state.json").exists():
        print("Existing study retained; no assignments or fixtures changed")
        return
    sys.path.insert(0, str(ROOT.parent))
    from app.rewriter import _artist_name_hit
    fresh = [make_row(s) for s in SPECS]
    full = [make_row(s, True) for s in LONG]
    monitor_path = ROOT / "analysis/uni16_20260906/prompts/prompts-lofi-uni-16-v1-eval.yaml"
    monitoring_rows = yaml.safe_load(monitor_path.read_text())["rows"]
    training = yaml.safe_load((ROOT / "analysis/fresh_seed_20260910/train-one-prompt.yaml").read_text())["rows"]
    existing_lyrics = {r["lyrics"].strip() for r in [*training, *monitoring_rows]}
    all_new = [f["row"]["lyrics"].strip() for f in [*fresh, *full]]
    assert len(set(all_new)) == 15 and not set(all_new) & existing_lyrics
    for r in [*training, *monitoring_rows, *[f["row"] for f in [*fresh, *full]]]:
        assert not _artist_name_hit("", json.dumps(r)), "Prompt name validation failed"
    configs = {"off":dict(kind="off", scale=0.), "caption":dict(kind="caption", scale=0.)}
    weight_paths = {
        600: ROOT / "models/uni16-gan-v1/lofi-warmup600-a02/lofi-warmup600-a02_last.safetensors",
        2000: ROOT / "analysis/fresh_seed_stress_20260910/lofi-fresh-stress/lofi-fresh-stress_step2000.safetensors",
        3000: ROOT / "analysis/fresh_seed_break_20260911/lofi-fresh-break/lofi-fresh-break_step3000.safetensors",
        3400: ROOT / "analysis/fresh_seed_break_20260911/lofi-fresh-break/lofi-fresh-break_step3400.safetensors",
    }
    for step, path in weight_paths.items():
        meta = read(path.with_suffix(".json"))
        assert not _artist_name_hit("", json.dumps(meta)), "Checkpoint metadata name validation failed"
        for scale in (.5, .75, 1.):
            configs[config_id(step, scale)] = dict(kind="adapter", step=step, scale=scale,
                weights=str(path), sha256=sha(path), metadata=meta)
    monitoring = []
    for row in (2, 3):
        for variation, seed in enumerate((101, 303), 1):
            monitoring.append(dict(title=f"Monitoring arrangement {row-1}", variation=variation,
                brief=monitoring_rows[row]["neutral"], row=monitoring_rows[row], seed=seed,
                duration=20., source_row=row))
    confirmation = [dict(f, variation=i+1, seed=seed, duration=20.) for f in fresh
                    for i, seed in enumerate((1709, 2903, 4517))]
    long_fixtures = [dict(f, variation=i+1, seed=seed, duration=90.) for f in full
                     for i, seed in enumerate((6823, 7919))]
    protocol = dict(version=1, created=time.time(), concept="Lo-fi", checkpoints=configs,
        monitoring=monitoring, confirmation=confirmation, full=long_fixtures,
        style="Soft transients, a relaxed gently swung pocket, mellow harmony and light tape warmth; intelligible singing.",
        preview="Two-pass EBU R128 normalization, -18 LUFS / -2 dBTP, MP3 320 kbps. Original WAV retained.",
        physical_gpu=1, seed_retries=0, accept_short=True, accept_silent=True,
        selection="Blind ratings on reused monitoring cases; choose two distinct checkpoints, then one strength for each. Freeze both before any confirmation rendering.",
        assessment="Style, enjoyment and words rated independently, each 1 to 5. No automated quality score selects a checkpoint. New-song results describe one listener on 12 prompt clusters; three seeds are repeated measurements. Full-song checks use three additional prompts.",
        limits="Initial cases were heard before. Blinding cannot erase prior familiarity. LM arrangement changes are allowed. No release or publication of weights is automatic.")
    write(DATA / "protocol.json", protocol)
    (DATA / "confirmation-prompts.yaml").write_text(yaml.safe_dump(dict(rows=[f["row"] for f in fresh]), sort_keys=False))
    (DATA / "full-song-prompts.yaml").write_text(yaml.safe_dump(dict(rows=[f["row"] for f in full]), sort_keys=False))
    state = dict(study_id="lofi-release-"+secrets.token_hex(4), protocol_hash=digest(protocol),
        configs=configs, monitoring=monitoring, confirmation=confirmation, full=long_fixtures,
        clips={}, stages={}, ratings={}, selections={}, events=[])
    add_stage(state, "shortlist", monitoring, ["off", "caption", *[config_id(s) for s in weight_paths]])
    scores = {}
    for step, file in [(600, "fresh_seed_20260910/scores.json"),
                        (2000, "fresh_seed_stress_20260910/scores-2000.json"),
                        (3000, "fresh_seed_break_20260911/scores-3000.json"),
                        (3400, "fresh_seed_break_20260911/scores-3400.json")]:
        for record in read(ROOT / "analysis" / file)["records"]:
            if record["checkpoint"]["steps"] != step:
                continue
            assert record["checkpoint"]["sha256"] == configs[config_id(step)]["sha256"]
            row = int(Path(record["candidate"]["audio"]).parent.parent.name.split("-")[-1])
            scores[step, row, record["seed"]] = record
    for clip in state["clips"].values():
        cfg, fixture = configs[clip["config"]], clip["fixture"]
        key = "candidate" if cfg["kind"] == "adapter" else "positive_reference" if cfg["kind"] == "caption" else "baseline"
        source = scores[cfg.get("step", 600), fixture["source_row"], fixture["seed"]][key]
        assert sha(source["audio"]) == source["sha256"]
        if key != "candidate":
            assert len({scores[s, fixture["source_row"], fixture["seed"]][key]["sha256"] for s in weight_paths}) == 1
        clip["source"] = dict(path=source["audio"], sha256=source["sha256"])
    save(state, "created")
    write(DATA / "preflight.json", dict(checkpoint_hashes_checked=True, source_audio_hashes_checked=True,
        reference_hashes_match=True, names_checked=True, new_lyric_sheets_disjoint=True,
        new_prompt_count=12, long_prompt_count=3, protocol_hash=state["protocol_hash"]))
    print(f"Frozen study: {len(state['clips'])} existing clips; 36 new-song cases and 6 full-song cases reserved")


if __name__ == "__main__":
    main()
