"""Character-vibe score and native-resolution tile protocol from Anima PR 38."""
from io import BytesIO
from PIL import Image

VERSION = "final-boss-working-v4"
URL = "https://api.nano-gpt.com/api/v1/decisions"
MODEL = "inception/mercury-decide"
# Four 256-pixel tiles cover a 512-pixel image exactly. Every image is read the same way:
# the reader refuses a PNG above about 180 KB, and tiles of this size stay under it.
TILE = 256
# A larger tile would have to be read as four smaller native pieces, which reads more leniently.
TILE_BYTES = 170_000
PIECE = 160
# Quality falls from one at the OFF image's own reading to zero this far above it.
SPAN = 0.30
STATE = "This is an unscaled crop from an anime illustration."
QUESTIONS = {
    "broken_drawing": {
        "type": "score",
        "instructions": (
            "How cleanly is this crop drawn? A finished drawing has continuous outlines, smooth deliberate "
            "shading and shapes that read as real things such as fabric, armor, skin, hair, walls or sky. A "
            "broken drawing shatters into jagged shards or facets with no meaning, has stray streaks of "
            "color, a smeared, missing or mask-like face, or smears forms together. Plain flat areas, and "
            "deliberate sharp designs such as spikes on armor, are fine when their outlines are clean."
        ),
        "criteria": [
            "Cleanly drawn throughout",
            "Mostly clean with a small jagged or smeared spot",
            "Noticeable shards, streaks or smears in part of the crop",
            "Shards, streaks or smears across most of the crop",
            "Fragments only; nothing is drawn properly",
        ],
    }
}


def _png(picture, box):
    buffer = BytesIO()
    picture.crop(box).save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def tiles(path, size=TILE):
    """Cover the image with native tiles, each a list of PNG pieces. Pixels are never resampled."""
    with Image.open(path) as source:
        picture = source.convert("RGB")
    width, height = picture.size
    if min(width, height) < size:
        raise ValueError("Image is smaller than one native tile")
    out = []
    for top in sorted({min(y, height - size) for y in range(0, height, size)}):
        for left in sorted({min(x, width - size) for x in range(0, width, size)}):
            whole = _png(picture, (left, top, left + size, top + size))
            if len(whole) <= TILE_BYTES:
                out.append([whole])
                continue
            out.append([_png(picture, (x, y, x + PIECE, y + PIECE))
                        for y in (top, top + size - PIECE) for x in (left, left + size - PIECE)])
    return out



# Read on the whole image by the paired judge. Tiles cannot see whether a figure holds together.
FIGURE_STATE = "Look at the central character of this anime illustration."
FIGURE_QUESTIONS = {
    "figure": {
        "type": "noul",
        "instructions": (
            "Does the picture show one character whose head and body are both clearly drawn and connected? "
            "A helmet, mask or hood counts as a head when it sits on a body. Answer near zero when the head "
            "is missing, when the figure is an empty garment, robe or collar with no head, when it is a "
            "tree-like, flame-like or abstract mass with no person, when a beard or hair merges into the "
            "clothing as one shape, or when part of the face is covered by an accidental dark blot."
        ),
    }
}


# Asked once per image with its OFF image. Only one_figure is scored; the others are reported.
PAIR_STATE = (
    "Image 1 is the original illustration of one person. Image 2 is an edit meant to give that "
    "same person a final-boss design while keeping the picture."
)
PAIR_QUESTIONS = {
    "same_person": {
        "type": "noul",
        "instructions": (
            "Is the central character of image 2 recognizably the same individual as in image 1, with "
            "their face and hair still visible, now wearing a different design? Answer near zero when the "
            "face is hidden, missing or replaced, when the person has become a faceless cloak, creature, "
            "statue, silhouette or abstract shape, or when the head can no longer be found."
        ),
    },
    "one_figure": {
        "type": "noul",
        "instructions": (
            "Does image 2 show exactly one central character, as image 1 does? Answer near zero when "
            "there are two or more figures, a duplicated or split figure, or no clear figure at all."
        ),
    },
    "same_scene": {
        "type": "noul",
        "instructions": (
            "Is image 2 still set in the same place as image 1, with the same surroundings visible "
            "behind the character? A change of costume, pose or camera distance is fine. Answer near zero "
            "when the surroundings were replaced, washed in one color or lost."
        ),
    },
}


# Asked with the OFF image. A scene washed in one new color counts against the image.
RECOLOR_STATE = "Image 1 is the original illustration. Image 2 is an edit meant to redesign only the central character."
RECOLOR_QUESTIONS = {
    "recolored": {
        "type": "noul",
        "instructions": (
            "Compared with image 1, has the edit recolored the surroundings as well as the character, so that "
            "the walls, floor, sky or light took on one new color cast such as red or purple? Answer near zero "
            "when the surroundings keep the colors and lighting of image 1."
        ),
    }
}


def quality(broken, off_broken, one_figure, figure):
    """One for a finished drawing of one whole figure; zero when any of those fails."""
    if None in (broken, off_broken, one_figure, figure):
        return None
    drawn = 1 - min(1.0, max(0.0, broken - off_broken) / SPAN)
    return min(drawn, one_figure, figure)


# Anchors of the Boss distinction question: ordinary clothing, and a distinctive endgame antagonist.
BOSS_FLOOR = 0.25
BOSS_ENOUGH = 0.75


def effect(boss, original):
    """Zero for an ordinary character, one once the edit reads as a final boss, and no more."""
    floor = max(BOSS_FLOOR, original)
    if floor >= BOSS_ENOUGH:
        return 0.0
    return min(1.0, max(0.0, boss - floor) / (BOSS_ENOUGH - floor))


def score_case(boss, original, quality_, recolored):
    """A final boss, still a finished drawing of one whole figure, in the scene it started in."""
    if None in (boss, original, quality_, recolored):
        return None
    if not all(0 <= value <= 1 for value in (boss, original, quality_, recolored)):
        raise ValueError("Readings must lie in the unit interval")
    return effect(boss, original) * quality_ * (1 - recolored)



from itertools import combinations
from math import isfinite
from statistics import fmean


def summarize(case_scores, pairs):
    expected = {frozenset(p) for p in combinations(case_scores, 2)}
    repeats = {frozenset(p["cases"]): p["repeat"] for p in pairs}
    values = (*case_scores.values(), *repeats.values())
    if (
        not expected
        or len(repeats) != len(pairs)
        or set(repeats) != expected
        or any(not isfinite(v) or not 0 <= v <= 1 for v in values)
    ):
        raise ValueError(
            "Expected all distinct case pairs and scores in the unit interval"
        )
    scores = []
    for (a, b), repetition in repeats.items():
        left, right = case_scores[a], case_scores[b]
        scores.append((left + right - repetition * min(left, right)) / 2)
    return {
        "base": fmean(case_scores.values()),
        "repeat": fmean(repeats.values()),
        "score": fmean(scores),
    }
