"""The board's questions with the vibe phrase in place of the Final Boss wording."""

from . import image_score as ws

from .scene_grid import article


def build(vibe):
    a = article(vibe)
    effect = dict(
        state=("Image_1 is the original illustration. Image_2 is the edited illustration. Judge whether the "
               f"edited person has become an unmistakable {vibe}, from visible design of the character alone."),
        questions={"effect": {"type": "noul", "instructions": (
            f"Score {vibe} distinction from 0 to 1. 0 means no discernible {vibe} character. 0.25 means ordinary "
            f"clothing with no {vibe} cue. 0.50 means a credible hint of {a}, with some {vibe} cues. 0.75 means a "
            f"distinctive {vibe} with a recognizable, purposeful silhouette and signature costume details. 1.0 means "
            f"a visually unmistakable {vibe}. Judge the outfit, silhouette, headwear, accessories, and other visible "
            "evidence. Ignore scene, lighting, crop, and image polish.")}})
    repeat = dict(
        state="Compare the central character designs in these two illustrations from the same visual style.",
        questions={"repeat": {"type": "score", "instructions": (
            f"How much is the same underlying {vibe} design repeated across the characters? Consider the silhouette, "
            "body construction, headwear, costume shapes and signature visual idea together. Different people can be "
            f"put in essentially the same {vibe} design. Distinct designs can share a kind of garment: look for "
            "individual design ideas. Differences in setting, lighting, hue, camera or pose alone say little about "
            "the design. Ignore beauty and rendering damage; those are judged separately."),
            "criteria": ["Clearly different character designs with different signature forms",
                         "Mostly different designs with a few shared motifs",
                         "Related designs with substantial overlap and individual differences",
                         f"Mostly the same {vibe} template with minor design differences",
                         "Effectively the same character design or a repeat of the same illustration"]}})
    pair_state = ("Image 1 is the original illustration of one person. Image 2 is an edit meant to give that "
                  f"same person {a} design while keeping the picture.")
    separate = dict(
        state="Image 1 is the source. Image 2 is a proposed transformation of its central character.",
        questions={"separate": {"type": "noul", "instructions": (
            f"In image 2, does the original subject remain present as an ordinary person while a separate figure "
            f"supplies the {vibe} design? Look for two distinct figures. A single transformed person in a new "
            "costume, a mask or a new pose is a valid edit.")}})
    # Read on one image during training, where only the gain over OFF and the structural check matter.
    rubric = {"type": "score", "instructions": (
        f"How strongly does the main character visibly read as {a} from their own costume, accessories, "
        "expression or silhouette? Ignore the setting, lighting, theatrical effects and image prettiness."),
        "criteria": [f"The person looks ordinary or generic, with no convincing {vibe} cue.",
                     f"The person has a weak or ambiguous visual cue of {a}.",
                     f"Several visible character-design cues make the person clearly read as {a}.",
                     f"The person's own iconic design makes them unmistakable as {a}."]}
    return dict(effect=effect, repeat=repeat, pair=dict(state=pair_state, questions=ws.PAIR_QUESTIONS),
                figure=dict(state=ws.FIGURE_STATE, questions=ws.FIGURE_QUESTIONS),
                recolor=dict(state=ws.RECOLOR_STATE, questions=ws.RECOLOR_QUESTIONS),
                separate=separate, rubric=rubric)
