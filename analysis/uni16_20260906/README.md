# Uni16: a listening palette trained with the winning GAN recipe

The new version replaces the proposed 28-control attribute palette with **16 controls: two vocal choices and fourteen recognizable sounds**. The product aim is repeat preference: a listener finds a sound they want to retain across different songs. This is an authored hypothesis to test with listening, not a claim that these sixteen choices have already won a preference study.

[Live training and listening page](../../eval/listen/uni16-gan-v1/index.html) · [Status](status.json) · [Catalog and combinations](catalog.json) · [Prompt audit](prompt-audit.json) · [Frozen campaign](manifest.json).

## Review of the existing uni catalog

The actual studio registry contains 28 LM controls. Their sidecars show 800 updates with the non-GAN uni lyric-hold recipe, using two to four prompt rows each. The four refreshed exports and the older uni exports coexist. The [per-control audit](current-catalog-audit.json) records the exact weight, prompt, row count, target and training method for every current slider. All referenced prompt files passed the project's existing name validator; that is a backstop, not an exhaustive entity detector.

Many slots represent opposing performance attributes: loud/quiet, fast/slow, live/studio, tender/fierce, grit/smooth, joy/somber, yearning/settled and hurt/numb. Rap specifies delivery, while the old Pop is the separate opposite of Trip-hop. The new Pop gets its own positive musical description. Rhyme and Prose belong to lyric composition, so neither occupies a slot in this sound palette.

## The 16 controls

| Control | Persistent listening character |
|---|---|
| Female | One adult female lead; phrasing and instrumentation stay with the song |
| Male | One adult male lead; phrasing and instrumentation stay with the song |
| Pop | Clear hooks, crisp drums and a polished chorus |
| Hip-Hop | Rapped verses, deep sub bass, spacious trap drums |
| R&B | Warm electric piano, syncopated bass and fluid phrasing |
| Indie Rock | Chiming guitars, moving bass and a human drum kit |
| Pop Punk | Palm-muted power chords and a driving chorus |
| Metal | Precise heavy riffs, tight kicks and melodic chorus space |
| Country | Acoustic strum, twangy fills, pedal steel and an easy backbeat |
| Acoustic Folk | Fingerpicked strings and a warm small-room performance |
| House | Four-on-the-floor kick, offbeat hats and rolling bass |
| Disco / Funk | Elastic bass, clipped guitar and bright dance-band accents |
| K-pop | Sharp synth hooks, precise drum edits and a big chorus lift |
| Reggaeton | Dembow drums, rounded sub bass and clipped melodic hooks |
| Afrobeats | Interlocking percussion, melodic bass and buoyant guitar |
| Lo-fi | Soft swung drums, mellow keys and gentle tape warmth |

The starting audience is a broad mix of mainstream and genre listeners. R&B/hip-hop accounts for about a quarter of U.S. audio streams, dance/electronic leads U.S. streaming-share growth in the first half of 2026, and Latin casual listenership has broadened. These support including those families, not a precise ranking of our fourteen sound slots. [Luminate, July 2026](https://luminatedata.com/blog/luminate-2026-midyear-report-trends-in-music-television-film/). The older global listening survey found Pop the most popular genre and substantial diversity of tastes; it supplies background rather than a current subgenre ranking. [IFPI, December 2023](https://www.ifpi.org/ifpis-global-study-finds-were-listening-to-more-music-in-more-ways-than-ever/).

Indie Rock, Pop Punk and Metal deliberately cover distinct guitar preferences. House and Disco/Funk have different rhythm sections. Acoustic Folk and Lo-fi cover two different intimate listening textures. The narrower slots are product choices, not measured market-share conclusions. K-pop describes production and arrangement here; it does not translate lyrics or require a group. Reggaeton and Afrobeats similarly describe groove and instrumentation, without forcing language or accent.

## Prompt design

The official Music 3 caption guidance favors global musical context, explicit vocal configuration and an arrangement that develops by section. It also separates lyric text from caption instructions. We follow that structure with concise paired captions appropriate to contrastive slider training. [Official caption guide](https://github.com/MiniMax-AI/MiniMax-Music3/blob/main/skills/music-caption-rewriter/SKILL.md).

Every file contains four training or four evaluation rows with `Global Metadata:`, `Vocal Details:` and `Arrangement:`. The target describes a real playable source song. `neutral == target == negative`; only the positive caption adds the requested control. Each pair has identical tempo, lyric sheet and section tags. Four new six-line lyric sheets are used for training, with four different sheets reserved for evaluation. Target BPM varies across rows within a plausible range, but never changes within a neutral/positive pair.

Female and Male alter exactly one phrase: the lead's adult vocal gender. The old female teacher's mezzo, raised-formant, high-sibilant and prescribed-vibrato cues are removed. Gender controls do not hard-code a range, breathiness, rasp or arrangement. Genre controls cover unspecified, female and male leads while preserving each row's vocal gender. Hip-Hop additionally teaches rapped verses and a melodic chorus; R&B teaches its phrasing. The remaining genres keep the same generic melodic delivery. All captions and lyrics are newly authored without names or copied song lines.

## Training and evidence

Each control starts from a fresh zero-output rank-8, alpha-8 LM attention adapter. It follows the original constant-LR RpGAN/b_cap recipe for 600 updates, then migrates **its own full state** into the winning `gan_v2` baseline arm for updates 601–660. The critic and both optimizer states carry over. No original female weights initialize other concepts.

The preflight compares every original warmup setting and source fingerprint with the winning ancestor and compares the entire bounded recipe and critic configuration with baseline 660. Generator LR is 0.0005; critic LR is 0.00075; adversarial/FM/end weights are 1; batch is 4; seed is 7. The parameter-step limit is 2 during the continuation. Normalized FM, FM gradient limiting, effective-weight limiting, extra policy/hidden guards, EMA and LR decay remain off, matching the winner. Explicit lyric-hold loss remains zero; lyric preservation is measured in audio. The data change means this is recipe transfer, not a reproduction of the old audio or proof that 660 is optimal for every sound.

For each slider the queue renders **600 and 660**, two reserved prompts, seeds 7/23 and 20-second clips. Off and positive-caption reference audio are shared within each comparison. All first samples and failures remain visible. Audio scoring uses the existing component weights with separately frozen genre descriptions. Compare checkpoints within one slider; scores do not rank different genres or extend the old gender leaderboard. Rows 2/3 and seeds 101/303 are reserved for further individual confirmation.

After all sixteen complete, eight authored combinations receive a small listening screen: six style pairs and two vocal/style pairs. Each uses the same new fixture and seed 515, comparing Off, each component separately, and both together. This is one sample per case, not an established safe strength range or measured long-term preference. The intended listening question is whether the combination remains appealing while preserving words and a coherent groove.

## Operation

Run from `/ml2/music/sliders-conceptmod` with the existing conda interpreter. All training and rendering is serialized on physical GPU 1. CPU scoring uses the installed models and caches; nothing is installed. The persistent user service is `music-uni16-gan-v1-20260906.service`.

```bash
systemctl --user status music-uni16-gan-v1-20260906.service
journalctl --user -u music-uni16-gan-v1-20260906.service -n 30 --no-pager
```

`status.json` exposes the current child PID and log. Warmup states save every 150 updates and at 600; the bounded continuation saves every 30 updates. Interrupted stages resume their own state into fresh directories. A lock prevents duplicate runners. Source and prompt hashes are checked before every child job; completed weight exports are checked against full states for finiteness and exact tensor identity. Failed controls remain explicit while the independent queue continues.

All sixteen successful runs produce `studio-catalog.json`, a complete replacement registry pointing to their 660 exports. It is a new version prepared for review; the live studio registry and published catalog are not switched by the training runner. The gallery is a local training artifact, not a published release. Do not describe queued jobs as trained or a partial campaign as complete.

The listening template now lives in `report.py`. It reloads during progress refreshes, so template edits can appear without restarting model jobs. The reporter receives copies of the queue data; presentation errors are logged while training continues. `report.py` and the older page patch helper are deliberately outside the training fingerprint set. Training, rendering, scoring and prompt files remain checked. A shared fingerprint or disk-space blockage pauses the queue once, preserving completed stages and leaving later jobs queued.

On September 6, a page-only edit in the former combined `train.py` triggered its source check after Indie Rock finished update 600. Five sliders had already completed both training and scoring; the ten later sliders never launched. The [page-isolation amendment](amendments/0002-page-isolation/record.json) archives the incident and records the reviewed orchestration repair. The audio model, optimization recipe, prompts, completed weights and saved Indie Rock warmup state are preserved.

The [recorded label amendment](amendments/0001-label-filename/record.json) changes the unstarted Disco/Funk export label to `Disco Funk`, because slash characters create nested audio paths in the existing renderer. All caption pairs, lyrics and training settings are identical before and after; the original manifest and changed files remain archived. The active Female training was unaffected. The amended manifest is the current source of expected hashes.
