# Lo-fi release listening test

The test serves a shared, server-saved listening session at
<http://100.90.104.57:8890/>. The service binds to `0.0.0.0:8890`.
The original listening archive on port 8888 stays available.

1. **Checkpoints:** four previously monitored prompt/seed cases, each with
   steps 600, 2000, 3000 and 3400, adapter-off and a direct style-caption control.
   All 24 original recordings are imported with source hash checks.
2. **Strength:** after blind ratings and reveal, choose two different
   checkpoints. Both appear at 0.5, 0.75 and 1.0 on the same four cases, with
   controls. Existing recordings are reused; 16 new recordings are rendered.
3. **New songs:** choose one strength for each finalist. These choices freeze
   before rendering 12 new prompts × 3 seeds × 4 conditions = 144 recordings.
4. **Full songs:** the same frozen finalists and both controls render on three
   additional prompts × 2 seeds = 24 recordings, each capped at 90 seconds.

Sound, enjoyment and lyric clarity/fidelity use separate 1–5 ratings. Issues
and notes are optional. Every stage's ratings lock before labels are revealed.
The result table displays means without combining them into a selection score.
No LoRA is promoted or uploaded automatically. The user makes the listening
decisions. Initial cases were heard before; blinding cannot remove familiarity.
Three seeds per confirmation prompt are repeated measurements, not independent
prompt replications. Full-song endings must be listened to; early endings are
retained and do not automatically fail.

Sample assignments are shuffled per case and stage. The API never returns
unrevealed mappings, checkpoint paths or automatic scores. Opaque audio URLs
serve only registered recordings; the state and protocol are outside public
routes. This is one shared session for the user's Tailscale devices, not a
multi-rater experiment.

Previews use two-pass EBU R128 loudness normalization at -18 LUFS and a -2 dBTP
target, encoded as MP3 at 320 kbps. Original WAVs remain untouched and are
available via the Original levels toggle. Short and silent generations stay in
the set. Seeds are never bumped or substituted.

`prepare.py` freezes checkpoint hashes and all fixture definitions in
`data/protocol.json`, audits prompt/metadata names and lyric disjointness, and
creates the initial state exactly once. `study.py` owns state transitions and
atomic saves. `server.py` hosts the page, ratings API and render supervisor.
`worker.py` imports or renders missing audio on physical GPU 1, loading the
frozen inference snapshot used by the monitoring experiment. The GPU process
exits when its queue finishes. Source hashes, parameters, seed, output hashes,
normalization measurements and basic audio diagnostics are retained per clip.

The user service `music-lofi-release-test.service` restarts the server on failure
and is enabled for future logins. Interrupted rendering resumes on server
restart using the same seeds. A failed render is retained and reported; Retry
remaining recordings resumes without changing the protocol.

```bash
systemctl --user status music-lofi-release-test.service
journalctl --user -u music-lofi-release-test.service -n 30
tail -n 30 /ml2/music/sliders-conceptmod/analysis/lofi_release_test_20260912/data/render.log
```

State, ratings and the append-only event history inside the state document live
in `data/state.json`. Download revealed results from the page; unrevealed
assignments are excluded. Back up the complete `data/` directory to retain the
session and recordings. Do not rerun preparation with a different data path to
replace an in-progress study.
