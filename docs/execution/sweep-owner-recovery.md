# Retained Music3 sweep attempt ownership (O16)

`scripts/sweep_music3_critic.py` is a retained legacy source workflow. Its score,
training protocol and historical captures are unchanged; this repair is not a
new qualification of that workflow or its legacy selection metric. Current
model-independent products use their own declared core cohort and RAW metric.

Screening and extension now share one attempt admission/launch/reap path. Each
new child gets an atomically written `sweep-owner.json` recording both trainer
and controller host, boot UUID, PID and `/proc` start stamp. A nonblocking launch
lock prevents controllers from replacing one another's ownership during setup.
The exit code is certified only after the owning `Popen.wait()` completes.
A terminal status written before cleanup finishes cannot bypass the child wait.

Waits require exact local process proof. The shared dependency-light helper
`particle_sliders.execution.process_identity` distinguishes live, dead and
unknown without signaling a process. Foreign hosts, malformed records,
unreadable metadata and missing owner receipts cannot prove liveness. A dead
owner behind a training marker is refused immediately. A terminal marker also
needs a reaped successful exit receipt; if only a still-live controller can
supply it, the wrapper waits for that receipt. There is no elapsed training
cutoff. No dead or unknown attempt is automatically rerolled.

Pre-existing zero-step or partial directories are never deleted. Without a
terminal completion or explicit `complete`, `checkpoint_ready` or `paused`
status plus a nonempty `state.pt`, the wrapper refuses and asks for a new
immutable output directory. The trainer remains responsible for admitting the
full checkpoint schema, source signature and RAW provenance. Existing terminal
completion directories that predate owner receipts retain their historical
reuse behavior; that compatibility does not establish process provenance.
Legacy directory migration applies the same gate before any rename.

Only newly spawned children owned by this wrapper may be terminated when their
setup fails. All launched batch children are reaped even when a later launch,
receipt or board write fails. Captured sources, logs, checkpoints and partial
outputs are retained on every refused attempt.

CPU regression coverage uses tiny owned subprocesses and temporary directories,
including live-to-dead transitions, PID reuse, foreign/corrupt metadata, zombie
identity capture, immediate successful exit, terminal-before-exit ordering,
zero-step preservation, launch contention, and first-board failure while a
second child is still owned. It launches no model or GPU job.
