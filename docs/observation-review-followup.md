# Independent observation review follow-up

The independent review held clean candidate
36a086470513b01f3981ad5528517b73af21d544 for two reproduced P2 defects:
state_dict excludes nonpersistent buffers, and inference device collection
looked only at parameters. Preserve that commit and all 0.8.0 receipts.

Core 0.8.1 uses one private observation context for both public session types.
It snapshots all module buffer registrations/values in addition to state_dict,
restores original references and persistence flags on success/failure, and
retains mode/RNG isolation. Both session types collect CUDA devices through
_rng_devices, including network buffers and declared request/runtime devices.
No generator/network/optimizer law, checkpoint schema or native sampler changes.
The patch also removes the three trailing blank lines flagged by the reviewer.

New regressions cover both session types, persistent/nonpersistent and nested
buffers, success/failure, rebound/added buffer registrations, caller RNG/modes,
and CUDA devices represented only by FakeTensor metadata. Device tests mock
fork/seed/copy operations for simulated CUDA tensors on the CPU-only Torch
build; real CPU buffer restoration is independently exercised. No GPU RNG or
mixed-device pretrained qualification is claimed. Independent rereview remains
required before common-pin synchronization; derivative adoption is separate.
