# Hugging Face model releases

The action validates a reviewed `.github/hub-model-release.json` from the calling
repository. It publishes from a protected GitHub Actions environment on pushes
to `main`. The Hub Trusted Publisher should match the caller repository and its
numeric repository ID, workflow ref, `refs/heads/main`, environment, `push` event,
and `github-hosted` runner, and grant `repo.content.write` to the one model.

1. Build and qualify the release using the model project's existing tools.
   Include the quality evidence and the source revision in the release PR.
2. Stage the files on a Hub branch from its current `main`. Use this public
   staging workflow only for artifacts already cleared for public distribution.
   Add new versioned paths for historical checkpoints, samples, and reports.
3. From a checkout of the action at the exact commit pinned in the model's
   workflow, install `huggingface_hub==1.19.0` and generate the manifest. Run
   this command in the model's GitHub checkout and pass the actual repo and SHA.

   ```sh
   python /path/to/particle-sliders/.github/actions/hub-model-release/release.py \
     ntc-ai/model-name --stage <staged-sha> > .github/hub-model-release.json
   ```

4. Review the precise change list and staged files before merging. CI checks
   the current parent, commits the reviewed delta, and reads back the complete
   production tree. The Hub commit records the GitHub merge and staging SHAs.

CI refuses removals and overwrites of historical files. Root release metadata
and `.gitattributes` can change when included in the reviewed manifest. Regenerate
and review the manifest if Hub `main` advances with a distinct file tree. Consumers
can pin `base_revision` while a corrected release is prepared.

Projects that publish checksum manifests may pass `mutable-files` on the action
and repeat `--mutable-file` in the staging command. Supported additions are
`FORMULATION.md`, `manifest.json`, and `SHA256SUMS`.

For a model with a `release-manifest.json` whose `files` object maps paths to
`bytes` and `sha256`, set `verify-release-manifest: 'true'` in both calling jobs
and pass `--verify-release-manifest` when generating the staging manifest. The
action checks both changed tracked files and new or changed checksum entries
against the staged revision. Include the updated checksum in the same staged
revision as the file it covers. Validate other checksum formats in the caller.
