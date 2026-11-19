from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import release as hub


REPO = "ntc-ai/test-model"
BASE, STAGED, NEW = "a" * 40, "b" * 40, "c" * 40
ORIGINAL = {"bytes": 100, "git_blob": "d" * 40}
WEIGHT = {"bytes": 250, "sha256": "e" * 64}


class HubReleaseTest(unittest.TestCase):
    def setUp(self):
        self.base = {"README.md": ORIGINAL, "weights/v1.safetensors": WEIGHT}
        self.changes = {"weights/v2.safetensors": WEIGHT, ".gitattributes": ORIGINAL}
        self.staged = self.base | self.changes
        self.manifest = {"base_revision": BASE, "staged_revision": STAGED, "changes": self.changes}
        self.api, self.writer = Mock(), Mock()
        self.api.model_info.return_value = SimpleNamespace(id=REPO, private=False, disabled=False, sha=BASE)

    def trees(self, **extra):
        files = {BASE: self.base, STAGED: self.staged, **extra}
        return patch.object(hub, "tree", side_effect=lambda api, repo, sha: files[sha])

    def test_read_only_validation_and_no_op(self):
        baseline = {"base_revision": BASE, "staged_revision": BASE, "changes": {}}
        with self.trees():
            self.assertEqual(hub.release(self.api, REPO, self.manifest), ("ready", BASE))
            self.assertEqual(hub.release(self.api, REPO, baseline, self.writer), ("already-current", BASE))
        self.writer.create_commit.assert_not_called()

    def test_unexpected_changes_are_rejected(self):
        for payload, message in (
            ({"README.md": ORIGINAL}, "removes"),
            (self.base | {"weights/v1.safetensors": ORIGINAL}, "historical"),
            (self.base | {".config/secret": ORIGINAL}, "Unexpected release path"),
            (self.base | {"weights/../release": ORIGINAL}, "Unexpected release path"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                hub.reviewed_changes(self.base, payload)

    def test_unreviewed_delta_and_concurrent_updates_are_rejected(self):
        with self.trees(), self.assertRaisesRegex(ValueError, "reviewed manifest"):
            hub.release(self.api, REPO, self.manifest | {"changes": {}}, self.writer)
        self.api.model_info.return_value.sha = NEW
        with self.trees(**{NEW: self.base | {"external-change": WEIGHT}}), self.assertRaisesRegex(ValueError, "advanced"):
            hub.release(self.api, REPO, self.manifest, self.writer)
        self.writer.create_commit.assert_not_called()

    def test_checksum_metadata_updates_are_opt_in_and_keep_weights_immutable(self):
        base = self.base | {"SHA256SUMS": ORIGINAL}
        changed = base | {"SHA256SUMS": WEIGHT}
        with self.assertRaisesRegex(ValueError, "historical"):
            hub.reviewed_changes(base, changed)
        self.assertEqual(hub.reviewed_changes(base, changed, ["SHA256SUMS"]), {"SHA256SUMS": WEIGHT})
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            hub.reviewed_changes(base, changed, ["weights/v1.safetensors"])

    def checksum_files(self, previous, staged, card=b"new card"):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "base.json").write_text(json.dumps({"files": previous}))
        (root / "staged.json").write_text(json.dumps({"files": staged}))
        (root / "README.md").write_bytes(card)
        self.api.hf_hub_download.side_effect = lambda repo, path, *, revision: root / {
            (BASE, hub.CHECKSUM_MANIFEST): "base.json",
            (STAGED, hub.CHECKSUM_MANIFEST): "staged.json",
            (STAGED, "README.md"): "README.md",
        }[revision, path]
        self.base[hub.CHECKSUM_MANIFEST] = ORIGINAL
        self.staged[hub.CHECKSUM_MANIFEST] = ORIGINAL if previous == staged else WEIGHT
        self.staged["README.md"] = {"bytes": len(card), "git_blob": "f" * 40}
        self.manifest["changes"] = hub.reviewed_changes(self.base, self.staged)

    def test_stale_release_checksum_blocks_the_writer(self):
        stale = {"README.md": {"bytes": len(b"old card"), "sha256": sha256(b"old card").hexdigest()}}
        self.checksum_files(stale, stale)
        with self.trees(), self.assertRaisesRegex(ValueError, "SHA-256 does not match staged file README.md"):
            hub.release(self.api, REPO, self.manifest, self.writer, BASE, verify_release_manifest=True)
        self.writer.create_commit.assert_not_called()

    def test_changed_git_and_lfs_checksums_match_the_staged_revision(self):
        previous = {"README.md": {"bytes": len(b"old card"), "sha256": sha256(b"old card").hexdigest()}}
        staged = {
            "README.md": {"bytes": len(b"new card"), "sha256": sha256(b"new card").hexdigest()},
            "weights/v2.safetensors": WEIGHT,
        }
        self.checksum_files(previous, staged)
        with self.trees():
            self.assertEqual(hub.release(self.api, REPO, self.manifest, verify_release_manifest=True), ("ready", BASE))
        self.api.hf_hub_download.assert_any_call(REPO, "README.md", revision=STAGED)
        self.assertEqual(self.api.hf_hub_download.call_count, 3)

    def test_new_checksum_for_a_retained_file_is_verified(self):
        self.checksum_files({}, {"weights/v1.safetensors": {"bytes": 1, "sha256": WEIGHT["sha256"]}})
        with self.trees(), self.assertRaisesRegex(ValueError, "byte count or path does not match"):
            hub.release(self.api, REPO, self.manifest, verify_release_manifest=True)

    def test_changed_covered_file_cannot_drop_its_checksum(self):
        self.checksum_files({"README.md": WEIGHT}, {})
        with self.trees(), self.assertRaisesRegex(ValueError, "drops checksums for changed files"):
            hub.release(self.api, REPO, self.manifest, verify_release_manifest=True)

    def test_commit_copies_exact_delta_and_checks_production(self):
        self.writer.create_commit.return_value = SimpleNamespace(oid=NEW)
        self.api.model_info.side_effect = [
            self.api.model_info.return_value, SimpleNamespace(id=REPO, private=False, disabled=False, sha=NEW),
        ]
        with self.trees(**{NEW: self.staged}):
            self.assertEqual(hub.release(self.api, REPO, self.manifest, self.writer, BASE), ("published", NEW))
        call = self.writer.create_commit.call_args
        self.assertEqual(call.args, (REPO,))
        self.assertEqual((call.kwargs["revision"], call.kwargs["parent_commit"]), ("main", BASE))
        self.assertEqual({op.path_in_repo: op.src_revision for op in call.kwargs["operations"]},
                         {path: STAGED for path in self.changes})

    def test_successful_retry_verifies_full_tree(self):
        self.api.model_info.return_value.sha = NEW
        with self.trees(**{NEW: self.staged}):
            self.assertEqual(hub.release(self.api, REPO, self.manifest, self.writer), ("already-current", NEW))
        self.writer.create_commit.assert_not_called()

    def test_publisher_uses_the_upload_preflight_only_on_main(self):
        env = {
            "GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main",
            "GITHUB_SHA": BASE, "HF_TOKEN": "hf_jwt_test",
        }
        with patch.dict(hub.os.environ, env, clear=True), patch.object(hub, "get_session") as session, \
                patch.object(hub, "hf_raise_for_status"):
            session.return_value.post.return_value.json.return_value = {"commitOid": BASE}
            _, sha = hub.publisher(REPO)
            self.assertEqual(sha, BASE)
            session.return_value.post.assert_called_once_with(
                f"{hub.HUB}/api/models/{REPO}/preupload/main", json={"files": []}, timeout=20,
                headers={"Authorization": "Bearer hf_jwt_test"},
            )
            with patch.dict(hub.os.environ, {"GITHUB_EVENT_NAME": "pull_request"}):
                with self.assertRaisesRegex(ValueError, "push to main"):
                    hub.publisher(REPO)


if __name__ == "__main__":
    unittest.main()
