"""Verify and promote a reviewed revision of an existing public Hub model."""

import argparse
import json
import os
import re
from pathlib import Path, PurePosixPath

from huggingface_hub import CommitOperationCopy, HfApi
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import get_session, hf_raise_for_status


HUB = "https://huggingface.co"
MODEL_ID = re.compile(r"ntc-ai/[A-Za-z0-9][A-Za-z0-9._-]+")
COMMIT = re.compile(r"[a-f0-9]{40}")
MUTABLE_FILES = {
    ".gitattributes", "README.md", "catalog.json", "release-manifest.json",
    "source-provenance.json", "source.zip",
}
ADDITIONAL_METADATA = {"FORMULATION.md", "manifest.json", "SHA256SUMS"}


def revision(value):
    if not isinstance(value, str) or not COMMIT.fullmatch(value):
        raise ValueError("Expected a full lowercase commit SHA")
    return value


def tree(api, repo, sha):
    files = {}
    for entry in api.list_repo_tree(repo, repo_type="model", revision=sha, recursive=True):
        if isinstance(entry, RepoFile):
            identity = {"sha256": entry.lfs.sha256} if entry.lfs else {"git_blob": entry.blob_id}
            files[entry.path] = {"bytes": entry.size, **identity}
    if not files:
        raise ValueError(f"{repo} at {sha} has an empty file tree")
    return files


def reviewed_changes(base, staged, extra_metadata=()):
    extra_metadata = set(extra_metadata)
    if extra_metadata - ADDITIONAL_METADATA:
        raise ValueError("Unsupported mutable release metadata filename")
    mutable = MUTABLE_FILES | extra_metadata
    removed = base.keys() - staged.keys()
    if removed:
        raise ValueError(f"The staged release removes existing files: {sorted(removed)[:8]}")

    changes = {}
    for path, identity in staged.items():
        if base.get(path) == identity:
            continue
        relative = PurePosixPath(path)
        if (
            not relative.parts or relative.is_absolute() or relative.as_posix() != path
            or "\\" in path or any(ord(char) < 32 for char in path)
            or (path != ".gitattributes" and any(part.startswith(".") for part in relative.parts))
        ):
            raise ValueError(f"Unexpected release path: {path!r}")
        if path in base and path not in mutable:
            raise ValueError(f"The staged release overwrites a historical file: {path}")
        changes[path] = identity
    return changes


def main_revision(api, repo):
    info = api.model_info(repo, revision="main")
    if info.id != repo or info.private or info.disabled:
        raise ValueError(f"Expected an active public model at {repo}")
    return revision(info.sha)


def make_manifest(api, repo, base_sha, staged_sha, extra_metadata=()):
    base_sha, staged_sha = revision(base_sha), revision(staged_sha)
    old_tree = tree(api, repo, base_sha)
    staged_tree = old_tree if base_sha == staged_sha else tree(api, repo, staged_sha)
    return {
        "base_revision": base_sha,
        "staged_revision": staged_sha,
        "changes": reviewed_changes(old_tree, staged_tree, extra_metadata),
    }, staged_tree


def release(api, repo, manifest, writer=None, source_sha=None, extra_metadata=()):
    if set(manifest) != {"base_revision", "staged_revision", "changes"}:
        raise ValueError("Unexpected release manifest fields")
    base_sha, staged_sha = manifest["base_revision"], manifest["staged_revision"]
    verified, staged_tree = make_manifest(api, repo, base_sha, staged_sha, extra_metadata)
    if verified != manifest:
        raise ValueError("The staged tree does not match the reviewed manifest")

    head = main_revision(api, repo)
    if head == staged_sha or (head != base_sha and tree(api, repo, head) == staged_tree):
        return "already-current", head
    if head != base_sha:
        raise ValueError(f"Hub main advanced to {head}. Regenerate and review the release manifest.")
    if not manifest["changes"]:
        return "already-current", head
    if writer is None:
        return "ready", head

    commit = writer.create_commit(
        repo,
        repo_type="model",
        revision="main",
        parent_commit=head,
        commit_message=f"Publish model release from GitHub {revision(source_sha)}",
        commit_description=f"Reviewed Hub staging revision: {staged_sha}",
        operations=[CommitOperationCopy(path, path, src_revision=staged_sha) for path in sorted(manifest["changes"])],
    )
    if tree(api, repo, commit.oid) != staged_tree or main_revision(api, repo) != commit.oid:
        raise RuntimeError(f"Published revision {commit.oid} did not pass file tree and branch verification")
    return "published", commit.oid


def publisher(repo):
    if (
        os.getenv("GITHUB_ACTIONS") != "true" or os.getenv("GITHUB_EVENT_NAME") != "push"
        or os.getenv("GITHUB_REF") != "refs/heads/main"
    ):
        raise ValueError("Hub publication requires a GitHub Actions push to main")
    source_sha = revision(os.environ["GITHUB_SHA"])
    token = os.environ["HF_TOKEN"]
    if not token.startswith("hf_jwt_"):
        raise ValueError("Expected a short-lived Hub Trusted Publisher token")
    # The upload endpoint checks the repo publisher token without uploading files.
    response = get_session().post(
        f"{HUB}/api/models/{repo}/preupload/main",
        headers={"Authorization": f"Bearer {token}"}, json={"files": []}, timeout=20,
    )
    hf_raise_for_status(response)
    head = revision(response.json()["commitOid"])
    print(f"Verified upload preflight for {repo} at {head}", flush=True)
    return HfApi(endpoint=HUB, token=token), source_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Existing ntc-ai model repository")
    parser.add_argument("--manifest", type=Path, default=Path(".github/hub-model-release.json"))
    parser.add_argument("--mutable-file", action="append", default=[], choices=sorted(ADDITIONAL_METADATA))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stage", metavar="STAGED_SHA", help="Print the immutable manifest for review")
    mode.add_argument("--publish", action="store_true", help="Promote the reviewed staged revision")
    args = parser.parse_args()
    if not MODEL_ID.fullmatch(args.repo):
        parser.error("Expected an ntc-ai model repository ID")

    api = HfApi(endpoint=HUB, token=False)
    try:
        if args.stage:
            manifest, _ = make_manifest(api, args.repo, main_revision(api, args.repo), args.stage, args.mutable_file)
            print(json.dumps(manifest, indent=2, sort_keys=True))
        else:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            writer, sha = publisher(args.repo) if args.publish else (None, None)
            status, result = release(api, args.repo, manifest, writer, sha, args.mutable_file)
            print(f"{args.repo} {status} {result}; reviewed paths: {len(manifest['changes'])}")
    except (KeyError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
