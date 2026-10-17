"""Verify the installed text-slider package before native product execution."""
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
import json
from pathlib import Path


PIN_PATH = Path(__file__).with_name("text_package_pin.json")


def require_text_package(*, pg_source_profile="release"):
    """Reject source checkouts, wrong releases and changed native implementations.

    Distribution metadata alone cannot establish which package Python imports.
    Both module origins and exact installed-wheel source hashes must match.
    The explicitly selected ALZ profile permits only the reviewed PG overlay;
    the native core must still come from the exact installed wheel.
    This does not load a model or approve a pending package review.
    """
    if pg_source_profile not in ("release", "alz-continuous-v1"):
        raise ValueError("Unknown text PG source profile")
    manifest_bytes = PIN_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "text-package-pin-v1":
        raise RuntimeError("Unsupported text package pin manifest")
    checked = {}
    versions = {}
    for name, artifact in manifest["artifacts"].items():
        try:
            installed = distribution(name)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Install the pinned offline {name} wheel before text inference") from exc
        if installed.version != artifact["version"]:
            raise RuntimeError(f"{name} version differs from the text package pin")
        if installed.read_text("WHEEL") is None:
            raise RuntimeError(f"{name} must come from the pinned installed wheel")
        versions[name] = installed.version
        root = Path(installed.locate_file("")).resolve()
        owned = {str(path) for path in (installed.files or ())}
        expected_paths = {}
        # Check bytes before importing native code, including each package root.
        for module_name, record in artifact["critical_modules"].items():
            relative = record["file"]
            expected = Path(installed.locate_file(relative)).resolve()
            if relative not in owned or not expected.is_relative_to(root) or not expected.is_file():
                raise RuntimeError(f"{module_name} is not an installed distribution file")
            actual_hash = hashlib.sha256(expected.read_bytes()).hexdigest()
            if actual_hash != record["sha256"]:
                raise RuntimeError(f"{module_name} source differs from the pinned wheel")
            expected_paths[module_name] = expected
        package_name = artifact["import_package"]
        try:
            spec = find_spec(package_name)
        except (ImportError, ValueError) as exc:
            raise RuntimeError(f"Cannot resolve {package_name} installed import identity") from exc
        if spec is None or spec.origin is None:
            raise RuntimeError(f"{package_name} import shadows its installed distribution")
        package_origin = Path(spec.origin).resolve()
        allow_pg_overlay = name == "particlegan" and pg_source_profile == "alz-continuous-v1"
        if allow_pg_overlay:
            expected_hash = artifact["critical_modules"][package_name]["sha256"]
            if not package_origin.is_file() or hashlib.sha256(package_origin.read_bytes()).hexdigest() != expected_hash:
                raise RuntimeError("PG overlay root source differs from the pinned wheel")
        elif package_origin != expected_paths[package_name]:
            raise RuntimeError(f"{package_name} import shadows its installed distribution")
        for module_name, expected in expected_paths.items():
            if allow_pg_overlay and module_name == package_name:
                expected = package_origin
            module = import_module(module_name)
            actual = getattr(module, "__file__", None)
            if actual is None or Path(actual).resolve() != expected:
                raise RuntimeError(f"{module_name} import shadows its installed distribution")
            checked[module_name] = dict(path=str(expected), sha256=artifact["critical_modules"][module_name]["sha256"])
    # The shared validator checks every actual PG source against the exact
    # release/overlay manifest after the installed core has been verified.
    from particle_sliders.pg09 import validate_pg09_domain

    penalty_domain = "continuous" if pg_source_profile == "alz-continuous-v1" else "auto"
    domain = validate_pg09_domain(
        (1, 1024 if pg_source_profile == "alz-continuous-v1" else 32),
        response_domain="continuous_vector", pg_source_profile=pg_source_profile,
        penalty_input_domain=penalty_domain,
    )
    return dict(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                core_source_commit=manifest["core_source_commit"],
                review_status=manifest["review_status"], versions=versions,
                wheel_sha256={name: artifact["sha256"] for name, artifact in manifest["artifacts"].items()},
                critical_modules=checked, pg_source_profile=pg_source_profile,
                penalty_input_domain=penalty_domain,
                pg_sources_sha256=domain["pg_sources_sha256"],
                resolved_penalty_domain=domain["resolved_penalty_domain"], domain=domain)
