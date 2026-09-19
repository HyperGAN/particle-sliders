"""Physical GPU placement, separate from the immutable recipe and catalog."""
from common import WORK, read, sha


def route_catalog(catalog, routing, manifest_sha256):
    if routing.get("version") != 1 or routing.get("training_manifest_sha256") != manifest_sha256:
        raise ValueError("GPU routing belongs to a different campaign or version")
    overrides = routing.get("overrides", {})
    ids = {item["id"] for item in catalog}
    if set(overrides) - ids or any(type(gpu) is not int or gpu not in (0, 1) for gpu in overrides.values()):
        raise ValueError("GPU routing must map known sliders to physical GPU 0 or 1")
    return [dict(item, gpu=overrides.get(item["id"], item["gpu"])) for item in catalog]


def load_routes():
    manifest_sha = sha(WORK / "manifest.json")
    routing = read(WORK / "gpu-routing.json", dict(version=1,
        training_manifest_sha256=manifest_sha, overrides={}))
    catalog = read(WORK / "catalog.json")["sliders"]
    return route_catalog(catalog, routing, manifest_sha)
