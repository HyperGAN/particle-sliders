"""Run the pinned workload with an explicitly recorded cuBLAS SM-count target.

This changes kernel selection, not the training formulation. A host must pass
teacher, update and rendering parity before this launcher runs production work.
"""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install_kernel_policy(sm_count):
    """Apply the same SM heuristic to CUDA and every autograd cuBLAS handle.

    These process-local interposers change launch heuristics, not GPU ownership
    or available memory. Their exact binaries must pass the hardware replay.
    """
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    packages = {name: importlib.metadata.distribution(name) for name in
                ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12", "triton")}
    runtime = Path(packages["nvidia-cuda-runtime-cu12"].locate_file("nvidia/cuda_runtime"))
    cublas = Path(packages["nvidia-cublas-cu12"].locate_file("nvidia/cublas"))
    triton = Path(packages["triton"].locate_file("triton/backends/nvidia/include"))
    sources = [ROOT / "scripts/native" / name for name in
               ("anima_cublas_sm_target.cpp", "anima_cuda_sm_target.cpp")]
    identity = dict(sources={p.name: sha(p) for p in sources},
                    packages={k: v.version for k, v in packages.items()},
                    compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = ROOT / "artifacts/anima/host-policies" / key
    directory.mkdir(parents=True, exist_ok=True)
    libraries = []
    for source, dependency in zip(sources, (cublas / "lib/libcublas.so.12", runtime / "lib/libcudart.so.12")):
        library = directory / (source.stem + ".so")
        if not library.exists():
            temp = library.with_suffix(f".{os.getpid()}.tmp")
            command = ["g++", "-shared", "-fPIC", "-O2"]
            if "cuda_sm" in source.name:
                command += ["-I" + str(runtime / "include"), "-I" + str(triton)]
            command += [str(source), "-Wl,--no-as-needed", str(dependency),
                        "-Wl,-rpath," + str(dependency.parent), "-ldl", "-o", str(temp)]
            subprocess.run(command, check=True)
            temp.replace(library)
        libraries.append(library)
    identity["libraries"] = {p.name: sha(p) for p in libraries}
    manifest = directory / "policy.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Compiled host-policy identity changed")
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
    preload = ":".join(map(str, libraries))
    if os.environ.get("LUMEN_KERNEL_POLICY") != key:
        if os.environ.get("LD_PRELOAD"):
            raise ValueError("Start the host launcher without another LD_PRELOAD policy")
        environment = dict(os.environ, LD_PRELOAD=preload, LUMEN_KERNEL_POLICY=key,
                           LUMEN_SM_TARGET=str(sm_count), LUMEN_CUBLAS_SM_TARGET=str(sm_count))
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], environment)
    if (os.environ.get("LD_PRELOAD") != preload
            or os.environ.get("LUMEN_SM_TARGET") != str(sm_count)
            or os.environ.get("LUMEN_CUBLAS_SM_TARGET") != str(sm_count)):
        raise ValueError("Loaded host policy does not match the requested execution")
    return dict(identity=key, manifest=str(manifest), **identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--sm-count", type=int, required=True)
    parser.add_argument("--kernel-policy", action="store_true",
                        help="Match CUDA heuristics and all autograd cuBLAS handles")
    parser.add_argument("--single-thread-autograd", action="store_true")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--script", type=Path)
    target.add_argument("--module")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    policy = install_kernel_policy(args.sm_count) if args.kernel_policy else None
    from lumen_studio import execution  # Configure determinism before CUDA use.
    import torch
    if args.single_thread_autograd:
        torch.autograd.set_multithreading_enabled(False)
    properties = torch.cuda.get_device_properties(0)
    if not 1 <= args.sm_count <= properties.multi_processor_count:
        raise ValueError("SM-count target must fit the selected physical GPU")
    library = ctypes.CDLL(str(importlib.metadata.distribution("nvidia-cublas-cu12")
        .locate_file("nvidia/cublas/lib/libcublas.so.12")))
    library.cublasSetSmCountTarget.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.cublasSetSmCountTarget.restype = ctypes.c_int
    library.cublasGetSmCountTarget.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    library.cublasGetSmCountTarget.restype = ctypes.c_int
    handle = torch._C._cuda_getCurrentBlasHandle()
    if library.cublasSetSmCountTarget(handle, args.sm_count):
        raise RuntimeError("cuBLAS rejected the requested kernel-selection target")
    actual = ctypes.c_int()
    if library.cublasGetSmCountTarget(handle, ctypes.byref(actual)) or actual.value != args.sm_count:
        raise RuntimeError("cuBLAS SM-count setting did not take effect")
    uuid = subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid",
                                    "--format=csv,noheader"], text=True).strip()
    print(json.dumps(dict(cuda_host_policy=dict(gpu_uuid=uuid, name=properties.name,
        reported_sm_count=properties.multi_processor_count, cublas_sm_count_target=actual.value,
        autograd_multithreading=torch.autograd.is_multithreading_enabled(),
        kernel_policy=policy,
        determinism=execution.DETERMINISM))), flush=True)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [str(args.script) if args.script else args.module, *arguments]
    if args.script:
        runpy.run_path(str(args.script), run_name="__main__")
    else:
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
