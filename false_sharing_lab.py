import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

GIT_USERNAME = "mikazhan"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000

STRIDES = (1, 4, 8, 16)
REPEATS = 3
GIL_CONTRAST_ITERS = 1_000_000
MANIFEST_PATH = "cache_lab_manifest.json"
ALIGNMENT = 128


def detect_cache_line_size() -> int:
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def detect_cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           encoding="utf-8").strip()
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def detect_physical_cores():
    try:
        if sys.platform == "darwin":
            p = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], encoding="utf-8"))
            perf = None
            try:
                perf = int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                                                   encoding="utf-8", stderr=subprocess.DEVNULL))
            except Exception:
                pass
            return p, perf
        cores = set()
        if os.path.exists("/proc/cpuinfo"):
            phys = core = None
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("physical id"):
                        phys = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":")[1].strip()
                        cores.add((phys, core))
        return (len(cores) or None), None
    except Exception:
        return None, None


def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"

    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    affinity_supported = hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity")
    available_cores = sorted(os.sched_getaffinity(0)) if affinity_supported else list(range(logical_cores))
    if affinity_supported and len(available_cores) >= NUM_THREADS:
        step = 2 if len(available_cores) >= 2 * NUM_THREADS else 1
        core_map = [available_cores[(t * step) % len(available_cores)] for t in range(NUM_THREADS)]
    else:
        core_map = [None] * NUM_THREADS

    physical_cores, perf_cores = detect_physical_cores()
    ctx = dict(
        git_user=git_user.strip(),
        logical_cores=logical_cores,
        physical_cores=physical_cores,
        performance_cores=perf_cores,
        cache_line_size=line_size,
        cpu_sig=cpu_sig,
        cpu_model=detect_cpu_model(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        instance_seed=instance_seed,
        affinity_supported=affinity_supported,
        core_map=core_map,
    )
    print(f"[init] git_user={ctx['git_user']}  INSTANCE_SEED={instance_seed} (0x{instance_seed:016x})")
    print(f"[init] CPU={ctx['cpu_model']}  platform={ctx['platform']}")
    print(f"[init] logical_cores={logical_cores} physical_cores={physical_cores} "
          f"perf_cores={perf_cores} cache_line={line_size} B")
    print(f"[init] affinity_supported={affinity_supported} core_map={core_map}")
    return ctx


def thread_seeds(instance_seed: int):
    rng = np.random.default_rng(instance_seed)
    return [int(x) for x in rng.integers(1, 2**31 - 1, size=NUM_THREADS)]


@njit(nogil=True, cache=False)
def worker_strided_accumulate(buffer, slot, iters, seed):
    s = uint64(seed)
    j = slot
    for i in range(iters):
        v = buffer[j]
        nv = v + ((s ^ uint64(i)) & uint64(7)) + uint64(1)
        buffer[j] = nv
        j = slot + np.int64(nv >> uint64(60))
    return j


@njit(nogil=True, cache=False)
def worker_private_accumulate(buffer, slot, iters, seed):
    s = uint64(seed)
    acc = buffer[slot]
    for i in range(iters):
        acc = acc + ((s ^ uint64(i)) & uint64(7)) + uint64(1)
    buffer[slot] = acc
    return slot


def aligned_zeros(n_elems: int, align: int = ALIGNMENT) -> np.ndarray:
    raw = np.zeros(n_elems + align // 8, dtype=np.uint64)
    offset = (-raw.ctypes.data % align) // 8
    arr = raw[offset:offset + n_elems]
    assert arr.ctypes.data % align == 0
    return arr


def _rusage():
    if resource is None:
        return (0, 0)
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (r.ru_nvcsw, r.ru_nivcsw)


def run_threads(kernel, buffer, slots, iters, seeds, core_map):
    barrier = threading.Barrier(len(slots) + 1)
    starts = [0] * len(slots)
    ends = [0] * len(slots)
    errors = []

    def body(t):
        try:
            if core_map[t] is not None:
                os.sched_setaffinity(0, {core_map[t]})
            barrier.wait()
            starts[t] = time.perf_counter_ns()
            kernel(buffer, slots[t], iters, seeds[t])
            ends[t] = time.perf_counter_ns()
        except Exception as exc:
            errors.append(repr(exc))
            try:
                barrier.abort()
            except Exception:
                pass

    threads = [threading.Thread(target=body, args=(t,), name=f"worker-{t}") for t in range(len(slots))]
    cs0 = _rusage()
    for th in threads:
        th.start()
    barrier.wait()
    for th in threads:
        th.join()
    cs1 = _rusage()
    if errors:
        raise RuntimeError(errors)
    elapsed = max(ends) - min(starts)
    per_thread = [e - s for s, e in zip(starts, ends)]
    return elapsed, per_thread, (cs1[0] - cs0[0], cs1[1] - cs0[1])


def cache_lines_touched(buffer, slots, line):
    base = buffer.ctypes.data
    return sorted({(base + 8 * s) // line - base // line for s in slots})


def run_stride_benchmark(ctx, seeds, iters=None):
    iters = iters or ITERATIONS_PER_THREAD
    results = {}
    for stride in STRIDES:
        slots = [t * stride for t in range(NUM_THREADS)]
        reps = []
        final_vals = None
        for _ in range(REPEATS):
            buf = aligned_zeros(NUM_THREADS * stride + 32)
            elapsed, per_thread, (vcs, ivcs) = run_threads(
                worker_strided_accumulate, buf, slots, iters, seeds, ctx["core_map"])
            vals = [int(buf[s]) for s in slots]
            if final_vals is None:
                final_vals = vals
            elif vals != final_vals:
                raise AssertionError(f"non-deterministic counters for stride={stride}")
            reps.append(dict(elapsed_ns=elapsed, per_thread_ns=per_thread,
                             voluntary_ctx_switches=vcs, involuntary_ctx_switches=ivcs))
        med = sorted(r["elapsed_ns"] for r in reps)[len(reps) // 2]
        results[stride] = dict(
            stride=stride, byte_spacing=stride * 8, slots=slots,
            lines_touched_64B=cache_lines_touched(buf, slots, 64),
            lines_touched_native=cache_lines_touched(buf, slots, ctx["cache_line_size"]),
            elapsed_ns_median=med,
            ns_per_update=med / iters,
            ns_per_update_aggregate=med / (iters * NUM_THREADS),
            repeats=reps, final_counters=final_vals,
        )
        print(f"[stride {stride:>2} | {stride*8:>3} B] median {med:>14,d} ns  "
              f"{med/iters:7.3f} ns/update/thread  lines(64B)={results[stride]['lines_touched_64B']}  "
              f"lines({ctx['cache_line_size']}B)={results[stride]['lines_touched_native']}  "
              f"ctxsw(vol/invol, last rep)={reps[-1]['voluntary_ctx_switches']}/"
              f"{reps[-1]['involuntary_ctx_switches']}", flush=True)
    return results


def run_private_baseline(ctx, seeds, iters=None):
    iters = iters or ITERATIONS_PER_THREAD
    slots = list(range(NUM_THREADS))
    reps = []
    final_vals = None
    for _ in range(REPEATS):
        buf = aligned_zeros(NUM_THREADS + 32)
        elapsed, per_thread, (vcs, ivcs) = run_threads(
            worker_private_accumulate, buf, slots, iters, seeds, ctx["core_map"])
        final_vals = [int(buf[s]) for s in slots]
        reps.append(dict(elapsed_ns=elapsed, per_thread_ns=per_thread,
                         voluntary_ctx_switches=vcs, involuntary_ctx_switches=ivcs))
    med = sorted(r["elapsed_ns"] for r in reps)[len(reps) // 2]
    print(f"[private  | reg ] median {med:>14,d} ns  {med/iters:7.3f} ns/update/thread", flush=True)
    return dict(elapsed_ns_median=med, ns_per_update=med / iters, repeats=reps, final_counters=final_vals)


def run_single_thread_reference(seeds, iters=None):
    iters = iters or ITERATIONS_PER_THREAD
    buf = aligned_zeros(64)
    t0 = time.perf_counter_ns()
    worker_strided_accumulate(buf, 0, iters, seeds[0])
    el = time.perf_counter_ns() - t0
    print(f"[1 thread | L1  ] {el:>14,d} ns  {el/iters:7.3f} ns/update", flush=True)
    return dict(elapsed_ns=el, ns_per_update=el / iters)


def python_gil_worker(buf, slot, iters, seed):
    s = seed
    for i in range(iters):
        buf[slot] = buf[slot] + ((s ^ i) & 7) + 1


def run_gil_contrast(seeds, iters=None):
    iters = iters or GIL_CONTRAST_ITERS
    out = {}
    for nthreads in (1, NUM_THREADS):
        buf = [0] * 64
        ths = [threading.Thread(target=python_gil_worker, args=(buf, t, iters, seeds[t]))
               for t in range(nthreads)]
        cs0 = _rusage()
        t0 = time.perf_counter_ns()
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        el = time.perf_counter_ns() - t0
        cs1 = _rusage()
        out[nthreads] = dict(threads=nthreads, iters_per_thread=iters, elapsed_ns=el,
                             ns_per_update=el / (iters * nthreads),
                             voluntary_ctx_switches=cs1[0] - cs0[0],
                             involuntary_ctx_switches=cs1[1] - cs0[1])
        print(f"[GIL python | {nthreads} thr] {el:>14,d} ns total  "
              f"{el/(iters*nthreads):8.2f} ns/update  ctxsw={cs1[0]-cs0[0]}/{cs1[1]-cs0[1]}", flush=True)
    return out


def _extract_function(ir: str, needle: str) -> str:
    blocks = []
    cur = None
    for line in ir.splitlines():
        if line.startswith("define "):
            cur = [line]
        elif cur is not None:
            cur.append(line)
            if line.startswith("}"):
                blocks.append("\n".join(cur))
                cur = None
    for b in blocks:
        head = b.splitlines()[0]
        if needle in head and "cpython" not in head and "cfunc" not in head:
            return b
    return ""


def audit_llvm_and_parity(ctx, stride_results, private_result, seeds, iters=None):
    iters = iters or ITERATIONS_PER_THREAD
    import re
    expected = private_result["final_counters"]
    parity = {str(s): (r["final_counters"] == expected) for s, r in stride_results.items()}
    parity_ok = all(parity.values())
    i = np.arange(iters, dtype=np.uint64)
    analytic0 = int((((np.uint64(seeds[0]) ^ i) & np.uint64(7)) + np.uint64(1)).sum(dtype=np.uint64))
    analytic_ok = analytic0 == expected[0]

    llvm = worker_strided_accumulate.inspect_llvm()
    ir = "\n".join(llvm.values()) if isinstance(llvm, dict) else str(llvm)
    fn = _extract_function(ir, "worker_strided_accumulate")
    loads_i64 = len(re.findall(r"=\s*load\s+i64", fn))
    stores_i64 = len(re.findall(r"\bstore\s+i64\b", fn))
    volatile = len(re.findall(r"\bvolatile\b", fn))
    py_calls = re.findall(r"@(Py\w+|_Py\w+|NRT_\w+|numba_\w+)", fn)
    loop_blocks = len(re.findall(r"^\s*[\w.$-]+:\s*;.*preds", fn, flags=re.M))
    has_phi = "phi i64" in fn
    store_lines = [l.strip() for l in fn.splitlines() if re.search(r"\bstore\s+i64\b", l)]
    nogil_flag = bool(worker_strided_accumulate.targetoptions.get("nogil", False))

    llvm_ok = (stores_i64 > 0 and loads_i64 > 0 and not py_calls and nogil_flag)
    audit = dict(
        function_found=bool(fn), ir_lines=len(fn.splitlines()),
        load_i64=loads_i64, store_i64=stores_i64, volatile_qualifiers=volatile,
        python_object_calls=sorted(set(py_calls)), phi_i64_present=has_phi,
        basic_blocks=loop_blocks, nogil=nogil_flag, store_i64_lines=store_lines[:6],
        passed=llvm_ok,
    )
    print(f"[parity] per-layout bit-exact vs private baseline: {parity} ; analytic thread0 match: {analytic_ok}")
    print(f"[llvm] load i64={loads_i64} store i64={stores_i64} python_object_calls={audit['python_object_calls']} "
          f"nogil={nogil_flag} -> {'PASS' if llvm_ok else 'FAIL'}")
    for l in store_lines[:4]:
        print(f"[llvm]   {l}")
    return dict(parity=parity, parity_passed=parity_ok, analytic_thread0=analytic0,
                analytic_passed=analytic_ok, expected_counters=expected, llvm_audit=audit,
                llvm_passed=llvm_ok)


def main():
    ctx = init_system_context(GIT_USERNAME)
    seeds = thread_seeds(ctx["instance_seed"])
    print(f"[init] per-thread seeds={seeds}")

    wb = aligned_zeros(64)
    worker_strided_accumulate(wb, 0, 1000, 1)
    worker_private_accumulate(wb, 1, 1000, 1)
    print(f"[run] {NUM_THREADS} threads x {ITERATIONS_PER_THREAD:,} updates, {REPEATS} repeats per layout (median)")

    single = run_single_thread_reference(seeds)
    strides = run_stride_benchmark(ctx, seeds)
    private = run_private_baseline(ctx, seeds)
    gil = run_gil_contrast(seeds)
    audit = audit_llvm_and_parity(ctx, strides, private, seeds)

    t8 = strides[8]["elapsed_ns_median"]
    summary = {
        f"stride_{s}": dict(elapsed_ns=r["elapsed_ns_median"], ns_per_update=r["ns_per_update"],
                            slowdown_vs_stride8=r["elapsed_ns_median"] / t8)
        for s, r in strides.items()
    }
    summary["private_register"] = dict(elapsed_ns=private["elapsed_ns_median"],
                                       ns_per_update=private["ns_per_update"],
                                       slowdown_vs_stride8=private["elapsed_ns_median"] / t8)
    summary["single_thread_strided"] = dict(elapsed_ns=single["elapsed_ns"], ns_per_update=single["ns_per_update"])
    summary["gil_python_slowdown_4_vs_1_threads_per_update"] = (
        gil[NUM_THREADS]["ns_per_update"] / gil[1]["ns_per_update"])

    verification_passed = bool(audit["parity_passed"] and audit["analytic_passed"] and audit["llvm_passed"])

    print("=" * 78)
    for k, v in summary.items():
        if isinstance(v, dict):
            extra = f"  slowdown vs stride8 = {v['slowdown_vs_stride8']:.3f}x" if "slowdown_vs_stride8" in v else ""
            print(f"[summary] {k:<22} {v['elapsed_ns']:>14,d} ns  {v['ns_per_update']:7.3f} ns/update{extra}")
    print(f"[summary] stride1 / stride8 slowdown = {strides[1]['elapsed_ns_median'] / t8:.3f}x")
    print(f"[summary] stride8 / stride16 ratio   = {t8 / strides[16]['elapsed_ns_median']:.3f}x")
    print(f"[verification_passed] {verification_passed}")
    print("=" * 78)

    try:
        with open(os.path.abspath(__file__), "rb") as fh:
            src_hash = hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        src_hash = None
    import numba
    manifest = dict(
        git_username=ctx["git_user"],
        instance_seed=str(ctx["instance_seed"]),
        instance_seed_hex=f"0x{ctx['instance_seed']:016x}",
        cpu_sig=ctx["cpu_sig"], cpu_model=ctx["cpu_model"], platform=ctx["platform"],
        logical_cores=ctx["logical_cores"], physical_cores=ctx["physical_cores"],
        performance_cores=ctx["performance_cores"],
        cache_line_size=ctx["cache_line_size"],
        affinity_supported=ctx["affinity_supported"], core_map=ctx["core_map"],
        num_threads=NUM_THREADS, iterations_per_thread=ITERATIONS_PER_THREAD, repeats=REPEATS,
        buffer_alignment_bytes=ALIGNMENT, thread_seeds=seeds,
        single_thread_reference=single,
        stride_results={str(k): v for k, v in strides.items()},
        private_baseline=private,
        gil_contrast={str(k): v for k, v in gil.items()},
        summary=summary,
        parity=audit["parity"], parity_passed=audit["parity_passed"],
        analytic_thread0=str(audit["analytic_thread0"]), analytic_passed=audit["analytic_passed"],
        expected_counters=[str(x) for x in audit["expected_counters"]],
        llvm_audit=audit["llvm_audit"],
        verification_passed=verification_passed,
        versions=dict(python=platform.python_version(), numpy=np.__version__, numba=numba.__version__),
        source_sha256=src_hash,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[manifest] wrote {MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].strip():
        GIT_USERNAME = sys.argv[1].strip()
    main()
