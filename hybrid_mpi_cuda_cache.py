import os
import sys
import json
import math
import time
import hashlib
import platform
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2
GLOBAL_NX = 512
GLOBAL_NY = 512
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15

PROC_DIMS = (2, 2)
REPORT_EVERY = 100
CFL_SAFETY = 0.8
KAPPA_MIN, KAPPA_MAX = 0.08, 0.24
N_PULSES = 3
PARITY_TOL = 1e-12
SHM_SLOTS = 64
SHM_ALIGN = 128
SHM_STRIDES = (1, 8, 16)
SHM_REPEATS = 3
MANIFEST_PATH = "hybrid_lab_manifest.json"
TAG_TO_UP, TAG_TO_DOWN, TAG_TO_LEFT, TAG_TO_RIGHT = 10, 11, 12, 13


def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def get_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            encoding="utf-8",
        )
        return out.strip().splitlines()[0].strip()
    except Exception:
        return "unknown"


def detect_cpu_line_size() -> int:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


def detect_cpu_topology():
    model, physical = platform.processor() or "unknown", None
    try:
        cores = set()
        phys = None
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    cores.add((phys, line.split(":")[1].strip()))
        physical = len(cores) or None
    except Exception:
        pass
    return model, physical


def log0(ctx, msg):
    if ctx["rank"] == 0:
        print(msg, flush=True)


# ==============================================================================
# TASK 1: Hybrid MPI Cartesian Grid + Node-Local MPI-3 Shared-Memory Window
# ==============================================================================
def init_hybrid_context(github_user: str) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}.")

    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    gpu_name = comm.bcast(get_gpu_name() if rank == 0 else None, root=0)
    cpu_sig = f"{platform.machine()}::{os.cpu_count()}"
    line_bytes = detect_cpu_line_size()

    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    cart_comm = comm.Create_cart(dims=list(PROC_DIMS), periods=[False, False], reorder=False)
    cart_rank = cart_comm.Get_rank()
    row_rank, col_rank = cart_comm.Get_coords(cart_rank)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    node_size = node_comm.Get_size()
    itemsize = np.dtype(np.uint64).itemsize
    nbytes = (SHM_SLOTS + SHM_ALIGN // itemsize) * itemsize if node_rank == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=node_comm)
    buf, _ = win.Shared_query(0)
    raw = np.ndarray(buffer=buf, dtype=np.uint64, shape=(SHM_SLOTS + SHM_ALIGN // itemsize,))
    offset = (-raw.ctypes.data % SHM_ALIGN) // itemsize
    shm_view = raw[offset:offset + SHM_SLOTS]
    assert shm_view.ctypes.data % 64 == 0

    cpu_model, physical_cores = detect_cpu_topology()
    ctx = dict(
        comm=comm, cart_comm=cart_comm, node_comm=node_comm, win=win, shm_view=shm_view,
        rank=cart_rank, size=size, node_rank=node_rank, node_size=node_size,
        coords=(row_rank, col_rank),
        nbr_up=nbr_up, nbr_down=nbr_down, nbr_left=nbr_left, nbr_right=nbr_right,
        gpu_uuid=gpu_uuid, gpu_name=gpu_name, cpu_sig=cpu_sig, cpu_model=cpu_model,
        logical_cores=os.cpu_count(), physical_cores=physical_cores,
        cpu_line_bytes=line_bytes, instance_seed=instance_seed, github_user=github_user.strip(),
        ny=GLOBAL_NY // PROC_DIMS[0], nx=GLOBAL_NX // PROC_DIMS[1],
        shm_base_addr=int(shm_view.ctypes.data),
    )

    def fmt(n):
        return "PROC_NULL" if n == MPI.PROC_NULL else str(n)

    for r in range(size):
        cart_comm.Barrier()
        if r == cart_rank:
            print(f"[rank {cart_rank}] coords=({row_rank},{col_rank}) node_rank={node_rank}/{node_size} "
                  f"up={fmt(nbr_up)} down={fmt(nbr_down)} left={fmt(nbr_left)} right={fmt(nbr_right)}",
                  flush=True)
    cart_comm.Barrier()
    log0(ctx, f"[init] MPI ranks={size}, Cartesian dims={list(PROC_DIMS)}, periods=[False, False], node_comm size={node_size}")
    log0(ctx, f"[init] GPU={gpu_name} UUID={gpu_uuid}")
    log0(ctx, f"[init] CPU={cpu_model} logical={os.cpu_count()} physical={physical_cores} line={line_bytes} B sig={cpu_sig}")
    log0(ctx, f"[init] GITHUB_USERNAME={ctx['github_user']} INSTANCE_SEED={instance_seed} (0x{instance_seed:016x})")
    log0(ctx, f"[init] shm window: {SHM_SLOTS} uint64 slots, base=0x{ctx['shm_base_addr']:x} "
              f"(mod 64 = {ctx['shm_base_addr'] % 64}, mod 128 = {ctx['shm_base_addr'] % 128})")
    return ctx


# ==============================================================================
# TASK 2: Rank-Local GPU Subdomain & Halo Pack/Unpack Kernels
# ==============================================================================
def seeded_parameters(instance_seed: int):
    rng = np.random.default_rng(instance_seed)
    return dict(
        kappa_phase=rng.uniform(0.0, 1.0, size=2),
        kappa_freq=rng.integers(1, 4, size=2),
        pulse_cx=rng.uniform(0.2, 0.8, size=N_PULSES),
        pulse_cy=rng.uniform(0.2, 0.8, size=N_PULSES),
        pulse_amp=rng.uniform(0.8, 1.5, size=N_PULSES),
        pulse_sigma=rng.uniform(0.05, 0.10, size=N_PULSES),
        src_cx=rng.uniform(0.3, 0.7),
        src_cy=rng.uniform(0.3, 0.7),
        src_amp=rng.uniform(0.01, 0.03),
        src_sigma=0.03,
    )


def physical_fields(X, Y, p):
    mid = 0.5 * (KAPPA_MIN + KAPPA_MAX)
    amp = 0.5 * (KAPPA_MAX - KAPPA_MIN)
    kappa = mid + amp * (np.sin(2.0 * np.pi * (p["kappa_freq"][0] * X + p["kappa_phase"][0]))
                         * np.cos(2.0 * np.pi * (p["kappa_freq"][1] * Y + p["kappa_phase"][1])))
    kappa = np.clip(kappa, KAPPA_MIN, KAPPA_MAX)
    u0 = np.zeros_like(X)
    for k in range(N_PULSES):
        r2 = (X - p["pulse_cx"][k]) ** 2 + (Y - p["pulse_cy"][k]) ** 2
        u0 += p["pulse_amp"][k] * np.exp(-r2 / (2.0 * p["pulse_sigma"][k] ** 2))
    r2s = (X - p["src_cx"]) ** 2 + (Y - p["src_cy"]) ** 2
    S = p["src_amp"] * np.exp(-r2s / (2.0 * p["src_sigma"] ** 2))
    return kappa, u0, S


def initialize_fields(ctx):
    ny, nx = ctx["ny"], ctx["nx"]
    row_rank, col_rank = ctx["coords"]
    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)
    gi = row_rank * ny + np.arange(ny + 2, dtype=np.float64)
    gj = col_rank * nx + np.arange(nx + 2, dtype=np.float64)
    Y, X = np.meshgrid(gi * dy, gj * dx, indexing="ij")

    p = seeded_parameters(ctx["instance_seed"])
    kappa, u0, S = physical_fields(X, Y, p)
    u0[0, :] = u0[ny + 1, :] = 0.0
    u0[:, 0] = u0[:, nx + 1] = 0.0
    S[0, :] = S[ny + 1, :] = 0.0
    S[:, 0] = S[:, nx + 1] = 0.0
    u0, kappa, S = (np.ascontiguousarray(a) for a in (u0, kappa, S))

    kappa_max = ctx["cart_comm"].allreduce(float(kappa[1:ny + 1, 1:nx + 1].max()), op=MPI.MAX)
    kappa_min = ctx["cart_comm"].allreduce(float(kappa[1:ny + 1, 1:nx + 1].min()), op=MPI.MIN)
    dt_max = dx * dx / (4.0 * kappa_max)
    dt = CFL_SAFETY * dt_max
    phys = dict(dx=dx, dy=dy, dt=dt, dt_max=dt_max, kappa_max=kappa_max, kappa_min=kappa_min,
                params={k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in p.items()})
    log0(ctx, f"[fields] local padded shape={u0.shape}, dx=dy={dx:.6e}, kappa in [{kappa_min:.5f}, {kappa_max:.5f}]")
    log0(ctx, f"[fields] dt_max(CFL)={dt_max:.6e}, dt={dt:.6e} (safety {CFL_SAFETY})")
    return u0, kappa, S, phys


@cuda.jit
def pack_halos_kernel(u, send_buf, ny, nx):
    k = cuda.grid(1)
    if k < nx:
        send_buf[k] = u[1, k + 1]
        send_buf[nx + k] = u[ny, k + 1]
    if k < ny:
        send_buf[2 * nx + k] = u[k + 1, 1]
        send_buf[2 * nx + ny + k] = u[k + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, recv_buf, ny, nx):
    k = cuda.grid(1)
    if k < nx:
        u[0, k + 1] = recv_buf[k]
        u[ny + 1, k + 1] = recv_buf[nx + k]
    if k < ny:
        u[k + 1, 0] = recv_buf[2 * nx + k]
        u[k + 1, nx + 1] = recv_buf[2 * nx + ny + k]


class HaloBuffers:
    def __init__(self, ny, nx):
        self.ny, self.nx = ny, nx
        n = 2 * nx + 2 * ny
        self.d_send = cuda.device_array(n, dtype=np.float64)
        self.d_recv = cuda.to_device(np.zeros(n, dtype=np.float64))
        self.h_send = cuda.pinned_array(n, dtype=np.float64)
        self.h_recv = cuda.pinned_array(n, dtype=np.float64)
        self.h_send[:] = 0.0
        self.h_recv[:] = 0.0
        self.threads = 256
        self.blocks = (max(nx, ny) + self.threads - 1) // self.threads

    def segs(self, buf):
        nx, ny = self.nx, self.ny
        return dict(top=buf[0:nx], bottom=buf[nx:2 * nx],
                    left=buf[2 * nx:2 * nx + ny], right=buf[2 * nx + ny:2 * nx + 2 * ny])


def post_halo_exchange(ctx, bufs, d_u, stream_boundary):
    pack_halos_kernel[bufs.blocks, bufs.threads, stream_boundary](d_u, bufs.d_send, bufs.ny, bufs.nx)
    bufs.d_send.copy_to_host(bufs.h_send, stream=stream_boundary)
    stream_boundary.synchronize()
    cart = ctx["cart_comm"]
    s, r = bufs.segs(bufs.h_send), bufs.segs(bufs.h_recv)
    return [
        cart.Irecv([r["top"], MPI.DOUBLE], source=ctx["nbr_up"], tag=TAG_TO_DOWN),
        cart.Irecv([r["bottom"], MPI.DOUBLE], source=ctx["nbr_down"], tag=TAG_TO_UP),
        cart.Irecv([r["left"], MPI.DOUBLE], source=ctx["nbr_left"], tag=TAG_TO_RIGHT),
        cart.Irecv([r["right"], MPI.DOUBLE], source=ctx["nbr_right"], tag=TAG_TO_LEFT),
        cart.Isend([s["top"], MPI.DOUBLE], dest=ctx["nbr_up"], tag=TAG_TO_UP),
        cart.Isend([s["bottom"], MPI.DOUBLE], dest=ctx["nbr_down"], tag=TAG_TO_DOWN),
        cart.Isend([s["left"], MPI.DOUBLE], dest=ctx["nbr_left"], tag=TAG_TO_LEFT),
        cart.Isend([s["right"], MPI.DOUBLE], dest=ctx["nbr_right"], tag=TAG_TO_RIGHT),
    ]


def complete_halo_exchange(bufs, reqs, d_u, stream_boundary):
    MPI.Request.Waitall(reqs)
    bufs.d_recv.copy_to_device(bufs.h_recv, stream=stream_boundary)
    unpack_halos_kernel[bufs.blocks, bufs.threads, stream_boundary](d_u, bufs.d_recv, bufs.ny, bufs.nx)


# ==============================================================================
# TASK 3: 2D Shared-Memory Tiled Stencil & Dual-Stream Pipeline
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_new, kappa, src, r_start, r_end, c_start, c_end,
                         dt, inv_dx2, inv_dy2, lam):
    smem = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    sy = ty + 1
    sx = tx + 1
    lo_r = r_start - 1
    hi_r = r_end + 1
    lo_c = c_start - 1
    hi_c = c_end + 1

    if i <= hi_r and j <= hi_c:
        smem[sy, sx] = u[i, j]
    if ty == 0 and i - 1 >= lo_r and j <= hi_c:
        smem[0, sx] = u[i - 1, j]
    if ty == TILE_DIM - 1 and i + 1 <= hi_r and j <= hi_c:
        smem[SHARED_DIM - 1, sx] = u[i + 1, j]
    if tx == 0 and j - 1 >= lo_c and i <= hi_r:
        smem[sy, 0] = u[i, j - 1]
    if tx == TILE_DIM - 1 and j + 1 <= hi_c and i <= hi_r:
        smem[sy, SHARED_DIM - 1] = u[i, j + 1]

    cuda.syncthreads()

    if i <= r_end and j <= c_end:
        uc = smem[sy, sx]
        u_im = smem[sy - 1, sx]
        u_ip = smem[sy + 1, sx]
        u_jm = smem[sy, sx - 1]
        u_jp = smem[sy, sx + 1]
        kc = kappa[i, j]
        k_ip = 0.5 * (kappa[i + 1, j] + kc)
        k_im = 0.5 * (kappa[i - 1, j] + kc)
        k_jp = 0.5 * (kappa[i, j + 1] + kc)
        k_jm = 0.5 * (kappa[i, j - 1] + kc)
        div_i = (k_ip * (u_ip - uc) - k_im * (uc - u_im)) * inv_dx2
        div_j = (k_jp * (u_jp - uc) - k_jm * (uc - u_jm)) * inv_dy2
        u_new[i, j] = uc + dt * (div_i + div_j - lam * uc * uc * uc + src[i, j])


def launch_stencil(st, box, stream, phys):
    r0, r1, c0, c1 = box
    grid = ((c1 - c0 + TILE_DIM) // TILE_DIM, (r1 - r0 + TILE_DIM) // TILE_DIM)
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        st["u"], st["un"], st["k"], st["s"], r0, r1, c0, c1,
        phys["dt"], 1.0 / (phys["dx"] ** 2), 1.0 / (phys["dy"] ** 2), LAMBDA_DAMP)


def make_state(u0, kappa, S, ny, nx):
    return dict(
        u=cuda.to_device(u0), un=cuda.to_device(u0), k=cuda.to_device(kappa), s=cuda.to_device(S),
        stream_boundary=cuda.stream(), stream_interior=cuda.stream(),
        full=(1, ny, 1, nx), interior=(2, ny - 1, 2, nx - 1),
        ring=[(1, 1, 1, nx), (ny, ny, 1, nx), (2, ny - 1, 1, 1), (2, ny - 1, nx, nx)],
    )


def step_serialized(ctx, st, bufs, phys):
    sb = st["stream_boundary"]
    reqs = post_halo_exchange(ctx, bufs, st["u"], sb)
    complete_halo_exchange(bufs, reqs, st["u"], sb)
    launch_stencil(st, st["full"], sb, phys)
    sb.synchronize()
    st["u"], st["un"] = st["un"], st["u"]


def step_overlapped(ctx, st, bufs, phys):
    sb, si = st["stream_boundary"], st["stream_interior"]
    reqs = post_halo_exchange(ctx, bufs, st["u"], sb)
    launch_stencil(st, st["interior"], si, phys)
    complete_halo_exchange(bufs, reqs, st["u"], sb)
    for box in st["ring"]:
        launch_stencil(st, box, sb, phys)
    sb.synchronize()
    si.synchronize()
    st["u"], st["un"] = st["un"], st["u"]


def global_energy(ctx, st, phys):
    ny, nx = ctx["ny"], ctx["nx"]
    interior = st["u"].copy_to_host()[1:ny + 1, 1:nx + 1]
    cart = ctx["cart_comm"]
    energy = cart.allreduce(float(interior.sum()) * phys["dx"] * phys["dy"], op=MPI.SUM)
    l2 = math.sqrt(cart.allreduce(float((interior ** 2).sum()), op=MPI.SUM) * phys["dx"] * phys["dy"])
    umax = cart.allreduce(float(np.abs(interior).max()), op=MPI.MAX)
    return energy, l2, umax


def run_stencil_mode(ctx, mode, u0, kappa, S, phys, bufs):
    st = make_state(u0, kappa, S, ctx["ny"], ctx["nx"])
    step_fn = step_serialized if mode == "serialized" else step_overlapped
    trace = []
    e, l2, m = global_energy(ctx, st, phys)
    trace.append(dict(step=0, energy=e, l2_norm=l2, max_abs_u=m))
    log0(ctx, f"[{mode:>10}] step {0:4d}  E={e:.12e}  ||u||2={l2:.6e}  max|u|={m:.6f}")
    wall = 0.0
    ev_ms = 0.0
    for start in range(0, STENCIL_STEPS, REPORT_EVERY):
        chunk = min(REPORT_EVERY, STENCIL_STEPS - start)
        cuda.synchronize()
        ctx["cart_comm"].Barrier()
        ev0, ev1 = cuda.event(), cuda.event()
        t0 = MPI.Wtime()
        ev0.record()
        for _ in range(chunk):
            step_fn(ctx, st, bufs, phys)
        ev1.record()
        ev1.synchronize()
        wall += MPI.Wtime() - t0
        try:
            ev_ms += cuda.event_elapsed_time(ev0, ev1)
        except Exception:
            pass
        e, l2, m = global_energy(ctx, st, phys)
        trace.append(dict(step=start + chunk, energy=e, l2_norm=l2, max_abs_u=m))
        log0(ctx, f"[{mode:>10}] step {start + chunk:4d}  E={e:.12e}  ||u||2={l2:.6e}  max|u|={m:.6f}")
    wall_ms = ctx["cart_comm"].allreduce(wall * 1e3, op=MPI.MAX)
    ev_ms = ctx["cart_comm"].allreduce(ev_ms, op=MPI.MAX)
    return dict(trace=trace, wall_ms=wall_ms, event_ms=ev_ms, final=st["u"].copy_to_host())


def ptx_audit():
    import re
    try:
        asm = tiled_stencil_kernel.inspect_asm()
    except Exception as exc:
        return dict(available=False, error=repr(exc), st_shared_f64=0, ld_shared_f64=0, bar_sync=0, passed=False)
    ptx = "\n".join(asm.values()) if isinstance(asm, dict) else str(asm)
    st_n = len(re.findall(r"\bst\.shared\.f64\b", ptx))
    ld_n = len(re.findall(r"\bld\.shared\.f64\b", ptx))
    st_any = len(re.findall(r"\bst\.shared\.", ptx))
    ld_any = len(re.findall(r"\bld\.shared\.", ptx))
    bar = len(re.findall(r"\bbar\.sync\b", ptx)) + len(re.findall(r"\bbarrier\.sync\b", ptx))
    return dict(available=True, st_shared_f64=st_n, ld_shared_f64=ld_n, st_shared_any=st_any,
                ld_shared_any=ld_any, bar_sync=bar, passed=bool(st_any > 0 and ld_any > 0 and bar > 0))


# ==============================================================================
# TASK 4: Multi-Rank Host Cache-Line False Sharing Sweep
# ==============================================================================
@njit(nogil=True, cache=False)
def host_shm_accumulate(shm_view, slot, iters, seed):
    s = uint64(seed)
    j = slot
    for i in range(iters):
        v = shm_view[j]
        nv = v + ((s ^ uint64(i)) & uint64(7)) + uint64(1)
        shm_view[j] = nv
        j = slot + np.int64(nv >> uint64(60))
    return j


def expected_counter(seed, iters):
    total = np.uint64(0)
    chunk = 10_000_000
    for a in range(0, iters, chunk):
        i = np.arange(a, min(a + chunk, iters), dtype=np.uint64)
        total += (((np.uint64(seed) ^ i) & np.uint64(7)) + np.uint64(1)).sum(dtype=np.uint64)
    return int(total)


def benchmark_shm_strides(ctx, rank_seed):
    node_comm, win, view = ctx["node_comm"], ctx["win"], ctx["shm_view"]
    rank = ctx["node_rank"]
    line = ctx["cpu_line_bytes"]
    base = ctx["shm_base_addr"]
    expected = expected_counter(rank_seed, SHM_ITERS)

    host_shm_accumulate(np.zeros(8, dtype=np.uint64), 0, 1000, 1)
    node_comm.Barrier()

    results = {}
    parity = {}
    configs = [("solo_stride1", 1, True)] + [(f"stride_{s}", s, False) for s in SHM_STRIDES]
    for name, stride, solo in configs:
        slot = rank * stride
        reps = []
        for _ in range(SHM_REPEATS):
            win.Fence()
            if rank == 0:
                view[:] = 0
            win.Fence()
            node_comm.Barrier()
            t0 = time.perf_counter_ns()
            if (not solo) or rank == 0:
                host_shm_accumulate(view, slot, SHM_ITERS, rank_seed)
            t1 = time.perf_counter_ns()
            node_comm.Barrier()
            t2 = time.perf_counter_ns()
            win.Fence()
            own = t1 - t0
            reps.append(dict(rank_kernel_ns_max=node_comm.allreduce(own, op=MPI.MAX),
                             barrier_to_barrier_ns=node_comm.allreduce(t2 - t0, op=MPI.MAX)))
            ok = (int(view[slot]) == expected) if ((not solo) or rank == 0) else True
            parity_ok = node_comm.allreduce(int(ok), op=MPI.MIN) == 1
        med = sorted(r["rank_kernel_ns_max"] for r in reps)[len(reps) // 2]
        slots = [r * stride for r in range(ctx["node_size"])] if not solo else [0]
        lines64 = sorted({(base + 8 * s) // 64 - base // 64 for s in slots})
        lines_native = sorted({(base + 8 * s) // line - base // line for s in slots})
        results[name] = dict(stride=stride, byte_spacing=8 * stride, solo=solo, slots=slots,
                             cpu_lines_64B=lines64, cpu_lines_native=lines_native,
                             latency_ns_median=med, ns_per_update=med / SHM_ITERS, repeats=reps)
        parity[name] = parity_ok
        log0(ctx, f"[shm {name:<13}] {med:>14,d} ns  {med/SHM_ITERS:7.3f} ns/update  "
                  f"slots={slots} lines(64B)={lines64}  parity={parity_ok}")
    s1 = results["stride_1"]["latency_ns_median"]
    s8 = results["stride_8"]["latency_ns_median"]
    s16 = results["stride_16"]["latency_ns_median"]
    ratios = dict(false_sharing_slowdown_stride1_vs_stride8=s1 / s8,
                  stride1_vs_stride16=s1 / s16, stride8_vs_stride16=s8 / s16)
    log0(ctx, f"[shm ratios] stride1/stride8={ratios['false_sharing_slowdown_stride1_vs_stride8']:.3f}x  "
              f"stride1/stride16={ratios['stride1_vs_stride16']:.3f}x  stride8/stride16={ratios['stride8_vs_stride16']:.3f}x")
    return dict(results=results, ratios=ratios, parity=parity, parity_passed=all(parity.values()),
                expected_counter_rank0=str(node_comm.bcast(expected, root=0)))


def llvm_audit():
    import re
    llvm = host_shm_accumulate.inspect_llvm()
    ir = "\n".join(llvm.values()) if isinstance(llvm, dict) else str(llvm)
    blocks, cur = [], None
    for line in ir.splitlines():
        if line.startswith("define "):
            cur = [line]
        elif cur is not None:
            cur.append(line)
            if line.startswith("}"):
                blocks.append("\n".join(cur))
                cur = None
    fn = ""
    for b in blocks:
        head = b.splitlines()[0]
        if "host_shm_accumulate" in head and "cpython" not in head and "cfunc" not in head:
            fn = b
            break
    loads = len(re.findall(r"=\s*load\s+i64", fn))
    stores = len(re.findall(r"\bstore\s+i64\b", fn))
    py_calls = sorted(set(re.findall(r"@(Py\w+|_Py\w+|NRT_\w+)", fn)))
    nogil = bool(host_shm_accumulate.targetoptions.get("nogil", False))
    return dict(function_found=bool(fn), load_i64=loads, store_i64=stores, python_object_calls=py_calls,
                nogil=nogil, store_i64_lines=[l.strip() for l in fn.splitlines() if re.search(r"\bstore\s+i64\b", l)][:6],
                passed=bool(fn and loads > 0 and stores > 0 and not py_calls and nogil))


# ==============================================================================
# TASK 5: Cross-Tier Verification & Manifest Export
# ==============================================================================
def main(github_user: str):
    ctx = init_hybrid_context(github_user)
    u0, kappa, S, phys = initialize_fields(ctx)
    bufs = HaloBuffers(ctx["ny"], ctx["nx"])

    warm = make_state(u0, kappa, S, ctx["ny"], ctx["nx"])
    step_overlapped(ctx, warm, bufs, phys)
    step_serialized(ctx, warm, bufs, phys)
    cuda.synchronize()
    ctx["cart_comm"].Barrier()
    log0(ctx, f"[gpu] kernels compiled; running {STENCIL_STEPS} steps per mode")

    ser = run_stencil_mode(ctx, "serialized", u0, kappa, S, phys, bufs)
    ovl = run_stencil_mode(ctx, "overlapped", u0, kappa, S, phys, bufs)
    parity_max_diff = ctx["cart_comm"].allreduce(float(np.abs(ser["final"] - ovl["final"]).max()), op=MPI.MAX)
    gpu_parity_ok = parity_max_diff < PARITY_TOL
    energies = [t["energy"] for t in ovl["trace"]]
    energy_monotonic = all(energies[k + 1] <= energies[k] for k in range(len(energies) - 1))
    traces_match = all(abs(a["energy"] - b["energy"]) < 1e-14 for a, b in zip(ser["trace"], ovl["trace"]))
    ptx = ctx["cart_comm"].bcast(ptx_audit() if ctx["rank"] == 0 else None, root=0)

    rank_seed = int(np.random.default_rng(ctx["instance_seed"]).integers(1, 2**31 - 1, size=4)[ctx["node_rank"]])
    shm = benchmark_shm_strides(ctx, rank_seed)
    llvm = ctx["cart_comm"].bcast(llvm_audit() if ctx["rank"] == 0 else None, root=0)

    topo = np.zeros((ctx["size"], 6), dtype=np.int64)
    topo[ctx["rank"]] = [ctx["coords"][0], ctx["coords"][1], ctx["nbr_up"], ctx["nbr_down"],
                         ctx["nbr_left"], ctx["nbr_right"]]
    topo_all = np.zeros_like(topo)
    ctx["cart_comm"].Allreduce(topo, topo_all, op=MPI.SUM)

    verification_passed = bool(gpu_parity_ok and ptx["passed"] and llvm["passed"]
                               and shm["parity_passed"] and energy_monotonic)

    if ctx["rank"] == 0:
        print("=" * 78)
        print(f"[gpu bench] gpu_serialized_ms = {ser['wall_ms']:.3f}  (CUDA events {ser['event_ms']:.3f})")
        print(f"[gpu bench] gpu_overlapped_ms = {ovl['wall_ms']:.3f}  (CUDA events {ovl['event_ms']:.3f})")
        print(f"[gpu bench] overlap speedup = {ser['wall_ms'] / ovl['wall_ms']:.4f}x")
        print(f"[gpu parity] max |serialized - overlapped| = {parity_max_diff:.3e} -> {'PASS' if gpu_parity_ok else 'FAIL'}")
        print(f"[energy] monotonic={energy_monotonic}  serialized/overlapped traces identical={traces_match}")
        print(f"[ptx] st.shared.f64={ptx['st_shared_f64']} ld.shared.f64={ptx['ld_shared_f64']} "
              f"bar.sync={ptx['bar_sync']} -> {'PASS' if ptx['passed'] else 'FAIL'}")
        print(f"[llvm] load i64={llvm['load_i64']} store i64={llvm['store_i64']} "
              f"python_object_calls={llvm['python_object_calls']} nogil={llvm['nogil']} -> {'PASS' if llvm['passed'] else 'FAIL'}")
        for l in llvm["store_i64_lines"][:4]:
            print(f"[llvm]   {l}")
        print(f"[shm parity] {shm['parity']}")
        print(f"[verification_passed] {verification_passed}")
        print("=" * 78)

        try:
            with open(os.path.abspath(__file__), "rb") as fh:
                src_hash = hashlib.sha256(fh.read()).hexdigest()
        except Exception:
            src_hash = None
        import numba
        manifest = dict(
            github_username=ctx["github_user"],
            gpu_uuid=ctx["gpu_uuid"], gpu_name=ctx["gpu_name"],
            cpu_sig=ctx["cpu_sig"], cpu_model=ctx["cpu_model"],
            logical_cores=ctx["logical_cores"], physical_cores=ctx["physical_cores"],
            cpu_line_bytes=ctx["cpu_line_bytes"],
            instance_seed=str(ctx["instance_seed"]), instance_seed_hex=f"0x{ctx['instance_seed']:016x}",
            mpi_ranks=ctx["size"], cart_dims=list(PROC_DIMS), cart_periods=[False, False],
            node_comm_size=ctx["node_size"], mpi_proc_null=int(MPI.PROC_NULL),
            topology=[dict(rank=r, coords=[int(topo_all[r, 0]), int(topo_all[r, 1])],
                           up=int(topo_all[r, 2]), down=int(topo_all[r, 3]),
                           left=int(topo_all[r, 4]), right=int(topo_all[r, 5])) for r in range(ctx["size"])],
            global_grid=[GLOBAL_NY, GLOBAL_NX], local_padded_shape=[ctx["ny"] + 2, ctx["nx"] + 2],
            tile_dim=TILE_DIM, shared_dim=SHARED_DIM, stencil_steps=STENCIL_STEPS, lambda_damp=LAMBDA_DAMP,
            dx=phys["dx"], dy=phys["dy"], dt=phys["dt"], dt_max_cfl=phys["dt_max"],
            kappa_max=phys["kappa_max"], kappa_min=phys["kappa_min"], seeded_params=phys["params"],
            energy_trace=ovl["trace"], energy_trace_serialized=ser["trace"],
            energy_monotonic=energy_monotonic, energy_traces_identical=traces_match,
            gpu_serialized_ms=ser["wall_ms"], gpu_overlapped_ms=ovl["wall_ms"],
            gpu_serialized_event_ms=ser["event_ms"], gpu_overlapped_event_ms=ovl["event_ms"],
            gpu_overlap_speedup=ser["wall_ms"] / ovl["wall_ms"],
            parity_max_diff=parity_max_diff, gpu_parity_passed=gpu_parity_ok,
            ptx_audit=ptx, ptx_audit_passed=ptx["passed"],
            shm_window=dict(slots=SHM_SLOTS, alignment_bytes=SHM_ALIGN, base_mod_64=ctx["shm_base_addr"] % 64,
                            base_mod_128=ctx["shm_base_addr"] % 128, iters_per_rank=SHM_ITERS,
                            repeats=SHM_REPEATS),
            shm_results=shm["results"], shm_ratios=shm["ratios"],
            shm_parity=shm["parity"], shm_parity_passed=shm["parity_passed"],
            shm_expected_counter_rank0=shm["expected_counter_rank0"],
            llvm_audit=llvm, llvm_audit_passed=llvm["passed"],
            verification_passed=verification_passed,
            tasks_completed=[1, 2, 3, 4, 5],
            versions=dict(python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                          mpi=MPI.Get_library_version().splitlines()[0]),
            source_sha256=src_hash,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[manifest] wrote {MANIFEST_PATH}")

    ctx["cart_comm"].Barrier()
    ctx["win"].Free()


if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        raise SystemExit("usage: mpirun -np 4 python3 hybrid_mpi_cuda_cache.py <github_username>")
    main(sys.argv[1].strip())
