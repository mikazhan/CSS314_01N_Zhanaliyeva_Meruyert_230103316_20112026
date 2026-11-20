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
from numba import cuda, float64


TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15


PROC_DIMS = (2, 2)
REPORT_EVERY = 100
CFL_SAFETY = 0.8         
KAPPA_MIN, KAPPA_MAX = 0.08, 0.24
N_PULSES = 3
PARITY_TOL = 1e-12
MANIFEST_PATH = "submission_manifest.json"


TAG_TO_UP, TAG_TO_DOWN, TAG_TO_LEFT, TAG_TO_RIGHT = 10, 11, 12, 13


def get_gpu_uuid() -> str:
    """Query active NVIDIA GPU UUID via nvidia-smi."""
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


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive deterministic 64-bit integer seed bound to student GitHub handle and GPU UUID."""
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def log0(ctx, msg):
    if ctx["rank"] == 0:
        print(msg, flush=True)


# ==============================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    
    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    gpu_name = get_gpu_name() if rank == 0 else None
    gpu_name = comm.bcast(gpu_name, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    
    cart_comm = comm.Create_cart(dims=list(PROC_DIMS), periods=[False, False], reorder=False)
    cart_rank = cart_comm.Get_rank()
    row_rank, col_rank = cart_comm.Get_coords(cart_rank)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)    
    nbr_left, nbr_right = cart_comm.Shift(1, 1)  

    ny = GLOBAL_NY // PROC_DIMS[0]
    nx = GLOBAL_NX // PROC_DIMS[1]

    ctx = dict(
        comm=comm, cart_comm=cart_comm, rank=cart_rank, size=size,
        coords=(row_rank, col_rank),
        nbr_up=nbr_up, nbr_down=nbr_down, nbr_left=nbr_left, nbr_right=nbr_right,
        gpu_uuid=gpu_uuid, gpu_name=gpu_name, instance_seed=instance_seed,
        github_user=github_user, ny=ny, nx=nx,
    )

    def fmt(n):
        return "PROC_NULL" if n == MPI.PROC_NULL else str(n)

    for r in range(size):
        cart_comm.Barrier()
        if r == cart_rank:
            print(f"[rank {cart_rank}] coords=({row_rank},{col_rank}) local=({ny}+2)x({nx}+2) "
                  f"up={fmt(nbr_up)} down={fmt(nbr_down)} left={fmt(nbr_left)} right={fmt(nbr_right)}",
                  flush=True)
    cart_comm.Barrier()
    log0(ctx, f"[init] MPI ranks={size}, Cartesian dims={list(PROC_DIMS)}, periods=[False, False]")
    log0(ctx, f"[init] GPU={gpu_name} UUID={gpu_uuid}")
    log0(ctx, f"[init] GITHUB_USERNAME={github_user} INSTANCE_SEED={instance_seed} (0x{instance_seed:016x})")
    return ctx


# ==============================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization
# ==============================================================================
def seeded_parameters(instance_seed: int):
    """All ranks derive identical physical parameters from INSTANCE_SEED (no communication)."""
    rng = np.random.default_rng(instance_seed)
    params = dict(
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
    return params


def physical_fields(X, Y, p):
    """Evaluate kappa(x,y), u0(x,y), S(x,y) on coordinate arrays (X = column dir j, Y = row dir i)."""
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


def initialize_subdomains(ctx):
    ny, nx = ctx["ny"], ctx["nx"]
    row_rank, col_rank = ctx["coords"]
    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)

    
    gi = row_rank * ny + np.arange(ny + 2, dtype=np.float64)
    gj = col_rank * nx + np.arange(nx + 2, dtype=np.float64)
    Y, X = np.meshgrid(gi * dy, gj * dx, indexing="ij")  

    p = seeded_parameters(ctx["instance_seed"])
    kappa, u0, S = physical_fields(X, Y, p)

    
    u0[0, :] = 0.0
    u0[ny + 1, :] = 0.0
    u0[:, 0] = 0.0
    u0[:, nx + 1] = 0.0
   
    S[0, :] = S[ny + 1, :] = 0.0
    S[:, 0] = S[:, nx + 1] = 0.0

    u0 = np.ascontiguousarray(u0)
    kappa = np.ascontiguousarray(kappa)
    S = np.ascontiguousarray(S)

    local_kmax = np.array([kappa[1:ny + 1, 1:nx + 1].max()])
    local_kmin = np.array([kappa[1:ny + 1, 1:nx + 1].min()])
    kmax = np.zeros(1)
    kmin = np.zeros(1)
    ctx["cart_comm"].Allreduce(local_kmax, kmax, op=MPI.MAX)
    ctx["cart_comm"].Allreduce(local_kmin, kmin, op=MPI.MIN)
    kappa_max = float(kmax[0])
    dt_max = dx * dx / (4.0 * kappa_max)
    dt = CFL_SAFETY * dt_max

    phys = dict(dx=dx, dy=dy, dt=dt, dt_max=dt_max, kappa_max=kappa_max, kappa_min=float(kmin[0]),
                params={k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in p.items()})
    log0(ctx, f"[init] local padded shape={u0.shape}, dx=dy={dx:.6e}, kappa in [{phys['kappa_min']:.5f}, "
              f"{kappa_max:.5f}]")
    log0(ctx, f"[init] dt_max(CFL)={dt_max:.6e}, dt={dt:.6e} (safety {CFL_SAFETY})")
    return u0, kappa, S, phys


# ==============================================================================
# TASK 3: GPU Halo Pack/Unpack Kernels & Pinned MPI Exchange
# ==============================================================================

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
    """Device 1D send/recv vectors + page-locked (pinned) host staging buffers."""

    def __init__(self, ny, nx):
        self.ny, self.nx = ny, nx
        n = 2 * nx + 2 * ny
        self.n = n
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


def exchange_halos_async(ctx, bufs, d_u, stream_boundary):
    """Stage A + B(MPI part): GPU pack -> async D2H into pinned -> post Irecv/Isend.
    Returns MPI requests; the caller overlaps compute and then calls finish_halos()."""
    ny, nx = bufs.ny, bufs.nx
    cart = ctx["cart_comm"]
   
    pack_halos_kernel[bufs.blocks, bufs.threads, stream_boundary](d_u, bufs.d_send, ny, nx)
    bufs.d_send.copy_to_host(bufs.h_send, stream=stream_boundary)
    stream_boundary.synchronize()  

    s = bufs.segs(bufs.h_send)
    r = bufs.segs(bufs.h_recv)
    reqs = [
        
        cart.Irecv([r["top"], MPI.DOUBLE], source=ctx["nbr_up"], tag=TAG_TO_DOWN),
        cart.Irecv([r["bottom"], MPI.DOUBLE], source=ctx["nbr_down"], tag=TAG_TO_UP),
        cart.Irecv([r["left"], MPI.DOUBLE], source=ctx["nbr_left"], tag=TAG_TO_RIGHT),
        cart.Irecv([r["right"], MPI.DOUBLE], source=ctx["nbr_right"], tag=TAG_TO_LEFT),
        
        cart.Isend([s["top"], MPI.DOUBLE], dest=ctx["nbr_up"], tag=TAG_TO_UP),
        cart.Isend([s["bottom"], MPI.DOUBLE], dest=ctx["nbr_down"], tag=TAG_TO_DOWN),
        cart.Isend([s["left"], MPI.DOUBLE], dest=ctx["nbr_left"], tag=TAG_TO_LEFT),
        cart.Isend([s["right"], MPI.DOUBLE], dest=ctx["nbr_right"], tag=TAG_TO_RIGHT),
    ]
    return reqs


def finish_halos(bufs, reqs, d_u, stream_boundary):
    """Stage C (first half): Waitall -> async H2D -> GPU unpack into ghost rows/cols 0 and n+1."""
    MPI.Request.Waitall(reqs)
    bufs.d_recv.copy_to_device(bufs.h_recv, stream=stream_boundary)
    unpack_halos_kernel[bufs.blocks, bufs.threads, stream_boundary](d_u, bufs.d_recv, bufs.ny, bufs.nx)


# ==============================================================================
# TASK 4: 2D Shared-Memory Tiled Stencil & Dual-Stream Orchestrator
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_new, kappa, src, r_start, r_end, c_start, c_end,
                         dt, inv_dx2, inv_dy2, lam):
    smem = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty   # global (local-array) row
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx   # global (local-array) column
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


def launch_stencil(d_u, d_un, d_k, d_s, box, stream, phys):
    r0, r1, c0, c1 = box
    grid = ((c1 - c0 + 1 + TILE_DIM - 1) // TILE_DIM, (r1 - r0 + 1 + TILE_DIM - 1) // TILE_DIM)
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        d_u, d_un, d_k, d_s, r0, r1, c0, c1,
        phys["dt"], 1.0 / (phys["dx"] ** 2), 1.0 / (phys["dy"] ** 2), LAMBDA_DAMP)


def region_boxes(ny, nx):
    full = (1, ny, 1, nx)
    interior = (2, ny - 1, 2, nx - 1)
    ring = [
        (1, 1, 1, nx),           
        (ny, ny, 1, nx),        
        (2, ny - 1, 1, 1),        
        (2, ny - 1, nx, nx),     
    ]
    return full, interior, ring


def step_serialized(ctx, st, bufs, phys):
    """Reference: blocking-style halo exchange, then one full-domain stencil, single stream."""
    s = st["stream_boundary"]
    reqs = exchange_halos_async(ctx, bufs, st["u"], s)
    finish_halos(bufs, reqs, st["u"], s)
    launch_stencil(st["u"], st["un"], st["k"], st["s"], st["full"], s, phys)
    s.synchronize()
    st["u"], st["un"] = st["un"], st["u"]


def step_overlapped(ctx, st, bufs, phys):
    sb, si = st["stream_boundary"], st["stream_interior"]
    
    reqs = exchange_halos_async(ctx, bufs, st["u"], sb)
    
    launch_stencil(st["u"], st["un"], st["k"], st["s"], st["interior"], si, phys)
   
    finish_halos(bufs, reqs, st["u"], sb)
    for box in st["ring"]:
        launch_stencil(st["u"], st["un"], st["k"], st["s"], box, sb, phys)
    sb.synchronize()
    si.synchronize()
    st["u"], st["un"] = st["un"], st["u"]


# ==============================================================================
# TASK 5: Distributed Reductions, PTX Audit & Comparative Benchmark
# ==============================================================================
def global_diagnostics(ctx, st, phys, h_prev):
    ny, nx = ctx["ny"], ctx["nx"]
    h_u = st["u"].copy_to_host()
    interior = h_u[1:ny + 1, 1:nx + 1]
    loc = np.array([
        interior.sum() * phys["dx"] * phys["dy"],
        0.0 if h_prev is None else float(((interior - h_prev) ** 2).sum()),
        float((interior ** 2).sum()),
        float(np.abs(interior).max()),
    ])
    glob_sum = np.zeros(3)
    ctx["cart_comm"].Allreduce(loc[:3].copy(), glob_sum, op=MPI.SUM)
    glob_max = np.zeros(1)
    ctx["cart_comm"].Allreduce(loc[3:4].copy(), glob_max, op=MPI.MAX)
    energy = float(glob_sum[0])
    l2_residual = math.sqrt(glob_sum[1] * phys["dx"] * phys["dy"])  # ||u^n - u^(n-1)||_L2
    l2_norm = math.sqrt(glob_sum[2] * phys["dx"] * phys["dy"])
    return energy, l2_residual, l2_norm, float(glob_max[0]), interior.copy()


def make_state(u0, kappa, S, ctx):
    ny, nx = ctx["ny"], ctx["nx"]
    full, interior, ring = region_boxes(ny, nx)
    return dict(
        u=cuda.to_device(u0), un=cuda.to_device(u0),
        k=cuda.to_device(kappa), s=cuda.to_device(S),
        stream_boundary=cuda.stream(), stream_interior=cuda.stream(),
        full=full, interior=interior, ring=ring,
    )


def run_mode(ctx, mode, u0, kappa, S, phys, bufs, steps):
    st = make_state(u0, kappa, S, ctx)
    step_fn = step_serialized if mode == "serialized" else step_overlapped
    trace = []
    h_prev = None
    step_time = 0.0
    ev_total_ms = 0.0

    e, r, n, m, h_prev = global_diagnostics(ctx, st, phys, None)
    trace.append(dict(step=0, energy=e, l2_residual=r, l2_norm=n, max_abs_u=m))
    log0(ctx, f"[{mode:>10}] step {0:4d}  E={e:.12e}  dU_L2=n/a        ||u||2={n:.6e}  max|u|={m:.6f}")

    for start in range(0, steps, REPORT_EVERY):
        chunk = min(REPORT_EVERY, steps - start)
        cuda.synchronize()
        ctx["cart_comm"].Barrier()
        ev0, ev1 = cuda.event(), cuda.event()  
        t0 = MPI.Wtime()
        ev0.record()
        for _ in range(chunk):
            step_fn(ctx, st, bufs, phys)
        ev1.record()
        ev1.synchronize()
        cuda.synchronize()
        step_time += MPI.Wtime() - t0
        try:
            ev_total_ms += cuda.event_elapsed_time(ev0, ev1)
        except Exception:
            pass
       
        e, r, n, m, h_cur = global_diagnostics(ctx, st, phys, h_prev)
        trace.append(dict(step=start + chunk, energy=e, l2_residual=r, l2_norm=n, max_abs_u=m))
        log0(ctx, f"[{mode:>10}] step {start + chunk:4d}  E={e:.12e}  dU_L2({REPORT_EVERY} steps)={r:.3e}  "
                  f"||u||2={n:.6e}  max|u|={m:.6f}")
        h_prev = h_cur

    wall_ms = np.array([step_time * 1e3])
    evm = np.array([ev_total_ms])
    wall_max = np.zeros(1)
    ev_max = np.zeros(1)
    ctx["cart_comm"].Allreduce(wall_ms, wall_max, op=MPI.MAX)
    ctx["cart_comm"].Allreduce(evm, ev_max, op=MPI.MAX)
    final_local = st["u"].copy_to_host()
    return dict(trace=trace, wall_ms=float(wall_max[0]), event_ms=float(ev_max[0]), final=final_local)


def ptx_audit():
    try:
        asm = tiled_stencil_kernel.inspect_asm()
    except Exception as exc:  
        return dict(available=False, error=repr(exc), st_shared_f64=0, ld_shared_f64=0,
                    bar_sync=0, passed=False)
    if isinstance(asm, dict):
        ptx = "\n".join(asm.values())
    else:
        ptx = str(asm)
    import re
    st_count = len(re.findall(r"\bst\.shared\.f64\b", ptx))
    ld_count = len(re.findall(r"\bld\.shared\.f64\b", ptx))
    st_any = len(re.findall(r"\bst\.shared\.", ptx))
    ld_any = len(re.findall(r"\bld\.shared\.", ptx))
    bar = len(re.findall(r"\bbar\.sync\b", ptx)) + len(re.findall(r"\bbarrier\.sync\b", ptx))
    return dict(available=True, st_shared_f64=st_count, ld_shared_f64=ld_count,
                st_shared_any=st_any, ld_shared_any=ld_any, bar_sync=bar,
                ptx_lines=len(ptx.splitlines()),
                passed=bool(st_any > 0 and ld_any > 0 and bar > 0))


def warmup(ctx, u0, kappa, S, phys, bufs):
    """JIT-compile every kernel once (excluded from timing)."""
    st = make_state(u0, kappa, S, ctx)
    step_overlapped(ctx, st, bufs, phys)
    step_serialized(ctx, st, bufs, phys)
    cuda.synchronize()
    ctx["cart_comm"].Barrier()


def run_solver(github_user: str, steps: int = STEPS, write_manifest: bool = True):
    ctx = init_distributed_context(github_user)
    u0, kappa, S, phys = initialize_subdomains(ctx)
    bufs = HaloBuffers(ctx["ny"], ctx["nx"])

    warmup(ctx, u0, kappa, S, phys, bufs)
    log0(ctx, f"[run] kernels compiled; running {steps} steps per mode")

    ser = run_mode(ctx, "serialized", u0, kappa, S, phys, bufs, steps)
    ovl = run_mode(ctx, "overlapped", u0, kappa, S, phys, bufs, steps)

    
    loc_diff = np.array([float(np.abs(ser["final"] - ovl["final"]).max())])
    glob_diff = np.zeros(1)
    ctx["cart_comm"].Allreduce(loc_diff, glob_diff, op=MPI.MAX)
    parity_max_diff = float(glob_diff[0])
    parity_ok = parity_max_diff < PARITY_TOL

    
    topo = np.zeros((ctx["size"], 6), dtype=np.int64)
    topo[ctx["rank"]] = [ctx["coords"][0], ctx["coords"][1],
                         ctx["nbr_up"], ctx["nbr_down"], ctx["nbr_left"], ctx["nbr_right"]]
    topo_all = np.zeros_like(topo)
    ctx["cart_comm"].Allreduce(topo, topo_all, op=MPI.SUM)

    audit = ptx_audit() if ctx["rank"] == 0 else None
    audit = ctx["cart_comm"].bcast(audit, root=0)

    energies = [t["energy"] for t in ovl["trace"]]
    monotone = all(energies[k + 1] <= energies[k] for k in range(len(energies) - 1))
    speedup = ser["wall_ms"] / ovl["wall_ms"] if ovl["wall_ms"] > 0 else float("nan")

    if ctx["rank"] == 0:
        print("=" * 78)
        print(f"[bench] serialized_ms = {ser['wall_ms']:.3f}  (CUDA events: {ser['event_ms']:.3f} ms)")
        print(f"[bench] overlapped_ms = {ovl['wall_ms']:.3f}  (CUDA events: {ovl['event_ms']:.3f} ms)")
        print(f"[bench] overlap speedup = {speedup:.4f}x")
        print(f"[parity] max |u_serialized - u_overlapped| = {parity_max_diff:.3e} -> "
              f"{'PASS' if parity_ok else 'FAIL'} (tol {PARITY_TOL:g})")
        print(f"[ptx] st.shared.f64={audit['st_shared_f64']} ld.shared.f64={audit['ld_shared_f64']} "
              f"bar.sync={audit['bar_sync']} -> {'PASS' if audit['passed'] else 'FAIL'}")
        print(f"[energy] monotonic non-increasing over reported steps: {monotone}")
        print("=" * 78)

    manifest = None
    if ctx["rank"] == 0:
        try:
            with open(os.path.abspath(__file__), "rb") as fh:
                src_hash = hashlib.sha256(fh.read()).hexdigest()
        except Exception:
            src_hash = None
        import numba
        manifest = dict(
            github_username=github_user,
            gpu_uuid=ctx["gpu_uuid"],
            gpu_name=ctx["gpu_name"],
            instance_seed=str(ctx["instance_seed"]),
            instance_seed_hex=f"0x{ctx['instance_seed']:016x}",
            mpi_ranks=ctx["size"],
            cart_dims=list(PROC_DIMS),
            cart_periods=[False, False],
            topology=[dict(rank=r, coords=[int(topo_all[r, 0]), int(topo_all[r, 1])],
                           up=int(topo_all[r, 2]), down=int(topo_all[r, 3]),
                           left=int(topo_all[r, 4]), right=int(topo_all[r, 5]))
                      for r in range(ctx["size"])],
            mpi_proc_null=int(MPI.PROC_NULL),
            global_grid=[GLOBAL_NY, GLOBAL_NX],
            local_padded_shape=[ctx["ny"] + 2, ctx["nx"] + 2],
            tile_dim=TILE_DIM, shared_dim=SHARED_DIM,
            steps=steps, lambda_damp=LAMBDA_DAMP,
            dx=phys["dx"], dy=phys["dy"], dt=phys["dt"], dt_max_cfl=phys["dt_max"],
            kappa_max=phys["kappa_max"], kappa_min=phys["kappa_min"],
            seeded_params=phys["params"],
            energy_trace=ovl["trace"],
            energy_trace_serialized=ser["trace"],
            energy_monotonic=monotone,
            serialized_ms=ser["wall_ms"], overlapped_ms=ovl["wall_ms"],
            serialized_event_ms=ser["event_ms"], overlapped_event_ms=ovl["event_ms"],
            overlap_speedup=speedup,
            parity_max_diff=parity_max_diff, parity_passed=parity_ok,
            ptx_audit=audit, ptx_audit_passed=audit["passed"],
            tasks_completed=[1, 2, 3, 4, 5],
            versions=dict(python=platform.python_version(), numpy=np.__version__,
                          numba=numba.__version__, mpi=MPI.Get_library_version().splitlines()[0]),
            solver_sha256=src_hash,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        if write_manifest:
            with open(MANIFEST_PATH, "w") as fh:
                json.dump(manifest, fh, indent=2)
            print(f"[manifest] wrote {MANIFEST_PATH}")
    return dict(ctx=ctx, serialized=ser, overlapped=ovl, parity=parity_max_diff,
                audit=audit, manifest=manifest, phys=phys)


if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        raise SystemExit("usage: mpirun -np 4 python3 distributed_cuda_solver.py <github_username>")
    run_solver(sys.argv[1].strip())
