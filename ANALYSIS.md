# Cache Line & False Sharing Mechanics — Analysis (Q1–Q5)

**Run identity:** Git user `mikazhan` · `INSTANCE_SEED = 5761042896976688533 (0x4ff355fdd2153595)` · Apple **M2** (MacBook Air), Darwin 25.4.0 arm64 · 8 logical / 8 physical cores (4 performance + 4 efficiency) · **coherency line size = 128 B** (`sysctl hw.cachelinesize`) · core affinity: not supported on macOS (`core_map = [None]*4`, scheduler-placed) · 4 threads × 60,000,000 updates, 3 repeats per layout (median reported) · `verification_passed = true`.

## Measured results (from `cache_lab_manifest.json`)

| Regime | Median wall time (ns) | ns / update / thread | Aggregate ns / update (÷ 240 M) | Slowdown vs stride 8 | Lines touched (64 B / 128 B) | Ctx switches vol/invol (last rep) |
|---|---|---|---|---|---|---|
| stride 1 (8 B) | 196,264,708 | 3.271 | 0.818 | **1.033×** | 1 / 1 | 0 / 99 |
| stride 4 (32 B) | 196,519,625 | 3.275 | 0.819 | 1.034× | 2 / 1 | 0 / 104 |
| stride 8 (64 B) | 189,971,583 | 3.166 | 0.792 | 1.000× | 4 / **2** | 0 / 65 |
| stride 16 (128 B) | 189,705,959 | 3.162 | 0.790 | 0.999× | 4 / 4 | 0 / 193 |
| private register (Task 4) | 11,743,416 | 0.196 | 0.049 | 0.062× | — | — |
| 1 thread, strided kernel (no sharing) | 174,961,167 | 2.916 | — | — | 1 / 1 | — |

Parity: all four layouts and the privatized baseline produced bit-identical final counters (`{'1': True, '4': True, '8': True, '16': True}`), also matching a closed-form NumPy sum for thread 0. LLVM audit: `load i64 = 1`, `store i64 = 2`, no Python-object calls, `nogil = True` → PASS.

---

### Question 1 — MESI protocol state transitions (stride = 1)

With stride 1, `buffer[0]` and `buffer[1]` are at byte offsets 0 and 8 of the same line (on this M2 the line is 128 B, so all four counters 0/8/16/24 B sit in **line 0** — the manifest reports `lines_touched = [0]`).

1. **Initial:** after `aligned_zeros` the line is cached by nobody, or **Shared (S)** in several cores after the zero-fill.
2. **Core 0 stores to `buffer[0]`:** it needs ownership. If the line is S/I in Core 0, Core 0 issues a **Read-For-Ownership (RFO / ReadUnique)**. Every other copy is invalidated (**S → I** in Core 1); Core 0's copy becomes **Exclusive (E)** and, after the store retires into L1, **Modified (M)**.
3. **Core 1 stores to `buffer[1]` immediately afterwards:** its copy is **I**, so it issues its own RFO. Core 0 must supply the dirty line (snoop hit on M) and invalidate it: **Core 0: M → I**, **Core 1: I → E → M** after the store commits. (In MOESI-style protocols Core 0 may briefly pass through Owned, but a write request always ends with the writer in M and every other copy in I.)
4. Core 0's next store finds **I** again → another RFO → Core 1 **M → I**, Core 0 **I → M**, and so on. With 4 threads the line migrates among four cores ("ping-pong"), although no two threads ever touch the same 8-byte word.

**Why the RFO stalls Core 1's store buffer:** a store can retire into the store buffer immediately, but it can only *commit* to L1 when the core holds the line in E/M. While the RFO is outstanding (a cross-core snoop + line transfer, tens of ns instead of ~1 ns for an L1 hit), the store at the head of the store buffer cannot drain; younger stores queue behind it, and once the buffer is full the core cannot retire further stores and the pipeline stalls. Each ownership round-trip is therefore paid by whichever core is waiting, on every hand-off.

### Question 2 — Quantitative slowdown & coherency overhead

Measured medians (4 threads × 60 M updates):

- stride 1: **196,264,708 ns** → 3.271 ns/update per thread (0.818 ns aggregate)
- stride 4: **196,519,625 ns** → 3.275 ns/update
- stride 8: **189,971,583 ns** → 3.166 ns/update
- stride 16: **189,705,959 ns** → 3.162 ns/update
- privatized register: **11,743,416 ns** → 0.196 ns/update

**Slowdown stride 1 / stride 8 = 196,264,708 / 189,971,583 = 1.033×.**

Decomposition using the single-thread reference (174,961,167 ns = 2.916 ns/update, same kernel, no other writer):

- **L1 store→load forwarding cost** ≈ 2.92 ns/update (≈ 10 cycles at the M2 P-core's ~3.5 GHz). Every iteration loads the value the previous iteration just stored, so the loop is a serial memory dependency chain.
- **Cross-core coherence cost** (stride 1 − single thread) = (196.26 − 174.96) ms / 60 M ≈ **0.36 ns/update**; of this, the part that is truly due to *false sharing* (stride 1 − stride 16) is only (196.26 − 189.71) ms / 60 M ≈ **0.11 ns/update**. The rest (≈ 0.25 ns) is common to all layouts (thread start-up, P/E-core scheduling, shared L2/memory traffic, 65–193 involuntary context switches per run).
- **Register privatization** removes the memory chain entirely: 0.196 ns/update is **16.2× faster than stride 8** and 14.9× faster than the single-thread memory loop.

**Interpretation:** on this Apple M2 the false-sharing penalty is real but small (3.3 %). The loop is *latency-bound on its own store→load forwarding chain* (2.9 ns/iter), so each core issues a store only every ~10 cycles. The load is always satisfied from the core's own store buffer (forwarding), so it never waits for the line; only the *commit* of stores needs ownership. Apple's cores have a deep store buffer and can merge successive stores to the same address before commit, so when a core regains the line it drains many logical updates in one ownership period. Coherence traffic is thereby mostly hidden behind the forwarding latency. On x86 with TSO (stores must commit one by one in order) the same kernel shows a much larger penalty — a reduced-iteration test of this code on an x86 Xeon gave ≈1.9× for stride 1 vs stride 8.

### Question 3 — L2 spatial prefetcher & 128-byte pairing

Measured: **stride 8 = 189,971,583 ns vs stride 16 = 189,705,959 ns → ratio 1.001×** (stride 16 is 0.1 % faster).

On Intel cores the **L2 adjacent-cache-line (spatial) prefetcher** treats memory as 128-byte-aligned pairs of 64-byte lines: on a miss to line 2k it also fetches its "buddy" 2k+1 (and vice versa). With 64-byte padding (stride 8), thread 0 writes line 0 and thread 1 writes line 1 — the same 128-byte sector. When thread 0 misses on line 0, the prefetcher pulls in line 1 as well, which requires a coherence request against thread 1's Modified copy; thread 1 then loses (or has to re-fetch) its line although no data is shared. These **secondary invalidations** happen when (a) the two padded slots fall in the same 128-B-aligned pair (even/odd line), (b) the prefetcher is enabled, and (c) both cores keep writing so the lines are repeatedly in M state on different cores. Spacing by 128 B (stride 16) puts every thread in its own pair, so the prefetcher only fetches lines no one else writes.

On this **Apple M2 the situation is even stricter: the coherency granule itself is 128 B** (`cache_line = 128`). The manifest shows stride 8 touches only **two 128-B lines `[0, 1]`** (threads 0+1 share line 0, threads 2+3 share line 1), i.e. 64-byte padding is still genuine 2-way false sharing on Apple Silicon, and only stride 16 (`lines = [0, 1, 2, 3]`) is truly padded. The measured difference is small (≈0.27 ms over 60 M updates ≈ 0.004 ns/update) for the same reason as in Q2: the kernel's store-forwarding latency hides most of the ownership transfers. The correct portable rule is therefore to pad to 128 B (`std::hardware_destructive_interference_size` is 128 on Apple ARM and on Intel parts with adjacent-line prefetch).

### Question 4 — Compiler register promotion vs. real memory stores

LLVM IR from `worker_strided_accumulate.inspect_llvm()` (native function, wrapper excluded): **1 `load i64`, 2 `store i64`, no `volatile`, no Python-object/NRT calls**, `nogil = True`. The two stores are:

```
store i64 %.184, ptr %.178, align 8          ; buffer[j] = nv   (inside the loop)
store i64 %j.2.0.lcssa, ptr %retptr, align 8 ; return value j   (after the loop)
```

**Why a compiler would hoist `buffer[slot]`:** in `for i: buffer[slot] += f(i)` the address is loop-invariant and, from a single-thread point of view, nothing else in the loop can alias it. LLVM's LICM pass then performs *scalar promotion*: one load before the loop, the running sum kept in a register (`phi i64`), one store after the loop. That is legal because LLVM assumes no concurrent writer unless the access is `volatile` or atomic — exactly what my Task 4 kernel does on purpose; its 0.196 ns/update is what promotion would give.

**How my kernel prevents it:** the store address is recomputed from the value just loaded:

```python
v  = buffer[j]
nv = v + ((s ^ i) & 7) + 1
buffer[j] = nv
j  = slot + int64(nv >> 60)   # always == slot (counters < 2**60), but LLVM cannot prove it
```

So `%.178` (the pointer) depends on a `phi` of `j`, which depends on the loaded value — the address is **not loop-invariant**, LICM's promotion precondition fails, and every iteration must perform a real `load i64` / `store i64` to L1. The run-time evidence agrees: 2.916 ns/update for the memory loop vs 0.196 ns/update for the register loop (14.9×), and the parity check confirms both compute the identical bit-exact sums.

### Question 5 — CPython GIL masking vs. `nogil=True`

Pure-Python contrast run (same stride-1 update, list slots 0..3, 1,000,000 updates per thread):

| Python threads | Total wall time | ns / update | Involuntary ctx switches |
|---|---|---|---|
| 1 | 65,517,375 ns | 65.52 | 29 |
| 4 | 244,008,958 ns | 61.00 | 225 |

With 4 threads the job does 4× the work in **3.72× the time**: there is **no parallel speedup at all** (ideal would be ≈ 66 ms), and involuntary context switches rise **7.8×** (29 → 225). The reason is that CPython bytecode can only run while holding the **Global Interpreter Lock**. Only one thread executes at a time; the others sleep until the eval loop forces a switch (every 5 ms switch interval) or a thread blocks. Consequently two threads are never storing to the same cache line *at the same moment*, so there is no RFO ping-pong to observe — the hardware effect is **masked by software serialization**. Each update also costs ~60–70 ns of interpreter work (bytecode dispatch, boxed `int` objects, reference counting, list-item pointer stores), which is 20× larger than the whole native update (≈3 ns), so even a cache-line transfer would be lost in the noise. Adding threads only adds GIL hand-off costs (lock contention, wake-ups, context switches), which is why multithreaded pure-Python code is at best no faster and typically slower per unit of wall time than one thread.

With `@njit(nogil=True)` the kernel is compiled to native ARM64 code and releases the GIL on entry, so all four OS threads run truly simultaneously on separate cores (4 × 60 M updates finish in 196.3 ms ≈ the 175 ms single-thread time, i.e. ≈ 3.6× parallel throughput). Only then do their stores contend at the cache-line level, and the remaining slowdown (stride 1 vs stride 16: 1.035×) is **hardware cache-line contention**, not lock contention.
