"""
Make a Training Step Fast: Profile, Fix, Prove It

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - op_cost
# ---- provided: a simulated GPU, the model, the data, a training job and a noisy timer
import math

import numpy as np

GPU = {"tflops": {"fp32": 20.0, "bf16": 160.0, "fp16": 160.0, "fp8": 320.0},
       "bandwidth_gbs": 2000.0,
       "launch_us": 5.0}

BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1}

LINK = {"gpus": 8, "latency_us": 20.0, "bandwidth_gbs": 50.0}

MODEL = {"layers": 4, "d": 512}
MAX_LEN = 512

LENGTHS = [512, 497, 431, 388, 342, 301, 266, 240, 213, 190, 171, 160, 148, 139, 130, 122, 115, 108, 101, 96, 90, 85,
           80, 76, 71, 67, 64, 60, 57, 54, 51, 48, 46, 44, 41, 39, 37, 35, 33, 31, 30, 28, 27, 26, 25, 24, 23, 700]

LOAD_US = 40000.0

GRAD_STATS = [(8.0, 2.0e-9), (6.5, 1.8e-9), (5.2, 1.5e-9), (4.4, 1.4e-9), (3.9, 1.2e-9), (3.1, 1.1e-9),
              (2.8, 1.0e-9), (2.5, 9.0e-10), (48.0, 8.5e-10), (2.2, 8.0e-10), (2.0, 7.5e-10), (1.8, 7.0e-10),
              (1.7, 6.6e-10), (1.6, 6.2e-10), (1.5, 6.0e-10), (1.4, 5.8e-10)]


def allreduce_us(nbytes, link=LINK):
    """Ring all-reduce time: latency + 2 (n - 1) / n of the bytes over the link."""
    n = link["gpus"]
    return link["latency_us"] + 2 * (n - 1) / n * nbytes / (link["bandwidth_gbs"] * 1e3)


def _ew(name, group, phase, reads, size, flops, precision, grads=()):
    return {"name": name, "group": group, "kind": "elementwise", "phase": phase, "reads": reads, "size": size,
            "flops": flops, "bytes": (reads + 1) * size, "precision": precision, "grads": list(grads)}


def _mm(name, group, phase, flops, nbytes, precision, kind="matmul", grads=()):
    return {"name": name, "group": group, "kind": kind, "phase": phase, "flops": flops, "bytes": nbytes,
            "precision": precision, "grads": list(grads)}


def build_ops(tokens, attn_pairs, precision):
    """The kernels of one training step in launch order: forward, backward, then the optimizer."""
    d, e, T = MODEL["d"], BYTES[precision], tokens
    fwd, bwd_layers = [], []
    shapes = {"qkv": (d, 3 * d), "proj": (d, d), "up": (d, 4 * d), "down": (4 * d, d)}

    for i in range(MODEL["layers"]):
        p = f"L{i}."
        layer = []

        for w in ("qkv", "attn", "proj", "add1", "norm1", "up", "gelu", "down", "add2", "norm2"):
            if w in shapes:
                a, b = shapes[w]
                layer.append(("mm", w, 2 * T * a * b, e * (a * b + T * a + T * b), [(p + w, 4 * a * b)]))
            elif w == "attn":
                layer.append(("attn", w, 4 * d * attn_pairs, e * (T * 3 * d + T * d), []))
            elif w in ("add1", "add2"):
                layer.append(("ew", w, 2, e * T * d, T * d, []))
            elif w == "gelu":
                layer.append(("ew", w, 1, e * T * 4 * d, 8 * T * d, []))
            else:
                layer.append(("ew", w, 1, e * T * d, 5 * T * d, [(p + w, 4 * d)]))

        for spec in layer:
            if spec[0] == "ew":
                fwd.append(_ew(p + spec[1], spec[1], "forward", spec[2], spec[3], spec[4], precision))
            else:
                fwd.append(_mm(p + spec[1], spec[1], "forward", spec[2], spec[3], precision,
                               kind="matmul" if spec[0] == "mm" else "attention"))

        back = []
        for spec in reversed(layer):
            g = spec[1] + ".grad"
            if spec[0] == "ew":
                back.append(_ew(p + g, g, "backward", spec[2] + 1, spec[3], 2 * spec[4], precision, spec[5]))
            else:
                back.append(_mm(p + g, g, "backward", 2 * spec[2], 2 * spec[3], precision,
                                kind="matmul" if spec[0] == "mm" else "attention", grads=spec[4]))

        bwd_layers.append(back)

    bwd = [op for back in reversed(bwd_layers) for op in back]
    opt = []

    for i in range(MODEL["layers"]):
        for w in ("qkv", "proj", "norm1", "up", "down", "norm2"):
            size = 4 * (shapes[w][0] * shapes[w][1] if w in shapes else d)
            n = size // 4
            opt += [_ew(f"L{i}.{w}.adam_m", "adam_m", "optimizer", 2, size, 3 * n, "fp32"),
                    _ew(f"L{i}.{w}.adam_v", "adam_v", "optimizer", 2, size, 4 * n, "fp32"),
                    _ew(f"L{i}.{w}.adam_step", "adam_step", "optimizer", 3, size, 6 * n, "fp32")]

    return fwd + bwd + opt


def run_step(config):
    """Simulate one steady-state step on one of the 8 GPUs."""
    precision = config.get("precision", "fp32")

    if config.get("pack"):
        rows = config["pack"](LENGTHS, MAX_LEN)["rows"]
        pairs = sum(min(L, MAX_LEN) ** 2 for L in LENGTHS)
    else:
        rows = [[i] for i in range(len(LENGTHS))]
        pairs = len(rows) * MAX_LEN ** 2

    tokens = len(rows) * MAX_LEN
    ops = build_ops(tokens, pairs, precision)

    if config.get("fuse"):
        ops = config["fuse"](ops)

    trace, host, gpu, grads = [], 0.0, 0.0, []
    comm_end = None

    for op in ops:
        if op["phase"] == "optimizer" and comm_end is None:
            if config.get("comm"):
                buckets = config["comm"](grads, LINK)
            else:
                total = sum(g["bytes"] for g in grads)
                start = max(g["ready"] for g in grads)
                buckets = [{"params": [g["name"] for g in grads], "bytes": total,
                            "start": start, "end": start + allreduce_us(total)}]

            for k, b in enumerate(buckets):
                trace.append({"lane": "comm", "name": f"allreduce_{k}", "kind": "collective",
                              "start": b["start"], "end": b["end"]})

            comm_end = max(b["end"] for b in buckets)

        trace.append({"lane": "host", "name": "launch " + op["name"], "kind": "launch",
                      "start": host, "end": host + GPU["launch_us"]})

        host += GPU["launch_us"]
        start = max(host, gpu, comm_end or 0.0)
        gpu = start + op_cost(op)["time_us"]

        trace.append({"lane": "gpu", "name": op["name"], "group": op["group"], "kind": "kernel",
                      "phase": op["phase"], "start": start, "end": gpu})

        for name, nbytes in op["grads"]:
            grads.append({"name": name, "bytes": nbytes, "ready": gpu})

    compute = max(e["end"] for e in trace)
    wait = LOAD_US if config.get("workers", 0) == 0 else max(0.0, LOAD_US / config["workers"] - compute)

    for e in trace:
        e["start"] += wait
        e["end"] += wait

    trace.insert(0, {"lane": "host", "name": "next(batch)", "kind": "dataloader_wait",
                     "start": 0.0, "end": wait})

    return {"trace": trace, "step_us": wait + compute, "tokens": tokens, "kernels": len(ops)}


def train_loss(config):
    """Final loss of a full training run with this config."""
    precision = config.get("precision", "fp32")

    if precision == "fp16":
        res = config["scaler"](GRAD_STATS) if config.get("scaler") else None
        lost = res["underflow_steps"] if res else sum(lo < 2.0 ** -24 for _, lo in GRAD_STATS)
        return round(2.0 + 0.002 + 0.4 * lost / len(GRAD_STATS), 4)

    return {"fp32": 2.0, "bf16": 2.0031, "fp8": 2.0688}[precision]


def measure(config, runs=15, seed=0):
    """Time training steps using a noisy timer (microseconds)."""
    true = run_step(config)["step_us"]
    rng = np.random.default_rng(seed)
    out = []

    for k in range(runs):
        t = true * (1 + 0.03 * rng.standard_normal())

        if k == 0:
            t += 3.0 * true
        elif k == 1:
            t += 0.5 * true

        if rng.random() < 0.1:
            t *= 1.4

        out.append(round(t, 1))

    return out


def op_cost(op, gpu=GPU):
    """Calculate the Roofline cost of one GPU kernel."""

    # Peak compute throughput for the kernel's precision.
    tflops = gpu["tflops"][op["precision"]]

    # Compute execution time in microseconds.
    compute_us = op["flops"] / (tflops * 1e6)

    # Memory transfer time in microseconds.
    memory_us = op["bytes"] / (gpu["bandwidth_gbs"] * 1e3)

    # Roofline time is determined by the slower resource.
    time_us = max(compute_us, memory_us)

    # Compute takes precedence when both times are equal.
    bound = "compute" if compute_us >= memory_us else "memory"

    # Arithmetic intensity in FLOPs per byte.
    intensity = op["flops"] / op["bytes"]

    # Dictionary key order must match the expected output.
    return {
        "time_us": time_us,
        "compute_us": compute_us,
        "memory_us": memory_us,
        "bound": bound,
        "intensity": intensity
    }

# Step 2 - analyze_trace
def analyze_trace(trace, launch_us=5.0, top=3):
    """Analyze GPU activity, idle time, communication and performance bottlenecks."""

    # Handle an empty trace.
    if not trace:
        return {
            "step_us": 0.0,
            "gpu_busy_us": 0.0,
            "gpu_idle_share": 0.0,
            "data_wait_us": 0.0,
            "exposed_comm_us": 0.0,
            "small_kernels": 0,
            "top_groups": [],
            "verdict": "compute_bound"
        }

    # Calculate total step duration.
    first_start = min(event["start"] for event in trace)
    last_end = max(event["end"] for event in trace)
    step_us = last_end - first_start

    # Calculate the union length of intervals.
    # Overlapping intervals are counted only once.
    def union_length(intervals):
        if not intervals:
            return 0.0

        intervals = sorted(intervals)

        total = 0.0
        current_start, current_end = intervals[0]

        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                total += current_end - current_start
                current_start, current_end = start, end

        total += current_end - current_start

        return float(total)

    # Separate GPU kernel and communication events.
    kernels = [
        event for event in trace
        if event["kind"] == "kernel"
    ]

    collectives = [
        event for event in trace
        if event["kind"] == "collective"
    ]

    # Extract GPU kernel execution intervals.
    kernel_intervals = [
        (event["start"], event["end"])
        for event in kernels
    ]

    # Total GPU busy time, excluding double-counted overlaps.
    gpu_busy_us = union_length(kernel_intervals)

    # Fraction of the training step during which the GPU is idle.
    gpu_idle_share = (
        (step_us - gpu_busy_us) / step_us
        if step_us > 0 else 0.0
    )

    # Calculate total data-loader waiting time.
    # Initializing with 0.0 ensures a floating-point result
    # even when there are no data-loader wait events.
    data_wait_us = sum(
        (
            event["end"] - event["start"]
            for event in trace
            if event["kind"] == "dataloader_wait"
        ),
        0.0
    )

    # Extract collective execution intervals.
    collective_intervals = [
        (event["start"], event["end"])
        for event in collectives
    ]

    # Combine kernel and collective intervals.
    combined_intervals = kernel_intervals + collective_intervals

    # Calculate the union of all kernel and collective intervals.
    combined_busy_us = union_length(combined_intervals)

    # Exposed communication is collective time not overlapped by kernels.
    exposed_comm_us = max(
        0.0,
        combined_busy_us - gpu_busy_us
    )

    # Count kernels shorter than the launch overhead.
    # A kernel exactly equal to launch_us is not counted.
    small_kernels = sum(
        1
        for event in kernels
        if (event["end"] - event["start"]) < launch_us
    )

    # Accumulate execution time for every kernel group.
    group_times = {}

    for event in kernels:
        group = event["group"]
        duration = event["end"] - event["start"]

        group_times[group] = (
            group_times.get(group, 0.0) + duration
        )

    # Sort groups by rounded execution time (descending),
    # using group name as the tie-breaker.
    top_groups = sorted(
        [
            (group, round(duration, 3))
            for group, duration in group_times.items()
        ],
        key=lambda item: (-item[1], item[0])
    )[:top]

    # Determine the primary performance bottleneck.
    # Apply the rules in the required priority order.
    if step_us > 0 and data_wait_us >= 0.2 * step_us:
        verdict = "input_bound"

    elif step_us > 0 and exposed_comm_us >= 0.2 * step_us:
        verdict = "communication_bound"

    elif gpu_idle_share >= 0.2:
        verdict = "launch_bound"

    else:
        verdict = "compute_bound"

    # Return all required profiling metrics.
    return {
        "step_us": step_us,
        "gpu_busy_us": gpu_busy_us,
        "gpu_idle_share": gpu_idle_share,
        "data_wait_us": data_wait_us,
        "exposed_comm_us": exposed_comm_us,
        "small_kernels": small_kernels,
        "top_groups": top_groups,
        "verdict": verdict
    }

# Step 3 - summarize_runs
def summarize_runs(times, warmup=2):
    """Drop the warmup runs, then return n, median, q1, q3, iqr and mean."""

    # Discard the initial warmup measurements.
    # Convert the remaining timings to floating-point values.
    measured = np.asarray(times[warmup:], dtype=float)

    # Number of measurements retained after warmup.
    n = int(len(measured))

    # Handle cases where no measurements remain.
    if n == 0:
        return {
            "n": 0,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "mean": None
        }

    # Calculate the median execution time.
    median = float(np.median(measured))

    # Calculate the 25th and 75th percentiles.
    # NumPy uses linear interpolation by default.
    q1, q3 = np.percentile(measured, [25, 75])

    # Convert percentile results into standard Python floats.
    q1 = float(q1)
    q3 = float(q3)

    # Interquartile range measures the spread of the middle 50%.
    iqr = float(q3 - q1)

    # Calculate the arithmetic mean.
    mean = float(np.mean(measured))

    # Return the benchmark statistics.
    return {
        "n": n,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "mean": mean
    }

# Step 4 - choose_workers (not yet solved)
# TODO: implement

# Step 5 - pack_sequences (not yet solved)
# TODO: implement

# Step 6 - fuse_elementwise (not yet solved)
# TODO: implement

# Step 7 - dynamic_loss_scale (not yet solved)
# TODO: implement

# Step 8 - bucket_allreduce (not yet solved)
# TODO: implement

# Step 9 - plan_parallelism (not yet solved)
# TODO: implement

# Step 10 - compare_runs (not yet solved)
# TODO: implement

# Step 11 - speedup_report (not yet solved)
# TODO: implement

# Step 12 - release_check (not yet solved)
# TODO: implement

