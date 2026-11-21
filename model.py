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

# Step 4 - choose_workers
def choose_workers(config, candidates=(1, 2, 4, 8, 16, 32)):
    """Find the fewest data-loader workers that eliminate data wait."""

    table = []
    best_workers = None

    # Evaluate each candidate in the specified order.
    for workers in candidates:

        # Copy the original configuration to avoid modifying it.
        candidate_config = config.copy()
        candidate_config["workers"] = workers

        # Simulate one training step using this worker count.
        result = run_step(candidate_config)

        # Extract the total data-loader waiting time from the trace.
        wait_us = sum(
            (
                event["end"] - event["start"]
                for event in result["trace"]
                if event["kind"] == "dataloader_wait"
            ),
            0.0
        )

        # Record worker count, step duration and data-loading wait.
        table.append((
            workers,
            result["step_us"],
            wait_us
        ))

        # Identify the first candidate with zero data-loading wait.
        if wait_us == 0 and best_workers is None:
            best_workers = workers

    # Return the optimal worker count and results for all candidates.
    return {
        "workers": best_workers,
        "table": table
    }

# Step 5 - pack_sequences
def pack_sequences(lengths, max_len):
    """First-fit decreasing packing into rows of max_len, with offsets and padding shares."""

    # Clip sequences that exceed the maximum permitted length.
    clipped_lengths = [
        min(length, max_len) for length in lengths
    ]

    # Sort samples from longest to shortest.
    # Original index provides deterministic ordering for equal lengths.
    sorted_indices = sorted(
        range(len(lengths)),
        key=lambda i: (-clipped_lengths[i], i)
    )

    # Store packed rows and the number of tokens used in each row.
    rows = []
    row_used = []

    # Apply First-Fit Decreasing packing.
    for index in sorted_indices:
        length = clipped_lengths[index]

        # Find the first existing row with sufficient remaining capacity.
        placed = False

        for row_index in range(len(rows)):
            if row_used[row_index] + length <= max_len:
                rows[row_index].append(index)
                row_used[row_index] += length
                placed = True
                break

        # Create a new row if no existing row has enough space.
        if not placed:
            rows.append([index])
            row_used.append(length)

    # Calculate cumulative token offsets for each packed row.
    cu_seqlens = []

    for row in rows:
        offsets = [0]
        cumulative = 0

        for index in row:
            cumulative += clipped_lengths[index]
            offsets.append(cumulative)

        cu_seqlens.append(offsets)

    # Calculate the number of tokens before and after packing.
    tokens_before = len(lengths) * max_len
    tokens_after = len(rows) * max_len

    # Total real tokens, including sequences clipped to max_len.
    real_tokens = sum(clipped_lengths)

    # Calculate padding proportions before packing.
    pad_share_before = (
        (tokens_before - real_tokens) / tokens_before
        if tokens_before > 0 else 0.0
    )

    # Calculate padding proportions after packing.
    pad_share_after = (
        (tokens_after - real_tokens) / tokens_after
        if tokens_after > 0 else 0.0
    )

    # Return packing information and padding statistics.
    return {
        "rows": rows,
        "cu_seqlens": cu_seqlens,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "pad_share_before": float(pad_share_before),
        "pad_share_after": float(pad_share_after)
    }

# Step 6 - fuse_elementwise
def fuse_elementwise(ops):
    """Merge runs of adjacent same-size elementwise ops in one phase into single kernels."""

    # Store the fused operations in a new list.
    fused_ops = []

    for op in ops:

        # Check whether the current operation can be merged
        # with the immediately preceding operation.
        if (
            fused_ops
            and op["kind"] == "elementwise"
            and fused_ops[-1]["kind"] == "elementwise"
            and fused_ops[-1]["size"] == op["size"]
            and fused_ops[-1]["phase"] == op["phase"]
        ):
            previous = fused_ops[-1]

            # Combine operation names and groups.
            previous["name"] = previous["name"] + "+" + op["name"]
            previous["group"] = previous["group"] + "+" + op["group"]

            # Eliminate the intermediate memory read.
            # The output of the previous operation stays in registers.
            previous["reads"] = previous["reads"] + op["reads"] - 1

            # Recalculate total memory traffic.
            previous["bytes"] = (
                (previous["reads"] + 1) * previous["size"]
            )

            # Add the floating-point operations.
            previous["flops"] += op["flops"]

            # Preserve gradient metadata in the original order.
            previous["grads"] = (
                previous.get("grads", []) + op.get("grads", [])
            )

        else:
            # Copy the operation so the original input is not modified.
            new_op = dict(op)

            # Make an independent copy of the gradients list.
            if "grads" in new_op:
                new_op["grads"] = list(new_op["grads"])

            # Start a new fused operation or preserve a non-elementwise op.
            fused_ops.append(new_op)

    return fused_ops

# Step 7 - dynamic_loss_scale
def dynamic_loss_scale(grad_stats, init_scale=65536.0, growth_interval=4,
                       growth=2.0, backoff=0.5,
                       fp16_max=65504.0, fp16_min=2.0 ** -24):
    """Simulate a dynamic FP16 loss scaler with overflow protection and scale growth."""

    # Initialize the loss scale and consecutive clean-step counter.
    scale = float(init_scale)
    clean_steps = 0

    # Store the scale used for each training step.
    scales = []

    # Track training steps skipped due to gradient overflow.
    skipped = []

    # Count steps where small gradients underflow to zero.
    underflow_steps = 0

    # Process gradient statistics for each training step.
    for step_index, (largest, smallest) in enumerate(grad_stats):

        # Record the scale applied during this step.
        scales.append(scale)

        # Calculate the largest gradient after loss scaling.
        scaled_largest = largest * scale

        # Check whether the scaled gradient exceeds FP16's maximum.
        if scaled_largest > fp16_max:

            # Skip the optimizer update because of overflow.
            skipped.append(step_index)

            # Reduce the loss scale.
            scale *= backoff

            # Reset consecutive clean steps.
            clean_steps = 0

        else:
            # The step is applied successfully.
            scaled_smallest = smallest * scale

            # Detect whether the smallest gradient underflows.
            if scaled_smallest < fp16_min:
                underflow_steps += 1

            # Count this non-overflow step toward scale growth.
            clean_steps += 1

            # Increase the scale after enough consecutive clean steps.
            if clean_steps >= growth_interval:
                scale *= growth

                # Reset the clean-step counter after scale growth.
                clean_steps = 0

    # Return the complete loss-scaling statistics.
    return {
        "scales": scales,
        "skipped": skipped,
        "underflow_steps": underflow_steps,
        "final_scale": float(scale)
    }

# Step 8 - bucket_allreduce
def bucket_allreduce(grads, link=LINK, cap_mb=25.0):
    """Group gradients into buckets in backward order and schedule one all-reduce per bucket."""

    # Convert the bucket capacity from MB to bytes.
    cap_bytes = cap_mb * 1e6

    # Store the resulting communication buckets.
    buckets = []

    # Track gradients and total bytes in the current bucket.
    current_params = []
    current_bytes = 0
    current_ready = 0.0

    # The previous all-reduce must finish before the next one starts.
    previous_end = 0.0

    def flush_bucket():
        """Schedule the current bucket's all-reduce operation."""
        nonlocal current_params, current_bytes, current_ready, previous_end

        if not current_params:
            return

        # Start only after every gradient in the bucket is ready
        # and the preceding all-reduce has finished.
        start = max(current_ready, previous_end)

        # Calculate the communication duration for this bucket.
        end = start + allreduce_us(current_bytes, link)

        # Record the bucket details.
        buckets.append({
            "params": current_params,
            "bytes": current_bytes,
            "start": start,
            "end": end
        })

        # Update the previous communication completion time.
        previous_end = end

        # Reset the current bucket.
        current_params = []
        current_bytes = 0
        current_ready = 0.0

    # Process gradients in their original backward-production order.
    for grad in grads:
        grad_bytes = grad["bytes"]

        # Start a new bucket if adding this gradient would exceed capacity.
        # A gradient larger than the capacity is allowed its own bucket.
        if current_params and current_bytes + grad_bytes > cap_bytes:
            flush_bucket()

        # Add the gradient to the current bucket.
        current_params.append(grad["name"])
        current_bytes += grad_bytes

        # Track when all gradients in this bucket are ready.
        current_ready = max(current_ready, grad["ready"])

    # Schedule any remaining gradients.
    flush_bucket()

    return buckets

# Step 9 - plan_parallelism
def plan_parallelism(model, cluster):
    """Evaluate tensor-parallel, data-parallel and ZeRO plans, retaining those that fit."""

    # Convert the model's parameter count from billions to individual parameters.
    P = model["params_b"] * 1e9

    # Extract cluster specifications.
    gpus = cluster["gpus"]
    gpus_per_node = cluster["gpus_per_node"]
    mem_gb = cluster["mem_gb"]
    tflops = cluster["tflops"]
    intra_gbs = cluster["intra_gbs"]
    inter_gbs = cluster["inter_gbs"]
    tp_penalty = cluster["tp_penalty"]

    # Tensor-parallel communication size belongs to the model.
    tp_gb = model["tp_gb"]

    # Store feasible plans with their unrounded metrics for sorting.
    fits_internal = []

    # Store plans that exceed available GPU memory.
    rejected = []

    # Evaluate the supported tensor-parallel sizes.
    for tp in (1, 2, 4, 8):

        # TP must divide the GPU count and remain within one node.
        if gpus % tp != 0 or tp > gpus_per_node:
            continue

        # Calculate the data-parallel degree.
        dp = gpus // tp

        # Try ZeRO stages 0-3 when DP > 1.
        # With DP == 1, only stage 0 is applicable.
        zero_stages = range(4) if dp > 1 else (0,)

        for zero in zero_stages:

            # Calculate memory requirements in bytes.
            weights = 2 * P / tp
            gradients = 2 * P / tp
            optimizer_states = 12 * P / tp

            # ZeRO stage 3 partitions model weights across DP ranks.
            if zero >= 3:
                weights /= dp

            # ZeRO stages 2 and 3 partition gradients.
            if zero >= 2:
                gradients /= dp

            # ZeRO stages 1, 2 and 3 partition optimizer states.
            if zero >= 1:
                optimizer_states /= dp

            # Calculate total GPU memory in GB.
            memory_gb = (
                (weights + gradients + optimizer_states) / 1e9
                + model["act_gb"] / tp
            )

            # Calculate computation time in milliseconds.
            compute_ms = (
                model["step_tflop"]
                / (gpus * tflops)
                * 1000
                * (1 + tp_penalty * (tp - 1))
            )

            # Calculate tensor-parallel communication time.
            if tp == 1:
                tensor_parallel_ms = 0.0
            else:
                tensor_parallel_ms = (
                    tp_gb
                    * 2 * (tp - 1) / tp
                    / intra_gbs
                    * 1000
                )

            # Calculate data-parallel communication volume in bytes.
            B = 2 * P / tp

            # ZeRO stage 3 has an additional 1.5x communication cost.
            if zero == 3:
                B *= 1.5

            # Calculate data-parallel communication time.
            if dp == 1:
                data_parallel_ms = 0.0
            else:
                data_parallel_ms = (
                    2 * (dp - 1) / dp
                    * B / (inter_gbs * 1e9)
                    * 1000
                )

            # Calculate the total estimated training-step time.
            step_ms = (
                compute_ms
                + tensor_parallel_ms
                + data_parallel_ms
            )

            # Check whether the plan fits into GPU memory.
            if memory_gb <= mem_gb:

                # Keep unrounded metrics for accurate sorting.
                fits_internal.append((
                    step_ms,
                    memory_gb,
                    tp,
                    zero
                ))

            else:
                # Record rejected plans in the required format.
                rejected.append((
                    tp,
                    zero,
                    f"needs {memory_gb:.1f} GB"
                ))

    # Sort by exact step time, memory requirement, TP size and ZeRO stage.
    fits_internal.sort(
        key=lambda plan: (
            plan[0],
            plan[1],
            plan[2],
            plan[3]
        )
    )

    # Construct the final output with numerical values rounded to 2 decimals.
    fits = [
        {
            "tp": tp,
            "dp": gpus // tp,
            "zero": zero,
            "memory_gb": round(memory_gb, 2),
            "step_ms": round(step_ms, 2)
        }
        for step_ms, memory_gb, tp, zero in fits_internal
    ]

    return {
        "fits": fits,
        "rejected": rejected
    }

# Step 10 - compare_runs
def compare_runs(base_times, cand_times, warmup=2):
    """Compare benchmark medians and determine whether the speedup is measurable."""

    # Summarize baseline and candidate timings after removing warmup runs.
    base_stats = summarize_runs(base_times, warmup)
    cand_stats = summarize_runs(cand_times, warmup)

    # Extract the median execution times.
    base = base_stats["median"]
    cand = cand_stats["median"]

    # Handle cases where no measurements remain after warmup.
    if base is None or cand is None:
        return {
            "speedup": None,
            "verdict": "no measurable change",
            "base": base,
            "cand": cand
        }

    # Calculate the speedup relative to the baseline.
    speedup = float(base / cand) if cand != 0 else None

    # Determine whether the candidate is measurably faster or slower.
    # Comparisons are strict, as required.
    if cand_stats["q3"] < base_stats["q1"]:
        verdict = "faster"

    elif cand_stats["q1"] > base_stats["q3"]:
        verdict = "slower"

    else:
        verdict = "no measurable change"

    # Return the speedup, verdict and both median timings.
    return {
        "speedup": speedup,
        "verdict": verdict,
        "base": base,
        "cand": cand
    }

# Step 11 - speedup_report
def speedup_report(steps, min_share=0.01):
    """Attribute the end-to-end speedup to each fix in the order they were applied."""

    # The first entry represents the baseline configuration.
    baseline_time = steps[0][1]

    # The final entry represents the fully optimized configuration.
    final_time = steps[-1][1]

    # Total time saved across all optimizations.
    total_saved = baseline_time - final_time

    # Store the contribution of each individual optimization.
    fixes = []

    # Track optimizations that contributed less than the minimum threshold.
    did_not_matter = []

    # Process each optimization, excluding the baseline.
    for i in range(1, len(steps)):
        label, after = steps[i]

        # The preceding step's time is the time before this optimization.
        before = steps[i - 1][1]

        # Calculate the time saved by this optimization.
        saved_us = before - after

        # Calculate its share of the total end-to-end time saved.
        # Return 0.0 when the total time saved is zero.
        share = (
            saved_us / total_saved
            if total_saved != 0 else 0.0
        )

        # Calculate the speedup introduced by this optimization.
        speedup = before / after

        # Record the optimization's performance contribution.
        fixes.append({
            "fix": label,
            "saved_us": saved_us,
            "share": share,
            "speedup": speedup
        })

        # Identify optimizations whose savings are below the threshold.
        if saved_us < min_share * before:
            did_not_matter.append(label)

    # Calculate the overall end-to-end speedup.
    overall_speedup = baseline_time / final_time

    # Identify the optimization with the largest time savings.
    # max() preserves the first occurrence when values are tied.
    biggest = (
        max(fixes, key=lambda item: item["saved_us"])["fix"]
        if fixes else None
    )

    # Return the complete performance attribution report.
    return {
        "speedup": overall_speedup,
        "fixes": fixes,
        "biggest": biggest,
        "did_not_matter": did_not_matter
    }

# Step 12 - release_check
def release_check(base_times, cand_times, base_loss, cand_loss,
                  loss_tol=0.01, min_speedup=1.05, warmup=2):
    """Ship only a candidate that is measurably faster without changing the loss."""

    # Compare baseline and candidate benchmark results.
    comparison = compare_runs(base_times, cand_times, warmup)

    # Extract the measured speedup and statistical verdict.
    speedup = comparison["speedup"]
    verdict = comparison["verdict"]

    # Collect release rejection reasons in the required order.
    reasons = []

    # Check whether the candidate is measurably faster.
    if verdict != "faster":
        reasons.append(
            "not faster beyond the noise: " + verdict
        )

    # Check whether the candidate meets the minimum speedup.
    if speedup is not None and speedup < min_speedup:
        reasons.append(
            f"speedup {speedup:.3f} is below {min_speedup}"
        )

    # Check whether the training loss has changed beyond tolerance.
    loss_change = cand_loss - base_loss

    if abs(loss_change) > loss_tol:
        reasons.append(
            f"loss changed by {loss_change:+.4f}"
        )

    # Approve the candidate only when there are no rejection reasons.
    return {
        "ship": len(reasons) == 0,
        "speedup": round(speedup, 4) if speedup is not None else None,
        "reasons": reasons
    }

