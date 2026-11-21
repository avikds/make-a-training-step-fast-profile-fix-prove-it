"""
Make a Training Step Fast: Profile, Fix, Prove It scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Make a training step fast: profile it, fix what the profile shows, and prove each fix with a fair benchmark.

Story: a small transformer trains on 8 data-parallel GPUs and each step takes far too long. The profile shows
the GPU waiting for data, then doing 3x more work than needed because of padding, in a slow precision, with the
gradient all-reduce exposed after backward. Fix one thing at a time, re-profile after each, and let a release
check decide which candidate may ship.
"""


def show(label, cfg):
    res = run_step(cfg)
    a = analyze_trace(res["trace"])
    print(f"{label:<26} step {res['step_us'] / 1000:8.2f} ms | GPU idle {a['gpu_idle_share']:.0%} | "
          f"data wait {a['data_wait_us'] / 1000:6.2f} ms | exposed comm {a['exposed_comm_us'] / 1000:5.2f} ms | {a['verdict']}")
    return res["step_us"]


def main():
    cfg = {"workers": 0, "precision": "fp32"}
    steps = [("baseline", show("baseline", cfg))]
    for label, change in [("loader workers", {}), ("bf16", {"precision": "bf16"}), ("pack sequences", {"pack": pack_sequences}),
                          ("bucket all-reduce", {"comm": bucket_allreduce}), ("fuse elementwise", {"fuse": fuse_elementwise})]:
        cfg = dict(cfg, **change)
        cfg["workers"] = choose_workers(cfg)["workers"]  # re-profile after every fix and keep the GPU fed
        label = f"{label} ({cfg['workers']}w)"
        steps.append((label, show(label, cfg)))
    steps.append(("32 workers", show("32 workers", dict(cfg, workers=32))))

    rep = speedup_report(steps)
    print("end-to-end speedup:", round(rep["speedup"], 2), "| biggest fix:", rep["biggest"], "| did not matter:", rep["did_not_matter"])
    for f in rep["fixes"]:
        print(f"    {f['fix']:<26} saved {f['saved_us'] / 1000:8.2f} ms  share {f['share']:6.1%}")

    print("fp16 loss scaler:", {k: v for k, v in dynamic_loss_scale(GRAD_STATS).items() if k != "scales"})
    print("fp16 loss without / with the scaler:", train_loss({"precision": "fp16"}),
          train_loss({"precision": "fp16", "scaler": dynamic_loss_scale}))

    base = {"workers": 0, "precision": "fp32"}
    for label, cand in [("optimized bf16", cfg), ("optimized fp8", dict(cfg, precision="fp8")),
                        ("one more worker", dict(cfg, workers=cfg["workers"] + 1))]:
        ref = cfg if label == "one more worker" else base
        print(f"release {label}:", release_check(measure(ref, seed=1), measure(cand, seed=2), train_loss(ref), train_loss(cand)))

    big = {"params_b": 7.0, "act_gb": 40.0, "step_tflop": 2.6e3, "tp_gb": 6.0}
    cluster = {"gpus": 8, "gpus_per_node": 8, "mem_gb": 80.0, "tflops": 64.0, "intra_gbs": 300.0, "inter_gbs": 50.0,
               "tp_penalty": 0.04}
    plans = plan_parallelism(big, cluster)
    print("7B plans that fit on 8 GPUs:", len(plans["fits"]), "| best:", plans["fits"][0] if plans["fits"] else None)


if __name__ == "__main__":
    main()

