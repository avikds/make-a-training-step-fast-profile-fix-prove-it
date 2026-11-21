# Make a Training Step Fast: Profile, Fix, Prove It

A small transformer trains on 8 GPUs and each step takes 155 ms. Profile it the way a performance engineer does: cost every kernel with the roofline, read the trace to find what the GPU is waiting for, and measure with warmup and medians instead of single runs. Then fix what the profile shows, one thing at a time: data-loader workers, bf16, sequence packing with per-sample attention, gradient buckets that hide the all-reduce behind backward, and kernel fusion. Add a dynamic loss scaler for fp16 and plan how to split a 7B model across the GPUs. Finish by attributing the 32x speedup to each fix with Amdahl's law and letting a release check refuse the fp8 version that changes the loss. The GPU is simulated, so every number is exact and every step is testable.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** op_cost
- [x] **2.** analyze_trace
- [x] **3.** summarize_runs
- [x] **4.** choose_workers
- [x] **5.** pack_sequences
- [ ] **6.** fuse_elementwise
- [ ] **7.** dynamic_loss_scale
- [ ] **8.** bucket_allreduce
- [ ] **9.** plan_parallelism
- [ ] **10.** compare_runs
- [ ] **11.** speedup_report
- [ ] **12.** release_check

---

Built on Deep-ML.
