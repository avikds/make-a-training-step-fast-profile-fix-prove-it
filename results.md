```
baseline                   step   154.70 ms | GPU idle 27% | data wait  40.00 ms | exposed comm  1.78 ms | input_bound
loader workers (1w)        step   114.70 ms | GPU idle 2% | data wait   0.00 ms | exposed comm  1.78 ms | compute_bound
bf16 (4w)                  step    17.78 ms | GPU idle 10% | data wait   0.00 ms | exposed comm  1.78 ms | compute_bound
pack sequences (8w)        step     6.05 ms | GPU idle 30% | data wait   0.00 ms | exposed comm  1.78 ms | communication_bound
bucket all-reduce (8w)     step     5.07 ms | GPU idle 16% | data wait   0.00 ms | exposed comm  0.80 ms | compute_bound
fuse elementwise (16w)     step     4.85 ms | GPU idle 17% | data wait   0.00 ms | exposed comm  0.81 ms | compute_bound
32 workers                 step     4.85 ms | GPU idle 17% | data wait   0.00 ms | exposed comm  0.81 ms | compute_bound
end-to-end speedup: 31.87 | biggest fix: bf16 (4w) | did not matter: ['32 workers']
    loader workers (1w)        saved    40.00 ms  share  26.7%
    bf16 (4w)                  saved    96.92 ms  share  64.7%
    pack sequences (8w)        saved    11.73 ms  share   7.8%
    bucket all-reduce (8w)     saved     0.98 ms  share   0.7%
    fuse elementwise (16w)     saved     0.21 ms  share   0.1%
    32 workers                 saved     0.00 ms  share   0.0%
fp16 loss scaler: {'skipped': [0, 1, 2, 8], 'underflow_steps': 0, 'final_scale': 16384.0}
fp16 loss without / with the scaler: 2.402 2.002
release optimized bf16: {'ship': True, 'speedup': 31.4498, 'reasons': []}
release optimized fp8: {'ship': False, 'speedup': 47.9856, 'reasons': ['loss changed by +0.0688']}
release one more worker: {'ship': False, 'speedup': 0.9868, 'reasons': ['not faster beyond the noise: no measurable change', 'speedup 0.987 is below 1.05']}
7B plans that fit on 8 GPUs: 12 | best: {'tp': 2, 'dp': 4, 'zero': 2, 'memory_gb': 39.25, 'step_ms': 5511.25}
```
