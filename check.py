from time import perf_counter

t0 = perf_counter()
# === features ===
t1 = perf_counter()

# === masks ===
t2 = perf_counter()

# === build R/M ===
t3 = perf_counter()

print({
    "features_s": t1 - t0,
    "masks_s": t2 - t1,
    "tensor_build_s": t3 - t2,
    "total_s": t3 - t0,
})
