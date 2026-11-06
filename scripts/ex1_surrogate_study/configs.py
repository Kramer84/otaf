"""Candidate definitions. NEW recipe: AdamW, no weight decay, cosine LR to lr_end, shuffled mini-batches, PyTorch default init, no weight clamp.
OLD recipe: train_neural_networks.py settings reimplemented in ex1lib.train (validation every epoch, 4 unshuffled batches per epoch, growing LR)."""
BASE_DATA = dict(n=200_000, domain="inflated", m=1.35)
OLD = dict(hidden=[16, 8, 4], act="gelu", lr=3e-3, wd=1e-4, batch=30000, clamp=True, init="xavier002", shuffle=False, sched="growth", gamma=1.0001, val_every=4, epochs_max=2500, **BASE_DATA)
NEW = dict(batch=4096, steps=40_000, lr=2e-3, lr_end=1e-6, sched="cos", wd=0.0, **BASE_DATA)
def new(hidden, act, **k):
    c = dict(NEW, hidden=hidden, act=act); c.update(k); return c
SIZES = {"16": [16, 8, 4], "32": [32, 16, 8], "64": [64, 32, 16], "64x4": [64, 64, 64, 64]}
STAGES = {
 # A: what do the training settings do at the baseline size, activation and data?
 "A": {"A0_old_stop": dict(OLD, stop_r2=0.9998),
       "A1_old_nostop": dict(OLD),
       "A2_old_nostop_noclamp": dict(OLD, clamp=False),
       "A3_new_gelu16": new([16, 8, 4], "gelu")},
 # B: activation x size, 200k inflated, NEW recipe
 "B": {**{f"B_{a}{k}": new(h, a) for a in ["gelu", "relu", "leaky"] for k, h in SIZES.items() if not (a == "gelu" and k == "16")},
       **{f"B_relu_1h{w}": new([w], "relu") for w in [3, 8, 32]}},
 # C: training points
 "C": {"C_relu16_1M": new([16, 8, 4], "relu", n=1_000_000), "C_relu16_5M": new([16, 8, 4], "relu", n=5_000_000),
       "C_gelu64_1M": new([64, 32, 16], "gelu", n=1_000_000), "C_gelu64_5M": new([64, 32, 16], "gelu", n=5_000_000)},
 # D: training domain, 1M points
 "D": {**{f"D_relu16_{d}": new([16, 8, 4], "relu", n=1_000_000, domain=d) for d in ["credal", "uniform", "credal+limit", "credal+kink", "credal+limit+kink"]},
       "D_gelu64_credal+limit+kink": new([64, 32, 16], "gelu", n=1_000_000, domain="credal+limit+kink")},
 # E: loss trial and the one upper-bound row (input |g4-g5| from the closed form: not available for the larger models)
 "E": {"E_relu16_wmse": new([16, 8, 4], "relu", loss="wmse"), "E_gelu64_wmse": new([64, 32, 16], "gelu", loss="wmse"),
       "E_UPPERBOUND_gelu16_absfeature": new([16, 8, 4], "gelu", abs_feature=True)},
 # F: finalists, repeated over seeds, surfaces at 5e6
 "F5": {"F_relu16_credal_1M": new([16, 8, 4], "relu", n=1_000_000, domain="credal"),
        "F_relu16_credal_200k": new([16, 8, 4], "relu", domain="credal"),
        "F_relu16_inflated_200k": new([16, 8, 4], "relu")},
 "F3": {"F_gelu16_credal_1M": new([16, 8, 4], "gelu", n=1_000_000, domain="credal"),
        "F_gelu64_credal_1M": new([64, 32, 16], "gelu", n=1_000_000, domain="credal"),
        "F_gelu64_inflated_200k": new([64, 32, 16], "gelu"),
        "F_relu16_uniform_1M": new([16, 8, 4], "relu", n=1_000_000, domain="uniform")},
 # G: does training find the few-neuron solution? one hidden layer, surfaces at 1e6
 "G": {"G_relu_1h3": new([3], "relu", domain="credal"), "G_relu_1h4": new([4], "relu", domain="credal"), "G_relu_1h8": new([8], "relu", domain="credal"),
       "G_relu_1h3_inflated": new([3], "relu"), "G_relu_1h8_inflated": new([8], "relu")},
}
