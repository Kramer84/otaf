"""Where do the largest slack errors of the ReLU 16-8-4 credal models sit, per seed?"""
import sys, json
import numpy as np, torch
import ex1lib as L
torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(0.2)
tb = L.TestBed()
out = {}
for cfg in ["F_relu16_credal_200k", "F_relu16_credal_1M"]:
  for sd in range(5):
    tag = f"{cfg}_s{sd}"; n = L.load_ckpt(L.HERE / "ckpt" / f"{tag}.pth")
    for name, lam in [("TT", L.lam_vec(1, 1)), ("RR", L.lam_vec(0, 0)), ("EE", L.lam_vec(2**-.5, 2**-.5)), ("credal", tb.lam32)]:
        l = torch.as_tensor(lam, dtype=torch.float32, device=L.DEV)
        x = (tb.z32 * l)
        with torch.no_grad():
            e = torch.cat([n(x[i:i + (1 << 19)]) for i in range(0, len(x), 1 << 19)]).double() - L.closed_form(x.double())
        ref = L.closed_form(x.double()); a = e.abs()
        k = int(a.argmax())
        band = (ref > -0.02) & (ref < 0.095)              # the band of all thresholds alpha in the study
        v = a.topk(1000).values.min()
        out[f"{tag}/{name}"] = dict(max_all=float(a.max()), s_at_max=float(ref[k]), max_in_band=float(a[band].max()), share_gt_1e4=float((a > 1e-4).double().mean()), share_in_band=float(band.double().mean()))
        print(tag, name, f"max {float(a.max()):.1e} at s={float(ref[k]):+.3f}, in s band [-0.02,0.095] max {float(a[band].max()):.1e}, share |e|>1e-4 {float((a > 1e-4).double().mean()):.1e}", flush=True)
json.dump(out, open(L.HERE / "results" / "_tail_check.json", "w"), indent=1)
