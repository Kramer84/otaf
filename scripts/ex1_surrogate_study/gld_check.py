"""Best candidate only: Pf from a VSL GLD fit (gldpy-gpu), against direct sampling on the same points.
At TT, RR, EE on the 5e6 test sample: float32 fit with xtol 1e-6 as in plot_pf_2d_4dof_model.py (pf_at_alphas) and float64 as in ex1_grid_large.py.
The 50x50 GLD grid was dropped at the author's request (scope change).
Applied to the closed form, the best surrogate and the Nov. checkpoint.
Usage: python ex1_surrogate_study/gld_check.py BEST_CKPT_TAG"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[1]))
import ex1lib as L
from gldpy_gpu import gpu as gld_gpu
from plot_pf_2d_4dof_model import pf_at_alphas
torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(0.4)
tag = sys.argv[1]
tb = L.TestBed()
gld = gld_gpu.GLD("VSL")
A = list(L.ALPHAS)
models = {"closed_form": lambda x: L.closed_form(x), "best:" + tag: L.load_ckpt(L.HERE / "ckpt" / f"{tag}.pth"), "B0_nov_checkpoint": L.load_ckpt(L.SCRIPTS / "artifacts/model1_4_dof_surrogate.pth")}
out = {"best": tag}
for name, fn in models.items():
    out[name] = {}
    for k, v in L.LAM.items():
        s = tb.predict(fn, L.lam_vec(*v)).cpu().numpy()
        s64 = closed_form64 = L.closed_form(tb.zr * L.SIGMA * L.lam_vec(*v)) if name == "closed_form" else s.astype(np.float64)
        direct = float((s64 < 0).mean())
        g32 = pf_at_alphas(s.astype(np.float32), [0.0], gld, "gpu")[0]
        t = torch.as_tensor(s64, dtype=torch.float64, device=gld.device)
        p = gld.fit_LMM(t)
        g64 = float(gld.CDF_num(0.0, p, xtol=1e-9))
        del t
        out[name][k] = dict(direct=direct, gld_float32=g32, gld_float64=g64, params64=p.detach().cpu().numpy().ravel().tolist())
        print(name, k, f"direct {direct:.4e} gld32 {g32:.4e} gld64 {g64:.4e}", flush=True)
(L.HERE / "results" / "_gld_check.json").write_text(json.dumps(L.jsonable(out), indent=1))
