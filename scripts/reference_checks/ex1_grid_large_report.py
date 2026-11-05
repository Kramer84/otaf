"""Summary numbers of ex1_grid_large.py results. Usage: python ex1_grid_large_report.py file.npz [...]"""
import sys
import numpy as np
for f in sys.argv[1:]:
    d = np.load(f)
    done = ~np.isnan(d["mc_sur"][:, 0, 0]); print("\n==", f, "rows done", done.sum(), "of", len(done), "n", d["n"])
    sc = d["scales"]
    for ia, al in enumerate(d["alphas"]):
        print(f"-- alpha={al}")
        for k in ("mc_exact", "mc_sur", "gld_exact", "gld_sur"):
            g = d[k][:, :, ia]
            imin, imax = np.unravel_index(np.nanargmin(g), g.shape), np.unravel_index(np.nanargmax(g), g.shape)
            print(f"{k:10s} TT(1,1) {g[-1,-1]:.3e} RR(0,0) {g[0,0]:.3e} TR(1,0) {g[0,-1]:.3e} min {g[imin]:.3e} at l1={sc[imin[1]]:.2f},l2={sc[imin[0]]:.2f} max {g[imax]:.3e} at l1={sc[imax[1]]:.2f},l2={sc[imax[0]]:.2f}")
        ref = d["mc_exact"][:, :, ia]
        for k in ("mc_sur", "gld_exact", "gld_sur"):
            e = (d[k][:, :, ia] - ref) / ref
            print(f"  rel.err {k:10s} mean {np.nanmean(e):+.3f} med {np.nanmedian(e):+.3f} p5 {np.nanpercentile(e,5):+.3f} p95 {np.nanpercentile(e,95):+.3f} min {np.nanmin(e):+.3f} max {np.nanmax(e):+.3f}")
    for k in ("bias", "rmse", "bias_band", "rmse_band", "wrong_sign"):
        g = d[k]; print(f"{k:10s} median {np.nanmedian(g):+.2e} max|.| {np.nanmax(np.abs(g)):.2e} corner TT {g[-1,-1]:+.2e} RR {g[0,0]:+.2e}")
