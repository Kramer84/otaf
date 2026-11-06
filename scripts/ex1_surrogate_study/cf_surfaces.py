"""Direct sampling of the closed form over the grid on the test sample, for n = 1e6 and 5e6 (noise floor and paired surface errors)."""
import ex1lib as L, torch
torch.set_num_threads(2)
tb = L.TestBed()
for n in (1_000_000, 5_000_000):
    print(n, tb.cf_surface(n).shape, flush=True)
