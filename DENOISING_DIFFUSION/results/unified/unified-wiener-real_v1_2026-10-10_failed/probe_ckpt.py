import sys, types, numpy as np, torch
sys.path.insert(0, "experiments")
import train_unified as tu
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("round", ck["round"], "steps", ck["steps"], "K", ck["K"], "args lr", ck["args"].get("lr"), "snr", ck["args"].get("snr"))
tu.K = int(ck["K"])
net = tu.build_net(ck["arm"], int(ck["base"])); net.load_state_dict(ck["model_state_dict"]); net.eval()
bad = [n for n, p in net.named_parameters() if not torch.isfinite(p).all()]
mx = max(float(p.abs().max()) for p in net.parameters())
print("non-finite params:", bad[:5], "| max |param|", round(mx, 2))
print("conv_out |w| max", float(net.conv_out.weight.abs().max()))
tu.REAL.update(snr=(10.0, 150.0), chan_a=0.5, beam_max=1.6)
a = types.SimpleNamespace(hydro="9015", analytic="an01", n_line=1, p_ai=0.5, train_sgv2=False)
cubes = tu.load_train_cubes(a)
rng = np.random.default_rng(0)
acts = {}
def hook(name):
    def f(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        if torch.is_tensor(t):
            acts[name] = max(acts.get(name, 0.0), float(t.detach().abs().max()))
            if not torch.isfinite(t).all(): acts[name + "_NONFINITE"] = 1.0
    return f
for n, m in net.named_modules():
    if isinstance(m, (torch.nn.Conv2d, torch.nn.GroupNorm)) or "attn" in n.lower():
        m.register_forward_hook(hook(n))
probs = np.ones(len(cubes)) / len(cubes)
worst = 0
for trial in range(6):
    for c in cubes:
        c.redraw(rng, [256])
    for _ in range(3):
        x, b0, cc = tu.make_batch(cubes, probs, 12, 128, rng, "wiener")
        with torch.no_grad():
            out = net(x, torch.zeros(x.shape[0], dtype=torch.long, device=tu.dev), None)
        worst = max(worst, float(out.abs().max()))
print("max |net output|", round(worst, 3), "| x range of last batch", float(x.min()), float(x.max()))
top = sorted(acts.items(), key=lambda kv: -kv[1])[:8]
print("largest activations (fp16 limit 65504):"); [print(f"  {n:45s} {v:.4g}") for n, v in top]
print("any non-finite:", [k for k in acts if k.endswith("_NONFINITE")][:5])
