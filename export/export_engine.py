"""Export the Vega physics engine (+ both gated adapters) to ONNX, fp32.

VegaEngine.forward has three data-dependent branches that do not trace. Each is rewritten as
an unconditional equivalent, and the rewrite is checked against the original on golden.npz
before the ONNX file is trusted:
  * memory reader: only run when some chunk is valid -> always run; MemoryReader already
    multiplies its output by cmask.any(), so items without chunks get exactly zero
  * score rail: only built when some item is a score question -> always built; torch.where
    already selects it per item
  * halt: settled particles are frozen and the loop breaks once all have settled -> always
    run n_steps; a frozen particle's z, p, U and grad do not change, so the extra steps are
    no-ops and steps_used is set once, at settling
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "vega")
from vega_api import Vega

MODEL_DIR = "vega"
OUT = "out/engine.onnx"


class ExportEngine(torch.nn.Module):
    def __init__(self, e):
        super().__init__()
        self.e = e

    def forward(self, feats, opts, opt_mask, qtype, chunks, chunk_mask):
        e = self.e
        slots, o = e._mix(feats, opts, False)
        gates = e._set_gates(slots)
        s = e.world(slots[:, 0])
        ch = (chunks.float() - e.feat_mu[-1, 0]) / e.feat_sd[-1, 0]
        s = s + e.memory(torch.cat([slots[:, 1], slots[:, 2]], -1), ch, chunk_mask)
        s_next = e.foresight(s)
        probe = e.probe(slots[:, 1]) + e.qtype_emb(qtype)
        imp = e.impulse(slots[:, 2])
        ctx = torch.cat([s, s_next, probe, imp], -1)
        z = e.to_z0(ctx)
        p = e.to_p0(ctx)
        c, log_a, sigma = self.landscape(o, opt_mask, qtype, probe)
        B = z.size(0)
        h = z.new_zeros(B, e.A_log.numel())
        settled = torch.zeros(B, dtype=torch.bool)
        steps_used = torch.full((B,), e.max_steps, dtype=torch.long)
        U, grad = e.potential_and_grad(z, c, log_a, sigma, opt_mask)
        A = e.A_log.exp()
        for t in range(e.max_steps):
            x = torch.cat([z, p, grad, grad.norm(dim=-1, keepdim=True)], -1)
            u = e.ssm_in(x)
            bu, dt_raw, fr_raw = u[:, :-2], u[:, -2:-1], u[:, -1:]
            dt = e.dt_max * torch.sigmoid(dt_raw)
            gamma = F.softplus(fr_raw)
            h = torch.exp(-A * dt) * h + dt * bu
            force = e.force_scale * torch.tanh(e.ssm_out(h))
            g_eff = grad / (1 + grad.norm(dim=-1, keepdim=True) / e.force_max)
            p_new = p * torch.exp(-gamma * dt) - dt * g_eff + dt * force
            z_new = z + dt * p_new
            keep = settled[:, None]
            p = torch.where(keep, p, p_new)
            z = torch.where(keep, z, z_new)
            U, grad = e.potential_and_grad(z, c, log_a, sigma, opt_mask)
            kin = 0.5 * (p ** 2).sum(-1)
            now = (kin < e.settle_eps) & (grad.norm(dim=-1) < 10 * e.settle_eps) & ~settled
            steps_used = torch.where(now, torch.full_like(steps_used, t + 1), steps_used)
            settled = settled | now
        E, d2 = e.well_energy(z, c, log_a, sigma, opt_mask)
        nearest = (d2 / (2 * sigma ** 2)).masked_fill(~opt_mask, 1e4).min(-1).values
        return -E, 0.5 * (p ** 2).sum(-1), nearest, steps_used, gates

    def landscape(self, o, opt_mask, qtype, probe):
        e = self.e
        B, K, _ = o.shape
        out = e.well(torch.cat([o, probe[:, None, :].expand(B, K, -1)], -1))
        c = out[..., :e.D] + e.qtype_emb(qtype)[:, None, :]
        log_a = out[..., e.D].clamp(-4, 4)
        sigma = 0.3 + F.softplus(out[..., e.D + 1])
        m = opt_mask.float()
        r0 = (c * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
        u = F.normalize(e.rail_dir(probe), dim=-1)
        gaps = F.softplus(out[..., e.D + 2]) * m
        t = torch.cumsum(gaps, 1)
        t = t - (t * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
        rail = r0[:, None, :] + t[..., None] * u[:, None, :]
        c = torch.where((qtype == 1)[:, None, None], rail, c)
        return c, log_a, sigma


def inputs_for(npz, gold, key):
    case, qid = key.split("/")
    g = next(c for c in gold["cases"] if c["id"] == case)["questions"][qid]
    k = g["k"]
    return (torch.from_numpy(npz[key + "/feats"])[None], torch.from_numpy(npz[key + "/opts"])[None],
            torch.ones(1, k, dtype=torch.bool), torch.tensor([g["qtype"]]),
            torch.from_numpy(npz[key + "/chunks"])[None], torch.from_numpy(npz[key + "/chunk_mask"])[None]), g


@torch.no_grad()
def main():
    v = Vega(MODEL_DIR, backbone="qwen", device="cpu")
    assert v.engine.adapter_mode == "auto" and len(v.adapters) == 2, (v.engine.adapter_mode, len(v.adapters))
    m = ExportEngine(v.engine).eval()
    gold = json.load(open("../golden/golden.json"))
    npz = np.load("../golden/golden.npz")
    keys = sorted({k.rsplit("/", 1)[0] for k in npz.files})
    # 1. the unconditional rewrite against the reference engine outputs in golden.json
    worst = 0.0
    for key in keys:
        x, g = inputs_for(npz, gold, key)
        logits, kin, near, steps, gates = m(*x)
        d = max(np.abs(logits[0].numpy() - g["logits"]).max(), abs(float(kin[0]) - g["kinetic"]),
                abs(float(near[0]) - g["nearest"]), np.abs(gates[0].numpy() - g["gates"]).max())
        assert int(steps[0]) == g["steps_used"], (key, int(steps[0]), g["steps_used"])
        worst = max(worst, d)
    print("rewrite vs reference engine: max |d| %.2e over %d questions" % (worst, len(keys)))
    assert worst < 1e-4, worst
    # 2. export, dynamic batch and option count
    os.makedirs("out", exist_ok=True)
    x, _ = inputs_for(npz, gold, keys[0])
    torch.onnx.export(m, x, OUT, dynamo=False, opset_version=17,
                      input_names=["feats", "opts", "opt_mask", "qtype", "chunks", "chunk_mask"],
                      output_names=["logits", "kinetic", "nearest", "steps_used", "gates"],
                      dynamic_axes={"feats": {0: "B"}, "opts": {0: "B", 1: "K"}, "opt_mask": {0: "B", 1: "K"},
                                    "qtype": {0: "B"}, "chunks": {0: "B"}, "chunk_mask": {0: "B"},
                                    "logits": {0: "B", 1: "K"}, "kinetic": {0: "B"}, "nearest": {0: "B"},
                                    "steps_used": {0: "B"}, "gates": {0: "B"}})
    print("wrote %s (%.1f MB)" % (OUT, os.path.getsize(OUT) / 1e6))
    # 3. the config the browser reads: prompt + pooling settings and every calibration block, in the
    # engine's adapter order (gates[:, j] belongs to v.adapters[j])
    keys = ("max_len", "opt_max_tokens", "query_first_max_state", "n_steps", "memory_chunks", "memory_min_tokens",
            "memory_chunk_tokens", "feature_layers", "option_layers", "calibration")
    web = {k: v.cfg[k] for k in keys}
    web["calibration"] = {k: web["calibration"][k] for k in ("theta", "conformal", "abstain_threshold",
                                                             "unbound_threshold", "default_risk")}
    web["adapters"] = [{"name": a["name"], "gate_threshold": a["threshold"],
                        "calibration": {k: a["meta"]["calibration"].get(k) for k in
                                        ("theta", "conformal", "abstain_threshold", "unbound_threshold")}}
                       for a in v.adapters]
    with open("out/vega_web.json", "w") as f:
        json.dump(web, f, indent=1)
    print("wrote out/vega_web.json")


if __name__ == "__main__":
    main()
