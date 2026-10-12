"""Run the ONNX backbone + engine pair over golden.json, at batch 1, exactly as the browser will.

Reports backbone drift (pooled features vs golden.npz) separately from the end result, then the
parity-gate metrics: argmax agreement, max |dp|, mean KL(p_ref || p_onnx).

    python verify_onnx.py backbone_q8 [engine]
"""
import json, math, sys
import numpy as np
import onnxruntime as ort

GOLD = "../golden/golden.json"
NPZ = "../golden/golden.npz"
FEATURE_LAYERS = (13, 19)
CHUNKS, CHUNK_MIN, CHUNK_TOKENS = 16, 128, 128


def session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def empty_past(sess):
    feeds = {}
    for i in sess.get_inputs():
        if i.name.startswith("past_"):
            shape = [1 if d == "batch_size" else 0 if isinstance(d, str) else d for d in i.shape]
            feeds[i.name] = np.zeros(shape, dtype=np.float32)
    return feeds


def pool(h13, h19, sp, n):
    """Perception._pooled_batch at B=1: span means, final token, chunk means of the state span."""
    mean = lambda h, a, e: h[a:e].mean(0) if e > a else np.zeros(h.shape[1], np.float32)
    feats = np.stack([np.stack([mean(h, *sp["state"]), mean(h, *sp["question"]), h[n - 1]]) for h in (h13, h19)])
    opts = np.stack([mean(h19, a, e)[None] for a, e in sp["options"]])
    chunks = np.zeros((CHUNKS, h19.shape[1]), np.float32)
    cmask = np.zeros(CHUNKS, bool)
    s0, s1 = sp["state"]
    m = s1 - s0
    if m >= max(CHUNK_MIN, 2):
        k = max(2, min(CHUNKS, m // CHUNK_TOKENS))
        cid = np.arange(m) * k // m
        for j in range(k):
            chunks[j] = h19[s0:s1][cid == j].mean(0)
        cmask[:k] = True
    return feats.astype(np.float32), opts.astype(np.float32), chunks, cmask


def calibrate(theta, qtype, logits, kinetic, nearest, steps, k, n_steps=12):
    phys = np.array([math.log1p(kinetic), math.log1p(nearest), steps / n_steps, math.log(max(k, 2))])
    th = np.array(theta[qtype])
    tau = math.exp(min(max(th[0] + (th[1:] * phys).sum(), math.log(0.2)), math.log(20.0)))
    z = np.asarray(logits, np.float64) / tau
    p = np.exp(z - z.max())
    return p / p.sum()


def main(backbone, engine="engine"):
    gold = json.load(open(GOLD))
    npz = np.load(NPZ)
    cfg = json.load(open("vega/vega_config.json"))
    ads = [json.load(open("vega/adapters/%s.json" % a)) for a in gold["adapters"]]
    bb = session("out/%s.onnx" % backbone)
    en = session("out/%s.onnx" % engine)
    past = empty_past(bb)
    rows = []
    for case in gold["cases"]:
        for qid, g in case["questions"].items():
            key = "%s/%s" % (case["id"], qid)
            ids = np.array([g["input_ids"]], dtype=np.int64)
            n = ids.shape[1]
            h13, h19 = bb.run(["h13", "h19"], dict(past, input_ids=ids, attention_mask=np.ones_like(ids)))
            feats, opts, chunks, cmask = pool(h13[0], h19[0], g["spans"], n)
            rf = npz[key + "/feats"]
            d_feat = float(np.abs(feats - rf).max() / (np.abs(rf).max() + 1e-9))
            k = g["k"]
            logits, kin, near, steps, gates = en.run(None, {
                "feats": feats[None], "opts": opts[None], "opt_mask": np.ones((1, k), bool),
                "qtype": np.array([g["qtype"]], np.int64), "chunks": chunks[None], "chunk_mask": cmask[None]})
            route = int(gates[0].argmax()) if gates[0].max() >= ads[0].get("gate_threshold", 0.5) else -1
            theta = (cfg["calibration"] if route < 0 else ads[route]["calibration"])["theta"]
            p = calibrate(theta, g["qtype"], logits[0], float(kin[0]), float(near[0]), int(steps[0]), k)
            pr = np.array(g["probs"])
            rows.append({"case": key, "k": k, "L": n, "d_feat": d_feat,
                         "d_logit": float(np.abs(logits[0] - g["logits"]).max()),
                         "d_p": float(np.abs(p - pr).max()),
                         "kl": float((pr * np.log(np.clip(pr, 1e-12, None) / np.clip(p, 1e-12, None))).sum()),
                         "argmax_ok": int(p.argmax()) == int(pr.argmax()), "route_ok": route == g["route"],
                         "steps": (int(steps[0]), g["steps_used"])})
    print("%-36s %3s %5s %9s %8s %7s %9s %s" % ("case", "k", "L", "d_feat", "d_logit", "d_p", "kl", "flags"))
    for r in rows:
        flags = ("" if r["argmax_ok"] else " ARGMAX") + ("" if r["route_ok"] else " ROUTE") + \
                ("" if r["steps"][0] == r["steps"][1] else " steps%s" % (r["steps"],))
        print("%-36s %3d %5d %9.2e %8.4f %7.4f %9.2e%s" % (r["case"], r["k"], r["L"], r["d_feat"], r["d_logit"],
                                                       r["d_p"], r["kl"], flags))
    n = len(rows)
    print("\nargmax %d/%d  route %d/%d  max|dp| %.4f  mean KL %.2e" % (
        sum(r["argmax_ok"] for r in rows), n, sum(r["route_ok"] for r in rows), n,
        max(r["d_p"] for r in rows), np.mean([r["kl"] for r in rows])))


if __name__ == "__main__":
    main(*sys.argv[1:])
