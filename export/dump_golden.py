"""Golden reference: run the real Vega (PyTorch, CPU, fp32) over the fixtures and dump every
intermediate the ONNX/TS port must reproduce.

Per question, at batch 1 (the browser's shape): prompt ids and spans, the pooled perception
features the engine reads, the raw engine outputs, and the calibrated probabilities. Per case:
the full Vega.decide() response, which batches the questions of a request together.

Arrays too big for JSON (engine inputs) go to golden.npz, keyed "<case>/<qid>/<name>", so the
engine can be checked on its own, independent of backbone drift.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, "vega")
from vega_api import Vega
from vega_common import QTYPES, build_prompt, option_texts, serialize_state, softmax_np, thermal_features, thermal_tau, \
    route_adapters

MODEL_DIR = "vega"
BACKBONE = "qwen"
OUT = "../golden/golden.json"
NPZ = "../golden/golden.npz"


def one(v, text, q):
    """Vega._view for a single question, keeping the engine inputs."""
    cfg = v.cfg
    ids, spans = build_prompt(v.perc.tok, text, q, cfg["max_len"], opt_max_tokens=cfg["opt_max_tokens"],
                              query_first_max_state=cfg.get("query_first_max_state", 0))
    items = [{"ids": np.asarray(ids, dtype=np.int32), "spans": spans}]
    C = v.engine.memory_chunks
    feats, opts, chunks, cmask = v.perc.pooled(items, chunks=C, prefix_min=cfg.get("prefix_min_tokens", 4096),
                                               chunk_min_tokens=cfg.get("memory_min_tokens", 2048),
                                               chunk_tokens=cfg.get("memory_chunk_tokens", 128))
    K = len(spans["options"])
    opts = opts[None]                                           # [1, K, Lo, d]
    mask = torch.ones((1, K), dtype=torch.bool)
    qtype = torch.tensor([QTYPES[q["t"]]])
    out = v.engine(feats, opts, mask, qtype, halt=v.halt, return_traj=False, chunks=chunks, chunk_mask=cmask)
    logits = out["logits"].float().numpy()
    phys = thermal_features(out, mask, v.engine.max_steps).numpy()
    gates = out["gates"].float().numpy() if out.get("gates") is not None else None
    route = route_adapters(gates, v.adapters[0]["threshold"] if v.adapters else 0.5)
    r = -1 if route is None else int(route[0])
    tau = thermal_tau(v._cal(r)["theta"], qtype.numpy(), phys)
    p = softmax_np(logits, tau)[0]
    arrays = {"feats": feats[0].numpy(), "opts": opts[0].numpy(), "chunks": chunks[0].numpy(),
              "chunk_mask": cmask[0].numpy()}
    rec = {"input_ids": [int(i) for i in ids], "spans": {"state": list(spans["state"]), "question": list(spans["question"]),
                                                         "options": [list(s) for s in spans["options"]]},
           "n_tokens": len(ids), "k": K, "qtype": int(qtype[0]), "rendered_options": option_texts(q),
           "logits": [float(x) for x in logits[0]], "kinetic": float(out["kinetic"][0]),
           "nearest": float(out["nearest"][0]), "steps_used": int(out["steps_used"][0]),
           "gates": None if gates is None else [float(x) for x in gates[0]], "route": r,
           "phys": [float(x) for x in phys[0]], "tau": float(tau[0]), "probs": [float(x) for x in p],
           "n_chunks": int(cmask[0].sum())}
    return rec, arrays


@torch.no_grad()
def main():
    v = Vega(MODEL_DIR, backbone=BACKBONE, device="cpu")
    cases = json.load(open("../golden/fixtures.json"))
    out = {"model_dir": MODEL_DIR, "backbone": BACKBONE, "adapters": [a["name"] for a in v.adapters],
           "cases": []}
    arrays = {}
    for case in cases:
        text = serialize_state(case["state"])
        rec = {"id": case["id"], "questions": {}}
        for qid, qdef in case["questions"].items():
            q = Vega._internal(qdef)
            r, a = one(v, text, q)
            rec["questions"][qid] = r
            for k, x in a.items():
                arrays["%s/%s/%s" % (case["id"], qid, k)] = x
        rec["response"] = v.decide(case["state"], case["questions"])
        out["cases"].append(rec)
        print("%-28s %s" % (case["id"], "  ".join(
            "%s k=%d L=%d ch=%d steps=%d route=%d p=[%s]" % (
                qid, x["k"], x["n_tokens"], x["n_chunks"], x["steps_used"], x["route"],
                ",".join("%.3f" % y for y in x["probs"][:4]))
            for qid, x in rec["questions"].items())), flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    np.savez_compressed(NPZ, **arrays)
    print("\nwrote %s + %s (%d cases, %d questions)" % (OUT, NPZ, len(out["cases"]),
                                                        sum(len(c["questions"]) for c in out["cases"])))


if __name__ == "__main__":
    main()
