"""Golden reference for the image path: VegaVision (PyTorch, CPU, fp32) over golden/images.

Per question it dumps the expanded prompt ids and spans, the image grid, the 3-row M-RoPE
position ids, the pooled engine inputs and the engine's raw outputs and probabilities. Pixel
values and engine inputs go to golden_vision.npz so the TS image processor and the ONNX vision
encoder can each be checked on their own.

The image path is zero-shot (the engine was trained on text features) and, as in
vega_vision.py, feeds the engine no memory chunks.
"""
import json, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "vega")
from vega_vision import VegaVision, VISION_STATE
from vega_common import QTYPES, build_prompt, route_adapters, softmax_np, thermal_features, thermal_tau

IMAGES = "../golden/images"
OUT = "../golden/golden_vision.json"
NPZ = "../golden/golden_vision.npz"

CASES = [
    {"id": "invoice", "image": "invoice_640x480.png", "questions": {
        "kind": {"type": "choice", "instructions": "What kind of document is this?",
                 "criteria": {"invoice": "a bill", "receipt": "proof of payment", "letter": "correspondence",
                              "other": "none of these"}},
        "money": {"type": "boolean", "instructions": "Does this document ask for a payment?"}}},
    {"id": "letter", "image": "letter_500x700.png", "questions": {
        "kind": {"type": "choice", "instructions": "What kind of document is this?",
                 "criteria": ["invoice", "receipt", "letter", "form"]}}},
    {"id": "receipt", "image": "receipt_1400x900.png", "questions": {
        "kind": {"type": "choice", "instructions": "What kind of document is this?",
                 "criteria": {"invoice": "a bill", "receipt": "proof of payment", "letter": "correspondence",
                              "other": "none of these"}},
        "paid": {"type": "boolean", "instructions": "Has this been paid already?"}}},
]


@torch.no_grad()
def main():
    vv = VegaVision("vega", backbone="qwen", device="cpu")
    proc, model, cfg = vv.proc, vv.model, vv.cfg
    eng = vv.d.engine
    out = {"pad_id": vv.pad_id, "merge_size": proc.image_processor.merge_size,
           "image_processor": {k: getattr(proc.image_processor, k, None) for k in
                               ("patch_size", "temporal_patch_size", "merge_size", "image_mean", "image_std")},
           "size": dict(proc.image_processor.size), "cases": []}
    arrays = {}
    for case in CASES:
        image = Image.open(os.path.join(IMAGES, case["image"])).convert("RGB")
        im = proc.image_processor(images=[image], return_tensors="pt")
        pv, grid = im["pixel_values"].float(), im["image_grid_thw"]
        n_img = int(grid.prod()) // (proc.image_processor.merge_size ** 2)
        arrays[case["id"] + "/pixel_values"] = pv.numpy()
        rec = {"id": case["id"], "image": case["image"], "size": list(image.size), "grid_thw": grid[0].tolist(),
               "n_img": n_img, "questions": {}}
        for qid, qdef in case["questions"].items():
            q = vv.d._internal(qdef)
            ids, sp = build_prompt(vv.tok, VISION_STATE, q, cfg["max_len"], template=0,
                                   opt_max_tokens=cfg["opt_max_tokens"],
                                   query_first_max_state=cfg.get("query_first_max_state", 0))
            ids = list(ids)
            j = ids.index(vv.pad_id)
            ids = ids[:j] + [vv.pad_id] * n_img + ids[j + 1:]
            sh = lambda a: a + (n_img - 1) if a > j else a
            sp = {"state": (sp["state"][0], sh(sp["state"][1])), "question": tuple(map(sh, sp["question"])),
                  "options": [tuple(map(sh, o)) for o in sp["options"]]}
            t = torch.tensor([ids])
            pos, _ = model.model.get_rope_index(t, (t == vv.pad_id).long(), image_grid_thw=grid)
            cap = vv._run(t, pv, grid)
            h = {i: cap[i][0].float() for i in vv.capture}
            mean = lambda i, a, e: h[i][a:e].mean(0)
            feats = torch.stack([torch.stack([mean(i, *sp["state"]), mean(i, *sp["question"]), h[i][len(ids) - 1]])
                                 for i in vv.feature_layers])[None]
            opts = torch.stack([torch.stack([mean(i, a, e) for i in vv.option_layers]) for a, e in sp["options"]])[None]
            K = opts.size(1)
            mask = torch.ones(1, K, dtype=torch.bool)
            qtype = torch.tensor([QTYPES[q["t"]]])
            o = eng(feats, opts, mask, qtype, halt=vv.d.halt, return_traj=False, chunks=None, chunk_mask=None)
            logits = o["logits"].float().numpy()
            phys = thermal_features(o, mask, eng.max_steps).numpy()
            gates = o["gates"].float().numpy() if o.get("gates") is not None else None
            route = route_adapters(gates, vv.d.adapters[0]["threshold"] if vv.d.adapters else 0.5)
            r = -1 if route is None else int(route[0])
            p = softmax_np(logits, thermal_tau(vv.d._cal(r)["theta"], qtype.numpy(), phys))[0]
            key = "%s/%s" % (case["id"], qid)
            arrays[key + "/feats"] = feats[0].numpy()
            arrays[key + "/opts"] = opts[0].numpy()
            arrays[key + "/position_ids"] = pos[:, 0].numpy()
            rec["questions"][qid] = {
                "input_ids": ids, "spans": {"state": list(sp["state"]), "question": list(sp["question"]),
                                            "options": [list(s) for s in sp["options"]]},
                "k": K, "qtype": int(qtype[0]), "logits": [float(x) for x in logits[0]],
                "steps_used": int(o["steps_used"][0]), "route": r, "probs": [float(x) for x in p]}
        ref = vv.decide_image(image, case["questions"])
        for qid, a in ref["answers"].items():
            d = np.abs(np.array(list(a["probs"].values())) - rec["questions"][qid]["probs"]).max()
            assert d < 1e-5, (case["id"], qid, d)
        rec["response"] = ref
        out["cases"].append(rec)
        print("%-8s %s grid=%s n_img=%d  %s" % (case["id"], case["image"], rec["grid_thw"], n_img, "  ".join(
            "%s L=%d route=%d p=[%s]" % (qid, len(x["input_ids"]), x["route"], ",".join("%.3f" % v for v in x["probs"]))
            for qid, x in rec["questions"].items())), flush=True)
    json.dump(out, open(OUT, "w"))
    np.savez_compressed(NPZ, **arrays)
    print("wrote", OUT, NPZ)


if __name__ == "__main__":
    main()
