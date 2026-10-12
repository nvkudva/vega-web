"""Parity for the multimodal split: embed_tokens + cut decoder (+ vision encoder for images) + engine.

Text: golden.json run through embed -> decoder with 1-d positions broadcast to M-RoPE's 3 rows.
Images: golden_vision pixel values -> vision encoder -> image features spliced over the image pad
tokens -> decoder with M-RoPE positions from rope_index() (checked against get_rope_index's dump).

    python verify_mm.py q8          # embed_tokens_quantized + decoder_q8 + vision_encoder_quantized
    python verify_mm.py fp32        # the unquantized graphs: proves the cut and the splice
"""
import json, sys
import numpy as np
import verify_onnx as vo

MM = "onnxmm/onnx"
GRAPHS = {"q8": ("embed_tokens_quantized", "decoder_q8", "vision_encoder_quantized"),
          "fp32": ("embed_tokens", "decoder_fp32", "vision_encoder")}


def rope_index(ids, pad_id, grid_thw, merge):
    """Qwen3.5 get_rope_index for one sequence with at most one image: text runs count up on all three
    rows; the image block is a (t, h, w) grid offset by the current position, which then advances by
    max(h, w) of the merged grid."""
    pos = np.zeros((3, len(ids)), np.int64)
    cur, i, n = 0, 0, len(ids)
    while i < n:
        if ids[i] == pad_id:
            t, h, w = grid_thw[0], grid_thw[1] // merge, grid_thw[2] // merge
            k = t * h * w
            tt, hh, ww = np.meshgrid(np.arange(t), np.arange(h), np.arange(w), indexing="ij")
            pos[:, i:i + k] = np.stack([tt.ravel(), hh.ravel(), ww.ravel()]) + cur
            cur += max(h, w)
            i += k
        else:
            pos[:, i] = cur
            cur += 1
            i += 1
    return pos


def decode(dec, past, emb, pos):
    L = emb.shape[1]
    h13, h19 = dec.run(["h13", "h19"], dict(past, inputs_embeds=emb, attention_mask=np.ones((1, L), np.int64),
                                            position_ids=pos[:, None, :]))
    return h13[0], h19[0]


def engine_probs(en, cfg, ads, feats, opts, chunks, cmask, g):
    k = g["k"]
    logits, kin, near, steps, gates = en.run(None, {
        "feats": feats[None], "opts": opts[None], "opt_mask": np.ones((1, k), bool),
        "qtype": np.array([g["qtype"]], np.int64), "chunks": chunks[None], "chunk_mask": cmask[None]})
    route = int(gates[0].argmax()) if gates[0].max() >= ads[0].get("gate_threshold", 0.5) else -1
    theta = (cfg["calibration"] if route < 0 else ads[route]["calibration"])["theta"]
    return vo.calibrate(theta, g["qtype"], logits[0], float(kin[0]), float(near[0]), int(steps[0]), k), route


def report(title, rows):
    print("\n== %s" % title)
    for r in rows:
        flag = "" if r["argmax_ok"] else "  ARGMAX"
        print("%-30s L=%5d d_feat %.2e  d_p %.4f  kl %.2e%s" % (r["case"], r["L"], r["d_feat"], r["d_p"], r["kl"], flag))
    print("argmax %d/%d  max|dp| %.4f  mean KL %.2e" % (sum(r["argmax_ok"] for r in rows), len(rows),
                                                      max(r["d_p"] for r in rows), np.mean([r["kl"] for r in rows])))


def row(case, L, feats, rf, p, pr):
    pr = np.asarray(pr)
    return {"case": case, "L": L, "d_feat": float(np.abs(feats - rf).max() / (np.abs(rf).max() + 1e-9)),
            "d_p": float(np.abs(p - pr).max()), "argmax_ok": int(p.argmax()) == int(pr.argmax()),
            "kl": float((pr * np.log(np.clip(pr, 1e-12, None) / np.clip(p, 1e-12, None))).sum())}


def main(kind="q8"):
    e_name, d_name, v_name = GRAPHS[kind]
    emb = vo.session("%s/%s.onnx" % (MM, e_name))
    dec = vo.session("out/%s.onnx" % d_name)
    vis = vo.session("%s/%s.onnx" % (MM, v_name))
    en = vo.session("out/engine.onnx")
    past = vo.empty_past(dec)
    cfg = json.load(open("vega/vega_config.json"))
    ads = [json.load(open("vega/adapters/%s.json" % a)) for a in ("public_intents", "typed_decisions")]

    gold, npz = json.load(open(vo.GOLD)), np.load(vo.NPZ)
    rows = []
    for case in gold["cases"]:
        for qid, g in case["questions"].items():
            ids = np.array([g["input_ids"]], np.int64)
            e = emb.run(None, {"input_ids": ids})[0]
            pos = np.tile(np.arange(ids.shape[1], dtype=np.int64), (3, 1))
            h13, h19 = decode(dec, past, e, pos)
            feats, opts, chunks, cmask = vo.pool(h13, h19, g["spans"], ids.shape[1])
            p, _ = engine_probs(en, cfg, ads, feats, opts, chunks, cmask, g)
            rows.append(row("%s/%s" % (case["id"], qid), ids.shape[1], feats, npz["%s/%s/feats" % (case["id"], qid)],
                            p, g["probs"]))
    report("text, %s" % kind, rows)

    gv, nv = json.load(open("../golden/golden_vision.json")), np.load("../golden/golden_vision.npz")
    rows, rope_ok = [], True
    for case in gv["cases"]:
        pv = nv[case["id"] + "/pixel_values"]
        img = vis.run(None, {"pixel_values": pv, "image_grid_thw": np.array([case["grid_thw"]], np.int64)})[0]
        assert img.shape[0] == case["n_img"], (img.shape, case["n_img"])
        for qid, g in case["questions"].items():
            key = "%s/%s" % (case["id"], qid)
            ids = np.array(g["input_ids"], np.int64)
            pos = rope_index(ids, gv["pad_id"], case["grid_thw"], gv["merge_size"])
            rope_ok &= bool((pos == nv[key + "/position_ids"]).all())
            e = emb.run(None, {"input_ids": ids[None]})[0]
            e[0, ids == gv["pad_id"]] = img
            h13, h19 = decode(dec, past, e, pos)
            feats, opts, _, _ = vo.pool(h13, h19, g["spans"], len(ids))
            C = cfg["memory_chunks"]
            p, _ = engine_probs(en, cfg, ads, feats, opts, np.zeros((C, 1024), np.float32), np.zeros(C, bool), g)
            rows.append(row(key, len(ids), feats, nv[key + "/feats"], p, g["probs"]))
    report("image, %s (rope_index == get_rope_index: %s)" % (kind, rope_ok), rows)


if __name__ == "__main__":
    main(*sys.argv[1:])
