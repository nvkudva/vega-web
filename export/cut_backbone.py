"""Cut onnx-community's Qwen3.5-0.8B text graph down to what Vega reads.

Vega's Perception hooks decoder layers 13 and 19 and stops the forward there; the LM head
and layers 20-23 are never run. The onnx-community export fuses each residual add into the
next layer's SkipSimplifiedLayerNormalization, so a layer's output is not a named tensor.
Rebuild it as an explicit Add of the two summands (post-attention residual + MLP out),
make the two sums the only graph outputs, and drop every node they do not depend on.

    python cut_backbone.py model_quantized backbone_q8                        # text-only export
    python cut_backbone.py decoder_model_merged_quantized decoder_q8 onnxmm/onnx  # multimodal export
"""
import os, sys
import onnx
from onnx import helper

OUT_DIR = "out"
LAYERS = (13, 19)


def layer_out(i):
    return ("/model/layers.%d/post_attention_layernorm/output_3" % i, "/model/layers.%d/mlp/down_proj/MatMul/output_0" % i)


def main(src, dst, src_dir="onnxc/onnx"):
    m = onnx.load(os.path.join(src_dir, src + ".onnx"))
    g = m.graph
    produced = {o for n in g.node for o in n.output}
    for i in LAYERS:
        for t in layer_out(i):
            assert t in produced, "tensor %s not in graph: the export's naming changed" % t
        g.node.append(helper.make_node("Add", list(layer_out(i)), ["h%d" % i], name="/vega/h%d" % i))
    del g.output[:]
    g.output.extend([helper.make_tensor_value_info("h%d" % i, onnx.TensorProto.FLOAT, ["batch_size", "sequence_length", 1024])
                     for i in LAYERS])
    # keep only the ancestors of the outputs. A node's dependencies include the outer-scope
    # tensors its subgraphs (If/Loop bodies) read implicitly, not only its explicit inputs:
    # the multimodal export keeps parts of each linear-attention layer inside If branches.
    def reads(n):
        out = [x for x in n.input if x]
        for a in n.attribute:
            for sub in ([a.g] if a.type == onnx.AttributeProto.GRAPH else list(a.graphs)):
                inner = {o for sn in sub.node for o in sn.output} | {x.name for x in sub.initializer} | \
                        {x.name for x in sub.input}
                out += [x for sn in sub.node for x in reads(sn) if x not in inner]
        return out

    by_out = {o: n for n in g.node for o in n.output}
    need, stack, keep = set(), ["h%d" % i for i in LAYERS], set()
    while stack:
        t = stack.pop()
        if t in need:
            continue
        need.add(t)
        n = by_out.get(t)
        if n is not None and id(n) not in keep:
            keep.add(id(n))
            stack.extend(reads(n))
    nodes = [n for n in g.node if id(n) in keep]
    del g.node[:]
    g.node.extend(nodes)
    inits = [x for x in g.initializer if x.name in need]
    del g.initializer[:]
    g.initializer.extend(inits)
    ins = [x for x in g.input if x.name in need]
    del g.input[:]
    g.input.extend(ins)
    del g.value_info[:]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, dst + ".onnx")
    for f in (path, path + ".data"):
        if os.path.exists(f):
            os.remove(f)
    onnx.save(m, path, save_as_external_data=True, all_tensors_to_one_file=True, location=dst + ".onnx.data",
              size_threshold=1024)
    deepest = max(int(n.name.split("/")[2].split(".")[1]) for n in nodes if n.name.startswith("/model/layers."))
    print("%s: %d nodes, %d inputs %s, deepest layer %d, %.1f MB graph + %.1f MB data" % (
        path, len(nodes), len(ins), [x.name for x in ins][:3] + ["..."], deepest,
        os.path.getsize(path) / 1e6, os.path.getsize(path + ".data") / 1e6))


if __name__ == "__main__":
    main(*sys.argv[1:])
