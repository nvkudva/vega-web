"""Re-save onnx-community's embed_tokens and vision_encoder graphs under the names the browser loads.

ORT-web resolves external data by the file name recorded inside the graph, so a renamed copy must be
re-saved, not just renamed on disk. Also copies the image preprocessor config and adds the image-path
settings to vega_web.json.
"""
import json, os, shutil
import onnx

SRC = "onnxmm/onnx"
MAP = {"embed_tokens_quantized": "embed_q8", "vision_encoder_quantized": "vision_q8"}

for src, dst in MAP.items():
    m = onnx.load(os.path.join(SRC, src + ".onnx"))
    path = os.path.join("out", dst + ".onnx")
    for f in (path, path + ".data"):
        if os.path.exists(f):
            os.remove(f)
    onnx.save(m, path, save_as_external_data=True, all_tensors_to_one_file=True, location=dst + ".onnx.data",
              size_threshold=1024)
    print("%s: %.1f MB data" % (path, os.path.getsize(path + ".data") / 1e6))

shutil.copy("qwen/preprocessor_config.json", "out/preprocessor_config.json")
web = json.load(open("out/vega_web.json"))
web.update({"vision_max_pixels": 1024 * 32 * 32, "vision_merge_size": 2})   # VegaVision's defaults
json.dump(web, open("out/vega_web.json", "w"), indent=1)
print("updated out/vega_web.json")
