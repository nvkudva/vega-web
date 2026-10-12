"""Merge fixtures.json + golden.json into app/public/parity.json for the browser parity page."""
import json

fx = {c["id"]: c for c in json.load(open("../golden/fixtures.json"))}
gold = json.load(open("../golden/golden.json"))
cases = [{"id": c["id"], "state": fx[c["id"]]["state"], "questions": fx[c["id"]]["questions"],
          "expect": {q: {k: x[k] for k in ("input_ids", "spans", "rendered_options", "probs", "k")}
                     for q, x in c["questions"].items()},
          "response": c["response"]} for c in gold["cases"]]
json.dump({"cases": cases}, open("../app/public/parity.json", "w"), ensure_ascii=False)
print("%d cases" % len(cases))

# image path: the browser preprocesses the same files, so grid and expanded ids must match exactly
import shutil, os
gv = json.load(open("../golden/golden_vision.json"))
os.makedirs("../app/public/parity", exist_ok=True)
vcases = []
for c in gv["cases"]:
    shutil.copy("../golden/images/" + c["image"], "../app/public/parity/" + c["image"])
    vcases.append({"id": c["id"], "image": "/parity/" + c["image"], "grid_thw": c["grid_thw"],
                   "questions": next(x for x in __import__("dump_golden_vision").CASES if x["id"] == c["id"])["questions"],
                   "expect": {q: {"input_ids": x["input_ids"], "probs": x["probs"]} for q, x in c["questions"].items()}})
json.dump({"cases": vcases}, open("../app/public/parity_vision.json", "w"))
print("%d image cases" % len(vcases))
