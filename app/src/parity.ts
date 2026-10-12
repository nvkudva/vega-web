// Two gates in one page:
//   1. tokenizer + buildPrompt produce identical token ids and spans to the PyTorch
//      reference. Anything less and every number after it is meaningless.
//   2. end-to-end decide() output matches the reference response.
//   3. image as state: grid and expanded ids identical, probabilities vs vega_vision.
// ?vision skips gate 2 (its 6k-token case takes minutes).
import { buildPrompt, optionTexts, serializeState, toInternal, type Span } from "./vega/prompt";
import { loadTokenizer } from "./vega/tokenizer";
import { VegaSession } from "./vega/session";
import { VISION_STATE, expandImage, preprocess } from "./vega/vision";
import type { Questions, State, VegaConfig, VegaResponse } from "./vega/types";

interface Expect {
  input_ids: number[];
  spans: { state: Span; question: Span; options: Span[] };
  rendered_options: string[];
  probs: number[];
  k: number;
}
interface Case { id: string; state: State; questions: Questions; expect: Record<string, Expect>; response: VegaResponse }

const el = document.getElementById("out")!;
const log = (s = "") => { el.textContent += s + "\n"; };
const eq = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);
const base = new URLSearchParams(location.search).get("base") || import.meta.env.VITE_MODELS_BASE || "/models/v1";

function firstDiff(a: number[], b: number[]) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return n;
}

(async () => {
  el.textContent = "";
  const data: { cases: Case[] } = await (await fetch("/parity.json")).json();
  const cfg: VegaConfig = await (await fetch(`${base}/vega_web.json`)).json();
  const tok = await loadTokenizer(base);

  log("=== gate 1: token ids and spans ===");
  let nq = 0, idsOk = 0, spOk = 0, optOk = 0;
  const failures: string[] = [];
  for (const c of data.cases) {
    for (const [qid, qdef] of Object.entries(c.questions)) {
      nq++;
      const q = toInternal(qdef);
      const opts = optionTexts(q);
      const e = c.expect[qid];
      const { ids, spans } = buildPrompt(tok, serializeState(c.state), q, cfg);
      const o = JSON.stringify(opts) === JSON.stringify(e.rendered_options);
      const i = eq(ids, e.input_ids);
      const s = JSON.stringify(spans) === JSON.stringify(e.spans);
      if (o) optOk++; if (i) idsOk++; if (s) spOk++;
      if (!(o && i && s)) {
        const d = firstDiff(ids, e.input_ids);
        failures.push(`  ${c.id}/${qid}  opts:${o ? "ok" : "DIFF"} ids:${i ? "ok" : `DIFF@${d} len ${ids.length} vs ${e.input_ids.length}`} spans:${s ? "ok" : "DIFF"}`);
        if (!o) failures.push(`    got  ${JSON.stringify(opts)}\n    want ${JSON.stringify(e.rendered_options)}`);
        if (!i) failures.push(`    got  [...${ids.slice(Math.max(0, d - 2), d + 4)}...]\n    want [...${e.input_ids.slice(Math.max(0, d - 2), d + 4)}...]`);
        if (!s) failures.push(`    got  ${JSON.stringify(spans)}\n    want ${JSON.stringify(e.spans)}`);
      }
    }
  }
  log(`options ${optOk}/${nq}   ids ${idsOk}/${nq}   spans ${spOk}/${nq}`);
  failures.forEach((f) => log(f));
  if (idsOk !== nq || spOk !== nq || optOk !== nq) { log("\nGATE 1 FAILED - stopping"); return; }
  log("GATE 1 PASS\n");
  if (new URLSearchParams(location.search).has("tokens")) return;

  const t0 = performance.now();
  const vega = await VegaSession.load(base, (p) => {
    if (p.loaded === p.total) log(`loaded ${p.file}${p.cached ? " (cached)" : ""}`);
  });
  log(`session ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  if (!new URLSearchParams(location.search).has("vision")) {
    log("=== gate 2: end-to-end vs reference response ===");
    let maxDp = 0, argmaxOk = 0, total = 0, setOk = 0, abstainOk = 0, ms = 0, klSum = 0;
    for (const c of data.cases) {
      const t = performance.now();
      const res = await vega.decide(c.state, c.questions);
      const dt = performance.now() - t;
      ms += dt;
      log(`  ${c.id}: ${res.n_tokens} tokens, ${dt.toFixed(0)}ms`);
      for (const [qid, want] of Object.entries(c.response.answers)) {
        total++;
        const a = res.answers[qid];
        const got = Object.values(a.probs), ref = Object.values(want.probs);
        const dp = Math.max(...got.map((v, i) => Math.abs(v - ref[i])));
        maxDp = Math.max(maxDp, dp);
        klSum += ref.reduce((a, r, i) => a + r * Math.log(Math.max(r, 1e-12) / Math.max(got[i], 1e-12)), 0);
        if (got.indexOf(Math.max(...got)) === ref.indexOf(Math.max(...ref))) argmaxOk++;
        if (JSON.stringify(a.prediction_set) === JSON.stringify(want.prediction_set)) setOk++;
        if (a.abstain === want.abstain) abstainOk++;
        else log(`  ${c.id}/${qid}  abstain ${a.abstain} vs ${want.abstain} (confidence ${a.confidence.toFixed(4)} vs ${want.confidence.toFixed(4)})`);
        if (dp > 0.02) log(`  ${c.id}/${qid}  d_p ${dp.toFixed(4)}`);
      }
    }
    const kl = klSum / total;
    log(`\n${total} questions | argmax ${argmaxOk}/${total} | mean KL ${kl.toExponential(2)} | prediction set ${setOk}/${total} | abstain ${abstainOk}/${total} | max d_p ${maxDp.toExponential(2)} | ${(ms / data.cases.length).toFixed(0)}ms per request`);
    // the gate is laya-web's: every argmax kept and mean KL <= 1e-3; max d_p is reported, not gated,
    // since it is set by degenerate inputs (an empty state) where 8-bit drift is largest
    log(argmaxOk === total && kl <= 1e-3 ? "GATE 2 PASS" : "GATE 2 FAILED");
  }

  log("\n=== gate 3: image as state vs vega_vision ===");
  interface VCase { id: string; image: string; grid_thw: number[]; questions: Questions; expect: Record<string, { input_ids: number[]; probs: number[] }> }
  const vdata: { cases: VCase[] } = await (await fetch("/parity_vision.json")).json();
  const padId = tok.encode("<|image_pad|>")[0];
  let vq = 0, vIds = 0, vArg = 0, vMax = 0, vKl = 0, vGrid = 0;
  for (const c of vdata.cases) {
    const blob = await (await fetch(c.image)).blob();
    const pre = await preprocess(base, cfg, blob);
    const gridOk = JSON.stringify(pre.grid) === JSON.stringify(c.grid_thw);
    if (gridOk) vGrid++;
    const t = performance.now();
    const res = await vega.decide("", c.questions, blob);
    log(`  ${c.id}: grid ${pre.grid}${gridOk ? "" : ` vs ${c.grid_thw} DIFF`}, ${res.n_tokens} tokens, ${(performance.now() - t).toFixed(0)}ms`);
    for (const [qid, e] of Object.entries(c.expect)) {
      vq++;
      const p0 = buildPrompt(tok, VISION_STATE, toInternal(c.questions[qid]), cfg);
      const { ids } = expandImage(p0.ids, p0.spans, padId, pre.nTokens);
      if (eq(ids, e.input_ids)) vIds++;
      else log(`    ${qid} ids DIFF@${firstDiff(ids, e.input_ids)} len ${ids.length} vs ${e.input_ids.length}`);
      const got = Object.values(res.answers[qid].probs);
      const dp = Math.max(...got.map((v, i) => Math.abs(v - e.probs[i])));
      vMax = Math.max(vMax, dp);
      vKl += e.probs.reduce((a, r, i) => a + r * Math.log(Math.max(r, 1e-12) / Math.max(got[i], 1e-12)), 0);
      if (got.indexOf(Math.max(...got)) === e.probs.indexOf(Math.max(...e.probs))) vArg++;
      log(`    ${qid}: [${got.map((v) => v.toFixed(3))}] vs [${e.probs.map((v) => v.toFixed(3))}]`);
    }
  }
  log(`\n${vq} questions | grid ${vGrid}/${vdata.cases.length} | ids ${vIds}/${vq} | argmax ${vArg}/${vq} | mean KL ${(vKl / vq).toExponential(2)} | max d_p ${vMax.toExponential(2)}`);
  log(vGrid === vdata.cases.length && vIds === vq && vArg === vq && vKl / vq <= 1e-3 ? "GATE 3 PASS" : "GATE 3 FAILED");
})().catch((e) => log("ERROR: " + (e?.stack ?? e)));
