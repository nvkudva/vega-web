// Port of vega_common.thermal_features / thermal_tau / softmax_np / route_adapters and the
// answer assembly in vega_api.Vega.decide, for views=1.
import { QTYPES, type Answer, type Calibration, type InternalQ, type VegaConfig } from "./types";

export interface EngineOut {
  logits: number[]; // first k only
  kinetic: number;
  nearest: number;
  steps: number;
  gates: number[];
}

/** The adapter whose gate is most open if it reaches the threshold, else -1 (base engine). */
export function route(gates: number[], threshold: number): number {
  if (!gates.length) return -1;
  const best = gates.indexOf(Math.max(...gates));
  return gates[best] >= threshold ? best : -1;
}

/** Thermal calibration: log tau = b + w . [log1p(kinetic), log1p(nearest), steps/n_steps, log k]. */
export function calibrate(theta: number[][], qtype: number, e: EngineOut, nSteps: number): number[] {
  const k = e.logits.length;
  const phys = [Math.log1p(e.kinetic), Math.log1p(e.nearest), e.steps / Math.max(1, nSteps), Math.log(Math.max(k, 2))];
  const th = theta[qtype];
  const logTau = th[0] + phys.reduce((a, v, i) => a + th[i + 1] * v, 0);
  const tau = Math.exp(Math.min(Math.max(logTau, Math.log(0.2)), Math.log(20)));
  const z = e.logits.map((v) => v / tau);
  const m = Math.max(...z);
  const ex = z.map((v) => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / s);
}

export function decideOne(cfg: VegaConfig, q: InternalQ, e: EngineOut, alpha = "0.1", risk?: string): Answer {
  const threshold = cfg.adapters[0]?.gate_threshold ?? 0.5;
  const r = route(e.gates, threshold);
  const cal: Calibration = r < 0 ? cfg.calibration : cfg.adapters[r].calibration;
  const p = calibrate(cal.theta, QTYPES[q.t], e, cfg.n_steps);
  const k = p.length;
  const labels: (string | number | boolean)[] =
    q.t === "choice" ? Object.keys(q.crit as object) : q.t === "score" ? [...Array(k).keys()] : [false, true];
  const conf = Math.max(...p);
  const best = p.indexOf(conf);
  const th = cal.conformal[q.t]?.[alpha];
  // LAC set; the top option is always kept so the set is never empty
  const pset = th === undefined ? null : labels.filter((_, i) => p[i] >= Math.max(1 - th, 1e-12) || i === best);
  const rk = risk ?? cfg.calibration.default_risk ?? "0.1";
  const a: Answer = {
    // str() of Python's False / True: the reference response keys booleans "False" and "True"
    probs: Object.fromEntries(labels.map((l, i) => [typeof l === "boolean" ? (l ? "True" : "False") : String(l), p[i]])),
    confidence: conf,
    prediction_set: pset,
    abstain: conf < (cal.abstain_threshold[rk] ?? 0),
    unbound: e.nearest > (cal.unbound_threshold ?? Infinity),
    settle_steps: e.steps,
    energy: -Math.max(...e.logits),
  };
  if (cfg.adapters.length) {
    a.adapter = r >= 0 ? cfg.adapters[r].name : null;
    a.gate = e.gates.length ? Math.max(...e.gates) : 0;
  }
  if (q.t === "choice") a.choice = labels[best] as string;
  else if (q.t === "score") {
    a.score = best;
    a.expected_score = p.reduce((s, v, i) => s + i * v, 0);
  } else a.boolean = p[1];
  return a;
}
