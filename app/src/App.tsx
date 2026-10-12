import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DEFAULT_REQUEST, PRESETS } from "./presets";
import { VegaSession, cachedWeightBytes, deleteWeightCache, type LoadProgress } from "./vega/session";
import { Distribution } from "./Distribution";
import { JsonEditor, JsonView } from "./JsonCode";
import { Icon } from "./Icon";
import type { Questions, State, VegaConfig, VegaResponse } from "./vega/types";

// embed_q8 + decoder_q8 (.onnx + .data) + engine.onnx; the vision encoder loads on first image
const TOTAL_BYTES = 840_384_247;
const WEIGHTS_LABEL = "8-bit backbone + fp32 engine, 840 MB (+123 MB for images)";
const mb = (n: number) => (n / 1e6).toFixed(1);

interface Loading { files: Record<string, LoadProgress>; done: boolean }
interface Picked { blob: Blob; url: string; name: string }

export default function App() {
  const vega = useRef<VegaSession | null>(null);
  const [load, setLoad] = useState<Loading>({ files: {}, done: false });
  const [cfg, setCfg] = useState<VegaConfig | null>(null);
  const [request, setRequest] = useState(DEFAULT_REQUEST);
  const [response, setResponse] = useState<{ res: VegaResponse; questions: Questions } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ms, setMs] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [preset, setPreset] = useState(PRESETS[0].id);
  const [image, setImage] = useState<Picked | null>(null);
  const [cachedBytes, setCachedBytes] = useState<number | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [theme, setTheme] = useState<"system" | "light" | "dark">(() => {
    try { return (localStorage.getItem("vega-theme") as "light" | "dark") ?? "system"; } catch { return "system"; }
  });

  // Following prefers-color-scheme alone leaves no way out of an OS setting the
  // reader does not want for this page.
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      if (theme === "system") localStorage.removeItem("vega-theme");
      else localStorage.setItem("vega-theme", theme);
    } catch { /* private mode */ }
  }, [theme]);

  useEffect(() => {
    let live = true;
    const threads = Number(new URLSearchParams(location.search).get("threads")) || undefined;
    VegaSession.load(import.meta.env.VITE_MODELS_BASE || "/models/v1", (p: LoadProgress) => {
      if (live) setLoad((s) => ({ ...s, files: { ...s.files, [p.file]: p } }));
    }, threads).then((s) => {
      if (!live) return;
      vega.current = s;
      setCfg(s.cfg);
      setLoad((st) => ({ ...st, done: true }));
      cachedWeightBytes().then((n) => live && setCachedBytes(n));
    }).catch((e) => live && setError(String(e?.message ?? e)));
    return () => { live = false; };
  }, []);

  const loaded = useMemo(
    () => Object.values(load.files).reduce((a, f) => a + f.loaded, 0),
    [load.files],
  );
  const fromCache = Object.values(load.files).some((f) => f.cached);

  const parsed = useMemo(() => {
    try {
      const v = JSON.parse(request);
      if (!v || typeof v !== "object" || !("questions" in v)) throw new Error("needs a questions object");
      return { ok: true as const, v };
    } catch (e) {
      return { ok: false as const, why: (e as Error).message };
    }
  }, [request]);

  const run = useCallback(async () => {
    if (!parsed.ok || !vega.current) return;
    setBusy(true); setError(null);
    const { state = "", questions } = parsed.v as { state?: State; questions: Questions };
    const t0 = performance.now();
    try {
      const res = await vega.current.decide(state, questions, image?.blob);
      setResponse({ res, questions }); setMs(performance.now() - t0);
    } catch (e) {
      setError(String((e as Error)?.message ?? e)); setResponse(null);
    } finally {
      setBusy(false);
    }
  }, [parsed, image]);

  // revoke each preview URL once it is replaced or the page goes away
  useEffect(() => () => { if (image) URL.revokeObjectURL(image.url); }, [image]);
  const pick = (blob: Blob | null, name = "") => setImage(blob ? { blob, url: URL.createObjectURL(blob), name } : null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); run(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [run]);

  const nQuestions = parsed.ok ? Object.keys((parsed.v as { questions: object }).questions ?? {}).length : 0;

  // Switching scenario invalidates whatever is on the right; leaving it there would
  // read as this scenario's answer.
  const choosePreset = (p: (typeof PRESETS)[number]) => {
    setPreset(p.id);
    setRequest(p.request);
    if (p.image) fetch(p.image).then((r) => r.blob()).then((b) => pick(b, p.image!.split("/").pop()));
    else pick(null);
    setResponse(null);
    setError(null);
    setMs(null);
  };
  const active = PRESETS.find((p) => p.id === preset);
  const nAnswers = response ? Object.keys(response.res.answers).length : 0;

  return (
    <div className="shell">
      <header className="bar">
        <div className="bar-inner">
          <div className="bar-head">
            <h1 className="wordmark">Vega 0.8B — typed decision model — running fully in browser</h1>
            <p className="tagline">
              It is not a chat model. It is a System One decision model like{" "}
              <a href="https://typesafe.ai" target="_blank" rel="noreferrer">Jev</a>. A frozen
              Qwen3.5-0.8B reads the state — any text or JSON — and each typed question once; a
              small physics engine on top settles a particle into one well per option, and the
              occupancy of those wells, thermally calibrated, is the answer. No API needed: it runs
              on onnxruntime-web over WebAssembly, so nothing leaves the browser. Question types:{" "}
              <b>boolean</b> answers yes or no, <b>choice</b> picks one named option, <b>score</b>{" "}
              returns the expectation over ordered levels. Every answer also carries a conformal
              prediction set and an abstain flag.{" "}
              <a href="https://huggingface.co/nvkudva/vega-onnx-q8" target="_blank" rel="noreferrer">
                More on Hugging Face<Icon name="external" />
              </a>
            </p>
          </div>
          <div className="bar-aside">
          <dl className="readout">
            <div><dt>Checkpoint</dt><dd>Qwen3.5-0.8B + engine</dd></div>
            <div><dt>Weights</dt><dd>{WEIGHTS_LABEL}</dd></div>
            <div><dt>Context</dt><dd>{cfg ? `${cfg.max_len} tok` : "—"}</dd></div>
            <div><dt>Last run</dt><dd>{ms === null ? "—" : `${Math.round(ms)} ms`}</dd></div>
            <button
              className="theme-toggle"
              onClick={() => setTheme((t) => (t === "dark" ? "light" : t === "light" ? "system" : "dark"))}
              title="Switch between dark, light and your system setting"
            >
              {theme}
            </button>
          </dl>
          <div className="weights" aria-live="polite">
            {!load.done ? (
              <>
                <div className="weights-line">
                  <span>{fromCache ? "Reading weights from cache" : "Downloading weights"}</span>
                  <b>{mb(loaded)} / {mb(TOTAL_BYTES)} MB</b>
                </div>
                <div className="rule"><span style={{ width: `${Math.min(100, (loaded / TOTAL_BYTES) * 100)}%` }} /></div>
              </>
            ) : (
              <div className="weights-line">
                <span>{cachedBytes ? "Weights cached in this browser" : "Weights loaded, not cached"}</span>
                <b>{mb(cachedBytes || loaded)} MB</b>
                {cachedBytes ? (
                  <button
                    className={confirmDelete ? "weights-delete armed" : "weights-delete"}
                    onClick={async () => {
                      if (!confirmDelete) { setConfirmDelete(true); setTimeout(() => setConfirmDelete(false), 4000); return; }
                      await deleteWeightCache();
                      setCachedBytes(0);
                      setConfirmDelete(false);
                    }}
                    title="Free the cached weights. The model stays loaded for this session; the next visit downloads again."
                  >
                    {confirmDelete ? "Confirm delete" : "Delete"}
                  </button>
                ) : null}
              </div>
            )}
          </div>
          </div>
        </div>
      </header>

      <main>
        <section className="panel">
          <div className="panel-head">
            <h2>Request <span className="panel-qualifier">(editable)</span></h2>
            <span className="panel-note">{nQuestions} question{nQuestions === 1 ? "" : "s"}</span>
          </div>
          <section className="presets-block" aria-labelledby="presets-heading">
          <h3 className="presets-heading" id="presets-heading">Presets</h3>
          <nav className="presets" aria-label="Example scenarios">
            {PRESETS.map((p) => (
              <button
                key={p.id}
                className="preset"
                aria-pressed={p.id === preset}
                onClick={() => choosePreset(p)}
              >
                {p.label}
              </button>
            ))}
          </nav>
          {active && <p className="preset-note">{active.note}</p>}
          </section>
          <JsonEditor
            value={request}
            onChange={(v: string) => { setRequest(v); setPreset(""); }}
            label="Request JSON"
          />
          <div className="image-row">
            {image ? (
              <>
                <img className="image-thumb" src={image.url} alt="" />
                <span className="image-name">{image.name || "image"}</span>
                <span className="image-note">is the state; the JSON state is ignored</span>
                <button className="image-clear" onClick={() => pick(null)}>Remove</button>
              </>
            ) : (
              <label className="image-pick">
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) pick(f, f.name); e.target.value = ""; }}
                />
                Add an image as the state
                <span className="image-note">zero-shot · first use downloads 123 MB more</span>
              </label>
            )}
          </div>
          <div className="actions">
            <button onClick={run} disabled={!load.done || busy || !parsed.ok}>
              {busy ? "Running" : "Run request"}
            </button>
            <span className={`status${parsed.ok ? "" : " bad"}`}>
              {!load.done ? "Loading model" : !parsed.ok ? parsed.why : busy ? "One forward pass per question" : "Cmd+Enter to run"}
            </span>
          </div>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Response — Distribution</h2>
            <span className="panel-note">{response ? `${nAnswers} answered` : ""}</span>
          </div>
          {error ? (
            <pre className="code json-view" style={{ color: "var(--warn)" }}>{error}</pre>
          ) : response ? (
            <div className="dist-scroll">
              {Object.entries(response.res.answers).map(([id, ans]) => (
                <Distribution key={id} id={id} type={response.questions[id].type} answer={ans} />
              ))}
            </div>
          ) : (
            <pre className="code json-view empty">Each answer is one probability distribution over the options you listed.</pre>
          )}
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Response — JSON</h2>
            <span className="panel-note">
              {response ? `${response.res.n_tokens} input tokens` : ""}
            </span>
          </div>
          {response && !error ? (
            <JsonView src={JSON.stringify(response.res, null, 2)} />
          ) : (
            <pre className="code json-view empty">The raw decide() response, ready to paste into your own integration.</pre>
          )}
        </section>
      </main>

      <footer>
        <div className="footer-inner">
          <section className="credit-group">
            <h2>Credits</h2>
            <ul className="credits">
              <li>
                <Icon name="person" />
                <a href="https://github.com/NandhaKishorM" target="_blank" rel="noreferrer">Nandakishor M</a>
                <span className="credit-note">Convai Innovations</span>
              </li>
              <li>
                <Icon name="model" />
                <a href="https://huggingface.co/nandakishorm/vega-08b-public-intents" target="_blank" rel="noreferrer">nandakishorm/vega-08b-public-intents</a>
              </li>
              <li>
                <Icon name="model" />
                <a href="https://huggingface.co/Qwen/Qwen3.5-0.8B" target="_blank" rel="noreferrer">Qwen/Qwen3.5-0.8B</a>
                <span className="credit-note">frozen backbone</span>
              </li>
              <li>
                <Icon name="scale" />
                <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank" rel="noreferrer">Apache 2.0</a>
              </li>
            </ul>
          </section>
          <section className="credit-group">
            <h2>Source</h2>
            <ul className="credits">
              <li>
                <Icon name="person" />
                <a href="https://github.com/nvkudva" target="_blank" rel="noreferrer">nvkudva</a>
                <span className="credit-note">quantization and this page</span>
              </li>
              <li>
                <Icon name="model" />
                <a href="https://huggingface.co/nvkudva/vega-onnx-q8" target="_blank" rel="noreferrer">nvkudva/vega-onnx-q8</a>
                <span className="credit-note">8-bit</span>
              </li>
              <li>
                <Icon name="source" />
                <a href="https://github.com/nvkudva/vega-web" target="_blank" rel="noreferrer">nvkudva/vega-web</a>
              </li>
            </ul>
          </section>
        </div>
      </footer>
    </div>
  );
}
