import type { Answer } from "./vega/types";

/**
 * One question's answer is one distribution over its options, so each block is a
 * single series: one validated fill on a neutral track, no categorical palette and
 * therefore no legend. Rank is carried by ink weight on the label, never by hue.
 */

interface Row { label: string; p: number; index: number }

function rows(a: Answer): Row[] {
  return Object.entries(a.probs).map(([label, p], index) => ({ label, p, index }));
}

const pct = (p: number) => (p * 100 >= 9.95 ? (p * 100).toFixed(0) : (p * 100).toFixed(1));

function Bars({ data, winner }: { data: Row[]; winner: number }) {
  return (
    <ol className="bars">
      {data.map((r) => (
        <li
          key={r.label}
          className={r.index === winner ? "bar-row is-top" : "bar-row"}
          title={`${r.label} — ${r.p.toFixed(4)}`}
        >
          <span className="bar-label">{r.label}</span>
          <span className="bar-track">
            {/* min-width keeps a near-zero probability visible as a sliver rather than nothing */}
            <span className="bar-fill" style={{ width: `max(2px, ${r.p * 100}%)` }} />
          </span>
          <span className="bar-value">{pct(r.p)}<span className="pct">%</span></span>
        </li>
      ))}
    </ol>
  );
}

export function Distribution({ id, type, answer }: { id: string; type: string; answer: Answer }) {
  const data = rows(answer);
  const winner = data.reduce((best, r) => (r.p > data[best].p ? r.index : best), 0);
  const summary =
    answer.choice !== undefined ? answer.choice
    : answer.expected_score !== undefined ? answer.expected_score.toFixed(2)
    : answer.boolean!.toFixed(4);

  return (
    <section
      className="dist"
      role="img"
      aria-label={`${id}: ${data.map((r) => `${r.label} ${pct(r.p)} percent`).join(", ")}`}
    >
      <header className="dist-head">
        <h3>{id}</h3>
        <span className="dist-type">{type}</span>
        <span className="dist-answer">{summary}</span>
      </header>

      <Bars data={data} winner={winner} />

      <footer className="dist-foot">
        <span>
          {answer.expected_score !== undefined
            ? `expectation, ${data.length} levels`
            : answer.boolean !== undefined
              ? "p(yes)"
              : `${data.length} options`}
          {answer.abstain && " · abstain"}
          {answer.unbound && " · unbound"}
        </span>
        <span className="dist-meta">
          <span>confidence {answer.confidence.toFixed(3)}</span>
          {answer.prediction_set && <span>set {"{"}{answer.prediction_set.map(String).join(", ")}{"}"}</span>}
          {answer.adapter && <span>{answer.adapter}</span>}
          <span>{answer.settle_steps} steps</span>
        </span>
      </footer>
    </section>
  );
}
