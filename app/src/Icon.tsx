/** Generic 14px line glyphs rather than brand marks: one consistent set, and no
 *  redrawing of anyone's logo. */
const PATHS = {
  source: "M9.5 3.5 6 12.5M4.5 5.5 1.5 8l3 2.5M11.5 5.5 14.5 8l-3 2.5",
  model: "M8 1.8 14 5v6l-6 3.2L2 11V5zM2 5l6 3.2L14 5M8 8.2v6",
  person: "M8 8.5a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM2.5 14c.8-2.4 2.9-3.8 5.5-3.8s4.7 1.4 5.5 3.8",
  external: "M6.5 3H3.2A1.2 1.2 0 0 0 2 4.2v8.6A1.2 1.2 0 0 0 3.2 14h8.6a1.2 1.2 0 0 0 1.2-1.2V9.5M9.5 2.5H14V7M14 2.5 7.5 9",
  scale: "M2.5 13.5h11M8 13.5V4M4 6.5 8 4l4 2.5M2 10l2-3.5L6 10a2 2 0 0 1-4 0ZM10 10l2-3.5L14 10a2 2 0 0 1-4 0Z",
} as const;

export function Icon({ name }: { name: keyof typeof PATHS }) {
  return (
    <svg className="icon" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
      <path d={PATHS[name]} fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
