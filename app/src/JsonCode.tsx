import { useMemo } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { json } from "@codemirror/lang-json";
import { EditorView } from "@codemirror/view";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags as t } from "@lezer/highlight";

/**
 * CodeMirror rather than the painted-textarea overlay it replaces: the caret is the
 * editor's own, so it cannot drift out of alignment with the highlighted text, and
 * folding, line numbers and wrapping come with it.
 *
 * Colours reference the same CSS custom properties as the rest of the page, so light
 * and dark follow the page theme with no second palette to keep in sync. Every token
 * colour clears 4.5:1 against its surface.
 */

const highlight = HighlightStyle.define([
  { tag: [t.propertyName, t.definition(t.propertyName)], color: "var(--t-key)", fontWeight: "500" },
  { tag: [t.string, t.special(t.string)], color: "var(--t-str)" },
  { tag: t.number, color: "var(--t-num)" },
  { tag: [t.bool, t.null, t.atom, t.keyword], color: "var(--t-lit)" },
  { tag: [t.punctuation, t.separator, t.bracket, t.squareBracket, t.brace], color: "var(--t-punct)" },
  { tag: t.invalid, color: "var(--warn)" },
]);

const theme = EditorView.theme({
  "&": { color: "var(--t-punct)", backgroundColor: "transparent", height: "100%", fontSize: "12.5px" },
  ".cm-scroller": {
    fontFamily: "var(--mono)", lineHeight: "1.6",
    // room for the caret on the last line, and for the fold gutter's hover target
    padding: "10px 0 14px",
  },
  ".cm-content": { padding: "0 10px 0 4px", caretColor: "var(--ink)" },
  ".cm-gutters": {
    backgroundColor: "transparent", border: "none", color: "var(--muted)",
    opacity: "0.65", paddingRight: "2px",
  },
  ".cm-lineNumbers .cm-gutterElement": { padding: "0 6px 0 10px", minWidth: "2.5em" },
  ".cm-activeLine": { backgroundColor: "color-mix(in srgb, var(--signal) 7%, transparent)" },
  ".cm-activeLineGutter": { backgroundColor: "transparent", color: "var(--ink)", opacity: "1" },
  "&.cm-focused": { outline: "none" },
  "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": {
    backgroundColor: "color-mix(in srgb, var(--signal) 26%, transparent)",
  },
  ".cm-cursor, .cm-dropCursor": { borderLeftColor: "var(--ink)", borderLeftWidth: "1.5px" },
  ".cm-foldPlaceholder": {
    backgroundColor: "transparent", border: "1px solid var(--rule)",
    color: "var(--muted)", borderRadius: "2px", padding: "0 5px", margin: "0 2px",
  },
  ".cm-foldGutter .cm-gutterElement": { cursor: "pointer", color: "var(--muted)" },
  ".cm-foldGutter .cm-gutterElement:hover": { color: "var(--ink)" },
  ".cm-matchingBracket, &.cm-focused .cm-matchingBracket": {
    backgroundColor: "color-mix(in srgb, var(--signal) 20%, transparent)", outline: "none",
  },
  ".cm-nonmatchingBracket": { backgroundColor: "transparent", color: "var(--warn)" },
});

// Long single-line strings (a pasted email body) must wrap rather than force a
// horizontal scrollbar across the whole panel.
const base = [json(), theme, syntaxHighlighting(highlight), EditorView.lineWrapping];

const SETUP = {
  lineNumbers: true,
  foldGutter: true,
  highlightActiveLine: true,
  highlightActiveLineGutter: true,
  bracketMatching: true,
  closeBrackets: true,
  autocompletion: false,
  highlightSelectionMatches: false,
  searchKeymap: false,
} as const;

export function JsonEditor({
  value, onChange, label,
}: { value: string; onChange: (v: string) => void; label: string }) {
  const ext = useMemo(() => base, []);
  return (
    <div className="cm-host" aria-label={label}>
      <CodeMirror value={value} onChange={onChange} extensions={ext} basicSetup={SETUP} theme="none" />
    </div>
  );
}

export function JsonView({ src }: { src: string }) {
  const ext = useMemo(() => [...base, EditorView.editable.of(false)], []);
  return (
    <div className="cm-host cm-host-read">
      <CodeMirror
        value={src}
        extensions={ext}
        editable={false}
        theme="none"
        basicSetup={{ ...SETUP, highlightActiveLine: false, highlightActiveLineGutter: false, closeBrackets: false }}
      />
    </div>
  );
}
