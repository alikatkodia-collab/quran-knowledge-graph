/*
 * quran_linker.js — Sefaria-inspired Quranic citation linker.
 *
 * Drop-in client for any webpage. Scans visible text for Quranic citations
 * (English or Arabic) and converts them to interactive links that show
 * the Khalifa English text + Hafs Arabic on hover, with a deep-link to the
 * full QKG explorer.
 *
 * Usage (script tag):
 *   <script src="https://your-host/quran_linker.js" defer></script>
 *
 * Or as a module:
 *   <script type="module">
 *     import { linkPage } from "https://your-host/quran_linker.js";
 *     linkPage({ apiBase: "https://your-host", target: "main" });
 *   </script>
 *
 * Configuration (set window.QuranLinkerConfig before script load):
 *   {
 *     apiBase: "http://localhost:8085",
 *     target:  "body",          // CSS selector for the root to scan
 *     skipSelectors: ["pre", "code", "a", "[contenteditable]"],
 *     openInNewTab: true,
 *     style: "default"          // or "minimal" or false
 *   }
 */
(function () {
  "use strict";

  const DEFAULTS = {
    apiBase: typeof window !== "undefined" && window.location ?
              window.location.origin : "http://localhost:8085",
    target: "body",
    skipSelectors: ["pre", "code", "a", "script", "style", "textarea",
                    "[contenteditable]", ".quran-ref"],
    openInNewTab: true,
    style: "default",
    debounceMs: 250,
  };

  function cfg(key) {
    const ext = (typeof window !== "undefined" && window.QuranLinkerConfig) || {};
    return ext[key] !== undefined ? ext[key] : DEFAULTS[key];
  }

  // ── styling ──────────────────────────────────────────────────────────────
  const STYLE = `
.quran-ref {
  display: inline;
  border-bottom: 1px dotted currentColor;
  cursor: pointer;
  text-decoration: none;
  color: inherit;
}
.quran-ref:hover { color: #10b981; border-bottom-color: #10b981; }
.quran-tooltip {
  position: absolute;
  z-index: 100000;
  max-width: 380px;
  padding: 12px 14px;
  background: #060a14;
  color: #e5e7eb;
  border: 1px solid #10b981;
  border-radius: 6px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
  line-height: 1.5;
  box-shadow: 0 8px 24px rgba(0,0,0,.4);
  pointer-events: auto;
}
.quran-tooltip .quran-tooltip-ref {
  font-weight: 600;
  color: #10b981;
  margin-bottom: 4px;
}
.quran-tooltip .quran-tooltip-arabic {
  font-family: "Amiri", "Noto Naskh Arabic", "Scheherazade", serif;
  font-size: 18px;
  direction: rtl;
  text-align: right;
  margin: 6px 0;
  color: #f3f4f6;
}
.quran-tooltip .quran-tooltip-en { font-size: 13px; }
.quran-tooltip .quran-tooltip-link {
  display: inline-block;
  margin-top: 6px;
  font-size: 11px;
  color: #10b981;
  text-decoration: underline;
}
`;

  function injectStyle() {
    if (cfg("style") === false) return;
    if (document.getElementById("quran-linker-style")) return;
    const el = document.createElement("style");
    el.id = "quran-linker-style";
    el.textContent = STYLE;
    document.head.appendChild(el);
  }

  // ── tooltip ──────────────────────────────────────────────────────────────
  let activeTooltip = null;
  function clearTooltip() {
    if (activeTooltip) {
      activeTooltip.remove();
      activeTooltip = null;
    }
  }

  async function showTooltip(anchor, verseId) {
    clearTooltip();
    const tip = document.createElement("div");
    tip.className = "quran-tooltip";
    tip.innerHTML = `<div class="quran-tooltip-ref">${verseId}</div><div>Loading…</div>`;
    document.body.appendChild(tip);
    activeTooltip = tip;

    // Position
    const rect = anchor.getBoundingClientRect();
    tip.style.top  = (window.scrollY + rect.bottom + 4) + "px";
    tip.style.left = Math.min(window.scrollX + rect.left,
                              window.scrollX + window.innerWidth - 400) + "px";

    try {
      const r = await fetch(`${cfg("apiBase")}/api/verse/${encodeURIComponent(verseId)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const v = await r.json();
      if (v.error) {
        tip.innerHTML = `<div class="quran-tooltip-ref">${verseId}</div>
                        <div style="color:#ef4444">${v.error}</div>`;
        return;
      }
      tip.innerHTML = `
        <div class="quran-tooltip-ref">[${v.verse_id}] ${v.surah_name || ""}</div>
        ${v.arabic ? `<div class="quran-tooltip-arabic">${v.arabic}</div>` : ""}
        <div class="quran-tooltip-en">${v.text || ""}</div>
        <a class="quran-tooltip-link"
           href="${cfg("apiBase")}/?verse=${encodeURIComponent(v.verse_id)}"
           target="${cfg("openInNewTab") ? "_blank" : "_self"}"
           rel="noopener">explore in graph →</a>
      `;
    } catch (e) {
      tip.innerHTML = `<div class="quran-tooltip-ref">${verseId}</div>
                      <div style="color:#ef4444">Error loading: ${e.message}</div>`;
    }
  }

  // ── linkify ──────────────────────────────────────────────────────────────
  async function resolveText(text) {
    try {
      const r = await fetch(`${cfg("apiBase")}/api/resolve_refs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!r.ok) return [];
      const data = await r.json();
      return data.matches || [];
    } catch (e) {
      console.warn("[quran-linker] resolve_refs failed:", e);
      return [];
    }
  }

  function shouldSkip(el) {
    return cfg("skipSelectors").some(sel => {
      try { return el.closest(sel) != null; } catch { return false; }
    });
  }

  async function linkifyTextNode(node) {
    const text = node.nodeValue;
    if (!text || text.length < 4) return;
    const matches = await resolveText(text);
    if (matches.length === 0) return;

    // Group by span; expanded ranges share a span.
    const bySpan = new Map();
    for (const m of matches) {
      const key = `${m.start}:${m.end}`;
      if (!bySpan.has(key)) bySpan.set(key, []);
      bySpan.get(key).push(m);
    }
    const spans = Array.from(bySpan.entries())
      .map(([k, ms]) => {
        const [s, e] = k.split(":").map(Number);
        return { start: s, end: e, refs: ms };
      })
      .sort((a, b) => a.start - b.start);

    // Build replacement
    const frag = document.createDocumentFragment();
    let cursor = 0;
    for (const span of spans) {
      if (span.start < cursor) continue;
      if (span.start > cursor) {
        frag.appendChild(document.createTextNode(text.slice(cursor, span.start)));
      }
      // Use first ref as primary
      const primary = span.refs[0].verse_id;
      const a = document.createElement("a");
      a.className = "quran-ref";
      a.dataset.verse = primary;
      a.dataset.allVerses = span.refs.map(r => r.verse_id).join(",");
      a.href = `${cfg("apiBase")}/?verse=${encodeURIComponent(primary)}`;
      if (cfg("openInNewTab")) {
        a.target = "_blank";
        a.rel = "noopener";
      }
      a.textContent = text.slice(span.start, span.end);
      a.addEventListener("mouseenter", () => showTooltip(a, primary));
      a.addEventListener("mouseleave", clearTooltip);
      frag.appendChild(a);
      cursor = span.end;
    }
    if (cursor < text.length) {
      frag.appendChild(document.createTextNode(text.slice(cursor)));
    }
    node.parentNode.replaceChild(frag, node);
  }

  function collectTextNodes(root) {
    const out = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        if (!n.nodeValue || !n.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        if (n.parentElement && shouldSkip(n.parentElement)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    let n; while ((n = walker.nextNode())) out.push(n);
    return out;
  }

  async function linkPage(opts) {
    if (opts) Object.assign(DEFAULTS, opts);
    injectStyle();
    const root = document.querySelector(cfg("target"));
    if (!root) return console.warn("[quran-linker] target not found:", cfg("target"));
    const nodes = collectTextNodes(root);
    // Process sequentially to avoid hammering the API
    for (const n of nodes) {
      try { await linkifyTextNode(n); } catch (e) { /* swallow */ }
    }
    console.log(`[quran-linker] processed ${nodes.length} text nodes`);
  }

  // Auto-init on DOM ready unless caller suppresses it
  if (typeof window !== "undefined" &&
      !(window.QuranLinkerConfig && window.QuranLinkerConfig.manual)) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => linkPage());
    } else {
      linkPage();
    }
  }

  // Export
  if (typeof window !== "undefined") {
    window.QuranLinker = { linkPage, resolveText };
  }
})();
