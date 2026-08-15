"""Scope a stylesheet under one id and harden it against a host theme's CSS."""
import re

HARDEN = ("line-height", "margin", "margin-top", "margin-bottom", "margin-left",
          "margin-right", "font-size", "font-family", "font-weight", "display",
          "max-width", "width", "padding", "color", "background", "text-align",
          "grid-template-columns", "grid-row", "aspect-ratio", "letter-spacing")

def _split_top(css):
    """Yield (kind, header, body) for each top-level construct."""
    out, i, n = [], 0, len(css)
    while i < n:
        brace = css.find("{", i)
        if brace == -1:
            break
        depth, j = 1, brace + 1
        while j < n and depth:
            if css[j] == "{": depth += 1
            elif css[j] == "}": depth -= 1
            j += 1
        header, body = css[i:brace].strip(), css[brace+1:j-1]
        kind = "at" if header.startswith("@") else "rule"
        out.append((kind, header, body))
        i = j
    return out

def _harden(body):
    lines = []
    for decl in body.split(";"):
        if ":" not in decl:
            lines.append(decl); continue
        prop = decl.split(":", 1)[0].strip()
        if prop in HARDEN and "!important" not in decl:
            decl = decl.rstrip() + " !important"
        lines.append(decl)
    return ";".join(lines)

def scope(css, sel="#exxa-post"):
    parts = []
    for kind, header, body in _split_top(css):
        if kind == "at":
            at = header.split()[0]
            if at in ("@font-face", "@keyframes", "@-webkit-keyframes"):
                parts.append(f"{header}{{{body}}}")          # never scope these
            elif at in ("@media", "@supports"):
                parts.append(f"{header}{{{scope(body, sel)}}}")
            else:
                parts.append(f"{header}{{{body}}}")
            continue
        sels = []
        for s in header.split(","):
            s = s.strip()
            if not s: continue
            if s in (":root", "html", "body", "*"):
                sels.append(sel if s != "*" else f"{sel} *")
            elif s.startswith(sel):
                sels.append(s)
            else:
                sels.append(f"{sel} {s}")
        parts.append(f"{','.join(sels)}{{{_harden(body)}}}")
    return "\n".join(parts)
