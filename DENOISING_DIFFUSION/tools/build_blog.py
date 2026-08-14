#!/usr/bin/env python3
"""Build the proofing artifact for BLOG_MIDTERM.md."""
import re, base64, html, os, json

REPO = "/Users/Krishan/Projects/EXXA/DENOISING_DIFFUSION"
MD = os.path.join(REPO, "BLOG_MIDTERM.md")
FONT_DIR = "/tmp/fonts2"
OUT = "/private/tmp/claude-501/-Users-Krishan-Projects-EXXA/6a3fd662-00b4-4751-b4ef-432b3cc0d32b/scratchpad/blog.html"

# ---------------------------------------------------------------- assets
def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()

FONTS = {
    "sans400":     ("Inter_400.woff2", 400, "normal"),
    "sans600":     ("Inter_600.woff2", 600, "normal"),
    "sans700":     ("Inter_700.woff2", 700, "normal"),
    "serif400":    ("SourceSerif4_400.woff2", 400, "normal"),
    "serif400i":   ("SourceSerif4_400_italic.woff2", 400, "italic"),
    "serif600":    ("SourceSerif4_600.woff2", 600, "normal"),
    "mono400":     ("IBMPlexMono_400.woff2", 400, "normal"),
    "mono600":     ("IBMPlexMono_600.woff2", 600, "normal"),
}
font_data = {k: b64(os.path.join(FONT_DIR, fn)) for k, (fn, w, s) in FONTS.items()}

# figures are discovered from the markdown, downscaled to webp in /tmp/figs
_md_probe = open(MD, encoding="utf-8").read()
_fig_paths = re.findall(r'\*\*\[FIGURE\]\*\* `([^`]+)`', _md_probe)
img_data, FIG_DIMS = {}, {}
from PIL import Image as _PILImage

# Cache lives beside this script, not in /tmp: a /tmp clean used to break the build with a
# bare FileNotFoundError on a .webp nothing in this file ever wrote. Regenerate on miss.
_FIGCACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figcache")
os.makedirs(_FIGCACHE, exist_ok=True)
_MAXW = 1500
for _fp in _fig_paths:
    _key = os.path.basename(_fp)
    _webp = os.path.join(_FIGCACHE, _key.rsplit(".", 1)[0] + ".webp")
    _src = os.path.join(REPO, _fp)
    _im = _PILImage.open(_src)
    FIG_DIMS[_key] = _im.size
    if not os.path.exists(_webp) or os.path.getmtime(_webp) < os.path.getmtime(_src):
        _o = _im.convert("RGB")
        if _o.width > _MAXW:
            _o = _o.resize((_MAXW, round(_o.height * _MAXW / _o.width)), _PILImage.LANCZOS)
        _o.save(_webp, "WEBP", quality=82, method=6)
        print(f"  encoded {_key} -> {os.path.getsize(_webp)/1024:.0f} KB")
    img_data[_key] = b64(_webp)
print("assets encoded:", {k: len(v) for k, v in img_data.items()})

# ---------------------------------------------------------------- inline markdown -> html
def inline(text):
    text = html.escape(text, quote=False)
    parts = re.split(r'(`[^`]+`)', text)
    for i, p in enumerate(parts):
        if p.startswith('`') and p.endswith('`') and len(p) > 1:
            parts[i] = f'<code>{p[1:-1]}</code>'
    text = ''.join(parts)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2">\1</a>', text)
    # bare URLs not already inside an href
    text = re.sub(r'(?<!["\'>])\b(https?://[^\s<)]+)', r'<a href="\1">\1</a>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    return text

NUM_RE = re.compile(r'(?:^|\s)[-−+]?\d[\d.,]*%?')

def classify_code(body):
    lines = [l for l in body.splitlines() if l.strip()]
    if re.search(r'^(git |cd |pip |python )', body.strip()):
        return "shell"
    if len(lines) == 1:
        return "terminal" if ("=" not in lines[0] and "->" in lines[0]) else "formula"
    # data: at least two lines each carrying 2+ standalone numeric tokens (real columns)
    data_lines = sum(1 for l in lines if len(NUM_RE.findall(l)) >= 2)
    if data_lines >= 2:
        return "data"
    return "equation"

def colorize_data_numbers(text):
    """Wrap +N% / -N% tokens in semantic spans, inside data panels only."""
    def repl(m):
        sign, num = m.group(1), m.group(0)
        cls = "pos" if sign == "+" else "neg"
        return f'<span class="{cls}">{num}</span>'
    return re.sub(r'([+−–-])\d[\d.,]*\s?%', repl, text)

# ---------------------------------------------------------------- markdown parser
raw = open(MD, encoding="utf-8").read()
lines = raw.split("\n")

blocks = []  # list of dict(type=..., ...)
i = 0
fig_counter = 0

while i < len(lines):
    line = lines[i]

    if line.startswith("```"):
        j = i + 1
        body_lines = []
        while j < len(lines) and not lines[j].startswith("```"):
            body_lines.append(lines[j])
            j += 1
        body = "\n".join(body_lines)
        kind = classify_code(body)
        blocks.append({"type": "code", "kind": kind, "body": body})
        i = j + 1
        continue

    if line.startswith("> **[FIGURE]**"):
        m = re.search(r'`([^`]+)`', line)
        img_path = m.group(1)
        img_key = os.path.basename(img_path)
        j = i + 1
        cap_lines = []
        while j < len(lines) and lines[j].startswith(">"):
            content = lines[j][1:].strip()
            if content.startswith("*") and content.endswith("*") and len(content) > 1:
                content = content[1:-1]
            if content:
                cap_lines.append(content)
            j += 1
        fig_counter += 1
        blocks.append({"type": "figure", "num": fig_counter, "key": img_key,
                        "caption": " ".join(cap_lines)})
        i = j
        continue

    if line.startswith("> **Status note.**"):
        j = i
        note_lines = []
        while j < len(lines) and lines[j].startswith(">"):
            note_lines.append(lines[j][1:].strip())
            j += 1
        blocks.append({"type": "status", "text": " ".join(note_lines)})
        i = j
        continue

    if line.startswith("# "):
        blocks.append({"type": "h1", "text": line[2:].strip()})
        i += 1
        continue

    if line.startswith("### "):
        blocks.append({"type": "h3", "text": line[4:].strip()})
        i += 1
        continue

    if line.startswith("## "):
        blocks.append({"type": "h2", "text": line[3:].strip()})
        i += 1
        continue

    if line.strip() == "---":
        blocks.append({"type": "hr"})
        i += 1
        continue

    if line.startswith("- "):
        j = i
        items = []
        while j < len(lines) and lines[j].startswith("- "):
            items.append(lines[j][2:].strip())
            j += 1
        blocks.append({"type": "ulist", "items": items})
        i = j
        continue

    if re.match(r'^\d+\.\s', line):
        j = i
        items = []
        while j < len(lines) and (re.match(r'^\d+\.\s', lines[j]) or (lines[j].startswith("   ") and lines[j].strip())):
            if re.match(r'^\d+\.\s', lines[j]):
                items.append(lines[j].split(".", 1)[1].strip())
            else:
                items[-1] += " " + lines[j].strip()
            j += 1
        blocks.append({"type": "olist", "items": items})
        i = j
        continue

    if line.startswith("*") and line.endswith("*") and not line.startswith("**") and len(line) > 2 and i < 6:
        blocks.append({"type": "byline", "text": line.strip("*")})
        i += 1
        continue

    if line.strip() == "":
        i += 1
        continue

    # Layout directives, not content. They tell the builder how to group the figures that
    # follow; rendering them as text prints build instructions at the reader.
    if line.startswith("> **[FIGURE LAYOUT]**") or line.startswith("> **[FIGURE PAIR]**"):
        _n = 3 if "LAYOUT" in line else 2
        blocks.append({"type": "figgroup", "n": _n,
                       "mode": "feature" if _n == 3 else "pair"})
        j = i
        while j < len(lines) and lines[j].startswith(">"):
            j += 1
        i = j
        continue

    # ANY other blockquote. Without this, a '>' line matching none of the branches above
    # fell through to the paragraph accumulator below, whose condition refuses '>' lines --
    # so it consumed nothing, j stayed equal to i, and the parser spun forever appending
    # empty blocks. Line 7 ("> **Status.**", not "> **Status note.**") did exactly that:
    # the build climbed past 1 GB RSS and never reached the render.
    if line.startswith(">"):
        j = i
        note_lines = []
        while j < len(lines) and lines[j].startswith(">"):
            note_lines.append(lines[j][1:].strip())
            j += 1
        _note = " ".join(note_lines).strip()
        # The rendered block already carries a STATUS eyebrow, so a leading "Status."
        # in the prose just says it twice.
        _note = re.sub(r'^\*\*Status( note)?\.\*\*\s*', '', _note)
        blocks.append({"type": "status", "text": _note})
        i = j
        continue

    # plain paragraph — accumulate until blank line or block start
    j = i
    para_lines = []
    while j < len(lines) and lines[j].strip() != "" and not lines[j].startswith(("```", ">", "#", "---")) and not re.match(r'^\d+\.\s', lines[j]):
        para_lines.append(lines[j])
        j += 1
    blocks.append({"type": "p", "text": " ".join(para_lines)})
    # Never leave i unmoved: any future line the accumulator refuses would hang the build
    # rather than render imperfectly.
    i = j + 1 if j == i else j

print(f"parsed {len(blocks)} blocks")
for b in blocks[:15]:
    print(" ", b["type"], (b.get("text") or b.get("body") or "")[:50].replace("\n", "\\n"))

json.dump(blocks, open("/private/tmp/claude-501/-Users-Krishan-Projects-EXXA/6a3fd662-00b4-4751-b4ef-432b3cc0d32b/scratchpad/blocks.json", "w"), ensure_ascii=False, indent=1)
print("\nsaved blocks.json for inspection")

# ---------------------------------------------------------------- render

def slug_for_heading(text):
    m = re.match(r'^(\d+(?:\.\d+)?)\.?\s', text)
    if m:
        return "sec-" + m.group(1).replace(".", "-")
    return None

_slug_n = [0]  # fallback ids for headings with no leading number
out = []
list_buf = None
fig_group = None  # ('ol'|'ul', [items])

def flush_list():
    global list_buf
    if list_buf:
        tag, items = list_buf
        cls = "num-list" if tag == "ol" else "file-list"
        li_html = "".join(f"<li>{inline(it)}</li>" for it in items)
        out.append(f'<{tag} class="{cls}">{li_html}</{tag}>')
        list_buf = None

for b in blocks:
    t = b["type"]

    if t != "olist" and t != "ulist":
        flush_list()

    if t == "h1":
        out.append(f'<h1 class="title">{inline(b["text"])}</h1>')

    elif t == "byline":
        out.append(f'<p class="byline">{inline(b["text"])}</p>')

    elif t == "status":
        txt = b["text"].replace("**Status note.**", "").strip()
        out.append(f'<div class="status"><span class="status-tag">Status</span><p>{inline(txt)}</p></div>')

    elif t == "hr":
        out.append('<hr>')

    elif t == "h2":
        m = re.match(r'^(\d+)\.\s+(.*)$', b["text"])
        num, title = (m.group(1), m.group(2)) if m else ("", b["text"])
        slug = slug_for_heading(b["text"]) or f"sec-x{_slug_n[0]}"
        _slug_n[0] += 1
        out.append(f'<h2 id="{slug}"><span class="h-num">{num}</span><span class="h-title">{inline(title)}</span></h2>')

    elif t == "h3":
        m = re.match(r'^(\d+\.\d+)\s+(.*)$', b["text"])
        num, title = (m.group(1), m.group(2)) if m else ("", b["text"])
        slug = slug_for_heading(b["text"]) or f"sec-x{_slug_n[0]}"
        _slug_n[0] += 1
        out.append(f'<h3 id="{slug}"><span class="h-num">{num}</span><span class="h-title">{inline(title)}</span></h3>')

    elif t == "p":
        txt = b["text"].strip()
        if re.match(r'^\*Tags:.*\*$', txt):
            tags = [x.strip() for x in txt.strip("*")[5:].split(",")]
            chips = "".join(f'<span class="chip">{html.escape(x)}</span>' for x in tags)
            out.append(f'<div class="tags">{chips}</div>')
        else:
            out.append(f'<p>{inline(txt)}</p>')

    elif t == "code":
        kind, body = b["kind"], b["body"]
        if kind == "shell":
            lns = "\n".join(f'<span class="ln">{html.escape(l)}</span>' for l in body.splitlines())
            out.append(f'<div class="term outset"><div class="term-bar"><span></span><span></span><span></span><span class="term-label">terminal</span></div><pre class="term-body">{lns}</pre></div>')
        elif kind == "terminal":
            out.append(f'<div class="console outset">{html.escape(body)}</div>')
        elif kind in ("formula", "equation"):
            out.append(f'<div class="eq outset"><pre>{html.escape(body)}</pre></div>')
        elif kind == "data":
            colored = colorize_data_numbers(html.escape(body))
            out.append(f'<div class="data-panel outset"><span class="data-tag">Measured</span><pre>{colored}</pre></div>')

    elif t == "figgroup":
        fig_group = {"n": b["n"], "mode": b["mode"], "items": []}

    elif t == "figure" and fig_group is not None:
        fig_group["items"].append(b)
        if len(fig_group["items"]) >= fig_group["n"]:
            _cells = []
            for _fb in fig_group["items"]:
                _w, _h = FIG_DIMS.get(_fb["key"], (4, 3))
                _cells.append(
                    f'<figure><div class="fig-frame" style="aspect-ratio:{_w}/{_h}">'
                    f'<img src="data:image/webp;base64,{img_data[_fb["key"]]}" '
                    f'alt="{html.escape(_fb["caption"])}" loading="lazy"></div>'
                    f'<figcaption>{inline(_fb["caption"])}</figcaption></figure>')
            out.append(f'<div class="figgrid figgrid-{fig_group["mode"]} outset">'
                       + "".join(_cells) + '</div>')
            fig_group = None

    elif t == "figure":
        w, h = FIG_DIMS.get(b["key"], (4, 3))
        out.append(
            f'<figure class="outset">'
            f'<div class="fig-frame" style="aspect-ratio:{w}/{h}">'
            f'<img src="data:image/webp;base64,{img_data[b["key"]]}" alt="{html.escape(b["caption"])}" loading="lazy">'
            f'</div><figcaption>{inline(b["caption"])}</figcaption></figure>'
        )

    elif t == "olist":
        list_buf = ("ol", b["items"])

    elif t == "ulist":
        list_buf = ("ul", b["items"])

flush_list()
body_html = "\n".join(out)

print(f"\nrendered {len(out)} elements, {len(body_html)} chars body html")

# ---------------------------------------------------------------- CSS
def face(family, weight, style, key):
    return f"""@font-face {{
  font-family: '{family}';
  font-weight: {weight};
  font-style: {style};
  font-display: swap;
  src: url(data:font/woff2;base64,{font_data[key]}) format('woff2');
}}"""

font_faces = "\n".join([
    face("Inter", 400, "normal", "sans400"),
    face("Inter", 600, "normal", "sans600"),
    face("Inter", 700, "normal", "sans700"),
    face("Source Serif", 400, "normal", "serif400"),
    face("Source Serif", 400, "italic", "serif400i"),
    face("Source Serif", 600, "normal", "serif600"),
    face("Plex Mono", 400, "normal", "mono400"),
    face("Plex Mono", 600, "normal", "mono600"),
])

CSS = f"""
{font_faces}

/* Light only, deliberately. The reference blogs are light Medium posts and the
   destination is a light platform, so there is no dark variant to drift out of sync:
   a viewer in OS dark mode still gets the page as it will be published.
   Medium's own palette: white ground, #242424 ink,
   #6B6B6B secondary. No decorative accent, because the reference blogs have none:
   links are underlined ink. The only colour is semantic, inside data blocks, where
   this piece genuinely reports gains and regressions. */
:root {{
  --ground:     #ffffff;
  --ink:        #242424;
  --ink-soft:   #6b6b6b;
  --ink-faint:  #8c8c8c;
  --rule:       #e6e6e6;
  --panel:      #f7f7f5;
  --panel-rule: #e9e9e5;
  --pos:        #1a7f5a;
  --neg:        #b3261e;
}}



* {{ box-sizing: border-box; }}
html {{ -webkit-text-size-adjust: 100%; color-scheme: light; }}
body {{
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: 'Source Serif', Charter, Georgia, serif;
  font-size: 21px;
  line-height: 1.58;
  letter-spacing: -0.003em;
  -webkit-font-smoothing: antialiased;
}}
::selection {{ background: #b4d5fe; color: #242424; }}

a {{
  color: inherit;
  text-decoration: underline;
  text-decoration-thickness: 1px;
  text-underline-offset: 2px;
  text-decoration-color: var(--ink-faint);
}}
a:hover {{ text-decoration-color: var(--ink); }}
a:focus-visible {{ outline: 2px solid var(--ink); outline-offset: 2px; border-radius: 2px; }}

code {{
  font-family: 'Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace;
  font-size: 0.82em;
  background: var(--panel);
  padding: 0.12em 0.36em;
  border-radius: 2px;
}}
strong {{ font-weight: 600; }}
em {{ font-style: italic; }}

/* Single measure. Figures and data break out wider, the way Medium outsets images. */
.wrap {{ max-width: 680px; margin: 0 auto; padding: 0 24px 140px; }}
.outset {{
  width: min(860px, calc(100vw - 48px));
  margin-left: 50%;
  transform: translateX(-50%);
}}

/* ---- masthead ---- */
.masthead {{ padding: 72px 0 8px; }}
.eyebrow {{
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--ink-soft);
  display: block;
  margin-bottom: 20px;
}}
h1.title {{
  font-family: 'Inter', system-ui, sans-serif;
  font-weight: 700;
  font-size: clamp(32px, 5.2vw, 42px);
  line-height: 1.19;
  letter-spacing: -0.022em;
  margin: 0 0 24px;
  text-wrap: balance;
  color: var(--ink);
}}
p.byline {{
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 15px;
  color: var(--ink-soft);
  margin: 0 0 6px;
  line-height: 1.5;
  letter-spacing: 0;
}}

.status {{
  margin: 32px 0 0;
  padding: 0 0 0 20px;
  border-left: 3px solid var(--rule);
}}
.status-tag {{
  display: block;
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 12px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-weight: 600;
  color: var(--ink-faint);
  margin-bottom: 6px;
}}
.status p {{ margin: 0; font-size: 18px; color: var(--ink-soft); font-style: italic; line-height: 1.55; }}

hr {{ border: none; border-top: 1px solid var(--rule); margin: 52px 0; }}

/* ---- headings: sans, bold, plain. No numerals in circles. ---- */
h2 {{
  font-family: 'Inter', system-ui, sans-serif;
  font-weight: 700;
  font-size: 27px;
  line-height: 1.25;
  letter-spacing: -0.018em;
  margin: 56px 0 8px;
  scroll-margin-top: 20px;
  text-wrap: balance;
  color: var(--ink);
}}
h3 {{
  font-family: 'Inter', system-ui, sans-serif;
  font-weight: 600;
  font-size: 20px;
  line-height: 1.3;
  letter-spacing: -0.012em;
  margin: 40px 0 6px;
  scroll-margin-top: 20px;
  color: var(--ink);
}}
h2 .h-num, h3 .h-num {{
  color: var(--ink-faint);
  font-variant-numeric: tabular-nums;
  margin-right: 10px;
  font-weight: 600;
}}

p {{ margin: 0 0 28px; }}
p:last-child {{ margin-bottom: 0; }}
h2 + p, h3 + p {{ margin-top: 0; }}

/* ---- equations ---- */
.eq {{
  margin-block: 32px;
  padding: 20px 24px;
  background: var(--panel);
  border-radius: 3px;
  overflow-x: auto;
}}
.eq pre {{
  margin: 0;
  font-family: 'Plex Mono', ui-monospace, Menlo, monospace;
  font-size: 14px; line-height: 1.7;
  color: var(--ink);
  white-space: pre;
  text-align: center;
}}

/* ---- data panels: the tables this piece is built on ---- */
.data-panel {{
  position: relative;
  margin-block: 34px;
  background: var(--panel);
  border: 1px solid var(--panel-rule);
  border-radius: 3px;
  overflow-x: auto;
}}
.data-tag {{
  display: block;
  padding: 12px 20px 0;
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--ink-faint); font-weight: 600;
}}
.data-panel pre {{
  margin: 0;
  padding: 10px 20px 20px;
  font-family: 'Plex Mono', ui-monospace, Menlo, monospace;
  font-size: 13px; line-height: 1.8;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
  white-space: pre;
}}
.data-panel .pos {{ color: var(--pos); font-weight: 600; }}
.data-panel .neg {{ color: var(--neg); font-weight: 600; }}

/* ---- console + shell ---- */
.console, .term {{
  margin-block: 30px;
  background: var(--panel);
  border: 1px solid var(--panel-rule);
  border-radius: 3px;
  overflow-x: auto;
}}
.console {{
  padding: 14px 20px;
  font-family: 'Plex Mono', ui-monospace, Menlo, monospace;
  font-size: 13px;
  color: var(--ink);
  white-space: pre;
}}
.term-bar {{ display: none; }}
.term-body {{ margin: 0; padding: 16px 20px; overflow-x: auto; }}
.term-body .ln {{
  display: block;
  font-family: 'Plex Mono', ui-monospace, Menlo, monospace;
  font-size: 13px; line-height: 1.75;
  color: var(--ink);
}}
.term-body .ln::before {{ content: '$ '; color: var(--ink-faint); }}

/* ---- figures: outset, plain, sans caption centred under ---- */
figure {{ margin-block: 44px; }}
.fig-tag {{ display: none; }}
.fig-frame {{ width: 100%; background: transparent; }}
.fig-frame img {{ display: block; width: 100%; height: auto; }}
figcaption {{
  margin: 12px auto 0;
  max-width: 680px;
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  color: var(--ink-faint);
  text-align: center;
  letter-spacing: 0;
}}

/* ---- lists: plain, serif, Medium-style ---- */
ol.num-list, ul.file-list {{ margin: 0 0 28px; padding-left: 28px; }}
ol.num-list li, ul.file-list li {{ margin-bottom: 10px; padding-left: 6px; }}
ol.num-list {{ list-style: decimal; }}
ul.file-list {{ list-style: disc; }}
ul.file-list li::marker {{ color: var(--ink-faint); }}

/* margin-top/bottom, NOT the `margin` shorthand: this rule sits after .outset, so a
   shorthand would reset .outset's margin-left:50% to 0 and drag the whole grid 316px
   left, off the page. Measured x=-126 against a correct 190 before this. */
.figgrid {{ display: grid; gap: 16px; margin-top: 36px; margin-bottom: 36px; }}
/* min-width:0 keeps a grid item from refusing to shrink below its image's intrinsic width.
   Not what caused the off-page overflow (that was the margin shorthand above), but correct
   for a grid whose items are 1500px-wide images. */
.figgrid > figure {{ margin: 0; min-width: 0; }}
.figgrid > figure img {{ max-width: 100%; }}
/* No height override here. .fig-frame already carries an inline aspect-ratio, and forcing
   height:100% inside the row-spanning cell made aspect-ratio derive WIDTH from that height,
   which pushed the portrait sample grid clean out of its column. */
.figgrid .fig-frame {{ width: 100%; }}
.figgrid-pair {{ grid-template-columns: 1fr 1fr; align-items: start; }}
/* feature: the wide sample grid holds the left column, the two narrow plots stack right */
.figgrid-feature {{ grid-template-columns: 1fr 1.1fr; align-items: start; }}
.figgrid-feature > figure:first-child {{ grid-row: 1 / span 2; }}
@media (max-width: 720px) {{
  .figgrid-pair, .figgrid-feature {{ grid-template-columns: 1fr; }}
  .figgrid-feature > figure:first-child {{ grid-row: auto; }}
}}
.tags {{ margin: 36px 0 0; display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 13px;
  color: var(--ink-soft);
  background: var(--panel);
  border: 1px solid var(--panel-rule);
  padding: 5px 12px;
  border-radius: 20px;
}}

footer.colophon {{
  margin-top: 72px;
  padding-top: 24px;
  border-top: 1px solid var(--rule);
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 13px;
  color: var(--ink-faint);
  line-height: 1.6;
}}

@media (max-width: 700px) {{
  body {{ font-size: 19px; line-height: 1.55; }}
  .wrap {{ padding: 0 20px 90px; }}
  .masthead {{ padding: 44px 0 6px; }}
  h2 {{ font-size: 23px; margin-top: 44px; }}
  h3 {{ font-size: 18.5px; }}
  .outset {{ width: calc(100vw - 40px); }}
  figcaption {{ font-size: 13px; }}
}}

@media (prefers-reduced-motion: reduce) {{
  * {{ animation: none !important; transition: none !important; }}
}}

@media print {{
  body {{ background: #fff; color: #000; }}
}}
"""

print(f"CSS: {len(CSS)} chars")

# ---------------------------------------------------------------- assemble page
# split off masthead pieces (h1, bylines, status, first hr) from the flowing body
masthead_out = []
rest_out = []
seen_first_hr = False
for chunk in out:
    if not seen_first_hr and (chunk.startswith('<h1') or chunk.startswith('<p class="byline"') or chunk.startswith('<div class="status"')):
        masthead_out.append(chunk)
    elif not seen_first_hr and chunk == '<hr>':
        seen_first_hr = True
    else:
        rest_out.append(chunk)
masthead_html = "\n".join(masthead_out)
rest_html = "\n".join(rest_out)


OUT_HTML = os.path.join(REPO, "blog_midterm.html")
_desc = ("GSoC 2026 midterm write-up: denoising protoplanetary disk line emission with "
         "U-Net and conditional DDPM. ML4Sci / EXXA.")

_head = (
    '<!DOCTYPE html>\n<html lang="en" data-theme="light">\n<head>\n'
    '<meta charset="utf-8">\n'
    '<title>Denoising Protoplanetary Disks</title>\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    f'<meta name="description" content="{html.escape(_desc, quote=True)}">\n'
)

# Written in pieces. Assembling BODY and then doc held three ~2.5 MB copies of the same
# bytes at once, and on a machine already swapping that was enough to push the process into
# uninterruptible I/O for tens of minutes. Nothing here needs the whole document in memory.
with open(OUT_HTML, "w", encoding="utf-8") as _f:
    _f.write(_head)
    _f.write("<style>"); _f.write(CSS); _f.write("</style>\n")
    _f.write("</head>\n<body>\n")
    _f.write('<div class="wrap">\n  <div class="masthead">\n'
             '    <span class="eyebrow">GSoC 2026 &middot; ML4Sci / EXXA &middot; '
             'Midterm Report</span>\n')
    _f.write(masthead_html)
    _f.write("\n  </div>\n\n  <hr>\n\n")
    for _chunk in rest_out:
        _f.write(_chunk)
        _f.write("\n")
    _f.write('\n  <footer class="colophon">\n'
             '    Source: BLOG_MIDTERM.md &middot; every figure embedded from results/ '
             '&middot;\n    self-contained, no external requests\n  </footer>\n</div>\n')
    _f.write("</body>\n</html>\n")

print(f"\nwrote {OUT_HTML}")
print(f"size: {os.path.getsize(OUT_HTML)/1024/1024:.2f} MB")
