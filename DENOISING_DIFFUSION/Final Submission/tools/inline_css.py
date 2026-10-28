"""Fold a stylesheet into style="" attributes.

Blogger's theme beat our scoped rules on specificity (its selectors carry an id).
An inline declaration with !important outranks every stylesheet rule, so the
layout-critical properties are written onto the elements themselves.
"""
import re
from html.parser import HTMLParser
from scope_css import _split_top

VOID = {"br", "hr", "img", "meta", "link", "input", "source"}

def custom_props(css):
    """--name: value pairs, so var() can be resolved before inlining."""
    props = {}
    for kind, header, body in _split_top(css):
        if kind == "at":
            continue
        for d in body.split(";"):
            if ":" in d and d.strip().startswith("--"):
                k, v = d.split(":", 1)
                props[k.strip()] = v.strip()
    return props

def resolve(value, props, depth=0):
    if "var(" not in value or depth > 4:
        return value
    def sub(m):
        name = m.group(1).strip()
        fallback = (m.group(2) or "").lstrip(",").strip()
        return props.get(name, fallback or "inherit")
    return resolve(re.sub(r'var\(\s*(--[A-Za-z0-9_-]+)\s*((?:,[^()]*)?)\)', sub, value),
                   props, depth + 1)

def collect_rules(css):
    """(specificity, order, tag, classes, decls) for selectors simple enough to inline."""
    rules = []
    for kind, header, body in _split_top(css):
        if kind == "at":                      # @media/@font-face stay in the <style> block
            continue
        for sel in header.split(","):
            sel = sel.strip()
            if not sel or re.search(r'[\s>+~:\[]', sel):
                continue                      # descendant/pseudo: not inlinable
            m = re.fullmatch(r'([a-zA-Z0-9]*)((?:\.[A-Za-z0-9_-]+)*)', sel)
            if not m:
                continue
            tag, cls = m.group(1).lower(), set(c for c in m.group(2).split(".") if c)
            spec = (1 if cls else 0, len(cls), 1 if tag else 0)
            rules.append((spec, len(rules), tag, cls, body.strip()))
    return rules

def _decls(body):
    out = []
    for d in body.split(";"):
        if ":" in d:
            k, v = d.split(":", 1)
            out.append((k.strip(), v.strip().replace("!important", "").strip()))
    return out

class Inliner(HTMLParser):
    def __init__(self, rules, props=None):
        super().__init__(convert_charrefs=False)
        self.rules, self.out, self.props = rules, [], props or {}
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = set((a.get("class") or "").split())
        merged = {}
        for spec, order, rtag, rcls, body in sorted(self.rules, key=lambda r: (r[0], r[1])):
            if rtag and rtag != tag:
                continue
            if rcls and not rcls <= classes:
                continue
            if not rtag and not rcls:
                continue
            for k, v in _decls(body):
                if k.startswith("--"):
                    continue                 # the variable itself is not a real declaration
                merged[k] = resolve(v, self.props)
        if merged:
            own = a.get("style", "")
            css = "; ".join(f"{k}: {v} !important" for k, v in merged.items())
            a["style"] = (own + "; " if own else "") + css
        s = "".join(f' {k}="{v}"' for k, v in a.items())
        self.out.append(f"<{tag}{s}>" if tag not in VOID else f"<{tag}{s}>")
    def handle_endtag(self, tag):
        if tag not in VOID:
            self.out.append(f"</{tag}>")
    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
    def handle_data(self, d):    self.out.append(d)
    def handle_entityref(self, n): self.out.append(f"&{n};")
    def handle_charref(self, n):   self.out.append(f"&#{n};")
    def handle_comment(self, d):   self.out.append(f"<!--{d}-->")

def inline(html_fragment, css):
    p = Inliner(collect_rules(css), custom_props(css))
    p.feed(html_fragment)
    return "".join(p.out)
