"""Behaviour tests of the front end: the real index.html and app scripts run in Node with a small DOM.

Each test drives the page the way a user would (clicks, changed fields, stubbed server answers) and
checks what the page then does. Server payloads come from the real API where they matter.
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from survival_toolkit.app import app

_ROOT = Path(__file__).resolve().parents[1] / "src" / "survival_toolkit"
_STATIC_DIR = _ROOT / "static"
_TEMPLATE = _ROOT / "templates" / "index.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed for the front-end tests")

client = TestClient(app, base_url="http://127.0.0.1")

# A small DOM for the page: elements parsed from index.html, CSS-like selectors, events, timers, and
# stubbed fetch/Plotly/Blob, so the app's own scripts run unchanged.
_HARNESS_JS = r"""
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const VOID_TAGS = new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"]);
const SCRIPT_ORDER = [
  "app_shell.js", "app_downloads.js", "app_benchmark.js", "app_core.js", "app_workspace.js", "app_columns.js",
  "app_render.js", "app_predictive.js", "app_analyses.js", "app_models.js", "app_markers.js", "app.js",
];

function decodeEntities(text) {
  return text.replace(/&(#\d+|#x[0-9a-f]+|amp|lt|gt|quot|apos|nbsp|times);/gi, (match, code) => {
    const lower = code.toLowerCase();
    if (lower[0] === "#") return String.fromCodePoint(lower[1] === "x" ? parseInt(lower.slice(2), 16) : parseInt(lower.slice(1), 10));
    return { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: "\u00a0", times: "\u00d7" }[lower] ?? match;
  });
}

function escapeText(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function camelCase(name) {
  return name.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

class FakeText {
  constructor(text) {
    this.nodeType = 3;
    this.data = text;
    this.parentElement = null;
  }
  get textContent() { return this.data; }
  set textContent(value) { this.data = String(value ?? ""); }
  remove() { this.parentElement?.removeChild(this); }
}

class FakeClassList {
  constructor() { this.set = new Set(); }
  add(...names) { names.forEach((name) => this.set.add(name)); }
  remove(...names) { names.forEach((name) => this.set.delete(name)); }
  toggle(name, force) {
    const on = force === undefined ? !this.set.has(name) : Boolean(force);
    if (on) this.set.add(name); else this.set.delete(name);
    return on;
  }
  contains(name) { return this.set.has(name); }
  toString() { return [...this.set].join(" "); }
}

class FakeElement {
  constructor(tagName, ownerDocument) {
    this.nodeType = 1;
    this.tagName = String(tagName).toUpperCase();
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.childNodes = [];
    this.parentElement = null;
    this.attributes = {};
    this.dataset = {};
    this.style = {};
    this.classList = new FakeClassList();
    this.listeners = {};
    this.id = "";
    this.type = "";
    this.checked = false;
    this.disabled = false;
    this.selected = false;
    this.open = false;
    this.title = "";
    this.placeholder = "";
    this.scrollTop = 0;
    this.scrollLeft = 0;
    this._value = "";
    this._selectedIndex = -1;
  }

  get className() { return this.classList.toString(); }
  set className(value) { this.classList.set = new Set(String(value ?? "").split(/\s+/).filter(Boolean)); }

  get isConnected() {
    let node = this;
    while (node.parentElement) node = node.parentElement;
    return node === this.ownerDocument?.documentElement;
  }
  get offsetParent() { return this.isConnected ? this.parentElement : null; }
  get offsetWidth() { return 100; }
  get clientWidth() { return 100; }
  get firstChild() { return this.childNodes[0] || null; }
  get firstElementChild() { return this.children[0] || null; }

  get options() { return this.querySelectorAll("option"); }
  get selectedIndex() { return this._selectedIndex; }
  set selectedIndex(index) { this._selectedIndex = Number(index); }
  get selectedOptions() {
    const option = this.options[this._selectedIndex];
    return option ? [option] : [];
  }
  get value() {
    if (this.tagName === "SELECT") {
      const option = this.options[this._selectedIndex];
      return option ? option.value : "";
    }
    if (this.tagName === "OPTION") return this.attributes.value !== undefined ? this._value : this.textContent;
    return this._value;
  }
  set value(value) {
    if (this.tagName === "SELECT") {
      this._selectedIndex = this.options.findIndex((option) => option.value === String(value));
      return;
    }
    if (this.tagName === "OPTION") this.attributes.value = String(value);
    this._value = String(value ?? "");
  }

  get textContent() { return this.childNodes.map((node) => node.textContent).join(""); }
  set textContent(value) {
    this.childNodes.forEach((node) => { node.parentElement = null; });
    this.childNodes = [];
    this.children = [];
    const text = String(value ?? "");
    if (text) this.appendChild(new FakeText(text));
  }
  get innerText() { return this.textContent; }

  get innerHTML() { return this.childNodes.map((node) => serialize(node)).join(""); }
  set innerHTML(html) {
    this.childNodes.forEach((node) => { node.parentElement = null; });
    this.childNodes = [];
    this.children = [];
    if (this.tagName === "SELECT") this._selectedIndex = -1;
    parseHtmlInto(this, String(html ?? ""), this.ownerDocument);
  }

  getAttribute(name) {
    if (name === "class") return this.className;
    if (name === "id") return this.id || null;
    if (name.startsWith("data-")) return this.dataset[camelCase(name.slice(5))] ?? null;
    return this.attributes[name] ?? null;
  }
  setAttribute(name, value) {
    const text = String(value);
    if (name === "class") this.className = text;
    else if (name === "id") this.id = text;
    else if (name.startsWith("data-")) this.dataset[camelCase(name.slice(5))] = text;
    else if (name === "value") this.value = text;
    else if (name === "type") this.type = text;
    else this.attributes[name] = text;
  }
  hasAttribute(name) { return this.getAttribute(name) !== null; }
  removeAttribute(name) {
    if (name.startsWith("data-")) delete this.dataset[camelCase(name.slice(5))];
    else delete this.attributes[name];
  }

  appendChild(node) {
    if (node.parentElement) node.parentElement.removeChild(node);
    node.parentElement = this;
    this.childNodes.push(node);
    if (node.nodeType === 1) {
      this.children.push(node);
      const select = node.tagName === "OPTION" ? this.closest("select") : null;
      if (select) {
        const index = select.options.indexOf(node);
        if (node.selected || select._selectedIndex === -1) select._selectedIndex = index;
      }
    }
    return node;
  }
  append(...nodes) { nodes.forEach((node) => this.appendChild(typeof node === "string" ? new FakeText(node) : node)); }
  prepend(...nodes) {
    [...nodes].reverse().forEach((node) => {
      const child = typeof node === "string" ? new FakeText(node) : node;
      if (child.parentElement) child.parentElement.removeChild(child);
      child.parentElement = this;
      this.childNodes.unshift(child);
      if (child.nodeType === 1) this.children.unshift(child);
    });
  }
  removeChild(node) {
    this.childNodes = this.childNodes.filter((child) => child !== node);
    this.children = this.children.filter((child) => child !== node);
    node.parentElement = null;
    return node;
  }
  remove() { this.parentElement?.removeChild(this); }
  replaceWith(node) {
    const parent = this.parentElement;
    if (!parent) return;
    const index = parent.childNodes.indexOf(this);
    parent.removeChild(this);
    node.parentElement = parent;
    parent.childNodes.splice(index, 0, node);
    parent.children = parent.childNodes.filter((child) => child.nodeType === 1);
  }
  replaceChildren(...nodes) {
    this.textContent = "";
    this.append(...nodes);
  }
  insertAdjacentElement(position, node) {
    if (position !== "afterend" || !this.parentElement) return this.appendChild(node);
    const parent = this.parentElement;
    if (node.parentElement) node.parentElement.removeChild(node);
    node.parentElement = parent;
    parent.childNodes.splice(parent.childNodes.indexOf(this) + 1, 0, node);
    parent.children = parent.childNodes.filter((child) => child.nodeType === 1);
    return node;
  }
  contains(node) {
    for (let current = node; current; current = current.parentElement) if (current === this) return true;
    return false;
  }

  matches(selector) { return matchesSelector(this, selector); }
  closest(selector) {
    for (let node = this; node; node = node.parentElement) if (node.nodeType === 1 && matchesSelector(node, selector)) return node;
    return null;
  }
  querySelectorAll(selector) {
    const found = [];
    const visit = (node) => node.children.forEach((child) => {
      if (matchesSelector(child, selector, this)) found.push(child);
      visit(child);
    });
    visit(this);
    return found;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }

  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  removeEventListener(type, listener) { this.listeners[type] = (this.listeners[type] || []).filter((item) => item !== listener); }
  dispatchEvent(event) {
    event.target ||= this;
    let node = this;
    while (node) {
      event.currentTarget = node;
      (node.listeners?.[event.type] || []).slice().forEach((listener) => listener.call(node, event));
      if (!event.bubbles || event.propagationStopped) break;
      node = node.parentElement || (node === node.ownerDocument?.documentElement ? node.ownerDocument : null);
    }
    return !event.defaultPrevented;
  }
  click() {
    if (this.disabled) return;
    this.dispatchEvent(makeEvent("click", { bubbles: true }));
  }
  focus() { if (this.ownerDocument) this.ownerDocument.activeElement = this; }
  blur() {}
  scrollIntoView() {}
  getBoundingClientRect() { return { top: 0, bottom: 0, left: 0, right: 0, width: 100, height: 100 }; }
}

function makeEvent(type, { bubbles = false } = {}) {
  return {
    type,
    bubbles,
    target: null,
    currentTarget: null,
    defaultPrevented: false,
    propagationStopped: false,
    preventDefault() { this.defaultPrevented = true; },
    stopPropagation() { this.propagationStopped = true; },
  };
}

function serialize(node) {
  if (node.nodeType === 3) return escapeText(node.data);
  const tag = node.tagName.toLowerCase();
  const attrs = [];
  if (node.id) attrs.push(`id="${node.id}"`);
  if (node.className) attrs.push(`class="${node.className}"`);
  Object.entries(node.dataset).forEach(([key, value]) => attrs.push(`data-${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}="${value}"`));
  Object.entries(node.attributes).forEach(([key, value]) => attrs.push(`${key}="${value}"`));
  const open = `<${tag}${attrs.length ? ` ${attrs.join(" ")}` : ""}>`;
  return VOID_TAGS.has(tag) ? open : `${open}${node.childNodes.map(serialize).join("")}</${tag}>`;
}

const TOKEN_SOURCE = /<!--[\s\S]*?-->|<!doctype[^>]*>|<\/\s*([a-zA-Z][\w-]*)\s*>|<([a-zA-Z][\w-]*)((?:\s+[^\s=>\/]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?)*)\s*(\/?)>|([^<]+|<)/.source;
const ATTR_SOURCE = /([^\s=>\/]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/.source;

function parseHtmlInto(root, html, document) {
  const stack = [root];
  const pattern = new RegExp(TOKEN_SOURCE, "gi");
  let match;
  while ((match = pattern.exec(html))) {
    const [, closeTag, openTag, attrText, selfClose, text] = match;
    const parent = stack[stack.length - 1];
    if (text !== undefined) {
      if (text) parent.appendChild(new FakeText(decodeEntities(text)));
    } else if (closeTag) {
      const name = closeTag.toUpperCase();
      for (let index = stack.length - 1; index > 0; index -= 1) {
        if (stack[index].tagName === name) {
          stack.length = index;
          break;
        }
      }
    } else if (openTag) {
      const element = new FakeElement(openTag, document);
      const attrPattern = new RegExp(ATTR_SOURCE, "g");
      let attrMatch;
      while ((attrMatch = attrPattern.exec(attrText || ""))) {
        const name = attrMatch[1].toLowerCase();
        const value = decodeEntities(attrMatch[2] ?? attrMatch[3] ?? attrMatch[4] ?? "");
        if (name === "checked") element.checked = true;
        else if (name === "disabled") element.disabled = true;
        else if (name === "selected") element.selected = true;
        else if (name === "open") element.open = true;
        else element.setAttribute(name, value);
      }
      parent.appendChild(element);
      if (!selfClose && !VOID_TAGS.has(openTag.toLowerCase())) stack.push(element);
    }
  }
}

function parseCompound(text) {
  const parts = [];
  const pattern = /(#[\w-]+)|(\.[\w-]+)|(\[[^\]]+\])|(:[\w-]+)|([a-zA-Z*][\w-]*)/g;
  let match;
  while ((match = pattern.exec(text))) parts.push(match[0]);
  return parts;
}

function matchesCompound(element, compound, scope) {
  return parseCompound(compound).every((part) => {
    if (part[0] === "#") return element.id === part.slice(1);
    if (part[0] === ".") return element.classList.contains(part.slice(1));
    if (part === ":checked") return Boolean(element.checked || (element.tagName === "OPTION" && element.selected));
    if (part === ":scope") return element === scope;
    if (part[0] === ":") return false;
    if (part[0] === "[") {
      const attr = /^\[\s*([\w-]+)\s*(?:=\s*(?:"([^"]*)"|'([^']*)'|([^\]\s]+)))?\s*\]$/.exec(part);
      if (!attr) return false;
      const name = attr[1];
      const actual = name === "type" ? element.type : (name === "value" ? element.value : element.getAttribute(name));
      if (attr[2] === undefined && attr[3] === undefined && attr[4] === undefined) return actual !== null && actual !== undefined && actual !== "";
      return String(actual) === (attr[2] ?? attr[3] ?? attr[4]);
    }
    if (part === "*") return true;
    return element.tagName === part.toUpperCase();
  });
}

function matchesComplex(element, selector, scope) {
  const tokens = selector.trim().replace(/\s*>\s*/g, " > ").split(/\s+/).filter(Boolean);
  let index = tokens.length - 1;
  if (!matchesCompound(element, tokens[index], scope)) return false;
  let current = element;
  index -= 1;
  while (index >= 0) {
    if (tokens[index] === ">") {
      current = current.parentElement;
      index -= 1;
      if (!current || !matchesCompound(current, tokens[index], scope)) return false;
      index -= 1;
    } else {
      current = current.parentElement;
      while (current && !matchesCompound(current, tokens[index], scope)) current = current.parentElement;
      if (!current) return false;
      index -= 1;
    }
  }
  return true;
}

function matchesSelector(element, selectorList, scope = null) {
  if (element.nodeType !== 1) return false;
  return String(selectorList).split(",").some((selector) => selector.trim() && matchesComplex(element, selector, scope));
}

class FakeDocument {
  constructor(html) {
    this.nodeType = 9;
    this.listeners = {};
    this.activeElement = null;
    this.documentElement = new FakeElement("html", this);
    parseHtmlInto(this.documentElement, html, this);
    this.body = this.documentElement.querySelector("body") || this.documentElement;
  }
  createElement(tag) { return new FakeElement(tag, this); }
  createTextNode(text) { return new FakeText(text); }
  getElementById(id) { return this.documentElement.querySelector(`#${id}`); }
  querySelector(selector) { return this.documentElement.querySelector(selector); }
  querySelectorAll(selector) { return this.documentElement.querySelectorAll(selector); }
  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  removeEventListener(type, listener) { this.listeners[type] = (this.listeners[type] || []).filter((item) => item !== listener); }
  dispatchEvent(event) {
    event.target ||= this;
    (this.listeners[event.type] || []).slice().forEach((listener) => listener.call(this, event));
    return !event.defaultPrevented;
  }
}

function jsonResponse(status, body) {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: () => "application/json" },
    text: async () => text,
    json: async () => JSON.parse(text),
    blob: async () => ({ type: "application/json", parts: [text], size: text.length }),
  };
}

function createPage(staticDir, templatePath) {
  const html = fs.readFileSync(templatePath, "utf8").replace(/<script[\s\S]*?<\/script>/gi, "");
  const document = new FakeDocument(html);
  const timers = [];
  let timerSerial = 0;
  const requests = [];
  const page = { document, requests, timers, fetchHandler: null, downloads: [] };

  const schedule = (fn, delay = 0) => {
    timerSerial += 1;
    timers.push({ id: timerSerial, fn, delay: Number(delay) || 0 });
    return timerSerial;
  };
  const cancel = (id) => {
    const index = timers.findIndex((timer) => timer.id === id);
    if (index >= 0) timers.splice(index, 1);
  };

  class FakeBlob {
    constructor(parts = [], options = {}) {
      this.parts = parts;
      this.type = options.type || "";
    }
    text() { return Promise.resolve(this.parts.map((part) => (part?.parts ? part.parts.join("") : String(part))).join("")); }
    slice() { return new FakeBlob([]); }
    arrayBuffer() { return Promise.resolve(new ArrayBuffer(0)); }
  }

  class FakeFormData {
    constructor() { this.entries = []; }
    append(name, value) { this.entries.push([name, value]); }
    get(name) { return this.entries.find(([key]) => key === name)?.[1]; }
  }

  const fetchStub = (url, options = {}) => {
    const request = { url: String(url), method: options.method || "GET", body: options.body, signal: options.signal };
    request.json = () => JSON.parse(request.body || "{}");
    requests.push(request);
    const handler = page.fetchHandler;
    const answer = handler ? handler(request) : { status: 500, body: { detail: `No stub for ${request.url}` } };
    return new Promise((resolve, reject) => {
      const abortError = () => {
        const error = new Error("The operation was aborted.");
        error.name = "AbortError";
        return error;
      };
      if (options.signal?.aborted) {
        reject(abortError());
        return;
      }
      options.signal?.addEventListener?.("abort", () => reject(abortError()));
      Promise.resolve(answer).then(
        (value) => resolve(value && typeof value.text === "function" ? value : jsonResponse(value?.status ?? 200, value?.body ?? {})),
        reject,
      );
    });
  };

  const plotly = {
    downloadImageError: null,
    newPlot(el, data, layout = {}, config = {}) {
      el.data = data;
      el.layout = layout;
      el._fullLayout = { ...layout, height: layout?.height || 360 };
      el.config = config;
      return Promise.resolve(el);
    },
    purge(el) {
      delete el.data;
      delete el.layout;
      delete el._fullLayout;
    },
    relayout: () => Promise.resolve(),
    Plots: { resize() {} },
    downloadImage(el, options) {
      page.downloads.push({ kind: "image", options });
      return plotly.downloadImageError ? Promise.reject(plotly.downloadImageError) : Promise.resolve(options.filename);
    },
  };
  page.plotly = plotly;

  const context = {
    console,
    document,
    Element: FakeElement,
    HTMLElement: FakeElement,
    Blob: FakeBlob,
    FormData: FakeFormData,
    AbortController,
    structuredClone,
    URL: {
      createObjectURL: (blob) => {
        page.downloads.push({ kind: "blob", blob });
        return "blob:survstudio";
      },
      revokeObjectURL() {},
    },
    performance: { now: () => Date.now() },
    Plotly: plotly,
    fetch: fetchStub,
    setTimeout: schedule,
    clearTimeout: cancel,
    requestAnimationFrame: (fn) => schedule(fn, 16),
    cancelAnimationFrame: cancel,
  };
  vm.createContext(context);
  vm.runInContext("var window = this;", context);
  Object.assign(context, {
    location: { protocol: "http:", href: "http://127.0.0.1:8000/", origin: "http://127.0.0.1:8000" },
    history: { state: null, replaceState(state) { this.state = state; }, pushState(state) { this.state = state; } },
    innerWidth: 1280,
    innerHeight: 900,
    getComputedStyle: () => ({ display: "block" }),
    confirm: () => true,
    scrollTo() {},
    addEventListener() {},
    removeEventListener() {},
  });
  page.context = context;
  page.run = (code) => vm.runInContext(code, context);
  page.flushTimers = (limit = 500) => {
    for (let count = 0; count < limit && timers.length; count += 1) {
      timers.sort((left, right) => left.delay - right.delay || left.id - right.id);
      timers.shift().fn();
    }
  };
  page.settle = async (rounds = 25) => {
    for (let round = 0; round < rounds; round += 1) {
      await new Promise((resolve) => setImmediate(resolve));
      page.flushTimers();
    }
  };
  page.change = (selector, value) => {
    const element = document.querySelector(selector);
    if (value !== undefined) {
      if (element.type === "checkbox") element.checked = Boolean(value);
      else element.value = value;
    }
    element.dispatchEvent(makeEvent("input", { bubbles: true }));
    element.dispatchEvent(makeEvent("change", { bubbles: true }));
  };
  page.keydown = (options) => document.dispatchEvent({ ...makeEvent("keydown", { bubbles: true }), ...options });
  page.toasts = () => document.querySelectorAll("#toastContainer .toast").map((toast) => toast.textContent.replace(/\u00d7$/, ""));
  page.csvDownloads = () => Promise.all(page.downloads.filter((item) => item.kind === "blob").map((item) => item.blob.text()));

  for (const file of SCRIPT_ORDER) {
    vm.runInContext(fs.readFileSync(path.join(staticDir, file), "utf8"), context, { filename: file });
  }
  page.flushTimers();
  return page;
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolveFn, rejectFn) => { resolve = resolveFn; reject = rejectFn; });
  return { promise, resolve, reject };
}

// Loads a dataset payload through the landing page's example button.
async function loadDataset(page, payload) {
  const previous = page.fetchHandler;
  page.fetchHandler = (request) => (request.url.endsWith("/api/load-example")
    ? { status: 200, body: payload }
    : (previous ? previous(request) : { status: 500, body: { detail: `unexpected ${request.url}` } }));
  page.run("refs.loadExampleButton.click()");
  await page.settle();
  page.fetchHandler = previous;
}

// The page's own code catches its rejections; one that slips through is reported, not fatal.
process.on("unhandledRejection", (reason) => {
  console.error("unhandled rejection:", reason && reason.stack ? reason.stack : reason);
});

(async () => {
  const fixtures = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
  const page = createPage(fixtures.staticDir, fixtures.template);
  const body = new Function("page", "fixtures", "deferred", "loadDataset", "jsonResponse", `return (async () => {\n${fixtures.script}\n})();`);
  const result = await body(page, fixtures, deferred, loadDataset, jsonResponse);
  process.stdout.write(`\n@@RESULT@@${JSON.stringify(result === undefined ? null : result)}`);
})().catch((error) => {
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});
"""


def _run_page(tmp_path: Path, script: str, **fixtures) -> object:
    """Run `script` (the body of an async function of page, fixtures, deferred, loadDataset, jsonResponse)."""
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS_JS, encoding="utf-8")
    # The browser receives rendered HTML, including the local/hosted configuration.
    # Reading raw Jinja directives would incorrectly enable optional hosted fields.
    from jinja2 import Environment, FileSystemLoader
    rendered_template = tmp_path / "rendered-template.html"
    rendered_template.write_text(
        Environment(loader=FileSystemLoader(str(_TEMPLATE.parent)), autoescape=True)
        .get_template(_TEMPLATE.name).render(static_version="test"), encoding="utf-8",
    )
    data = tmp_path / "fixtures.json"
    data.write_text(
        json.dumps({"staticDir": str(_STATIC_DIR), "template": str(rendered_template), "script": script, **fixtures}),
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(harness), str(data)], capture_output=True, text=True, encoding="utf-8", timeout=120
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout.rsplit("@@RESULT@@", 1)[1])


@pytest.fixture(scope="module")
def example_dataset() -> dict:
    return client.post("/api/load-example").json()


def _test_predictions(models: list[str], seed: int) -> dict:
    rng = random.Random(seed)
    return {
        "row_ids": [str(index) for index in range(40)],
        "time": [float(index % 17 + 1) for index in range(40)],
        "event": [index % 3 != 0 for index in range(40)],
        "risk": {model: [rng.random() - index / 40 for index in range(40)] for model in models},
    }


def _summary(headline: str) -> dict:
    return {"status": "review", "headline": headline, "strengths": [], "cautions": [], "next_steps": []}


@pytest.fixture(scope="module")
def compare_payloads() -> dict:
    """ML and DL Compare All answers on one shared holdout split, and the server's intervals for them."""
    ml_predictions = _test_predictions(["Random Survival Forest", "LASSO-Cox", "Cox PH"], seed=1)
    dl_predictions = _test_predictions(["DeepHit", "DeepSurv"], seed=2)
    ml = {
        "comparison_table": [
            {"model": "Random Survival Forest", "c_index": 0.714, "evaluation_mode": "holdout", "rank": 1},
            {"model": "LASSO-Cox", "c_index": 0.681, "evaluation_mode": "holdout", "rank": 2},
            {"model": "Cox PH", "c_index": 0.672, "evaluation_mode": "holdout", "rank": 3},
        ],
        "evaluation_mode": "holdout",
        "evaluation_split_fingerprint": "holdout-seed42",
        "test_predictions": ml_predictions,
        "scientific_summary": _summary("ML done"),
        "manuscript_tables": {"model_performance_table": []},
    }
    dl = {
        "comparison_table": [
            {"model": "DeepHit", "c_index": 0.731, "evaluation_mode": "holdout", "rank": 1},
            {"model": "DeepSurv", "c_index": 0.703, "evaluation_mode": "holdout", "rank": 2},
        ],
        "evaluation_mode": "holdout",
        "evaluation_split_fingerprint": "holdout-seed42",
        "test_predictions": dl_predictions,
        "scientific_summary": _summary("DL done"),
        "manuscript_tables": {"model_performance_table": []},
    }
    intervals = client.post("/api/model-comparison-intervals", json={"predictions": [ml_predictions, dl_predictions]})
    assert intervals.status_code == 200, intervals.text
    return {"ml": ml, "dl": dl, "intervals": intervals.json()}


@pytest.fixture(scope="module")
def marker_payloads(example_dataset) -> dict:
    """Real marker evaluations of the example cohort (added-value and marginal lens), quick settings."""
    request = {
        "dataset_id": example_dataset["dataset_id"],
        "time_column": "os_months",
        "event_column": "os_event",
        "event_positive_value": 1,
        "marker_columns": ["biomarker_score", "immune_index"],
        "clinical_columns": ["age", "stage"],
        "categorical_clinical": ["stage"],
        "n_permutations": 49,
        "n_resamples": 8,
        "random_seed": 7,
    }
    added = client.post("/api/marker-evaluation", json=request)
    marginal = client.post("/api/marker-evaluation", json={**request, "clinical_columns": [], "categorical_clinical": []})
    assert added.status_code == 200, added.text
    assert marginal.status_code == 200, marginal.text
    return {"added": added.json(), "marginal": marginal.json()}


# ── Result currency and requests ────────────────────────────────


def test_trimmed_event_value_and_time_unit_keep_a_result_current(tmp_path: Path, example_dataset: dict) -> None:
    """H1: the server trims text settings when it echoes them; the page must not call the result stale."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      // An event value read with a stray space, and a time unit typed with one.
      page.run(`{ const option = document.createElement("option"); option.value = " 1"; option.textContent = " 1";
        refs.eventPositiveValue.appendChild(option); refs.eventPositiveValue.value = " 1"; }`);
      page.change("#timeUnitLabel", "Months ");
      page.fetchHandler = (request) => {
        const body = request.json();
        const echo = { ...body, event_positive_value: String(body.event_positive_value).trim(), time_unit_label: String(body.time_unit_label).trim() };
        return { status: 200, body: {
          analysis: { summary_table: [{ Group: "Overall", N: 360 }], risk_table: { rows: [], columns: [] }, pairwise_table: [], cohort: { n: 360 } },
          figure: { data: [{ x: [0, 1], y: [1, 0.5] }], layout: {} },
          request_config: echo,
        } };
      };
      page.run("refs.runKmButton.click()");
      await page.settle();
      const sent = page.requests.find((request) => request.url.endsWith("/api/kaplan-meier")).json();
      return {
        sentEventValue: sent.event_positive_value,
        sentTimeUnit: sent.time_unit_label,
        current: page.run("Boolean(currentGoalResult('km'))"),
        exportEnabled: !page.run("refs.downloadKmSummaryButton.disabled"),
        status: page.run("document.querySelector('[data-run-status=km]').textContent"),
      };
    """, dataset=example_dataset)

    assert result == {
        "sentEventValue": "1",
        "sentTimeUnit": "Months",
        "current": True,
        "exportEnabled": True,
        "status": "Up to date",
    }


def test_blank_numeric_fields_use_the_same_defaults_in_requests_and_currency(tmp_path: Path, example_dataset: dict) -> None:
    """H7: a blank field means its default everywhere, so ML and DL share the seed and results stay current."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.change("#dlRandomSeed", "");
      page.change("#dlEarlyStoppingMinDelta", "");
      page.change("#logrankWeight", "fleming_harrington");
      page.change("#fhPower", "");
      page.change("#riskTablePoints", "");
      page.fetchHandler = (request) => ({ status: 200, body: {
        analysis: { summary_table: [], risk_table: { rows: [], columns: [] }, pairwise_table: [], cohort: { n: 360 } },
        figure: { data: [{ x: [0], y: [1] }], layout: {} },
        request_config: request.json(),
      } });
      page.run("refs.runKmButton.click()");
      await page.settle();
      const km = page.requests.find((request) => request.url.endsWith("/api/kaplan-meier")).json();
      const dl = page.run("dlArchitectureRequestFields('deepsurv')");
      const ml = page.run("mlModelRequestFields('rsf')");
      return {
        mlSeed: ml.random_state,
        dlSeed: dl.random_seed,
        minDelta: dl.early_stopping_min_delta,
        fhPower: km.fh_p,
        riskTicks: km.risk_table_points,
        kmCurrent: page.run("Boolean(currentGoalResult('km'))"),
        dlValid: page.run("(() => { try { validateDlControls(); return true; } catch (error) { return error.message; } })()"),
      };
    """, dataset=example_dataset)

    assert result == {
        "mlSeed": 42,
        "dlSeed": 42,
        "minDelta": 0.0001,
        "fhPower": 1,
        "riskTicks": 6,
        "kmCurrent": True,
        "dlValid": True,
    }


def _signature_script(extra: str) -> str:
    return r"""
      await loadDataset(page, fixtures.dataset);
      page.run("activateTab('markers')");
      page.fetchHandler = (request) => {
        if (!request.url.endsWith("/api/discover-signature")) return { status: 500, body: { detail: "unexpected" } };
        const columns = [...fixtures.dataset.columns, { name: "sig_best", kind: "categorical", n_unique: 2, unique_preview: ["Signature+", "Signature-"], missing: 0, non_missing: 360 }];
        return { status: 200, body: {
          ...fixtures.dataset,
          dataset_id: "signature-snapshot",
          columns,
          derived_column: "sig_best",
          signature_request_config: request.json(),
          signature_analysis: {
            results_table: [{ Signature: "biomarker_score>=1.2", "P value": 0.0004, "Bootstrap support (p<alpha)": 0.0004 }],
            best_split: { Signature: "biomarker_score>=1.2", "BH adjusted p": 0.003, "Statistically significant": true },
            search_space: { tested_combinations: 12, significant_signatures: 3, combination_operator: "mixed", random_seed: 20260311, significance_level: 0.05 },
            derived_group: { auto_apply_recommended: true, outcome_informed: true },
            scientific_summary: { status: "review", headline: "Exploratory search.", strengths: [], cautions: [], next_steps: [] },
          },
        } };
      };
      page.run("refs.runSignatureSearchButton.click()");
      await page.settle();
    """ + extra


def test_signature_csv_is_exportable_after_discover_on_the_markers_tab(tmp_path: Path, example_dataset: dict) -> None:
    """H2/I3/H11: the search sends the Markers tab's candidates and its result stays current for export."""
    result = _run_page(tmp_path, _signature_script(r"""
      const sent = page.requests.find((request) => request.url.endsWith("/api/discover-signature")).json();
      const afterRun = {
        sentCandidates: sent.candidate_columns,
        csvEnabled: !page.run("refs.downloadSignatureButton.disabled"),
        tab: page.run("activeTabName()"),
        summary: page.run("refs.signatureSummary.textContent"),
        summaryCardShown: !page.run("refs.signatureSummary.closest('.table-card').classList.contains('result-hidden')"),
        rankingCell: page.run("refs.signatureShell.querySelectorAll('td')[2].textContent"),
        toasts: page.toasts(),
      };
      // The Cox covariates are not the search's candidates: changing them keeps the result exportable.
      page.run("setCheckedValues(refs.covariateChecklist, ['age']); renderSharedFeatureSummary();");
      const afterCoxChange = !page.run("refs.downloadSignatureButton.disabled");
      // Unticking a marker changes the candidates, so the export waits for a new search.
      page.run("setCheckedValues(refs.markerChecklist, ['biomarker_score']); renderMarkerSelectionLine(); syncDownloadButtonAvailability();");
      const afterMarkerChange = !page.run("refs.downloadSignatureButton.disabled");
      return { ...afterRun, afterCoxChange, afterMarkerChange };
    """), dataset=example_dataset)

    assert result["sentCandidates"] == ["biomarker_score", "immune_index", "age", "sex", "stage", "treatment"]
    assert result["csvEnabled"] is True
    assert result["tab"] == "markers"
    assert "Best signature" in result["summary"] and "biomarker_score>=1.2" in result["summary"]
    assert result["summaryCardShown"] is True
    # "Bootstrap support (p<alpha)" is a proportion, not a p-value.
    assert result["rankingCell"] == "4.00e-4"
    assert "Signature discovery complete. Group by switched to sig_best." in result["toasts"]
    assert result["afterCoxChange"] is True
    assert result["afterMarkerChange"] is False


def test_changing_the_endpoint_clears_outcome_informed_grouping_outputs(tmp_path: Path, example_dataset: dict) -> None:
    """H9: the search card and an optimal-cutpoint scan were computed for the old endpoint."""
    result = _run_page(tmp_path, _signature_script(r"""
      page.run("refs.cutpointPlot.classList.remove('hidden'); Plotly.newPlot(refs.cutpointPlot, [{ x: [1], y: [2] }], {})");
      const before = page.run("refs.signatureSummary.innerHTML.length > 0");
      page.change("#eventColumn", "pfs_event");
      return {
        before,
        summary: page.run("refs.signatureSummary.innerHTML"),
        cutpointHidden: page.run("refs.cutpointPlot.classList.contains('hidden')"),
        cutpointData: page.run("Boolean(refs.cutpointPlot.data)"),
      };
    """), dataset=example_dataset)

    assert result == {"before": True, "summary": "", "cutpointHidden": True, "cutpointData": False}


# ── Outcome pickers ─────────────────────────────────────────────


def test_time_picker_leaves_time_blank_and_offers_every_numeric_column(tmp_path: Path, example_dataset: dict) -> None:
    """H3: no ID column preselected; "All numeric" lets a real follow-up column be chosen with a warning."""
    result = _run_page(tmp_path, r"""
      const base = fixtures.dataset;
      const recordId = { name: "record_id", kind: "numeric", n_unique: 360, unique_preview: [1, 2, 3], missing: 0, non_missing: 360 };
      await loadDataset(page, {
        ...base,
        dataset_id: "no-likely-time",
        columns: [recordId, ...base.columns],
        numeric_columns: ["record_id", ...base.numeric_columns],
        suggestions: { ...base.suggestions, time_columns: [] },
      });
      const withoutLikely = { value: page.run("refs.timeColumn.value"), options: page.run("refs.timeColumn.options.map((o) => o.value)") };
      await loadDataset(page, { ...base, dataset_id: "one-likely-time", suggestions: { ...base.suggestions, time_columns: ["pfs_months"] } });
      const withLikely = { value: page.run("refs.timeColumn.value"), options: page.run("refs.timeColumn.options.map((o) => o.value)") };
      page.change("#showAllTimeColumns", true);
      const allNumeric = page.run("refs.timeColumn.options.map((o) => o.value)");
      page.change("#timeColumn", "os_months");
      return {
        withoutLikely,
        withLikely,
        allNumeric,
        chosen: page.run("refs.timeColumn.value"),
        warning: page.run("refs.timeColumnWarning.textContent"),
        warningClass: page.run("refs.timeColumnWarning.className"),
        runnable: page.run("(() => { try { return currentBaseConfig().time_column; } catch (error) { return error.message; } })()"),
      };
    """, dataset=example_dataset)

    assert result["withoutLikely"]["value"] == ""
    assert result["withoutLikely"]["options"][:2] == ["", "record_id"]
    assert result["withLikely"] == {"value": "pfs_months", "options": ["pfs_months"]}
    assert {"os_months", "pfs_months", "age"} <= set(result["allNumeric"])
    assert result["chosen"] == "os_months"
    assert "not one of the likely follow-up time columns" in result["warning"]
    assert "event-warning-warning" in result["warningClass"]
    assert result["runnable"] == "os_months"


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["No recurrence", "Recurrence"], "Recurrence"),
        (["Recurrence", "No recurrence"], "Recurrence"),
        (["0:No recurrence", "1:Recurrence"], "1:Recurrence"),
        (["No progression", "Progression"], "Progression"),
        (["Relapse-free", "Relapse"], "Relapse"),
        (["Alive", "Dead"], "Dead"),
        ([0, 1], "1"),
        (["No recurrence"], ""),
        (["Censored", "Not censored"], ""),
    ],
)
def test_event_value_is_never_a_negated_label(tmp_path: Path, values: list, expected: str) -> None:
    """H8: "No recurrence" contains an event word but names the censored side."""
    result = _run_page(tmp_path, r"""
      page.context.__values = fixtures.values;
      return page.run("inferEventPositiveSelection('status', __values)");
    """, values=values)

    assert result["value"] == expected
    if not expected:
        assert result["warning"]


# ── Tables ──────────────────────────────────────────────────────


def test_table_one_and_the_data_preview_show_data_labels_as_they_are(tmp_path: Path, example_dataset: dict) -> None:
    """H4: group levels and column names from the data are never p-values or rewritten headers."""
    dataset = json.loads(json.dumps(example_dataset))
    dataset["columns"] += [
        {"name": "ki67_p", "kind": "numeric", "n_unique": 300, "unique_preview": [], "missing": 0, "non_missing": 360},
        {"name": "7157", "kind": "numeric", "n_unique": 300, "unique_preview": [], "missing": 0, "non_missing": 360},
    ]
    for index, row in enumerate(dataset["preview"]):
        row["ki67_p"] = [12.5, 0.0002, -1.3][index % 3]
        row["7157"] = 2.1
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const previewHeaders = page.run("refs.datasetPreviewShell.querySelectorAll('th').map((th) => th.textContent)");
      const ki67Index = previewHeaders.indexOf("ki67_p");
      const previewKi67 = page.run(`refs.datasetPreviewShell.querySelectorAll('tbody tr').slice(0, 3).map((tr) => tr.children[${ki67Index}].textContent)`);
      page.fetchHandler = (request) => ({ status: 200, body: {
        analysis: {
          columns: ["Variable", "Statistic", "Overall (grouped subset)", "Test positive", "pT1a"],
          rows: [
            { Variable: "Cohort size", Statistic: "N", "Overall (grouped subset)": 150, "Test positive": 45, pT1a: 105 },
            { Variable: "age", Statistic: "Mean ± SD | Median [IQR]", "Overall (grouped subset)": "52.10 ± 9.80 | 51.00 [45.00, 60.00]", "Test positive": "45 (30.0%)", pT1a: 0.0004 },
          ],
          row_mask_hash: "rows",
        },
        request_config: request.json(),
      } });
      page.run("activateTab('tables'); refs.runCohortTableButton.click()");
      await page.settle();
      return {
        previewHeaders,
        previewKi67,
        tableHeaders: page.run("refs.cohortTableShell.querySelectorAll('th').map((th) => th.textContent)"),
        tableCells: page.run("refs.cohortTableShell.querySelectorAll('tbody tr').map((tr) => tr.children.map((td) => td.textContent))"),
        labels: page.run(`["Test positive", "raw pathology", "latest procedure", "p16", "Family-wise P", "Logrank p", "Global PH pvalue",
          "Selection-adjusted p-value", "Bootstrap support (p<alpha)", "Replication P (Holm)", "Permutation p (search-adjusted)"].map(isPValueLikeLabel)`),
      };
    """, dataset=dataset)

    assert result["previewHeaders"][-2:] == ["ki67_p", "7157"]
    assert result["previewHeaders"][:2] == ["patient_id", "os_months"]
    # The preview shows the file's values as they are (tests/test_frontend_review2.py: no rounding either).
    assert result["previewKi67"] == ["12.5", "0.0002", "-1.3"]
    assert result["tableHeaders"] == ["Variable", "Statistic", "Overall (grouped subset)", "Test positive", "pT1a"]
    assert result["tableCells"] == [
        ["Cohort size", "N", "150", "45", "105"],
        ["age", "Mean ± SD | Median [IQR]", "52.10 ± 9.80 | 51.00 [45.00, 60.00]", "45 (30.0%)", "4.00e-4"],
    ]
    assert result["labels"] == [False, False, False, False, True, True, True, True, False, True, True]


def test_checklist_notes_and_table_labels_ignore_object_prototype_names(tmp_path: Path) -> None:
    """H17: a column named "constructor" or "toString" gets no note or label from Object.prototype."""
    result = _run_page(tmp_path, r"""
      page.run(`renderChecklist(refs.markerChecklist, ["constructor", "toString", "age"], [], new Map([["age", "30% missing"]]));
        renderChecklist(refs.markerClinicalChecklist, ["constructor"], [], {});
        renderTable(refs.markersTableShell, [{ constructor: 1, toString: 2 }], ["constructor", "toString"], { rawHeaders: true });`);
      return {
        markerNotes: page.run("refs.markerChecklist.querySelectorAll('.check-item-note').map((note) => note.textContent)"),
        clinicalNotes: page.run("refs.markerClinicalChecklist.querySelectorAll('.check-item-note').length"),
        headers: page.run("refs.markersTableShell.querySelectorAll('th').map((th) => th.textContent)"),
        cells: page.run("refs.markersTableShell.querySelectorAll('td').map((td) => td.textContent)"),
      };
    """)

    assert result == {"markerNotes": ["30% missing"], "clinicalNotes": 0, "headers": ["constructor", "toString"], "cells": ["1", "2"]}


# ── Busy scopes and cancelled work ──────────────────────────────


def _ml_single_handler() -> str:
    return r"""
      page.fetchHandler = (request) => {
        if (!request.url.endsWith("/api/ml-model")) return { status: 500, body: { detail: "unexpected" } };
        return { status: 200, body: {
          importance_figure: { data: [{ type: "bar", x: [0.3], y: ["age"] }], layout: { height: 360 } },
          analysis: { model_stats: { c_index: 0.7, evaluation_mode: "holdout", n_patients: 360, n_features: 4 } },
          request_config: request.json(),
        } };
      };
    """


def test_ml_plots_are_marked_stale_not_deleted_while_settings_differ(tmp_path: Path, example_dataset: dict) -> None:
    """H5/I8: editing a setting (or switching the model) and undoing it brings the same plots back."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
    """ + _ml_single_handler() + r"""
      page.run("activateTab('benchmark'); reviewBenchmarkModel('rsf', 'single')");
      page.run("refs.runPredictiveWorkbenchButton.click()");
      await page.settle();
      const snapshot = () => page.run(`({ traces: (refs.mlImportancePlot.data || []).length, stale: refs.mlImportancePlot.classList.contains("plot-stale"),
        message: refs.mlImportancePlot.dataset.staleMessage || "" })`);
      const trained = snapshot();
      page.change("#mlNEstimators", "1000");
      const edited = snapshot();
      page.change("#mlNEstimators", "100");
      const undone = snapshot();
      page.change("#predictiveModelSelector", "gbs");
      page.change("#predictiveModelSelector", "rsf");
      const switchedBack = snapshot();
      return { trained, edited, undone, switchedBack };
    """, dataset=example_dataset)

    assert result["trained"] == {"traces": 1, "stale": False, "message": ""}
    assert result["edited"]["traces"] == 1 and result["edited"]["stale"] is True
    assert "Run Analysis to refresh feature importance" in result["edited"]["message"]
    assert result["undone"] == {"traces": 1, "stale": False, "message": ""}
    assert result["switchedBack"] == {"traces": 1, "stale": False, "message": ""}


def test_ctrl_enter_takes_the_same_busy_scope_as_a_click(tmp_path: Path, example_dataset: dict) -> None:
    """H6: Compare All and the workbench run under their scopes when started from the keyboard."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const hold = deferred();
      page.fetchHandler = () => hold.promise.then(() => ({ status: 500, body: { detail: "stopped" } }));
      page.run("activateTab('benchmark')");
      page.keydown({ key: "Enter", ctrlKey: true });
      await page.settle(3);
      const compareAll = page.run("({ predictive: isScopeBusy('predictive'), ml: isScopeBusy('ml'), button: refs.runPredictiveCompareAllButton.disabled })");
      hold.resolve();
      await page.settle();
      const hold2 = deferred();
      page.fetchHandler = () => hold2.promise.then(() => ({ status: 500, body: { detail: "stopped" } }));
      page.run("reviewBenchmarkModel('deepsurv', 'single')");
      page.keydown({ key: "Enter", ctrlKey: true });
      await page.settle(3);
      const workbench = page.run("({ dl: isScopeBusy('dl'), button: refs.runPredictiveWorkbenchButton.disabled })");
      hold2.resolve();
      await page.settle();
      return { compareAll, workbench, anyBusyAfter: page.run("Object.values(runtime.busyScopes).some(Boolean)") };
    """, dataset=example_dataset)

    assert result["compareAll"] == {"predictive": True, "ml": True, "button": True}
    assert result["workbench"] == {"dl": True, "button": True}
    assert result["anyBusyAfter"] is False


def test_going_home_cancels_runs_and_clears_the_cohort_banner(tmp_path: Path, example_dataset: dict) -> None:
    """H12: a running ML job is cancelled and the cohort-mismatch banner does not stay on the landing page."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.fetchHandler = () => deferred().promise;
      page.run("activateTab('benchmark'); reviewBenchmarkModel('rsf', 'single'); refs.runPredictiveWorkbenchButton.click()");
      await page.settle(3);
      const mlRequest = page.requests.find((request) => request.url.endsWith("/api/ml-model"));
      page.run("setAnalysisConsistencyBanner('Loaded analyses currently use different analyzable cohorts.', 'warning')");
      page.run("goHome({ syncHistory: false })");
      await page.settle();
      return {
        aborted: mlRequest.signal.aborted,
        mlBusy: page.run("isScopeBusy('ml')"),
        banner: page.run("refs.analysisConsistencyBanner.className"),
        dataset: page.run("state.dataset"),
      };
    """, dataset=example_dataset)

    assert result == {"aborted": True, "mlBusy": False, "banner": "runtime-banner hidden", "dataset": None}


def test_only_a_404_for_the_open_dataset_closes_the_workspace(tmp_path: Path, example_dataset: dict) -> None:
    """H13: a 404 for another dataset (old snapshot, history entry) is an ordinary error."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const current = page.run("state.dataset.dataset_id");
      page.fetchHandler = (request) => ({ status: 404, body: { detail: `Unknown dataset id: ${request.url.split("/").pop()}` } });
      const other = await page.run("fetchJSON('/api/dataset/0123abcdef').then(() => 'ok', (error) => error.message)");
      const openAfterOther = page.run("Boolean(state.dataset)");
      const own = await page.run(`fetchJSON('/api/dataset/${current}').then(() => 'ok', (error) => error.message)`);
      return { other, openAfterOther, own, openAfterOwn: page.run("Boolean(state.dataset)") };
    """, dataset=example_dataset)

    assert result["other"] == "Unknown dataset id: 0123abcdef"
    assert result["openAfterOther"] is True
    assert "no longer available" in result["own"]
    assert result["openAfterOwn"] is False


def test_a_history_entry_whose_cohort_is_gone_goes_home_with_a_reason(tmp_path: Path, example_dataset: dict) -> None:
    """H13: Back to a page of an expired cohort cannot show it; the landing page says why."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.fetchHandler = () => ({ status: 404, body: { detail: "Unknown dataset id: 0123abcdef" } });
      await page.run("restoreHistoryState({ view: 'workspace', datasetId: '0123abcdef', tab: 'km' })");
      return { dataset: page.run("state.dataset"), banner: page.run("refs.runtimeBanner.textContent") };
    """, dataset=example_dataset)

    assert result["dataset"] is None
    assert result["banner"] == "That page could not be restored: Unknown dataset id: 0123abcdef. Load the cohort again to continue."


def test_validate_and_marker_buttons_keep_their_own_conditions_after_a_busy_cycle(tmp_path: Path, example_dataset: dict) -> None:
    """H14/I19: a finished run does not re-enable Validate without a model, or the list buttons with a file."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const cycle = () => page.run("setScopeBusy('markers', true, refs.runMarkersButton); setScopeBusy('markers', false, refs.runMarkersButton);");
      const validateBefore = page.run("refs.runMarkerValidationButton.disabled");
      cycle();
      const validateAfter = page.run("refs.runMarkerValidationButton.disabled");
      page.run("state.markerMatrix = { matrix_id: 'm1', filename: 'expr.tsv', n_markers: 20, n_matched: 300, n_patients: 360, id_column: 'patient_id' }; renderMarkerMatrixState();");
      cycle();
      const listButtonsWithFile = page.run("[refs.selectAllMarkersButton.disabled, refs.clearMarkersButton.disabled]");
      page.run("state.markerMatrix = null; setScopeBusy('markers', true, refs.runMarkersButton); renderMarkerMatrixState();");
      const listButtonsWhileBusy = page.run("[refs.selectAllMarkersButton.disabled, refs.clearMarkersButton.disabled]");
      page.run("setScopeBusy('markers', false, refs.runMarkersButton)");
      // A re-render while a marker file is still uploading keeps Attach off.
      page.run("setButtonLoading(refs.attachMarkerMatrixButton, true); refreshMarkerSelections();");
      const attachWhileUploading = page.run("refs.attachMarkerMatrixButton.disabled");
      page.run("setButtonLoading(refs.attachMarkerMatrixButton, false); renderMarkerMatrixState();");
      return {
        validateBefore,
        validateAfter,
        listButtonsWithFile,
        listButtonsWhileBusy,
        listButtonsIdle: page.run("[refs.selectAllMarkersButton.disabled, refs.clearMarkersButton.disabled]"),
        attachWhileUploading,
        attachIdle: page.run("refs.attachMarkerMatrixButton.disabled"),
        validateScope: page.run("runScopeForButton(refs.runMarkerValidationButton)"),
      };
    """, dataset=example_dataset)

    assert result == {
        "validateBefore": True,
        "validateAfter": True,
        "listButtonsWithFile": [True, True],
        "listButtonsWhileBusy": [True, True],
        "listButtonsIdle": [False, False],
        "attachWhileUploading": True,
        "attachIdle": False,
        "validateScope": "markers",
    }


def test_create_stays_disabled_while_a_grouping_is_being_created(tmp_path: Path, example_dataset: dict) -> None:
    """H15: re-syncing the derive controls mid-run, or finishing with Group by set, leaves Create off."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const hold = deferred();
      page.fetchHandler = (request) => (request.url.endsWith("/api/derive-group") ? hold.promise : { status: 500, body: { detail: "unexpected" } });
      page.run("activateTab('cox'); refs.derivePanel.classList.remove('hidden'); syncDeriveToggleButton();");
      page.run("refs.deriveButton.click()");
      await page.settle(3);
      page.change("#deriveMethod", "tertile_split");
      const whileRunning = page.run("refs.deriveButton.disabled");
      const columns = [...fixtures.dataset.columns, { name: "age_group", kind: "categorical", n_unique: 2, unique_preview: ["Low", "High"], missing: 0, non_missing: 360 }];
      hold.resolve({ status: 200, body: { ...fixtures.dataset, dataset_id: "derived-snapshot", columns, derived_column: "age_group",
        derive_summary: { method: "median_split", counts: [{ group: "Low", n: 180 }, { group: "High", n: 180 }] } } });
      await page.settle();
      return { whileRunning, group: page.run("refs.groupColumn.value"), afterRun: page.run("refs.deriveButton.disabled"), busy: page.run("isScopeBusy('derive')") };
    """, dataset=example_dataset)

    assert result == {"whileRunning": True, "group": "age_group", "afterRun": True, "busy": False}


def test_a_newer_cox_preview_cancels_the_one_in_flight(tmp_path: Path, example_dataset: dict) -> None:
    """H16: superseded preview requests are aborted, not just ignored."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.fetchHandler = () => deferred().promise;
      page.requests.length = 0;
      page.run("refreshCoxPreview({ force: true })");
      await page.settle(2);
      page.run("refreshCoxPreview({ force: true })");
      await page.settle(2);
      const previews = page.requests.filter((request) => request.url.endsWith("/api/cox-preview"));
      return previews.map((request) => request.signal.aborted);
    """, dataset=example_dataset)

    assert result[-2:] == [True, False]


def test_an_invalid_run_does_not_cancel_the_run_in_flight(tmp_path: Path, example_dataset: dict) -> None:
    """I18: a run checks its inputs before it cancels the previous request of its kind."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      page.fetchHandler = () => deferred().promise;
      page.run("runKaplanMeier(); runMlModel();");
      await page.settle(3);
      const km = page.requests.find((request) => request.url.endsWith("/api/kaplan-meier"));
      const ml = page.requests.find((request) => request.url.endsWith("/api/ml-model"));
      // Group by the endpoint's own time column, and too few trees: both runs refuse to start.
      page.run("refs.groupColumn.value = 'os_months'; refs.mlNEstimators.value = '5';");
      const kmError = await page.run("runKaplanMeier().then(() => 'ran', (error) => error.message)");
      const mlError = await page.run("runMlModel().then(() => 'ran', (error) => error.message)");
      return {
        kmError,
        mlError,
        kmAborted: km.signal.aborted,
        mlAborted: ml.signal.aborted,
        requests: page.requests.filter((request) => /kaplan-meier|ml-model/.test(request.url)).length,
      };
    """, dataset=example_dataset)

    assert "part of the survival endpoint" in result["kmError"]
    assert result["mlError"].startswith("Trees must be an integer between 10 and 1000.")
    assert result["kmAborted"] is False
    assert result["mlAborted"] is False
    assert result["requests"] == 2


def test_compare_all_stops_when_its_dataset_is_replaced(tmp_path: Path, example_dataset: dict) -> None:
    """I4: a cancelled ML phase starts no DL run on the new snapshot and does not pull the user away."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const columns = [...fixtures.dataset.columns, { name: "age_group", kind: "categorical", n_unique: 2, unique_preview: ["Low", "High"], missing: 0, non_missing: 360 }];
      page.fetchHandler = (request) => {
        if (request.url.endsWith("/api/ml-model")) return deferred().promise;
        if (request.url.endsWith("/api/derive-group")) {
          return { status: 200, body: { ...fixtures.dataset, dataset_id: "derived-snapshot", columns, derived_column: "age_group",
            derive_summary: { method: "median_split", counts: [] } } };
        }
        return { status: 500, body: { detail: "unexpected" } };
      };
      page.run("activateTab('benchmark'); refs.runPredictiveCompareAllButton.click()");
      await page.settle(3);
      // While ML runs, the user makes a grouping on the Cox tab: a new dataset snapshot.
      page.run("activateTab('cox'); refs.derivePanel.classList.remove('hidden'); refs.deriveButton.click()");
      await page.settle(40);
      return {
        deepRequests: page.requests.filter((request) => request.url.endsWith("/api/deep-model")).length,
        tab: page.run("activeTabName()"),
        ml: page.run("state.ml"),
        dataset: page.run("state.dataset.dataset_id"),
        busy: page.run("Object.entries(runtime.busyScopes).filter(([, busy]) => busy).map(([scope]) => scope)"),
      };
    """, dataset=example_dataset)

    assert result == {"deepRequests": 0, "tab": "cox", "ml": None, "dataset": "derived-snapshot", "busy": []}


def test_ml_compare_does_not_clear_a_newer_banner(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """I17: a run clears only the banner it set."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const hold = deferred();
      page.fetchHandler = (request) => hold.promise.then(() => ({ status: 200, body: { analysis: fixtures.compare.ml, request_config: request.json() } }));
      page.run("runCompareModels()");
      await page.settle(3);
      const during = page.run("refs.runtimeBanner.textContent");
      page.run("setRuntimeBanner('Uploading external.csv', 'info')");
      hold.resolve();
      await page.settle();
      return { during, after: page.run("refs.runtimeBanner.textContent") };
    """, dataset=example_dataset, compare=compare_payloads)

    assert result["during"].startswith("Screening Cox PH")
    assert result["after"] == "Uploading external.csv"


def test_a_rejected_upload_removes_its_uploading_banner(tmp_path: Path) -> None:
    """I14: the "Uploading" banner goes when the server rejects the file."""
    result = _run_page(tmp_path, r"""
      page.run("refs.datasetFile.files = [{ name: 'notes.docx' }]");
      page.fetchHandler = () => ({ status: 400, body: { detail: "Unsupported input file extension '.docx' for 'notes.docx'." } });
      page.change("#datasetFile");
      await page.settle();
      return { banner: page.run("refs.runtimeBanner.className"), toasts: page.toasts() };
    """)

    assert result["banner"] == "runtime-banner hidden"
    assert any("Unsupported input file extension" in toast for toast in result["toasts"])


def test_a_failed_image_export_is_reported(tmp_path: Path) -> None:
    """I20: Plotly's rejected download promise is caught and shown."""
    result = _run_page(tmp_path, r"""
      page.plotly.downloadImageError = new Error("image export is unavailable");
      page.run("Plotly.newPlot(refs.kmPlot, [{ x: [1], y: [1] }], {})");
      await page.run("downloadPlotImage(refs.kmPlot, 'km_curve', 'png')");
      await page.settle(2);
      return page.toasts();
    """)

    assert result == ["Saving the PNG image failed: image export is unavailable"]


# ── Prediction leaderboard ──────────────────────────────────────


def _compare_all_script(extra: str) -> str:
    return r"""
      await loadDataset(page, fixtures.dataset);
      let intervalCalls = 0;
      page.fetchHandler = (request) => {
        if (request.url.endsWith("/api/ml-model")) return { status: 200, body: { analysis: fixtures.compare.ml, request_config: request.json() } };
        if (request.url.endsWith("/api/deep-model")) return { status: 200, body: { analysis: fixtures.compare.dl, request_config: request.json() } };
        if (request.url.endsWith("/api/model-comparison-intervals")) {
          intervalCalls += 1;
          return fixtures.failIntervals && intervalCalls === 1
            ? { status: 503, body: { detail: "busy" } }
            : { status: 200, body: fixtures.compare.intervals };
        }
        return { status: 500, body: { detail: "unexpected" } };
      };
      page.run("activateTab('benchmark'); refs.runPredictiveCompareAllButton.click()");
      await page.settle(60);
      const headers = () => page.run("refs.benchmarkComparisonShell.querySelectorAll('th').map((th) => th.textContent.trim())");
    """ + extra


def test_leaderboard_keeps_its_intervals_on_every_re_render(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """I2: switching tabs re-renders the leaderboard chrome; the 95% CI and ΔC columns must stay."""
    result = _run_page(tmp_path, _compare_all_script(r"""
      const afterRun = headers();
      page.run("activateTab('km'); activateTab('benchmark');");
      const afterTabs = headers();
      page.run("renderWorkspaceChrome()");
      return { afterRun, afterTabs, afterChrome: headers(), title: page.run("refs.benchmarkComparisonPlot.layout.title.text") };
    """), dataset=example_dataset, compare=compare_payloads)

    for headers in (result["afterRun"], result["afterTabs"], result["afterChrome"]):
        assert "95% CI" in headers and "ΔC vs Cox PH (95% CI)" in headers
    assert result["title"] == "C-index on the same test patients, with 95% bootstrap intervals"


def test_leaderboard_keeps_outcome_exclusions_visible_and_removal_reasons_in_detail(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    payloads = json.loads(json.dumps(compare_payloads))
    for family in ["ml", "dl"]:
        for row in payloads[family]["comparison_table"]:
            row.update(input_rows=240, excluded_outcome_rows=16)
    payloads["ml"]["comparison_table"][0]["removed_feature_notes"] = "copy: redundant in Cox training design"
    result = _run_page(tmp_path, _compare_all_script(r"""
      return {
        note: page.run("refs.benchmarkTableNote.innerHTML"),
        removal: page.run("refs.benchmarkTableNote.querySelector('details').textContent"),
      };
    """), dataset=example_dataset, compare=payloads)
    leading = result["note"].split("<details", 1)[0]
    assert leading.count("Excluded 16 of 240 input rows") == 1
    assert "missing or non-finite" in leading
    assert "copy: redundant in Cox training design" in result["removal"]


def test_failed_interval_requests_are_retried(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """I21: one failed bootstrap request does not hide the intervals for good."""
    result = _run_page(tmp_path, _compare_all_script(r"""
      const afterFailure = { status: page.run("runtime.benchmarkIntervals.status"), headers: headers() };
      page.run("renderBenchmarkBoard()");
      const beforeRetryTime = page.run("runtime.benchmarkIntervals.status");
      page.run("runtime.benchmarkIntervals.retryAt = 0; renderBenchmarkBoard();");
      await page.settle();
      return { afterFailure, beforeRetryTime, afterRetry: page.run("runtime.benchmarkIntervals.status"), headers: headers(), calls: intervalCalls };
    """), dataset=example_dataset, compare=compare_payloads, failIntervals=True)

    assert result["afterFailure"]["status"] == "error"
    assert "95% CI" not in result["afterFailure"]["headers"]
    assert result["beforeRetryTime"] == "error"
    assert result["afterRetry"] == "ready"
    assert "95% CI" in result["headers"]
    assert result["calls"] == 2


def _locked_test_compare(compare_payloads: dict, *, failed: bool) -> dict:
    """A repeated-CV comparison with a locked test set whose best locked-test model is not rank 1."""
    locked = json.loads(json.dumps(compare_payloads))
    for family, rows in (
        ("ml", [("Random Survival Forest", 0.74, 0.66), ("LASSO-Cox", 0.70, 0.71), ("Cox PH", 0.69, 0.68)]),
        ("dl", [("DeepHit", 0.72, 0.73), ("DeepSurv", 0.71, 0.65)]),
    ):
        analysis = locked[family]
        analysis["evaluation_mode"] = "repeated_cv"
        analysis.pop("test_predictions")
        analysis["comparison_table"] = [
            {
                "model": model,
                "c_index": cv,
                "evaluation_mode": "repeated_cv",
                "rank": rank,
                "locked_test_c_index": None if failed else test,
                **({"locked_test_error": "failed"} if failed else {}),
            }
            for rank, (model, cv, test) in enumerate(rows, start=1)
        ]
    return locked


def test_locked_test_chart_follows_screen_rank_and_names_only_what_it_draws(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """I9/I10: ordered by the leaderboard's rank (not locked-test values), rank 1 marked, no interval claim."""
    result = _run_page(tmp_path, _compare_all_script(r"""
      const plot = page.run(`({ title: refs.benchmarkComparisonPlot.layout.title.text, order: refs.benchmarkComparisonPlot.layout.yaxis.categoryarray,
        hidden: refs.benchmarkComparisonPlot.classList.contains('hidden') })`);
      return { ...plot, rankOne: page.run("refs.benchmarkComparisonShell.querySelectorAll('tbody tr')[0].children[2].textContent") };
    """), dataset=example_dataset, compare=_locked_test_compare(compare_payloads, failed=False))

    assert result["hidden"] is False
    assert result["title"] == "Locked-test C-index"
    assert result["rankOne"] == "Random Survival Forest"
    # Plotly lists the first category at the bottom: rank 1 is last, i.e. on top.
    assert result["order"][-1] == "Random Survival Forest (ML) · rank 1"
    assert result["order"][-2] == "DeepHit (DL)"
    assert result["order"][0] == "Cox PH (ML)"


def test_a_failed_locked_test_refit_is_a_caution_not_an_exclusion(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """An error at stage "locked_test" leaves the model ranked by CV; only other errors exclude a model."""
    compare = _locked_test_compare(compare_payloads, failed=False)
    compare["ml"]["comparison_table"][0]["locked_test_c_index"] = None
    compare["ml"]["errors"] = [
        {"model": "Random Survival Forest", "stage": "locked_test", "error": "the refit did not converge"},
        {"model": "Gradient Boosted Survival", "error": "fit failed"},
    ]
    result = _run_page(tmp_path, _compare_all_script(r"""
      const rows = page.run("refs.benchmarkComparisonShell.querySelectorAll('tbody tr').map((tr) => tr.children.map((td) => td.textContent.trim()))");
      return {
        rows,
        note: page.run("refs.benchmarkTableNote.textContent"),
        summary: page.run("refs.benchmarkSummaryGrid.textContent"),
        cellTitle: page.run("refs.benchmarkComparisonShell.querySelector('tbody tr td .benchmark-row-note')?.getAttribute('title') || ''"),
      };
    """), dataset=example_dataset, compare=compare)

    rsf = next(row for row in result["rows"] if "Random Survival Forest" in row)
    gbs = next(row for row in result["rows"] if "Gradient Boosted Survival" in row)
    assert rsf[0] == "1"
    assert any("Locked-test refit failed; ranked by CV" in cell for cell in rsf)
    assert gbs[0] == "—" and any("(Excluded)" in cell for cell in gbs)
    assert "The locked-test refit of the rank-1 model (Random Survival Forest) failed" in result["note"]
    assert "Excluded from the current Classical ML compare run: Gradient Boosted Survival." in result["summary"]
    assert "compare run: Random Survival Forest" not in result["summary"]
    assert result["cellTitle"] == "the refit did not converge"


def test_chart_shows_an_empty_state_when_no_locked_test_value_exists(tmp_path: Path, example_dataset: dict, compare_payloads: dict) -> None:
    """I9: a locked test that failed for every model must not draw an empty chart."""
    result = _run_page(tmp_path, _compare_all_script(r"""
      return {
        hidden: page.run("refs.benchmarkComparisonPlot.classList.contains('hidden')"),
        data: page.run("Boolean(refs.benchmarkComparisonPlot.data)"),
        note: page.run("refs.benchmarkPlotNote.textContent"),
      };
    """), dataset=example_dataset, compare=_locked_test_compare(compare_payloads, failed=True))

    assert result == {"hidden": True, "data": False, "note": "No model has a locked-test C-index to chart. Review the table below."}


# ── Markers ─────────────────────────────────────────────────────


def _marker_run_script(payload_key: str, extra: str) -> str:
    return r"""
      await loadDataset(page, fixtures.dataset);
      page.fetchHandler = (request) => (request.url.endsWith("/api/marker-evaluation")
        ? { status: 200, body: { ...fixtures.markers.""" + payload_key + r""", request_config: request.json() } }
        : { status: 500, body: { detail: "unexpected" } });
      page.run("activateTab('markers'); refs.runMarkersButton.click()");
      await page.settle();
    """ + extra


def test_marker_table_rank_interval_and_note_are_spreadsheet_and_lens_safe(tmp_path: Path, example_dataset: dict, marker_payloads: dict) -> None:
    """I11/I15: "1 to 2" not "1-2" (a date in Excel); the HR note matches the marginal lens."""
    result = _run_page(tmp_path, _marker_run_script("marginal", r"""
      const headers = page.run("refs.markersTableShell.querySelectorAll('th').map((th) => th.textContent)");
      const column = headers.indexOf("Rank 95% Interval");
      const intervals = page.run(`refs.markersTableShell.querySelectorAll('tbody tr').map((tr) => tr.children[${column}].textContent)`);
      page.run("downloadMarkerTable()");
      const csv = (await page.csvDownloads()).pop();
      return { intervals, csv, note: page.run("refs.markersTableNote.textContent") };
    """), dataset=example_dataset, markers=marker_payloads)

    assert result["intervals"] and all(re.fullmatch(r"\d+ to \d+", value) for value in result["intervals"])
    rows = list(csv.reader(io.StringIO(result["csv"].lstrip("\ufeff"))))
    header_index = next(index for index, row in enumerate(rows) if row and row[0] == "Marker")
    column = rows[header_index].index("Rank 95% interval")
    exported = [row[column] for row in rows[header_index + 1:]]
    assert exported == result["intervals"]
    assert "from a Cox model with the marker alone" in result["note"]
    assert "added value over the clinical covariates" not in result["note"]


def test_validation_of_an_older_locked_model_is_discarded_and_its_upload_freed(
    tmp_path: Path, example_dataset: dict, marker_payloads: dict
) -> None:
    """I5/I7/H14: Run waits while validating; a result for an older recipe is not shown; the upload is deleted."""
    result = _run_page(tmp_path, _marker_run_script("added", r"""
      const firstHash = page.run("state.markers.analysis.locked_recipe.recipe_hash");
      const hold = deferred();
      page.fetchHandler = (request) => {
        if (request.url.endsWith("/api/upload")) return { status: 200, body: { dataset_id: "external-1", filename: "external.csv" } };
        if (request.url.endsWith("/api/marker-validation")) return hold.promise;
        if (request.method === "DELETE") return { status: 200, body: { status: "deleted" } };
        return { status: 500, body: { detail: "unexpected" } };
      };
      page.run("refs.markerValidationFile.files = [{ name: 'external.csv' }]; refs.runMarkerValidationButton.click()");
      await page.settle(3);
      const runDisabledWhileValidating = page.run("refs.runMarkersButton.disabled");
      // A newer locked model replaces the one being validated.
      page.run("state.markers = { ...state.markers, analysis: { ...state.markers.analysis, locked_recipe: { ...state.markers.analysis.locked_recipe, recipe_hash: 'newer' } } }");
      hold.resolve({ status: 200, body: { validation: { recipe_hash: firstHash, cohort: { n: 360, events: 250 }, metrics: { c_index: 0.7 }, markers: [], notes: [] }, figure: null, request_config: {} } });
      await page.settle();
      const deletes = page.requests.filter((request) => request.method === "DELETE").map((request) => request.url);
      return {
        runDisabledWhileValidating,
        shown: page.run("state.markerValidation"),
        summary: page.run("refs.markerValidationSummary.textContent"),
        deletes,
        validateAvailable: !page.run("refs.runMarkerValidationButton.disabled"),
      };
    """), dataset=example_dataset, markers=marker_payloads)

    assert result["runDisabledWhileValidating"] is True
    assert result["shown"] is None
    assert result["summary"] == ""
    assert result["deletes"] == ["/api/dataset/external-1"]
    assert result["validateAvailable"] is True


def test_a_new_evaluation_cancels_the_running_validation(tmp_path: Path, example_dataset: dict, marker_payloads: dict) -> None:
    """I5: runMarkerEvaluation invalidates the validation request of the previous locked model."""
    result = _run_page(tmp_path, _marker_run_script("added", r"""
      page.fetchHandler = (request) => {
        if (request.url.endsWith("/api/upload")) return { status: 200, body: { dataset_id: "external-2" } };
        if (request.url.endsWith("/api/marker-validation")) return deferred().promise;
        if (request.url.endsWith("/api/marker-evaluation")) return { status: 200, body: { ...fixtures.markers.added, request_config: request.json() } };
        return { status: 200, body: {} };
      };
      page.run("refs.markerValidationFile.files = [{ name: 'external.csv' }]; refs.runMarkerValidationButton.click()");
      await page.settle(3);
      const validation = page.requests.find((request) => request.url.endsWith("/api/marker-validation"));
      page.run("runMarkerEvaluation()");
      await page.settle();
      return {
        aborted: validation.signal.aborted,
        deletes: page.requests.filter((request) => request.method === "DELETE").map((request) => request.url),
      };
    """), dataset=example_dataset, markers=marker_payloads)

    assert result == {"aborted": True, "deletes": ["/api/dataset/external-2"]}


def test_validation_table_formats_replication_p_and_names_the_horizon_unit(tmp_path: Path, example_dataset: dict, marker_payloads: dict) -> None:
    """I13: "Replication P (Holm)" is a p-value; the observed/expected horizon has a unit."""
    result = _run_page(tmp_path, _marker_run_script("added", r"""
      page.run(`renderMarkerValidation({
        external_filename: "external.csv",
        figure: null,
        validation: {
          cohort: { n: 200, events: 90 },
          metrics: { c_index: 0.7, observed_expected_ratio: 1.08, horizon: 24.5 },
          markers: [{ marker: "biomarker_score", marginal: { hazard_ratio: 1.5 }, same_direction: true, replicated: true, replication_p_holm: 0.0004 }],
          notes: [],
        },
      })`);
      await page.settle(2);
      const headers = page.run("refs.markerValidationShell.querySelectorAll('th').map((th) => th.textContent)");
      const cells = page.run("refs.markerValidationShell.querySelectorAll('td').map((td) => td.textContent)");
      return { p: cells[headers.indexOf("Replication P (Holm)")], labels: page.run("refs.markerValidationSummary.querySelectorAll('.metric-pill span').map((span) => span.textContent)") };
    """), dataset=example_dataset, markers=marker_payloads)

    assert result["p"] == "<0.001"
    assert "Observed/expected at 24.5 months" in result["labels"]


def test_marker_summary_names_the_smith_null_and_a_clinical_only_model(tmp_path: Path) -> None:
    """The added-value null is the Smith scheme ("freedman_lane" is its old name); no selected marker means a clinical-only model."""
    result = _run_page(tmp_path, r"""
      const base = {
        primary_lens: "added_value",
        tier_counts: { robust: 0, suggestive: 0 },
        cohort: { n: 300, events: 120, n_markers_evaluated: 12 },
        settings: { alpha: 0.05, robust_frequency: 0.5, robust_direction: 0.9 },
        resampling: { n_valid: 20, fraction: 0.632 },
      };
      const selected = { markers: ["GENE_A"], clinical_only: false, apparent_c: 0.71, optimism_corrected_c: 0.69, signature_optimism: 0.03,
        signature_c_left_out: 0.70, clinical_c_left_out: 0.66, signature_gain_left_out: 0.035, n_clinical_replicates: 18 };
      const clinicalOnly = { markers: [], clinical_only: true, apparent_c: 0.66, optimism_corrected_c: 0.63, signature_optimism: 0.03,
        signature_c_left_out: 0.645, clinical_c_left_out: 0.64, signature_gain_left_out: 0.004, n_clinical_replicates: 1 };
      const summarize = (nullScheme, signature) => {
        page.context.__payload = { analysis: { ...base, null: { n_permutations: 199, lens2_null: nullScheme }, signature } };
        return page.run("markerSummary(__payload)");
      };
      const smith = summarize("smith", selected);
      const oldName = summarize("freedman_lane", selected);
      const raw = summarize("raw", selected);
      const clinical = summarize("smith", clinicalOnly);
      const text = (summary) => [...summary.cautions, ...summary.strengths].join(" | ");
      page.context.__banner = { analysis: { ...base, signature: clinicalOnly } };
      return {
        smithText: text(smith),
        oldNameText: text(oldName),
        rawText: text(raw),
        clinicalText: text(clinical),
        clinicalMetric: clinical.metrics.map((metric) => metric.label),
        banner: page.run("markerMetaBanner(__banner)"),
      };
    """)

    assert "Smith method" in result["smithText"]
    assert "Smith method" in result["oldNameText"]
    assert "Smith method" not in result["rawText"]
    assert "In the patients left out of each of 18 subsamples, the whole selection procedure reached C 0.700 against 0.660" in result["smithText"]
    assert "(+0.035)" in result["smithText"]
    assert "selected-marker" not in result["clinicalText"]
    assert "the final model is the clinical-only model" in result["clinicalText"]
    assert "The selection procedure's mean subsample-to-left-out C-index gap is 0.030" in result["clinicalText"]
    assert "the gap adjustment is a heuristic that includes training-size effects" in result["clinicalText"]
    assert "Nonlinear marker-covariate relations can inflate false positives" in result["smithText"]
    for scheme in ("smithText", "rawText", "oldNameText"):
        assert "Check its functional form and proportional-hazards assumptions" in result[scheme]
        assert "multiple-testing correction does not resolve an unsuitable clinical baseline" in result[scheme]
    assert "Residual permutation assumes" not in result["rawText"]
    assert "In the patients left out of the one subsample that could be scored, the whole selection procedure" in result["clinicalText"]
    assert "The selected markers add little" not in result["clinicalText"]
    assert "Clinical-only C (gap-adjusted)" in result["clinicalMetric"]
    assert "clinical-only model (no marker selected) C apparent=0.660" in result["banner"]


def test_validation_rows_follow_the_tested_fit_and_flag_what_is_not_estimable(tmp_path: Path) -> None:
    """Rows name the fit their replication used; a null replication p reads "not estimable"."""
    result = _run_page(tmp_path, r"""
      const validation = {
        metrics: { c_index: 0.7, c_index_ci: [0.65, 0.75], clinical_only_c_index: 0.66, clinical_only_c_index_ci: [0.6, 0.71] },
        markers: [
          { marker: "A", tested: "added_value", adjusted: { hazard_ratio: 1.5, ci_lower: 1.1, ci_upper: 2.0 }, marginal: { hazard_ratio: 1.9 },
            same_direction: true, replication_p_holm: 0.004, replicated: true },
          { marker: "B", tested: "marginal", adjusted: { hazard_ratio: 1.2 }, marginal: { hazard_ratio: 0.8, ci_lower: 0.6, ci_upper: 1.0 },
            same_direction: true, replication_p_holm: 0.2, replicated: false },
          { marker: "C", tested: null, adjusted: null, marginal: { hazard_ratio: 1.4 }, same_direction: false, replication_p_holm: null, replicated: false },
          { marker: "D", tested: null, adjusted: null, marginal: null, same_direction: false, absent: true, replication_p_holm: null, replicated: false },
        ],
      };
      page.context.__validation = validation;
      return {
        rows: page.run("markerValidationRows(__validation)"),
        metrics: page.run("markerValidationMetrics(__validation)").map((metric) => `${metric.label}: ${metric.value}`),
        oldRows: page.run(`markerValidationRows({ markers: [{ marker: "A", adjusted: { hazard_ratio: 1.5 }, marginal: { hazard_ratio: 1.9 },
          same_direction: true, replication_p_holm: 0.01, replicated: true }] })`),
      };
    """)

    rows = {row["Marker"]: row for row in result["rows"]}
    assert rows["A"]["Tested as"] == "added value" and rows["A"]["HR per unit"] == 1.5
    assert rows["B"]["Tested as"] == "marginal" and rows["B"]["HR per unit"] == 0.8
    assert rows["C"] == {
        "Marker": "C",
        "Inference status": "not assessed",
        "Tested as": "not estimable",
        "HR per unit": None,
        "CI lower": None,
        "CI upper": None,
        "Same direction": "not estimable",
        "Replication P (Holm)": "not estimable",
        "Replicated": "not estimable",
    }
    assert rows["D"]["Replication P (Holm)"] == "not measured" and rows["D"]["Tested as"] == "not measured"
    assert "Clinical-only C-index: 0.66 (0.6 to 0.71)" in result["metrics"]
    # A result from before the "tested" field shows the adjusted fit and no extra column.
    assert result["oldRows"] == [{
        "Marker": "A",
        "Inference status": "not assessed",
        "HR per unit": 1.5,
        "CI lower": None,
        "CI upper": None,
        "Same direction": "yes",
        "Replication P (Holm)": 0.01,
        "Replicated": "yes",
    }]


def test_discover_is_disabled_while_a_marker_file_is_attached(tmp_path: Path, example_dataset: dict) -> None:
    """I16: the disabled-but-ticked checklist boxes do not count as markers for the cut-point search."""
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const before = page.run("refs.runSignatureSearchButton.disabled");
      page.run("state.markerMatrix = { matrix_id: 'm1', filename: 'expr.tsv', n_markers: 20, n_matched: 300, n_patients: 360, id_column: 'patient_id' }; renderMarkerMatrixState(); renderMarkerSelectionLine();");
      return {
        before,
        after: page.run("refs.runSignatureSearchButton.disabled"),
        title: page.run("refs.runSignatureSearchButton.title"),
        evaluationEnabled: !page.run("refs.runMarkersButton.disabled"),
      };
    """, dataset=example_dataset)

    assert result["before"] is False
    assert result["after"] is True
    assert "not an attached marker file" in result["title"]
    assert result["evaluationEnabled"] is True


# ── CSV export ──────────────────────────────────────────────────


_CSV_PROBES = [
    "-",
    "--",
    "+ -",
    "-1+E1",
    "-A1",
    "=cmd|' /C calc'!A0",
    "@SUM(1)",
    " =1",
    "\t=1",
    "\x01=1",
    "+1",
    "'+1",
    "'-0.5",
    "-0.50 ± 1.20",
    "-12%",
    "-1,234",
    "-1.2 (−3.4 to 0.5)",
    "-0.42 ± 1.00 | -0.42 [-1.09, 0.28]",
    "-0.5–1.2",
    "−5",
    "-1e-5",
    "1e5",
    "-cmd",
    "-SUM(A1)",
    "Test positive",
]


def test_client_csv_cells_match_the_server_sanitiser(tmp_path: Path) -> None:
    """I6/I12: client-side CSV exports neutralise formulas exactly as the server's CSV export does."""
    from survival_toolkit import app as app_module

    server = getattr(app_module, "_sanitize_csv_cell", None)
    if server is None:
        pytest.skip("The server's CSV sanitiser is not available under its expected name.")
    result = _run_page(tmp_path, r"""
      page.context.__probes = fixtures.probes;
      return page.run("__probes.map((value) => window.SurvStudioDownloads.sanitizeCsvCell(value))");
    """, probes=_CSV_PROBES)

    assert result == [server(value) for value in _CSV_PROBES]


def test_csv_preamble_lines_are_single_quoted_cells(tmp_path: Path) -> None:
    """I6: a comma in a note (a file name) cannot open a new cell that holds a formula."""
    result = _run_page(tmp_path, r"""
      page.run(`window.SurvStudioDownloads.downloadCsv({
        filename: "out.csv",
        rows: [{ Marker: "=cmd|' /C calc'!A0", HR: 1.2, Evidence: "-", Code: "-1+E1" }],
        caption: "Marker evaluation",
        notes: ['markers from expr,=HYPERLINK("http://evil.example"),x.csv, N=100'],
      })`);
      return (await page.csvDownloads()).pop();
    """)

    rows = list(csv.reader(io.StringIO(result.lstrip("\ufeff"))))
    assert rows[0] == ["# Marker evaluation"]
    assert rows[1] == ["# Notes:"]
    assert rows[2] == ['# - markers from expr,=HYPERLINK("http://evil.example"),x.csv, N=100']
    assert rows[3] == ["Marker", "HR", "Evidence", "Code"]
    assert rows[4] == ["'=cmd|' /C calc'!A0", "1.2", "-", "'-1+E1"]


# ── Removed code ────────────────────────────────────────────────


def test_dead_front_end_code_is_gone() -> None:
    """H18/H19/I22: the file:// preview branch and unused helpers are removed."""
    sources = "\n".join(path.read_text(encoding="utf-8") for path in sorted(_STATIC_DIR.glob("app*.js")))
    for name in (
        "isFilePreview",
        "function formatPercent",
        "function slugifyDownloadToken(value, fallback = \"na\") {\n  return downloadHelpers",
        "buildMarkdownTable",
        "alternatePredictiveFamilyGoal",
        "evaluationModeLabel",
        "resetChecklistSearch",
        "getActiveRunButton",
        "benchmarkSingleRunSummary",
        "formatMaxTimeChip",
        "showBenchmarkStarterAction",
    ):
        assert name not in sources, name
