# ruff: noqa
# fmt: off
"""GENERATED FILE: do not edit by hand. Edit the notebook, then run `python tools/build_engine.py`.

Engine for the Streamlit app: every non-UI code cell of
notebooks/Transit_Agentic_AI_Submission_v5_with_evaluation.ipynb, in notebook order.
"""
import os as _os
import matplotlib as _matplotlib
_matplotlib.use("Agg")                     # no screen on a server: figures render off-screen
_os.environ.setdefault("MPLBACKEND", "Agg")
from _ipython_shim import display, HTML, Markdown, clear_output  # noqa: F401


# ====================================================================================================
# notebook cell 6: SETUP 3 · Live LLM client (OpenAI by default → OpenRouter → Hugging Face → templates)
# ====================================================================================================
#@title SETUP 3 · Live LLM client (OpenAI by default → OpenRouter → Hugging Face → templates)
import os, json, requests
from datetime import datetime, timezone
try:
    USE_OLLAMA
except NameError:
    USE_OLLAMA = False
try:
    OLLAMA_MODEL
except NameError:
    OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
iso = globals().get("iso") or (lambda dt: dt.isoformat(timespec="seconds"))
utcnow = globals().get("utcnow") or (lambda: datetime.now(timezone.utc))

LLM_PROVIDER = "openai"  #@param ["openai", "openrouter", "huggingface", "ollama"]
LLM_FALLBACKS = "openrouter, huggingface"  #@param {type:"string"}
LLM_CUSTOM_MODEL = ""  #@param {type:"string"}
LLM_STRUCT_MODE = "auto"  #@param ["auto", "native", "prompt"]
LLM_MAX_CALLS = 300  #@param {type:"integer"}
import re, time, warnings
from datetime import date
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ValidationError, field_validator
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Keys are read only from Colab Secrets or environment variables; they are never printed or saved.
CATALOG = {
    "openai": {
        "label": "OpenAI", "url": "https://api.openai.com/v1/chat/completions", "key_name": "OPENAI_API_KEY",
        "key_help": "https://platform.openai.com/api-keys", "token_param": "max_completion_tokens",
        "models": [("gpt-4.1-mini", "Fast, low cost, reliable JSON-schema output")],
    },
    "openrouter": {
        "label": "OpenRouter", "url": "https://openrouter.ai/api/v1/chat/completions", "key_name": "OPENROUTER_API_KEY",
        "key_help": "https://openrouter.ai/keys", "token_param": "max_tokens",
        "models": [("openrouter/free", "Free-models router: picks an available free model")],
    },
    "huggingface": {
        "label": "Hugging Face", "url": "https://router.huggingface.co/v1/chat/completions", "key_name": "HF_API_KEY", "alt_key_names": ["HF_TOKEN"],
        "key_help": "https://huggingface.co/settings/tokens", "token_param": "max_tokens",
        "models": [("Qwen/Qwen2.5-72B-Instruct", "Open model with reliable structured output")],
    },
    "ollama": {
        "label": "Ollama (local, inside this Colab runtime)", "url": f"{OLLAMA_URL}/v1/chat/completions",
        "key_name": None, "key_help": "", "token_param": "max_tokens",
        "models": [(OLLAMA_MODEL, "Local model pulled in SETUP 2 (no key, no cost)")],
    },
}


def read_secret(name):
    """Colab Secrets first, then environment variables."""
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v:
            return v.strip()
    except Exception:
        pass
    return (os.getenv(name) or "").strip()


_hf_key = read_secret("HF_API_KEY") or read_secret("HF_TOKEN")
if _hf_key:
    os.environ.setdefault("HF_TOKEN", _hf_key)          # lets the FLAN-T5 download in A2 authenticate too


def _provider_entry(name, model_override=""):
    cfg = CATALOG[name]
    key = "ollama" if cfg["key_name"] is None else next(
        (v for v in (read_secret(k) for k in [cfg["key_name"]] + cfg.get("alt_key_names", [])) if v), "")
    usable = bool(key) and (name != "ollama" or USE_OLLAMA)
    return {"provider": name, "label": cfg["label"], "url": cfg["url"], "key": key,
            "model": model_override or cfg["models"][0][0], "token_param": cfg["token_param"],
            "native_ok": True, "usable": usable, "timeout": 180 if name == "ollama" else 90}


_chain_names = [LLM_PROVIDER] + [p.strip() for p in LLM_FALLBACKS.split(",") if p.strip() in CATALOG and p.strip() != LLM_PROVIDER]
if USE_OLLAMA and "ollama" not in _chain_names:      # SETUP 2 started a local model: use it as the first fallback
    _chain_names.insert(1, "ollama")
LLM_CHAIN = [_provider_entry(n, LLM_CUSTOM_MODEL.strip() if n == LLM_PROVIDER else "") for n in _chain_names]
LLM_CHAIN = [p for p in LLM_CHAIN if p["usable"]]
LLM_READY = bool(LLM_CHAIN)
LLM = {"provider": LLM_CHAIN[0]["provider"] if LLM_READY else None, "model": LLM_CHAIN[0]["model"] if LLM_READY else None,
       "struct_mode": LLM_STRUCT_MODE, "native_used": False, "last_provider": None, "last_model": None}
LLM_STATS = {"calls": 0, "by_provider": {}, "failures": [], "fallbacks_used": 0, "provider_errors": {}}
print("LLM providers available (in order):", [f"{p['label']} · {p['model']}" for p in LLM_CHAIN] or "none → template text only")
for n in _chain_names:
    cfg = CATALOG[n]
    if cfg["key_name"] and not any(read_secret(k) for k in [cfg["key_name"]] + cfg.get("alt_key_names", [])):
        print(f"  {cfg['label']}: no {cfg['key_name']} in Colab Secrets (skipped)")


def _scrub(text):
    """Error bodies can echo part of a key; never print anything that looks like one."""
    return re.sub(r"(sk-|or-|hf_)[A-Za-z0-9_\-\*]{4,}", "[key hidden]", str(text))[:200]


class LLMUnavailable(RuntimeError):
    pass


def response_format_for(schema):
    """Native JSON-schema mode: hand the Pydantic schema to the provider."""
    return {"type": "json_schema",
            "json_schema": {"name": schema.__name__, "strict": False, "schema": schema.model_json_schema()}}


def _call_one(p, messages, temperature, max_tokens, schema):
    headers = {"Authorization": f"Bearer {p['key']}", "Content-Type": "application/json"}
    payload = {"model": p["model"], "messages": messages, "temperature": temperature, p["token_param"]: max_tokens}
    if p["provider"] == "openrouter":
        payload["reasoning"] = {"exclude": True}
    if schema is not None and LLM["struct_mode"] in ("native", "auto") and p["native_ok"]:
        payload["response_format"] = response_format_for(schema)
    last_error = None
    for attempt in range(3):
        try:
            r = requests.post(p["url"], headers=headers, json=payload, timeout=p["timeout"])
            if r.status_code == 429 and "insufficient_quota" in r.text:
                raise LLMUnavailable("quota exhausted (HTTP 429 insufficient_quota); add credit or billing for this key")
            if r.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {r.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code in (401, 403):
                raise LLMUnavailable(f"auth failed (HTTP {r.status_code}); check {CATALOG[p['provider']]['key_name']}")
            if r.status_code in (400, 404, 415, 422) and "response_format" in payload and LLM["struct_mode"] == "auto":
                p["native_ok"] = False
                payload.pop("response_format")
                continue
            if r.status_code in (400, 404, 422):
                raise LLMUnavailable(f"request rejected (HTTP {r.status_code}): {_scrub(r.text)}")
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            if content:
                if "response_format" in payload:
                    LLM["native_used"] = True
                return content
            last_error = "empty response"
        except LLMUnavailable:
            raise
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            last_error = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(2 * (attempt + 1))
    raise LLMUnavailable(last_error or "no response")


def _looks_like_reasoning(text):
    """Some free models return their working notes instead of an answer; those are not used."""
    head = text.lstrip()[:400].lower()
    return any(k in head for k in ("analyze the request", "thinking process", "let me think", "**goal:**", "step 1: understand"))


def call_llm(messages, temperature=0.2, max_tokens=1800, schema=None):
    """One chat request. Tries each provider in LLM_CHAIN in order and records which one answered."""
    if not LLM_READY:
        raise LLMUnavailable("No LLM provider configured (add a key in Colab Secrets). Deterministic rules still work.")
    if LLM_STATS["calls"] >= LLM_MAX_CALLS:
        raise LLMUnavailable(f"LLM call budget reached ({LLM_MAX_CALLS}); raise LLM_MAX_CALLS to continue")
    errors = []
    for i, p in enumerate(LLM_CHAIN):
        if p.get("disabled"):                 # circuit breaker: a provider that failed auth/quota/model is skipped
            errors.append(f"{p['label']}: skipped ({p['disabled']})")
            continue
        try:
            LLM_STATS["calls"] += 1
            out = _call_one(p, messages, temperature, max_tokens, schema)
            if schema is None and _looks_like_reasoning(out):
                raise LLMUnavailable("returned its reasoning instead of an answer (not used)")
            LLM_STATS["by_provider"][p["label"]] = LLM_STATS["by_provider"].get(p["label"], 0) + 1
            LLM_STATS["fallbacks_used"] += int(i > 0)
            LLM.update(last_provider=p["label"], last_model=p["model"])
            return out
        except LLMUnavailable as e:
            errors.append(f"{p['label']}: {e}")
            LLM_STATS["provider_errors"][p["label"]] = _scrub(e)
            if re.search(r"auth failed|quota exhausted|request rejected", str(e)):
                p["disabled"] = _scrub(e)
    LLM_STATS["failures"].append({"time": iso(utcnow()), "errors": errors})
    raise LLMUnavailable("All LLM providers failed — " + " | ".join(errors))


def llm_generate(prompt, system, max_tokens=320):
    """Plain-text generation for agent narratives and notes (never used in any score or routing decision)."""
    txt = call_llm([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=max_tokens)
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
    if not txt:
        raise LLMUnavailable("empty response")
    return txt


def llm_source():
    return f"{LLM['last_provider']} · {LLM['last_model']}" if LLM.get("last_provider") else "template"


def extract_json(text):
    """Find the JSON object in a model reply."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in the reply")
    return json.loads(text[start:end + 1])


def run_structured(schema, system, user, retries=2):
    """Get JSON that fits a Pydantic schema (native mode if enabled), validate it, retry with the error on failure."""
    schema_hint = json.dumps(schema.model_json_schema(), indent=1)
    messages = [
        {"role": "system", "content": system + "\n\nReturn ONLY one JSON object (no prose, no markdown) "
                                              "that matches this JSON schema:\n" + schema_hint},
        {"role": "user", "content": user},
    ]
    for attempt in range(retries + 1):
        raw = call_llm(messages, schema=schema)
        try:
            return schema.model_validate(extract_json(raw)), attempt
        except (ValueError, ValidationError) as e:          # bad JSON or schema mismatch
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"That output was invalid: {str(e)[:500]}\nReturn corrected JSON only."}]
    raise RuntimeError(f"{schema.__name__}: model never produced valid JSON after {retries + 1} tries. "
                       f"Try a different model or output mode.")


def mode_used():
    """Which structured-output mode actually did the work in this session."""
    if LLM["struct_mode"] == "prompt":
        return "prompt + validate"
    return "native JSON-schema" if LLM["native_used"] else "prompt + validate (native unavailable)"


def draw_workflow(steps, notes=None, title=""):
    """Draw the workflow as boxes and arrows (identical helper in C and D, kept once).
    steps = [(key, label, kind)], kind in input/llm/tool/output."""
    from matplotlib.patches import FancyBboxPatch
    notes = notes or {}
    colors = {"input": "#E5E7EB", "llm": "#DBEAFE", "tool": "#FEF3C7", "output": "#DCFCE7"}
    tags = {"input": "INPUT", "llm": "LLM AGENT", "tool": "PYTHON TOOL", "output": "OUTPUT"}
    n, w, gap = len(steps), 1.9, 0.5
    fig, ax = plt.subplots(figsize=(min(17, 2.3 * n), 2.6))
    ax.set_xlim(0, n * (w + gap)); ax.set_ylim(0, 2.6); ax.axis("off")
    for i, (key, label, kind) in enumerate(steps):
        x = i * (w + gap) + 0.05
        ax.add_patch(FancyBboxPatch((x, 0.75), w, 1.15, boxstyle="round,pad=0.04", fc=colors[kind], ec="#334155", lw=1.2))
        ax.text(x + w / 2, 1.50, label, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x + w / 2, 1.05, tags[kind], ha="center", va="center", fontsize=7, color="#475569")
        if key in notes:
            ax.text(x + w / 2, 0.45, notes[key], ha="center", va="center", fontsize=8, color="#0F766E")
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, 1.32), xytext=(x + w + 0.02, 1.32),
                        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.4))
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    plt.tight_layout(); plt.show()




if LLM_READY:
    try:
        reply = call_llm([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=20)
        print(f"✓ Connected: {llm_source()} -> {reply.strip()[:40]!r}")
        for _p in LLM_CHAIN:
            if _p["label"] in LLM_STATS["provider_errors"]:
                print(f"  ✗ {_p['label']} did not answer: {LLM_STATS['provider_errors'][_p['label']]}"
                      + (" → skipped for this session" if _p.get("disabled") else " → will be retried"))
    except LLMUnavailable as e:
        print(f"✗ Connection test failed: {e}")


# ====================================================================================================
# notebook cell 8: E1 · Imports, runtime settings and helpers
# ====================================================================================================
#@title E1 · Imports, runtime settings and helpers
import json, math, random, zlib, copy, threading, io, zipfile, os, html as _html, statistics
from datetime import datetime, timezone, timedelta
import pandas as pd
import requests

try:
    USE_OLLAMA
except NameError:
    USE_OLLAMA = False
try:
    OLLAMA_MODEL
except NameError:
    OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

# Service area: Newark Penn Station. Section A's weather connectors use this point, and the synthetic (fictional)
# network is drawn over a real basemap around it. Change freely.
SERVICE_AREA_LATLON = (40.7347, -74.1642)
CITY_CENTER = SERVICE_AREA_LATLON
# 1.0 = real review deadlines (5/10/20 min). Set e.g. 0.1 to make deadlines 10x shorter for demos.
DEMO_DEADLINE_SCALE = 1.0  #@param {type:"number"}
# Every timestamp the notebook shows or saves is in this time zone (EST/EDT for New Jersey).
LOCAL_TZ_NAME = "America/New_York"  #@param {type:"string"}
from zoneinfo import ZoneInfo
LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)
OUTPUT_DIR = "transit_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def utcnow():
    return datetime.now(timezone.utc)


def iso(dt):
    """ISO time in the local zone, e.g. 2026-11-02T01:41:00-05:00 (the offset shows EST or EDT)."""
    return dt.astimezone(LOCAL_TZ).isoformat(timespec="seconds")


def local_now():
    return utcnow().astimezone(LOCAL_TZ)


def local_label(dt=None):
    """Readable local time with its zone name, e.g. '2026-11-02 01:41 EST'."""
    return (dt or utcnow()).astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M %Z")


def parse_iso(s):
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def r3(x):
    return None if x is None else round(float(x), 3)


def stable_rng(*keys):
    """Seeded RNG so synthetic values are reproducible (Python's hash() is randomized)."""
    return random.Random(zlib.crc32("|".join(map(str, keys)).encode()))


def n_benefit(x, L, U):
    """n(x) = clip((x - L) / (U - L), 0, 1). Missing values return None (never imputed silently)."""
    if x is None:
        return None
    if U == L:
        return 0.0
    return float(min(1.0, max(0.0, (float(x) - L) / (U - L))))


def n_cost(x, L, U):
    """ninv(x) = 1 - n(x) for cost or delay burden."""
    v = n_benefit(x, L, U)
    return None if v is None else 1.0 - v

# ====================================================================================================
# notebook cell 9: E1 · Policy rules (flowchart section 4) — versioned assumptions
# ====================================================================================================
#@title E1 · Policy rules (flowchart section 4) — versioned assumptions
DEFAULT_POLICY = {
    "version": "1.0.0",
    # Confidence C = 0.40Q + 0.25F + 0.20G + 0.15M
    "confidence_weights": {"Q": 0.40, "F": 0.25, "G": 0.20, "M": 0.15},
    "confidence_threshold": 0.70,
    "source_quality": {"vehicle_apc": 1.00, "official_alert": 0.90, "trusted_external": 0.80,
                       "public_report_corroborated": 0.40},
    # Source precedence: vehicle data > official alert > trusted external feed > corroborated public report
    "source_precedence": ["vehicle_apc", "official_alert", "trusted_external", "public_report"],
    "official_sources": ["vehicle_apc", "official_alert"],   # sources allowed to verify a hard trigger (C-gate bypass)
    "single_source_agreement": 0.50,                         # G when only one source reports a value (assumption)
    "conflict_threshold_min": 8,                             # delay disagreement that counts as a conflict
    "conflict_penalty": 0.15,                                # C reduction while a conflict is unresolved
    "freshness_limit_s": 300,
    # Impact and priority
    "R_weights": {"D": 0.30, "H": 0.25, "K": 0.25, "E": 0.20},
    "V_weights": {"N": 0.35, "I": 0.25, "B": 0.20, "T": 0.20},
    "A_weights": {"S": 0.35, "L": 0.30, "X": 0.20, "U": 0.15},
    "P_weights": {"R": 0.40, "V": 0.35, "A": 0.25},
    "significance_R": 0.50,
    "hard_trigger": {"delay_min": 15, "gap_headway_ratio": 2.0, "unboardable_load_factor": 1.0,
                     "essential_severity_min": 0.8},
    # Priority-adjusted scenario rank: Si = Ci x [0.30DR + (0.15+0.10P)EB + (0.15+0.10P)AP + (0.25-0.10P)OF + (0.15-0.10P)CE]
    "scenario_base": {"DR": 0.30, "EB": 0.15, "AP": 0.15, "OF": 0.25, "CE": 0.15},
    "scenario_shift": {"DR": 0.0, "EB": 0.10, "AP": 0.10, "OF": -0.10, "CE": -0.10},
    # Documented normalization bounds (L, U)
    "bounds": {
        "delay_min": [0, 30], "headway_ratio": [1, 3], "transit_dependence": [0.05, 0.60],
        "income_disadvantage": [0.05, 0.40], "disability": [0.05, 0.25], "alternatives": [0, 3],
        "missed_transfers": [0, 40], "passups": [0, 40], "age_s": [0, 300], "agreement_spread_min": [0, 15],
        "dr_min": [0, 20], "eb_riders": [0, 300], "cost_usd": [0, 1000],
    },
    # Published categorical lookup tables
    "lookups": {
        "event_severity": {"none": 0.0, "minor": 0.3, "major": 0.7, "emergency": 1.0},
        "time_sensitivity": {"off_peak": 0.3, "midday": 0.4, "peak": 0.8, "shift_change": 0.9, "last_service": 1.0},
        "essential_severity": {"none": 0.0, "retail": 0.3, "school": 0.6, "employment_center": 0.6,
                               "clinic": 0.8, "hospital": 1.0, "dialysis": 1.0},
        "option_certainty": {"do_nothing": 1.00, "rider_alerts": 0.95, "protect_connection": 0.85,
                             "reroute": 0.75, "dispatch_second_bus": 0.80},
    },
    # Human review
    "deadlines_min": {"severe": 5, "significant": 10, "moderate": 20},
    "severity_tiers": {"severe_P": 0.70, "significant_P": 0.50},
    "high_impact": {"actions": ["reroute", "dispatch_second_bus"], "P_min": 0.70, "cost_min": 500},
    "max_revisions": 2,
    "max_operator_checks": 2,
    "review": {"min_reason_chars": 15, "fast_response_flag_s": 30, "approval_rate_flag": 0.90, "min_n_for_pattern": 5},
    # Planner defaults (constraints the planner can revise)
    "planner_defaults": {"budget_cap_usd": 1000, "max_hold_min": 5, "max_deadhead_min": 30,
                         "allow_alternate_depot": False, "excluded_options": []},
    "bus_hour_cost_usd": 160,
    # Review deadline on a revision: False = the original deadline still applies (dashboard default);
    # True = every rebuilt brief starts a new deadline.
    "deadline_resets_on_revision": False,
    # Reviewer governance gates built from Sections C (ethics training) and D (conflict of interest)
    "governance_gates": {"enabled": True,
                         "unknown_person": "warn",                    # "warn" or "block" names not on the training roster
                         "training_block_statuses": ["NON_COMPLIANT"],
                         "training_gate_decisions": ["Approve", "Revise"],  # Reject (→ safe default) is always allowed
                         "coi_gate_decisions": ["Approve"],
                         "coi_threshold": 0.70},
}

REVIEW_CHECKLIST = [
    "I checked the verified evidence, sources and timestamps",
    "I viewed every alternative and its component scores",
    "I considered equity and access impacts on affected areas",
    "I confirmed resource, road, policy and safety constraints",
]


def validate_policy(p):
    errs = []
    for k in ["confidence_weights", "R_weights", "V_weights", "A_weights", "P_weights", "scenario_base"]:
        s = sum(p[k].values())
        if abs(s - 1) > 1e-9:
            errs.append(f"{k} sums to {s:.4f}, not 1")
    if abs(sum(p["scenario_shift"].values())) > 1e-9:
        errs.append("scenario_shift must sum to 0 so Si weights always sum to 1")
    for c, b in p["scenario_base"].items():
        sh = p["scenario_shift"].get(c, 0)
        if not (0 <= b + min(0, sh) and b + max(0, sh) <= 1):
            errs.append(f"weight for {c} leaves [0,1] when P moves 0→1")
    for k, (L, U) in p["bounds"].items():
        if U <= L:
            errs.append(f"bound {k}: U must exceed L")
    return errs


def get_path(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def set_path(d, path, value):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


class PolicyManager:
    """Versioned policy store. Changes only via policy-owner approval; every version kept for rollback."""

    def __init__(self, policy):
        assert not validate_policy(policy), validate_policy(policy)
        self.versions = [{"version": policy["version"], "policy": copy.deepcopy(policy), "released_at": iso(utcnow()),
                          "approved_by": "initial release", "reason": "Initial documented policy assumptions",
                          "changes": [], "backtest_summary": ""}]
        self.current_index = 0

    @property
    def current(self):
        return self.versions[self.current_index]["policy"]

    def get(self, version):
        for v in self.versions:
            if v["version"] == version:
                return v["policy"]
        raise KeyError(version)

    def _bump(self):
        major, minor, patch = map(int, self.versions[-1]["version"].split("."))
        return f"{major}.{minor + 1}.0"

    def release(self, new_policy, approved_by, reason, changes, backtest_summary=""):
        new_policy = copy.deepcopy(new_policy)
        new_policy["version"] = self._bump()
        errs = validate_policy(new_policy)
        if errs:
            raise ValueError("; ".join(errs))
        self.versions.append({"version": new_policy["version"], "policy": new_policy, "released_at": iso(utcnow()),
                              "approved_by": approved_by, "reason": reason, "changes": changes,
                              "backtest_summary": backtest_summary})
        self.current_index = len(self.versions) - 1
        return new_policy["version"]

    def rollback(self, to_version, by, reason):
        old = self.get(to_version)
        return self.release(old, by, f"ROLLBACK to {to_version}: {reason}",
                            [{"path": "*", "old": self.current["version"], "new": f"rules of {to_version}"}])

# ====================================================================================================
# notebook cell 10: E1 · Synthetic transit network (routes, stops, community areas, essential destinations, depots)
# ====================================================================================================
#@title E1 · Synthetic transit network (routes, stops, community areas, essential destinations, depots)
KM_LAT = 110.574


def _pt(x_km, y_km):
    lat0, lon0 = CITY_CENTER
    return (lat0 + y_km / KM_LAT, lon0 + x_km / (111.320 * math.cos(math.radians(lat0))))


def km_between(a, b):
    lat0 = math.radians((a[0] + b[0]) / 2)
    dx = (b[1] - a[1]) * 111.320 * math.cos(lat0)
    dy = (b[0] - a[0]) * KM_LAT
    return math.hypot(dx, dy)


def _interp(waypoints, n):
    pts = [_pt(*w) for w in waypoints]
    segs = [km_between(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(segs)
    out = []
    for k in range(n):
        d = total * k / (n - 1)
        i = 0
        while i < len(segs) - 1 and d > segs[i]:
            d -= segs[i]
            i += 1
        f = 0 if segs[i] == 0 else min(1.0, d / segs[i])
        out.append((pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f))
    return out


STREETS = ["Oak", "Maple", "Pine", "Cedar", "Elm", "Harbor", "Union", "Market", "Lake", "Park", "River", "Hill",
           "Mill", "Grove", "Spring", "Bay", "Main", "Ash", "Birch", "Linden"]

ROUTE_SPECS = [
    dict(id="R1", name="Crosstown", mode="bus", headway_min=10, waypoints=[(-7, 0.2), (7, 0.2)], n_stops=14, color="#1f77b4"),
    dict(id="R2", name="Hospital Line", mode="bus", headway_min=15, waypoints=[(-1.2, -7), (-1.0, 0), (1.5, 6.5)], n_stops=14, color="#9467bd"),
    dict(id="R3", name="Industrial Connector", mode="bus", headway_min=20, waypoints=[(-6.5, -5.5), (5.5, 4.5)], n_stops=12, color="#8c564b"),
    dict(id="R4", name="Northside", mode="bus", headway_min=12, waypoints=[(-6, 4.2), (0, 5.2), (6, 4.2)], n_stops=12, color="#2ca02c"),
    dict(id="R5", name="Southside", mode="bus", headway_min=15, waypoints=[(-6.5, -4.2), (6.5, -4.0)], n_stops=13, color="#ff7f0e"),
    dict(id="L1", name="Rail Line", mode="rail", headway_min=8, waypoints=[(3, -7.5), (3, 7.5)], n_stops=8, color="#d62728"),
]

ESSENTIAL_DESTINATIONS = [
    dict(id="D1", name="General Hospital", type="hospital", xy=(-1.6, -4.9)),
    dict(id="D2", name="Kidney Dialysis Center", type="dialysis", xy=(-0.5, -2.0)),
    dict(id="D3", name="Westfield High School", type="school", xy=(2.0, 0.6)),
    dict(id="D4", name="Riverside Employment Center", type="employment_center", xy=(4.4, 3.4)),
    dict(id="D5", name="Southside Community Clinic", type="clinic", xy=(-4.0, -4.6)),
    dict(id="D6", name="Northgate Market", type="retail", xy=(-2.5, 5.2)),
]

DEPOTS = [
    dict(id="DEPOT-A", name="Depot A (southwest)", xy=(-7.0, -6.5), spare_buses=2, operators=2, operated_by="Agency (in-house)"),
    dict(id="DEPOT-B", name="Depot B (northeast)", xy=(6.5, 6.5), spare_buses=1, operators=1, operated_by="Metro Coach Services LLC"),
]
# Outside parties involved in actions (shown in the brief for transparency and screened by the Section D conflict-of-interest gate)
CONNECTION_PARTNERS = {"L1": "Northeast Rail Partners", "R1": "Agency (in-house)", "R2": "Agency (in-house)",
                       "R3": "Agency (in-house)", "R4": "Agency (in-house)", "R5": "Agency (in-house)"}


def build_network():
    rng = stable_rng("network")
    routes = {}
    for spec in ROUTE_SPECS:
        coords = _interp(spec["waypoints"], spec["n_stops"])
        stops = []
        for i, (lat, lon) in enumerate(coords):
            stops.append({"stop_id": f"{spec['id']}-S{i:02d}", "route_id": spec["id"], "seq": i,
                          "name": f"{STREETS[(i * 3 + len(spec['id']) * 7 + ord(spec['id'][-1])) % len(STREETS)]} St "
                                  f"({spec['id']} #{i})", "lat": lat, "lon": lon})
        routes[spec["id"]] = {**{k: v for k, v in spec.items() if k != "waypoints"}, "stops": stops,
                              "path": [(s["lat"], s["lon"]) for s in stops]}
    # Community areas: 4 x 4 grid of 4 km cells; area-level aggregates only (no individual data)
    areas = []
    for gx in range(4):
        for gy in range(4):
            x0, y0 = -8 + gx * 4, -8 + gy * 4
            cx, cy = x0 + 2, y0 + 2
            sw01 = (1 + (-(cx / 6) - (cy / 6)) / 2) / 2  # 0 (northeast) .. 1 (southwest)
            sw01 = max(0.0, min(1.0, sw01))
            areas.append({
                "area_id": f"A{gx + 1}{gy + 1}", "name": f"Area {'ABCD'[gx]}{gy + 1}",
                "bounds": [_pt(x0, y0), _pt(x0 + 4, y0 + 4)],
                "transit_dependence": round(0.08 + 0.42 * sw01 + rng.uniform(-0.03, 0.03), 3),
                "income_disadvantage": round(0.06 + 0.30 * sw01 + rng.uniform(-0.02, 0.02), 3),
                "disability": round(0.07 + 0.12 * sw01 + rng.uniform(-0.01, 0.01), 3),
                "population": int(rng.uniform(8000, 26000)),
            })
    dests = [{**d, "lat": _pt(*d["xy"])[0], "lon": _pt(*d["xy"])[1]} for d in ESSENTIAL_DESTINATIONS]
    depots = [{**d, "lat": _pt(*d["xy"])[0], "lon": _pt(*d["xy"])[1]} for d in DEPOTS]
    all_stops = [s for r in routes.values() for s in r["stops"]]
    for s in all_stops:
        p = (s["lat"], s["lon"])
        s["area_id"] = next((a["area_id"] for a in areas
                             if a["bounds"][0][0] <= p[0] <= a["bounds"][1][0] and a["bounds"][0][1] <= p[1] <= a["bounds"][1][1]),
                            areas[0]["area_id"])
        s["essential"] = [d["id"] for d in dests if km_between(p, (d["lat"], d["lon"])) <= 0.9]
        s["nearby_routes"] = sorted({o["route_id"] for o in all_stops if o["route_id"] != s["route_id"]
                                     and km_between(p, (o["lat"], o["lon"])) <= 0.45})
        s["transfer_to"] = sorted({o["route_id"] for o in all_stops if o["route_id"] != s["route_id"]
                                   and km_between(p, (o["lat"], o["lon"])) <= 0.6})
    return {"routes": routes, "areas": {a["area_id"]: a for a in areas}, "destinations": {d["id"]: d for d in dests},
            "depots": {d["id"]: d for d in depots}, "stops": {s["stop_id"]: s for s in all_stops}}


NET = build_network()
print(f"Synthetic network: {len(NET['routes'])} routes, {len(NET['stops'])} stops, {len(NET['areas'])} community areas, "
      f"{len(NET['destinations'])} essential destinations, {len(NET['depots'])} depots")

# ====================================================================================================
# notebook cell 12: E1 · Scenario builder and preset incidents (synthetic data inputs)
# ====================================================================================================
#@title E1 · Scenario builder and preset incidents (synthetic data inputs)
SPEC_DEFAULTS = dict(
    route_id="R1", stop_range=(3, 7), time_period="peak", delay_min=10.0, headway_ratio=1.5, cancelled=False,
    cancellation_risk=0.2, boardings=150, load_factor=0.8, passups=0, obstruction=False,
    event_type="none", event_severity="none", event_verified=False,
    sources=["vehicle_apc"], feed_age_s=20, alert_age_s=40, external_age_s=60,
    official_delay=None, external_delay=None, public_delay=None, public_corroborated=False,
    backup_available=True, backup_delay=None, vision_agent_failures=0, detour_open=True, detour_extra_min=6,
    reroute_allowed=True, segment_open=True, missed_transfers=None, last_service=False, missing_fields=[],
    depot_overrides={},
)


def _crowding(load):
    return "unboardable" if load >= 1.0 else "high" if load >= 0.85 else "medium" if load >= 0.6 else "low"


def make_raw(spec, name="Custom scenario"):
    s = {**SPEC_DEFAULTS, **spec}
    rid = s["route_id"]
    route = NET["routes"][rid]
    i0, i1 = s["stop_range"]
    i1 = min(i1, len(route["stops"]) - 1)
    affected = route["stops"][i0:i1 + 1]
    now = utcnow()
    risk = 1.0 if s["cancelled"] else s["cancellation_risk"]

    def obs(source_id, stype, age_s, fields, corroborated=True):
        return {"source_id": source_id, "source_type": stype, "timestamp": iso(now - timedelta(seconds=age_s)),
                "age_s": age_s, "corroborated": corroborated, "fields": fields}

    observations = []
    if "vehicle_apc" in s["sources"]:
        observations.append(obs(f"AVL-{rid}", "vehicle_apc", s["feed_age_s"],
                                {"delay_min": s["delay_min"], "headway_ratio": s["headway_ratio"],
                                 "cancelled": s["cancelled"], "cancellation_risk": risk}))
    if "official_alert" in s["sources"]:
        f = {"cancelled": s["cancelled"], "event_type": s["event_type"], "event_severity": s["event_severity"],
             "event_verified": s["event_verified"]}
        if s["official_delay"] is not None:
            f["delay_min"] = s["official_delay"]
        observations.append(obs("AGENCY-ALERTS", "official_alert", s["alert_age_s"], f))
    if "trusted_external" in s["sources"]:
        sid = "NWS-WEATHER" if s["event_type"] == "weather" else "511-TRAFFIC"
        f = {"event_type": s["event_type"], "event_severity": s["event_severity"]}
        ext = s["external_delay"] if s["external_delay"] is not None else (s["delay_min"] + 1 if ("vehicle_apc" in s["sources"] and s["delay_min"] is not None) else None)
        if ext is not None:
            f["delay_min"] = ext
        observations.append(obs(sid, "trusted_external", s["external_age_s"], f))
    if "public_report" in s["sources"]:
        observations.append(obs("SOCIAL-REPORTS", "public_report", 90,
                                {"delay_min": s["public_delay"] if s["public_delay"] is not None else s["delay_min"],
                                 "event_type": s["event_type"]}, corroborated=s["public_corroborated"]))
    for o in observations:
        for m in s["missing_fields"]:
            o["fields"].pop(m, None)

    area_ids = sorted({st["area_id"] for st in affected})
    areas = [NET["areas"][a] for a in area_ids]
    community = {k: round(statistics.mean(a[k] for a in areas), 3) for k in ["transit_dependence", "income_disadvantage", "disability"]}
    community["areas"] = area_ids
    for m in s["missing_fields"]:
        community.pop(m, None)
    sev = DEFAULT_POLICY["lookups"]["essential_severity"]
    ess_ids = sorted({d for st in affected for d in st["essential"]})
    ess_types = [NET["destinations"][d]["type"] for d in ess_ids]
    essential_type = max(ess_types, key=lambda t: sev[t]) if ess_types else "none"
    alternatives = min(len(st["nearby_routes"]) for st in affected)

    # Connecting service: first affected/downstream stop with a transfer (rail preferred)
    conn = {"connecting_route": None, "partner": None, "transfer_stop_id": None, "transfer_window_min": 4, "connecting_headway_min": None,
            "last_service": s["last_service"], "missed_transfers": 0}
    downstream = route["stops"][i0:]
    for pref in (["L1"], None):
        for st in downstream:
            cands = [r for r in st["transfer_to"] if (pref is None or r in pref)]
            if cands:
                conn.update(connecting_route=cands[0], transfer_stop_id=st["stop_id"], partner=CONNECTION_PARTNERS.get(cands[0]),
                            connecting_headway_min=NET["routes"][cands[0]]["headway_min"])
                break
        if conn["connecting_route"]:
            break
    if conn["connecting_route"]:
        conn["missed_transfers"] = s["missed_transfers"] if s["missed_transfers"] is not None else int(round(s["boardings"] * 0.25))

    vision = {"camera_feeds": [f"VISION-{st['stop_id']}" for st in affected[:3]] + [f"VISION-ONBOARD-{rid}"],
              "boardings": s["boardings"], "alightings": int(s["boardings"] * 0.8), "capacity": 60,
              "load_factor": s["load_factor"], "passups": s["passups"], "obstruction": s["obstruction"],
              "crowding_level": _crowding(s["load_factor"]), "anonymized": True, "identities_stored": False}
    for m in s["missing_fields"]:
        vision.pop(m, None)

    depots = []
    for d in NET["depots"].values():
        o = s["depot_overrides"].get(d["id"], {})
        depots.append({"id": d["id"], "name": d["name"], "lat": d["lat"], "lon": d["lon"],
                       "spare_buses": o.get("spare_buses", d["spare_buses"]), "operators": o.get("operators", d["operators"]),
                       "operated_by": d["operated_by"], "deadhead_min_override": o.get("deadhead_min"),
                       "dispatch_cost_override": o.get("dispatch_cost")})

    raw = {
        "scenario_name": name, "route_id": rid, "route_name": route["name"], "headway_min": route["headway_min"],
        "affected_stop_ids": [st["stop_id"] for st in affected], "time_period": "last_service" if s["last_service"] else s["time_period"],
        "event_type": s["event_type"], "observations": observations, "vision": vision, "connections": conn,
        "community": community, "access": {"essential_type": essential_type, "essential_names": [NET["destinations"][d]["name"] for d in ess_ids],
                                            "alternatives": alternatives},
        "resources": {"depots": depots, "bus_hour_cost_usd": DEFAULT_POLICY["bus_hour_cost_usd"]},
        "roads": {"detour_open": s["detour_open"], "detour_extra_min": s["detour_extra_min"], "segment_open": s["segment_open"],
                  "reroute_allowed": s["reroute_allowed"]},
        "faults": {"vision_agent_failures": s["vision_agent_failures"], "backup_available": s["backup_available"]},
        "generated_at": iso(now), "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()},
    }
    if s["backup_available"]:
        bd = s["backup_delay"] if s["backup_delay"] is not None else s["delay_min"]
        raw["backup_observation"] = obs(f"GTFSRT-MIRROR-{rid}", "trusted_external", 45,
                                        {"delay_min": bd, "headway_ratio": s["headway_ratio"], "cancelled": s["cancelled"],
                                         "cancellation_risk": risk})
    return raw


PRESETS = {
    "Severe · R2 Hospital Line trip cancelled (verified)": dict(
        route_id="R2", stop_range=(1, 6), time_period="peak", delay_min=24, headway_ratio=2.6, cancelled=True,
        boardings=140, load_factor=0.95, passups=6, event_type="mechanical", event_severity="major", event_verified=True,
        sources=["vehicle_apc", "official_alert"], official_delay=25, missed_transfers=34),
    "Significant · R1 Crosstown unboardable crowding at peak": dict(
        route_id="R1", stop_range=(3, 8), time_period="peak", delay_min=8, headway_ratio=1.6, boardings=210,
        load_factor=1.08, passups=22, sources=["vehicle_apc"]),
    "Moderate · R4 Northside winter-weather delays": dict(
        route_id="R4", stop_range=(4, 9), time_period="midday", delay_min=14, headway_ratio=1.95, cancellation_risk=0.5,
        boardings=90, load_factor=0.7, event_type="weather", event_severity="major", event_verified=True,
        sources=["vehicle_apc", "trusted_external", "official_alert"]),
    "Below threshold · R5 Southside 4-minute delay": dict(
        route_id="R5", stop_range=(2, 6), time_period="off_peak", delay_min=4, headway_ratio=1.2, cancellation_risk=0.05,
        boardings=60, load_factor=0.5, sources=["vehicle_apc", "trusted_external"]),
    "Conflicting data · R3 vehicle feed vs. public reports": dict(
        route_id="R3", stop_range=(3, 7), time_period="peak", delay_min=3, headway_ratio=1.1, cancellation_risk=0.05,
        boardings=80, load_factor=0.55, event_type="obstruction", sources=["vehicle_apc", "public_report"],
        public_delay=25, public_corroborated=True),
    "Feed outage · R1 vehicle feed stale (backup available)": dict(
        route_id="R1", stop_range=(6, 11), time_period="shift_change", delay_min=17, headway_ratio=2.1, cancellation_risk=0.4,
        boardings=120, load_factor=0.9, passups=4, sources=["vehicle_apc"], feed_age_s=900, backup_available=True, backup_delay=17),
    "Feed outage · R5 vehicle feed stale (no backup)": dict(
        route_id="R5", stop_range=(1, 5), delay_min=16, headway_ratio=2.0, sources=["vehicle_apc"], feed_age_s=1200,
        backup_available=False),
    "Uncorroborated social report only · R4": dict(
        route_id="R4", stop_range=(2, 6), sources=["public_report"], public_delay=30, public_corroborated=False,
        event_type="obstruction"),
    "Agent error · Vision Agent fails twice on R2": dict(
        route_id="R2", stop_range=(2, 7), delay_min=18, headway_ratio=2.2, sources=["vehicle_apc", "official_alert"],
        vision_agent_failures=2),
    "Hard-trigger bypass · Verified emergency closure on R3, sparse data": dict(
        route_id="R3", stop_range=(2, 6), time_period="peak", boardings=110, load_factor=0.75, event_type="road_closure",
        event_severity="emergency", event_verified=True, sources=["official_alert"], alert_age_s=240,
        missing_fields=["cancellation_risk"], detour_open=True, detour_extra_min=7),
}
print(f"{len(PRESETS)} preset scenarios ready")

# ====================================================================================================
# notebook cell 14: CV · Computer vision passenger counting (pretrained person detector → anonymized counts)
# ====================================================================================================
#@title CV · Computer vision passenger counting (pretrained person detector → anonymized counts)
USE_CV = True  #@param {type:"boolean"}
CV_SCORE_THRESHOLD = 0.6  #@param {type:"number"}
CV_VIDEO_FRAMES = 8  #@param {type:"integer"}
SHOW_CV_PREVIEW = True  #@param {type:"boolean"}
import numpy as np

# Sample transit imagery, downloaded at runtime. Only person counts leave this cell; images are not saved.
#  - bus.jpg: riders at the curb beside a city bus (Ultralytics sample asset, AGPL-3.0)
#  - vtest.avi: overhead camera of pedestrians on a walkway (OpenCV sample data), used as a station-approach camera
CV_SOURCES = [
    {"camera_id": "VISION-STOP-NEWARK-PENN", "kind": "image", "role": "bus stop curb camera",
     "url": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
     "stop_id": "NEWARK_PENN_BUS_LANE", "capacity": 50,
     "note": "generic street photo used as a stand-in for a Newark Penn curb camera"},
    {"camera_id": "VISION-APPROACH-NEWARK-PENN", "kind": "video", "role": "station approach camera",
     "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
     "stop_id": "NEWARK_PENN_APPROACH", "capacity": 50,
     "note": "public pedestrian video used as a stand-in for a station approach camera"},
]
CV_DIR = os.path.join(OUTPUT_DIR, "cv_cache")      # downloaded samples only; nothing derived from them is written here
os.makedirs(CV_DIR, exist_ok=True)
_DETECTOR = {"name": None, "model": None, "error": []}


def load_detector():
    """Faster R-CNN (MobileNetV3, COCO) from torchvision; OpenCV's HOG people detector if torchvision weights fail."""
    if _DETECTOR["name"]:
        return _DETECTOR["name"]
    try:
        import torch, torchvision
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
        w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        m = fasterrcnn_mobilenet_v3_large_fpn(weights=w).eval()
        _DETECTOR.update(name=f"torchvision Faster R-CNN MobileNetV3 ({w.name}, COCO)", model=("torch", m, w))
        return _DETECTOR["name"]
    except Exception as e:
        _DETECTOR["error"].append(f"torchvision: {type(e).__name__}: {str(e)[:120]}")
    try:
        import cv2
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _DETECTOR.update(name="OpenCV HOG + linear SVM people detector (fallback)", model=("hog", hog, None))
        return _DETECTOR["name"]
    except Exception as e:
        _DETECTOR["error"].append(f"opencv: {type(e).__name__}: {str(e)[:120]}")
    return None


def detect_people(rgb):
    """rgb: HxWx3 uint8 array → list of (x1, y1, x2, y2, score) for detected people."""
    kind, model, weights = _DETECTOR["model"]
    if kind == "torch":
        import torch
        t = torch.from_numpy(rgb).permute(2, 0, 1)
        with torch.no_grad():
            out = model([weights.transforms()(t)])[0]
        person = weights.meta["categories"].index("person")
        return [(*map(float, b), float(s)) for b, s, l in zip(out["boxes"], out["scores"], out["labels"])
                if int(l) == person and float(s) >= CV_SCORE_THRESHOLD]
    import cv2
    # HOG uses a fixed 64x128 window: small people need the full-size frame, large people a downscaled one
    h, w = rgb.shape[:2]
    hog_threshold = 0.5   # HOG scores are SVM margins, not probabilities
    keep = []
    for f in sorted({1.0, min(1.0, 480.0 / max(h, w))}):
        img = cv2.resize(rgb, (int(w * f), int(h * f))) if f < 1 else rgb
        rects, wts = model.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), winStride=(4, 4), padding=(8, 8), scale=1.05)
        keep += [(x / f, y / f, (x + bw) / f, (y + bh) / f, float(sc)) for (x, y, bw, bh), sc in zip(rects, np.ravel(wts))
                 if float(sc) >= hog_threshold]
    return _nms(keep)


def _nms(boxes, iou=0.4):
    """Drop overlapping duplicate detections of the same person."""
    boxes = sorted(boxes, key=lambda b: -b[4])
    kept = []
    for b in boxes:
        def ov(a, c):
            ix = max(0, min(a[2], c[2]) - max(a[0], c[0])); iy = max(0, min(a[3], c[3]) - max(a[1], c[1]))
            inter = ix * iy
            return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (c[2] - c[0]) * (c[3] - c[1]) - inter + 1e-9)
        if all(ov(b, k) < iou for k in kept):
            kept.append(b)
    return kept


def _download(url):
    path = os.path.join(CV_DIR, os.path.basename(url))
    if not os.path.exists(path):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def _frames(src, path):
    import cv2
    if src["kind"] == "image":
        img = cv2.imread(path)
        if img is None:
            raise ValueError("unreadable image")
        yield 0, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    for idx in np.linspace(0, n - 1, max(1, CV_VIDEO_FRAMES)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            yield int(idx), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()


def _privacy_preview(rgb, boxes, block=None):
    """Privacy: the WHOLE frame is pixelated (so a person the detector missed is not identifiable either),
    then detection boxes are drawn on top. Display only; never saved or passed downstream."""
    import cv2
    h, w = rgb.shape[:2]
    block = block or max(10, w // 60)
    small = cv2.resize(rgb, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    for x1, y1, x2, y2, _ in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (250, 204, 21), max(2, w // 200))
    return out


def run_computer_vision(sources=CV_SOURCES):
    """Each camera → anonymized aggregate: people counted, load factor, detector confidence. No images or identities kept."""
    results = {"detector": None, "cameras": [], "errors": [], "ran_at": iso(utcnow()), "previews": []}
    if not USE_CV:
        results["errors"].append("computer vision disabled")
        return results
    results["detector"] = load_detector()
    if not results["detector"]:
        results["errors"] += _DETECTOR["error"]
        return results
    for src in sources:
        try:
            path = _download(src["url"])
            counts, confs, preview = [], [], None
            for idx, rgb in _frames(src, path):
                boxes = detect_people(rgb)
                counts.append(len(boxes))
                confs += [b[4] for b in boxes]
                if preview is None or len(boxes) > preview[1]:
                    preview = (_privacy_preview(rgb, boxes), len(boxes), idx)
            if not counts:
                raise ValueError("no frames read")
            people = int(round(statistics.median(counts)))
            results["cameras"].append({
                "camera_id": src["camera_id"], "role": src["role"], "stop_id": src["stop_id"],
                "frames_analyzed": len(counts), "people_per_frame": counts, "person_count": people,
                "peak_count": max(counts), "capacity": src["capacity"], "load_factor": round(people / src["capacity"], 2),
                "mean_detector_score": round(float(np.mean(confs)), 2) if confs else None,   # probability for Faster R-CNN; SVM margin for HOG
                "observed_at": iso(utcnow()), "identities_stored": False, "images_stored": False, "note": src.get("note", ""),
            })
            if SHOW_CV_PREVIEW and preview is not None:
                results["previews"].append((src["camera_id"], preview))
        except Exception as e:
            results["errors"].append(f"{src['camera_id']}: {type(e).__name__}: {str(e)[:120]}")
    return results


def show_cv_previews(results):
    if not results["previews"]:
        return
    fig, axes = plt.subplots(1, len(results["previews"]), figsize=(6 * len(results["previews"]), 5))
    axes = np.atleast_1d(axes)
    for ax, (cam, (img, n, idx)) in zip(axes, results["previews"]):
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"{cam}\n{n} {'person' if n == 1 else 'people'} detected (frame {idx}) · frame pixelated for privacy", fontsize=9)
    plt.tight_layout(); plt.show()


CV_RESULTS = run_computer_vision()
print("Detector:", CV_RESULTS["detector"] or "unavailable")
for cam in CV_RESULTS["cameras"]:
    print(f"  {cam['camera_id']} ({cam['role']}): {cam['person_count']} counted (median of {cam['frames_analyzed']} frame(s), "
          f"peak {cam['peak_count']}), load factor {cam['load_factor']}, mean detector score {cam['mean_detector_score']}")
    print(f"      imagery: {cam['note']}")
for err in CV_RESULTS["errors"]:
    print("  ⚠️", err)
show_cv_previews(CV_RESULTS)
_ = CV_RESULTS.pop("previews", None)   # pixelated previews are display-only and are not carried into any case record


# ====================================================================================================
# notebook cell 16: A1 · Live data connectors: NJ Transit GTFS-realtime, US Census ACS, National Weather Service
# ====================================================================================================
#@title A1 · Live data connectors: NJ Transit GTFS-realtime, US Census ACS, National Weather Service
DATA_MODE = "live"  #@param ["live", "sample"]
NJT_GTFS_SYSTEM = "GTFSG2"  #@param ["GTFSG2", "GTFS"]
NJT_NEARBY_KM = 2.0  #@param {type:"number"}
ACS_YEAR = 2023  #@param {type:"integer"}
import re
from datetime import datetime, timezone, timedelta

LAT, LON = SERVICE_AREA_LATLON           # Newark Penn Station
LIVE = DATA_MODE == "live"


def _ago(seconds):
    return iso(datetime.now(timezone.utc) - timedelta(seconds=seconds))


# ---------------------------------------------------------------- NJ Transit GTFS / GTFS-realtime
# Login: form post → {"Authenticated": "True", "UserToken": ...}. Data calls send the token as a multipart form field
# and return GTFS-realtime protobuf (trip updates, vehicle positions, alerts). The token is kept in memory only.
NJT_BASE = "https://pcsdata.njtransit.com/api/{system}/"
_NJT = {"token": None, "system": None}


def njt_authenticate():
    user, pw = read_secret("NJ_TRANSIT_USERNAME"), read_secret("NJ_TRANSIT_PASSWORD")
    if not (user and pw):
        return "skipped (no NJ_TRANSIT_USERNAME / NJ_TRANSIT_PASSWORD in Colab Secrets)"
    last = "not attempted"
    for system in dict.fromkeys([NJT_GTFS_SYSTEM, "GTFSG2", "GTFS"]):
        url = NJT_BASE.format(system=system) + "authenticateUser"
        for mode in ("form", "multipart"):
            try:
                if mode == "form":
                    r = requests.post(url, data={"username": user, "password": pw}, timeout=20)
                else:
                    r = requests.post(url, files={"username": (None, user), "password": (None, pw)}, timeout=20)
                r.raise_for_status()
                body = r.json()
            except (requests.RequestException, ValueError) as e:
                last = f"{system}: request failed ({type(e).__name__})"
                continue
            token = body.get("UserToken") if isinstance(body, dict) else None
            if str(body.get("Authenticated", "")).lower() == "true" and token:
                _NJT.update(token=token, system=system)
                return f"authenticated ({system})"
            last = f"{system}: rejected (no token returned)"
    return last


def njt_feed(endpoint):
    from google.transit import gtfs_realtime_pb2
    url = NJT_BASE.format(system=_NJT["system"]) + endpoint
    r = requests.post(url, files={"token": (None, _NJT["token"])}, timeout=30)
    r.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(r.content)
    return feed


def _text(ts):
    return ts.translation[0].text.strip() if ts and len(ts.translation) else ""


def fetch_njt_service():
    """Live service feed near Newark Penn Station: the most-delayed nearby vehicle and any alert on its route."""
    out = {"status": njt_authenticate(), "feeds": {}, "service": None, "alert": None}
    if not _NJT["token"]:
        return out
    feeds = {}
    for ep in ("getVehiclePositions", "getTripUpdates", "getAlerts"):
        try:
            feeds[ep] = njt_feed(ep)
            out["feeds"][ep] = f"ok ({len(feeds[ep].entity)} entities)"
        except Exception as e:
            out["feeds"][ep] = f"failed ({type(e).__name__})"
    vp, tu, al = feeds.get("getVehiclePositions"), feeds.get("getTripUpdates"), feeds.get("getAlerts")
    if vp is None:
        return out
    updates = {}
    if tu is not None:
        for ent in tu.entity:
            if ent.HasField("trip_update"):
                t = ent.trip_update
                delays = [s.arrival.delay if s.HasField("arrival") else s.departure.delay
                          for s in t.stop_time_update if s.HasField("arrival") or s.HasField("departure")]
                updates[t.trip.trip_id] = {"delay_s": delays[0] if delays else (t.delay if t.HasField("delay") else None),
                                           "cancelled": t.trip.schedule_relationship == t.trip.CANCELED,
                                           "route_id": t.trip.route_id}
    nearby = []
    for ent in vp.entity:
        if not ent.HasField("vehicle"):
            continue
        v = ent.vehicle
        d = km_between((v.position.latitude, v.position.longitude), (LAT, LON))
        if d <= NJT_NEARBY_KM:
            u = updates.get(v.trip.trip_id, {})
            nearby.append({"route_id": v.trip.route_id or u.get("route_id"), "trip_id": v.trip.trip_id,
                           "vehicle_id": v.vehicle.id, "lat": v.position.latitude, "lon": v.position.longitude,
                           "distance_km": round(d, 2), "delay_s": u.get("delay_s"), "cancelled": u.get("cancelled", False),
                           "timestamp": v.timestamp or vp.header.timestamp})
    out["nearby_vehicles"] = len(nearby)
    if not nearby:
        return out
    pick = max(nearby, key=lambda n: (n["cancelled"], n["delay_s"] or 0))
    out["service"] = pick
    if al is not None:
        for ent in al.entity:
            if ent.HasField("alert") and any(ie.route_id == pick["route_id"] for ie in ent.alert.informed_entity):
                a = ent.alert
                out["alert"] = {"header": _text(a.header_text), "description": _text(a.description_text)[:400],
                                "effect": a.Effect.Name(a.effect) if a.effect else None}
                break
    return out


# ---------------------------------------------------------------- US Census ACS (tract-level equity inputs)
ACS_DETAILED = {"B08201_001E": "households", "B08201_002E": "households_no_vehicle",
                "B08301_001E": "workers", "B08301_010E": "workers_public_transit", "B19013_001E": "median_household_income"}
ACS_SUBJECT = {"S1701_C03_001E": "poverty_rate_pct", "S1810_C03_001E": "disability_rate_pct"}


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = _find_key(v, key)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_key(v, key)
            if hit is not None:
                return hit
    return None


def _acs(path, variables, geo, key):
    params = {"get": ",".join(["NAME"] + list(variables)), "for": f"tract:{geo['TRACT']}",
              "in": f"state:{geo['STATE']} county:{geo['COUNTY']}"}
    if key:
        params["key"] = key
    r = requests.get(f"https://api.census.gov/data/{ACS_YEAR}/{path}", params=params, timeout=30)
    r.raise_for_status()
    header, row = r.json()[:2]
    vals = dict(zip(header, row))
    out = {"NAME": vals.get("NAME")}
    for code, name in variables.items():
        try:
            v = float(vals.get(code))
            out[name] = None if v < 0 else v          # negative = Census "not available" sentinel
        except (TypeError, ValueError):
            out[name] = None
    return out


def fetch_census(lat, lon):
    key = read_secret("CENSUS_API_KEY")
    res = {"status": "not attempted", "tract": None}
    try:
        r = requests.get("https://geocoding.geo.census.gov/geocoder/geographies/coordinates",
                         params={"x": lon, "y": lat, "benchmark": "Public_AR_Current", "vintage": "Current_Current",
                                 "layers": "Census Tracts", "format": "json"}, timeout=30)
        r.raise_for_status()
        tracts = _find_key(r.json(), "Census Tracts")
        geo = tracts[0] if tracts else None
        if not geo:
            res["status"] = "geocoder returned no tract"
            return res
        d = _acs("acs/acs5", ACS_DETAILED, geo, key)
        s = _acs("acs/acs5/subject", ACS_SUBJECT, geo, key)
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
        res["status"] = f"failed ({type(e).__name__})"
        return res
    share = lambda a, b: round(a / b, 3) if (a is not None and b) else None
    res.update({
        "status": f"ok (ACS {ACS_YEAR} 5-year{'' if key else ', no key'})",
        "tract": geo.get("GEOID") or f"{geo['STATE']}{geo['COUNTY']}{geo['TRACT']}", "tract_name": d["NAME"],
        "zero_vehicle_share": share(d["households_no_vehicle"], d["households"]),
        "transit_commute_share": share(d["workers_public_transit"], d["workers"]),
        "poverty_rate": None if s["poverty_rate_pct"] is None else round(s["poverty_rate_pct"] / 100, 3),
        "disability_rate": None if s["disability_rate_pct"] is None else round(s["disability_rate_pct"] / 100, 3),
        "median_household_income": d["median_household_income"],
    })
    return res


# ---------------------------------------------------------------- National Weather Service (7 connectors)
NWS_BASE = "https://api.weather.gov"


def _nws_headers():
    email = read_secret("NWS_CONTACT_EMAIL")
    return {"User-Agent": f"TransitAIStudentProject ({email})" if email else "TransitAIStudentProject",
            "Accept": "application/geo+json"}


def get_weather(path, log):
    try:
        response = requests.get(NWS_BASE + path, headers=_nws_headers(), timeout=15)
        response.raise_for_status()
        log.append({"connector": path.split("?")[0], "status": "ok"})
        return response.json()
    except (requests.RequestException, ValueError) as e:
        log.append({"connector": path.split("?")[0], "status": f"failed ({type(e).__name__})"})
        return {}


def fetch_weather(lat, lon):
    log, fetched_at = [], _ago(0)
    points = get_weather(f"/points/{lat},{lon}", log)                         # 1 location and grid
    grid = points.get("properties", {})
    office, x, y = grid.get("gridId"), grid.get("gridX"), grid.get("gridY")
    alerts = get_weather(f"/alerts/active?point={lat},{lon}", log)            # 2 active alerts
    observation, hourly, grid_data = {}, {}, {}
    if office is not None and x is not None and y is not None:
        path = f"/gridpoints/{office}/{x},{y}"
        stations = get_weather(path + "/stations", log)                       # 3 nearby stations
        hourly = get_weather(path + "/forecast/hourly", log)                  # 4 hourly forecast
        grid_data = get_weather(path, log)                                    # 5 gridpoint data
        feats = stations.get("features", [])
        sid = feats[0].get("properties", {}).get("stationIdentifier") if feats else None
        if sid:
            observation = get_weather(f"/stations/{sid}/observations/latest", log)  # 6 latest observation
    alert_types = get_weather("/alerts/types", log)                           # 7 alert types
    op = observation.get("properties", {})
    return {
        "source": "NWS live" if any(r["status"] == "ok" for r in log) else "unavailable",
        "fetched_at": fetched_at, "current_conditions": op.get("textDescription"), "observed_at": op.get("timestamp"),
        "active_alerts": [{"event": a.get("properties", {}).get("event"), "severity": a.get("properties", {}).get("severity"),
                           "sent": a.get("properties", {}).get("sent")} for a in alerts.get("features", [])],
        "hourly_forecast": [{"time": p.get("startTime"), "forecast": p.get("shortForecast")}
                            for p in hourly.get("properties", {}).get("periods", [])[:3]],
        "grid_data_available": bool(grid_data), "alert_types_available": bool(alert_types), "connectors": log,
    }


_fetch_njt_once = fetch_njt_service


def fetch_njt_service():
    out = _fetch_njt_once()
    out["fetched_at"] = iso(utcnow())
    return out


# Live connectors run ONCE here (not on every dashboard click). Re-run this cell to refresh.
LIVE_CONTEXT = {
    "nj_transit": fetch_njt_service() if LIVE else {"status": "sample mode", "service": None, "alert": None, "feeds": {}},
    "census": fetch_census(LAT, LON) if LIVE else {"status": "sample mode", "tract": None},
    "weather": fetch_weather(LAT, LON) if LIVE else {"source": "sample mode", "active_alerts": []},
}
_n, _c, _w = LIVE_CONTEXT["nj_transit"], LIVE_CONTEXT["census"], LIVE_CONTEXT["weather"]
print(f"NJ Transit GTFS-RT: {_n['status']} · feeds: {_n.get('feeds') or '-'} · nearby vehicles: {_n.get('nearby_vehicles', '-')}")
if _n.get("service"):
    s = _n["service"]
    print(f"  selected route {s['route_id']} (vehicle {s['vehicle_id']}, {s['distance_km']} km away): "
          f"delay {None if s['delay_s'] is None else round(s['delay_s'] / 60, 1)} min, cancelled={s['cancelled']}, "
          f"alert: {(_n.get('alert') or {}).get('header') or 'none'}")
print(f"US Census ACS: {_c['status']}" + (f" · {_c['tract_name']} · zero-vehicle {_c['zero_vehicle_share']}, "
      f"poverty {_c['poverty_rate']}, disability {_c['disability_rate']}, transit commute {_c['transit_commute_share']}"
      if _c.get("tract") else ""))
print(f"NWS weather: {_w.get('source')} · {_w.get('current_conditions') or 'conditions not reported'} · active alerts: {len(_w.get('active_alerts', []))} · "
      f"connectors ok: {sum(r['status'] == 'ok' for r in _w.get('connectors', []))}/{len(_w.get('connectors', []))}")


# ====================================================================================================
# notebook cell 17: A2 · Data inputs (live where available, sample otherwise) + small language model + hand-off to the Signal Scout
# ====================================================================================================
#@title A2 · Data inputs (live where available, sample otherwise) + small language model + hand-off to the Signal Scout
USE_SLM = True  #@param {type:"boolean"}
USE_LLM_ORGANIZER = True  #@param {type:"boolean"}
SLM_MODEL_ID = "google/flan-t5-small"

# 1. DATA INPUTS — seven sources. Each value is live when its connector succeeded, otherwise the documented sample;
# every source says which ("source"), and every source carries "observed_at" so freshness can be measured.
def _cv_stop_camera():
    cams = [c for c in CV_RESULTS.get("cameras", []) if c["role"] == "bus stop curb camera"] or CV_RESULTS.get("cameras", [])
    return cams[0] if cams else None


def section_a_sample_inputs():
    """Name kept for the dashboard. Returns live inputs where available (DATA_MODE='live'), samples otherwise."""
    n, c, w = LIVE_CONTEXT["nj_transit"], LIVE_CONTEXT["census"], LIVE_CONTEXT["weather"]
    svc_live, alert = n.get("service"), n.get("alert")
    cam = _cv_stop_camera()
    if svc_live:
        # keep one decimal: rounding 14.7 → 15 would cross the 15-minute hard trigger on its own
        delay = None if svc_live["delay_s"] is None else max(0.0, round(svc_live["delay_s"] / 60, 1))
        service = {"route": svc_live["route_id"], "scheduled_headway_min": 15, "headway_source": "assumed (no static GTFS loaded)",
                   "delay_min": delay, "cancelled": bool(svc_live["cancelled"]), "trip_id": svc_live["trip_id"],
                   "vehicle_position": f"{svc_live['lat']:.5f},{svc_live['lon']:.5f} ({svc_live['distance_km']} km from Newark Penn)",
                   "source": f"NJ Transit {_NJT['system']} GTFS-realtime (live)",
                   "observed_at": iso(datetime.fromtimestamp(svc_live["timestamp"], timezone.utc))
                   if svc_live.get("timestamp") else _ago(0)}
    else:
        service = {"route": "NJ-001", "scheduled_headway_min": 15, "delay_min": 18, "cancelled": False,
                   "vehicle_position": "Newark Penn Station", "source": "sample_data", "observed_at": _ago(30)}
    if alert:
        text = f"{alert['header']}. {alert['description']}".strip(". ")
        events = {"source_type": "official_alert", "report": text, "report_source": "NJ Transit GTFS-realtime alert (live)",
                  "source": "NJ Transit GTFS-realtime alert (live)",
                  "road_closure": any(k in text.lower() for k in ("detour", "road closure", "closed"))}
    elif svc_live:
        events = {"source_type": "none", "report_source": "generated from the live vehicle feed (no active alert)",
                  "source": "NJ Transit GTFS-realtime vehicle feed (live; no alert)",
                  "report": f"Route {service['route']} is running {service['delay_min'] or 0:g} minutes late near Newark Penn "
                            f"Station. No active NJ Transit service alert for this route.", "road_closure": False}
    else:
        events = {"source_type": "official_alert", "report_source": "sample_data", "source": "sample_data", "road_closure": True,
                  "report": "Route NJ-001 has an 18 minute delay near Newark Penn Station due to a road closure."}
    w_live = w.get("source") == "NWS live"
    events.update(weather=(w.get("current_conditions") or "not reported") if w_live else "Clear",
                  weather_alerts=w.get("active_alerts", []) if w_live else [],
                  weather_source="NWS live" if w_live else "sample_data", observed_at=_ago(90) if not alert else _ago(0))
    if w_live:
        events["source"] += " + NWS weather (live)"
    if cam:
        # The camera sees one stop: it measures crowding there (load factor), not total riders on the route,
        # so route boardings stay the documented sample and are labeled as such.
        vision = {"boardings": 45, "boardings_source": "sample_data (route boardings are not visible from one camera)",
                  "waiting_at_stop": cam["person_count"], "load_factor": cam["load_factor"], "alightings": None,
                  "capacity": cam["capacity"], "pass_ups": 0, "pass_ups_source": "not observable in a still image (0 assumed)",
                  "obstruction": False, "identities_stored": False, "camera_id": cam["camera_id"],
                  "detector": CV_RESULTS["detector"], "source": "computer vision (anonymized counts)", "observed_at": cam["observed_at"]}
    else:
        vision = {"boardings": 45, "alightings": 12, "capacity": 50, "pass_ups": 5, "obstruction": False,
                  "identities_stored": False, "source": "sample_data", "observed_at": _ago(20)}
    if c.get("tract") and None not in (c["zero_vehicle_share"], c["poverty_rate"], c["disability_rate"]):
        community = {"transit_dependence": c["zero_vehicle_share"], "income_disadvantage": c["poverty_rate"],
                     "disability_barrier": c["disability_rate"], "transit_commute_share": c["transit_commute_share"],
                     "median_household_income": c["median_household_income"], "essential_destination": "Hospital",
                     "essential_destination_source": "sample_data",
                     "data_level": f"{c['tract_name']} (tract {c['tract']}, ACS {ACS_YEAR} 5-year)",
                     "source": "US Census ACS (live)", "observed_at": _ago(0)}
    else:
        community = {"transit_dependence": 0.75, "income_disadvantage": 0.65, "disability_barrier": 0.40,
                     "essential_destination": "Hospital", "data_level": "Area aggregate", "source": "sample_data",
                     "observed_at": _ago(30 * 86400)}
    return {
        "service_feeds": service,
        "vision_ridership": vision,
        "connecting_services": {"connecting_route": "NJ-002", "arrival_delay_min": 8, "transfer_window_min": 10,
                                "last_service": False, "source": "sample_data", "observed_at": _ago(60)},
        "incident_history": {"past_disruptions": 3, "average_recovery_min": 25, "previous_response": "Rider alert",
                             "source": "sample_data", "observed_at": _ago(86400)},
        "community_access": community,
        "operating_resources": {"available_buses": 2, "available_operators": 2, "estimated_dispatch_cost": 150,
                                "estimated_travel_min": 12, "source": "sample_data", "observed_at": _ago(120)},
        "event_signals": events,
    }


NJT_MAX_SNAPSHOT_AGE_S = 240   # refetch the vehicle feed if A1 ran more than 4 minutes ago (freshness limit is 300 s)


def refresh_live_service(max_age_s=NJT_MAX_SNAPSHOT_AGE_S):
    """A1 fetches once; if this cell runs later the snapshot may be stale, so take a new one before the hand-off."""
    n = LIVE_CONTEXT.get("nj_transit", {})
    fetched = n.get("fetched_at") or iso(utcnow())
    if LIVE and n.get("status", "").startswith("authenticated") and (utcnow() - parse_iso(fetched)).total_seconds() > max_age_s:
        LIVE_CONTEXT["nj_transit"] = fetch_njt_service()
        print(f"NJ Transit snapshot was older than {max_age_s} s → refetched: {LIVE_CONTEXT['nj_transit']['status']}")


refresh_live_service()
raw_inputs = section_a_sample_inputs()


# 2. STANDARDIZE AND COLLATE INPUT DATA
def collate_inputs(data):
    combined = []
    collated_at = datetime.now(timezone.utc)
    for source_name, source_data in data.items():
        observed = source_data.get("observed_at")
        age_s = int((collated_at - parse_iso(observed)).total_seconds()) if observed else None
        combined.append({
            "source": source_name,
            "timestamp": observed or iso(collated_at),
            "collated_at": iso(collated_at),
            "age_s": age_s,
            "data": source_data,
            "status": "unverified",
        })
    return combined


# 3. SMALL LANGUAGE MODEL - FLAN-T5
# loaded as a seq2seq model with explicit generation arguments.
# one whole-report prompt made FLAN-T5 small echo the report back (observed in an earlier run).
# It now answers one short question per field, the task format FLAN-T5 was trained on.
_SLM = {"tok": None, "model": None, "error": None}
SLM_QUESTIONS = {
    "route": "Which transit route is affected?",
    "delay": "How many minutes is the delay?",
    "location": "Where is the problem?",
    "cause": "What caused the delay?",
}


def load_slm():
    if _SLM["model"] is not None:
        return True
    if _SLM["error"]:
        return False
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print("Loading small language model...")
        _SLM["tok"] = AutoTokenizer.from_pretrained(SLM_MODEL_ID)
        _SLM["model"] = AutoModelForSeq2SeqLM.from_pretrained(SLM_MODEL_ID)
        return True
    except Exception as e:  # no internet / no transformers: deterministic fields still work
        _SLM["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        return False


def slm_answers(report):
    """Returns {field: short answer} or None. Supporting information only: never verified, never scored."""
    if not (USE_SLM and load_slm()):
        return None
    tok, model = _SLM["tok"], _SLM["model"]
    answers = {}
    for field_name, q in SLM_QUESTIONS.items():
        prompt = f"Answer from the report only. If it is not stated, answer 'unknown'.\nReport: {report}\nQuestion: {q}"
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        out = model.generate(**ids, max_new_tokens=24, do_sample=False)
        answers[field_name] = tok.decode(out[0], skip_special_tokens=True).strip()
    return answers


def interpret_event(report):
    """Kept for the dashboard: one readable line built from the per-field answers."""
    a = slm_answers(report)
    if a is None:
        return f"[SLM unavailable — {_SLM['error'] or 'disabled'}; deterministic fields only]"
    return "; ".join(f"{k}: {v}" for k, v in a.items())


# 4. CREATE STRUCTURED EVENT FIELDS (deterministic; numbers never come from a language model)
# broader delay patterns
DELAY_PATTERNS = [
    # a range reports the upper end, but only in delay wording ("delays of 10 to 15 minutes", "10-15 min late"),
    # so service frequency ("buses every 10 to 15 minutes") or transfer advice is never read as a delay
    r"delay(?:ed|s)?\s*(?:of|by)?\s*(?:up to|about|around|approximately)?\s*(\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b",
    r"(\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\s*(?:delay|late|behind)",
    r"(\d+(?:\.\d+)?)\s*-?\s*(?:minutes?|mins?)\s*(?:delay|late|behind)",
    r"delay(?:ed|s)?\s*(?:of|by)?\s*(?:up to|about|around|approximately)?\s*(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)",
]
CAUSE_KEYWORDS = {
    "road_closure": ["road closure", "closed road", "road closed", "detour"],
    "weather": ["snow", "ice", "icy", "storm", "flood", "weather", "heavy rain"],
    "obstruction": ["crash", "collision", "accident", "blocked", "obstruction", "disabled vehicle"],
    "mechanical": ["mechanical", "breakdown", "operator shortage", "vehicle failure"],
    "medical": ["medical", "sick passenger"],
}


def extract_delay(text):
    for p in DELAY_PATTERNS:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            v = max(float(g) for g in m.groups() if g)
            return int(v) if v.is_integer() else v
    return None


def classify_cause(text):
    t = text.lower()
    for cause, kws in CAUSE_KEYWORDS.items():
        # a keyword matches as a whole word or with a common ending ("flood" → "flooding", "storm" → "storms"),
        # so "ice" inside "service", "Icebreaker" or "Floodlights" is not weather
        if any(re.search(rf"\b{re.escape(k)}(?:s|es|ed|ing|y)?\b", t) for k in kws):
            return cause
    return "none"


def slm_cross_check(answers, delay_min, route):
    """Flags where the SLM reading disagrees with the deterministic fields (for the reviewer, not for scoring)."""
    if not answers:
        return []
    flags = []
    if delay_min is not None and str(delay_min) not in answers.get("delay", ""):
        flags.append(f"SLM delay '{answers.get('delay')}' differs from extracted {delay_min} min")
    if route and route.lower() not in answers.get("route", "").lower():
        flags.append(f"SLM route '{answers.get('route')}' differs from feed route {route}")
    return flags


# 5. LLM ORGANIZER (shared live client; organizes the record for the Signal Scout — does not score, verify or recommend)
def llm_organize(report, slm_text, weather, sources):
    if not USE_LLM_ORGANIZER:
        return None, "disabled"
    if not LLM_READY:
        return None, "skipped (no LLM key in Colab Secrets)"
    try:
        txt = llm_generate(json.dumps({"report": report, "slm_interpretation": slm_text, "weather": weather,
                                       "data_sources": sources}),
                           "Organize transit incident data for the Signal Scout agent in at most 6 short bullet points. "
                           "Separate confirmed live information from sample or unverified information, using the "
                           "data_sources labels. Do not calculate scores, verify incidents, or recommend operational actions.",
                           max_tokens=300)
        bullets = [l.strip() for l in txt.splitlines() if l.strip().startswith(("-", "•", "*")) and len(l.strip()) > 2][:6]
        if not bullets:
            return None, f"discarded: {llm_source()} did not return bullet points"
        return "\n".join(bullets), f"completed ({llm_source()})"
    except LLMUnavailable as e:
        return None, f"failed ({str(e)[:120]})"


def build_signal_scout_input(inputs):
    """Collate → SLM reading → deterministic fields → optional LLM organization → hand-off for the Signal Scout."""
    collated_data = collate_inputs(inputs)
    ev = inputs["event_signals"]
    event_report = ev["report"]
    answers = slm_answers(event_report)
    slm_text = ("; ".join(f"{k}: {v}" for k, v in answers.items()) if answers
                else f"[SLM unavailable — {_SLM['error'] or 'disabled'}; deterministic fields only]")
    delay = extract_delay(event_report)
    route = inputs["service_feeds"]["route"]
    weather_ctx = {"conditions": ev.get("weather"), "alerts": ev.get("weather_alerts", []),
                   "source": ev.get("weather_source", "sample_data")}
    llm_text, llm_status = llm_organize(event_report, slm_text, weather_ctx,
                                        {k: v.get("source", "sample_data") for k, v in inputs.items()})
    structured_event = {
        "route": route,
        "delay_min": delay,
        "cause": classify_cause(event_report),
        "report_text": event_report,
        "slm_interpretation": slm_text,
        "slm_answers": answers,
        "slm_cross_check": slm_cross_check(answers, delay, route),
        "llm_interpretation": llm_text,
        "source_type": ev.get("source_type", "public_report"),
        "verification_status": "pending",
    }
    return {
        "case_status": "new",
        "status": "awaiting_verification",
        "created_at": iso(utcnow()),
        "route": route,
        "event": structured_event,
        "all_inputs": collated_data,
        "external_feeds": {
            "data_mode": DATA_MODE,
            "nj_transit": {k: v for k, v in LIVE_CONTEXT["nj_transit"].items() if k in ("status", "feeds", "nearby_vehicles")},
            "census": LIVE_CONTEXT["census"],
            "weather": {k: v for k, v in LIVE_CONTEXT["weather"].items() if k != "connectors"},
            "weather_connectors": LIVE_CONTEXT["weather"].get("connectors", []),
            "computer_vision": {"detector": CV_RESULTS.get("detector"), "cameras": CV_RESULTS.get("cameras", [])},
            "source_of_each_input": {k: v.get("source", "sample_data") for k, v in inputs.items()},
            "llm": llm_status,
        },
        "verification_required": True,
        "confidence_score": None,
        "priority_score": None,
        "audit_log": [{"step": "input_preparation", "status": "completed",
                       "message": "Data collated and prepared for Signal Scout"}],
    }


signal_scout_input = build_signal_scout_input(raw_inputs)

# 6. EXPORT AND DISPLAY
with open(os.path.join(OUTPUT_DIR, "signal_scout_input.json"), "w") as file:
    json.dump(signal_scout_input, file, indent=4, default=str)

print("===== SOURCE OF EACH INPUT =====")
for k, v in signal_scout_input["external_feeds"]["source_of_each_input"].items():
    print(f"  {k:22s} {v}")
print("LLM organizer:", signal_scout_input["external_feeds"]["llm"])
if signal_scout_input["event"]["llm_interpretation"]:
    print(signal_scout_input["event"]["llm_interpretation"])
print("===== SMALL LANGUAGE MODEL OUTPUT (supporting only) =====")
print(signal_scout_input["event"]["slm_interpretation"])
for flag in signal_scout_input["event"]["slm_cross_check"]:
    print("  check:", flag)
print("===== STRUCTURED EVENT (deterministic) =====")
print({k: v for k, v in signal_scout_input["event"].items()
       if k not in ("slm_interpretation", "slm_answers", "slm_cross_check", "llm_interpretation")})
print("JSON file saved:", os.path.join(OUTPUT_DIR, "signal_scout_input.json"))


# ====================================================================================================
# notebook cell 18: A · INTEGRATION — adapter from Section A's hand-off to the shared raw bundle
# ====================================================================================================
#@title A · INTEGRATION — adapter from Section A's hand-off to the shared raw bundle
# Routes that are not in the synthetic network are drawn on a stand-in route; A's own values are kept.
EXTERNAL_ROUTE_MAP = {"NJ-001": "R2", "12": "R3"}
ESSENTIAL_NAME_TO_TYPE = {"hospital": "hospital", "dialysis": "dialysis", "clinic": "clinic", "school": "school",
                          "college": "school", "university": "school", "employment": "employment_center",
                          "job": "employment_center", "market": "retail", "grocery": "retail"}
NWS_SEVERITY_RANK = {"Unknown": 0, "Minor": 1, "Moderate": 2, "Severe": 3, "Extreme": 4}
NWS_TO_EVENT_SEVERITY = {"Unknown": "minor", "Minor": "minor", "Moderate": "minor", "Severe": "major", "Extreme": "emergency"}
SEVERITY_BY_CAUSE = {"road_closure": "major", "obstruction": "major", "weather": "minor", "mechanical": "minor",
                     "medical": "minor", "none": "none"}


def essential_type_of(name):
    n = (name or "").lower()
    return next((t for k, t in ESSENTIAL_NAME_TO_TYPE.items() if k in n), "none")


def nearest_depot_id(route_id, stop_range):
    stops = NET["routes"][route_id]["stops"][stop_range[0]:stop_range[1] + 1]
    mid = (statistics.mean(s["lat"] for s in stops), statistics.mean(s["lon"] for s in stops))
    return min(NET["depots"].values(), key=lambda d: km_between((d["lat"], d["lon"]), mid))["id"]


def transit_time_period(ts=None):
    """Service period from the local (New Jersey) time of the observation, instead of always assuming peak."""
    from zoneinfo import ZoneInfo
    t = (parse_iso(ts) if ts else utcnow()).astimezone(ZoneInfo("America/New_York"))
    h = t.hour + t.minute / 60
    if t.weekday() >= 5:
        return "midday" if 9 <= h < 20 else "off_peak"
    if 6 <= h < 9.5 or 15.5 <= h < 19:
        return "peak"
    if 9.5 <= h < 15.5:
        return "midday"
    return "off_peak"


def signal_scout_input_to_raw(ssi, name=None):
    """Section A record → raw bundle used by the Signal Scout / E2 engine. Numbers come only from structured fields
    or the deterministic regex; the SLM interpretation is attached as a note."""
    src = {r["source"]: r for r in ssi["all_inputs"]}
    svc, vis = src["service_feeds"]["data"], src["vision_ridership"]["data"]
    conn, hist = src["connecting_services"]["data"], src["incident_history"]["data"]
    comm, res = src["community_access"]["data"], src["operating_resources"]["data"]
    ev = src["event_signals"]["data"]
    ext_route = svc["route"]
    rid = EXTERNAL_ROUTE_MAP.get(ext_route, "R2")
    stop_range = (1, 5)
    headway = svc.get("scheduled_headway_min") or NET["routes"][rid]["headway_min"]
    delay = svc.get("delay_min")
    cancelled = bool(svc.get("cancelled"))
    # Gap between consecutive buses when one runs `delay` minutes late (or one trip is missing) ÷ scheduled headway
    gap_ratio = 2.0 if cancelled else (round((headway + (delay or 0)) / headway, 2) if headway else None)
    cause = ssi["event"]["cause"] if ssi["event"]["cause"] != "none" else ("road_closure" if ev.get("road_closure") else "none")
    stype = ev.get("source_type", "public_report")
    sources = ["vehicle_apc"] + ([stype] if stype in ("official_alert", "trusted_external", "public_report") else [])
    # weather no longer adds a generic trusted_external source. That path created a synthetic "511-TRAFFIC"
    # observation repeating the bus delay + 1 min, which inflated agreement (G) and confidence. Real NWS alerts are
    # added below as their own observation, with no delay value.
    cap = vis.get("capacity") or 60
    dep = nearest_depot_id(rid, stop_range)
    spec = dict(
        route_id=rid, stop_range=stop_range, time_period=transit_time_period(src["service_feeds"].get("observed_at")),
        delay_min=delay, headway_ratio=gap_ratio, headway_from_delay=True,   # gap is computed from the delay, not measured
        backup_available=False,   # no second live feed is connected: a stale live snapshot must go to monitoring, not to a simulated mirror
        cancelled=cancelled, cancellation_risk=1.0 if cancelled else 0.2, boardings=vis.get("boardings", 0),
        load_factor=vis.get("load_factor", round(vis.get("boardings", 0) / cap, 2)), passups=vis.get("pass_ups", 0),
        obstruction=bool(vis.get("obstruction")), event_type=cause, event_severity=SEVERITY_BY_CAUSE.get(cause, "minor"),
        event_verified=stype == "official_alert", sources=sources, official_delay=ssi["event"]["delay_min"],
        public_delay=ssi["event"]["delay_min"], public_corroborated=False,
        feed_age_s=src["service_feeds"]["age_s"] or 0, alert_age_s=src["event_signals"]["age_s"] or 0,
        last_service=bool(conn.get("last_service")),
        depot_overrides={dep: {"spare_buses": res.get("available_buses", 0), "operators": res.get("available_operators", 0),
                               "deadhead_min": res.get("estimated_travel_min"), "dispatch_cost": res.get("estimated_dispatch_cost")}},
    )
    raw = make_raw(spec, name or f"Section A hand-off · {ext_route} ({cause.replace('_', ' ')})")
    raw["external_route_id"] = ext_route                               # keep the real route label
    raw["route_name"] = f"{ext_route} (drawn on synthetic {rid})"
    raw["community"].update(transit_dependence=comm.get("transit_dependence"), income_disadvantage=comm.get("income_disadvantage"),
                            disability=comm.get("disability_barrier"), areas=[comm.get("data_level", "area aggregate")])
    et = essential_type_of(comm.get("essential_destination"))
    raw["access"].update(essential_type=et, essential_names=[comm.get("essential_destination")] if et != "none" else [])
    raw["connections"].update(external_connecting_route=conn.get("connecting_route"),
                              transfer_window_min=conn.get("transfer_window_min", 4),
                              connecting_arrival_delay_min=conn.get("arrival_delay_min"))
    raw["incident_history"] = hist
    raw["slm_interpretation"] = ssi["event"]["slm_interpretation"]
    raw["slm_cross_check"] = ssi["event"].get("slm_cross_check", [])
    raw["llm_interpretation"] = ssi["event"].get("llm_interpretation")       # supporting text only, never scored
    raw["external_feeds"] = ssi.get("external_feeds", {})
    # active NWS alerts become one trusted_external observation (event only, no delay).
    # Field precedence still applies: an agency alert's event fields outrank NWS.
    w_alerts = ev.get("weather_alerts") or []
    if w_alerts:
        top = max(w_alerts, key=lambda a: NWS_SEVERITY_RANK.get(a.get("severity"), 0))
        fetched = ssi.get("external_feeds", {}).get("weather", {}).get("fetched_at")
        age = int((utcnow() - parse_iso(fetched)).total_seconds()) if fetched else src["event_signals"]["age_s"] or 0
        raw["observations"].append({
            "source_id": "NWS-ALERTS", "source_type": "trusted_external", "timestamp": iso(utcnow() - timedelta(seconds=age)),
            "age_s": age, "corroborated": True,
            "fields": {"event_type": "weather", "event_severity": NWS_TO_EVENT_SEVERITY.get(top.get("severity"), "minor")},
            "nws_event": top.get("event"),
        })
    if vis.get("camera_id"):                                           # computer vision provenance (counts only)
        raw["vision"].update(camera_feeds=[vis["camera_id"]], detector=vis.get("detector"), source=vis.get("source"),
                             waiting_at_stop=vis.get("waiting_at_stop"), pass_ups_source=vis.get("pass_ups_source"))
    raw["weather_context"] = {"conditions": ev.get("weather"), "alerts": w_alerts, "source": ev.get("weather_source")}
    raw["handoff_from"] = "Section A — Inputs_SLM.ipynb"
    return raw


_raw_a = signal_scout_input_to_raw(signal_scout_input)
print("Section A hand-off converted:", _raw_a["scenario_name"], "| time period:", _raw_a["time_period"],
      "| delay:", _raw_a["spec"]["delay_min"], "min | observations (source, age s):",
      [(o["source_id"], o["age_s"]) for o in _raw_a["observations"]],
      "| weather:", _raw_a["weather_context"]["source"], _raw_a["weather_context"]["conditions"])

# ====================================================================================================
# notebook cell 20: E2 · Deterministic validation, exception paths and scoring engine (Signal Scout → verification → confidence → impact → planner → evaluator)
# ====================================================================================================
#@title E2 · Deterministic validation, exception paths and scoring engine (Signal Scout → verification → confidence → impact → planner → evaluator)
APPROVED_TOOLS = {"vision_agent", "data_verification", "feed_outage_path", "conflicting_data_path", "confidence_calculation",
                  "impact_analyzer", "significance_rules", "planner_copilot", "scenario_evaluator", "explanation_agent",
                  "outcome_evaluator", "insight_agent"}
SOURCE_REGISTRY = {"AVL-": "vehicle_apc", "APC-": "vehicle_apc", "VISION-": "vehicle_apc", "AGENCY-ALERTS": "official_alert",
                   "NWS-": "trusted_external", "511-": "trusted_external", "GTFSRT-MIRROR-": "trusted_external",
                   "SOCIAL-": "public_report"}
REQUIRED_FIELDS = ["delay_min", "headway_ratio", "cancellation_risk", "load_factor", "passups", "missed_transfers",
                   "event_severity", "transit_dependence", "income_disadvantage", "disability", "essential_type", "alternatives"]
OPTION_LABELS = {"do_nothing": "Do nothing (baseline)", "rider_alerts": "Send rider alerts",
                 "protect_connection": "Protect the connection (hold)", "reroute": "Reroute around blockage",
                 "dispatch_second_bus": "Dispatch a second bus"}


class AgentFailure(Exception):
    pass


class SignalScout:
    """Signal Scout Agent: routes tasks, enforces approved tools, records the trace, exceptions and alerts."""

    def __init__(self):
        self.trace, self.exceptions, self.alerts = [], [], []

    def step(self, stage, result, detail=""):
        self.trace.append({"stage": stage, "result": result, "detail": detail})

    def exception(self, path, tool, detail, outcome):
        self.exceptions.append({"path": path, "tool": tool, "detail": detail, "outcome": outcome})

    def alert(self, to, msg):
        self.alerts.append({"to": to, "message": msg})

    def invoke(self, tool, fn, *args, retry=False, **kw):
        if tool not in APPROVED_TOOLS:
            raise PermissionError(f"Signal Scout blocked unapproved tool '{tool}'")
        attempts = 2 if retry else 1
        last = None
        for a in range(1, attempts + 1):
            try:
                out = fn(*args, attempt=a, **kw) if retry else fn(*args, **kw)
                if a > 1:
                    self.exception("AGENT ERROR", tool, "Recovered on retry with the same inputs", "recovered")
                return out
            except AgentFailure as e:
                last = e
                self.exception("AGENT ERROR", tool, str(e), "retrying once with same inputs" if a < attempts else "failed twice")
        raise last


def vision_agent(raw, attempt=1):
    """Vision Agent (ridership and load). Uses anonymized camera/sensor aggregates; no identities stored."""
    if attempt <= raw.get("faults", {}).get("vision_agent_failures", 0):
        raise AgentFailure(f"Vision Agent inference error on attempt {attempt}")
    v = raw.get("vision") or {}
    if v.get("identities_stored"):
        raise AgentFailure("Vision payload contains identities; rejected")
    return {k: v.get(k) for k in ["boardings", "alightings", "capacity", "load_factor", "passups", "obstruction", "crowding_level"]}


def _source_type_ok(o):
    expected = next((t for p, t in SOURCE_REGISTRY.items() if o["source_id"].startswith(p)), None)
    return expected is not None and expected == o["source_type"]


def verify_observations(raw, policy):
    """Data verification: schema, timestamps, refresh, source identity, completeness."""
    rows = []
    for o in raw.get("observations", []):
        o = copy.deepcopy(o)
        problems = []
        for k in ["source_id", "source_type", "timestamp", "age_s", "fields"]:
            if k not in o:
                problems.append(f"missing {k}")
        try:
            parse_iso(o["timestamp"])
        except Exception:
            problems.append("bad timestamp")
        if not problems and not _source_type_ok(o):
            problems.append("unknown source identity")
        if problems:
            o["status"], o["note"] = "excluded", "; ".join(problems)
        elif o["source_type"] == "public_report" and not o.get("corroborated"):
            o["status"], o["note"] = "excluded", "Uncorroborated public report cannot trigger action"
        elif o["age_s"] > policy["freshness_limit_s"]:
            o["status"], o["note"] = "stale", f"Older than freshness limit ({o['age_s']} s > {policy['freshness_limit_s']} s)"
        else:
            o["status"], o["note"] = "accepted", "Schema, timestamp, freshness and identity verified"
        q_key = "public_report_corroborated" if o["source_type"] == "public_report" else o["source_type"]
        o["quality"] = policy["source_quality"].get(q_key, 0.0) if o["status"] != "excluded" else 0.0
        rows.append(o)
    return rows


def precedence_rank(policy, stype):
    order = policy["source_precedence"]
    return order.index(stype) if stype in order else len(order)


def resolve_fields(evidence, policy):
    """Take each field from the highest-precedence accepted source."""
    acc = sorted([e for e in evidence if e["status"] == "accepted"], key=lambda e: precedence_rank(policy, e["source_type"]))
    fields, origin = {}, {}
    for e in acc:
        for k, v in e["fields"].items():
            if k not in fields and v is not None:
                fields[k], origin[k] = v, e["source_type"]
    return fields, origin, acc


def run_analysis(raw, policy, operator_resolution=None, constraints=None):
    """Pure, deterministic analysis of one raw bundle under one policy version (used live and in backtests)."""
    sc = SignalScout()
    A = {"policy_version": policy["version"], "gate": None, "gate_reason": "", "significant": False, "hard_triggers": [],
         "official_hard_trigger": False, "confidence": None, "scores": None, "normalization": [], "options": [],
         "ranking": [], "missing": [], "tier": None, "conflict": None, "outage": None}
    sc.step("Signal Scout", "case opened", f"Route {raw['route_id']} · {len(raw.get('observations', []))} observations · tools restricted to approved list")

    # ---- Data verification
    evidence = sc.invoke("data_verification", verify_observations, raw, policy)
    sc.step("Data verification", f"{sum(e['status'] == 'accepted' for e in evidence)} accepted / {len(evidence)}",
            "; ".join(f"{e['source_id']}: {e['status']}" for e in evidence))
    for e in evidence:
        if e["note"].startswith("Uncorroborated"):
            sc.exception("SOCIAL REPORT", "data_verification", f"{e['source_id']} excluded: requires corroboration", "excluded")

    # ---- Vision agent (retry once with same inputs, then deterministic safe mode)
    try:
        vis = sc.invoke("vision_agent", vision_agent, raw, retry=True)
        evidence.append({"source_id": f"VISION-{raw['route_id']}", "source_type": "vehicle_apc", "timestamp": raw["generated_at"],
                         "age_s": 15, "corroborated": True, "fields": {k: v for k, v in vis.items() if v is not None},
                         "status": "accepted", "note": "Anonymized station/onboard camera + sensor aggregates (no identities)",
                         "quality": policy["source_quality"]["vehicle_apc"]})
        sc.step("Vision Agent", "ok", f"load factor {vis.get('load_factor')}, pass-ups {vis.get('passups')}")
    except AgentFailure:
        A.update(gate="SAFE_MODE", gate_reason="Vision Agent failed twice → deterministic safe mode; evidence preserved")
        sc.step("Agent error path", "SAFE MODE", A["gate_reason"])
        sc.alert("technical owner", "Vision Agent failed twice with the same inputs; safe mode engaged")
        sc.alert("planner", "Case routed to safe default because an agent failed; no resource action will be taken")
        A.update(evidence=evidence, trace=sc.trace, exceptions=sc.exceptions, alerts=sc.alerts)
        return A

    # ---- Feed outage path
    vehicle = [e for e in evidence if e["source_type"] == "vehicle_apc" and not e["source_id"].startswith("VISION-")]
    if vehicle and all(e["status"] == "stale" for e in vehicle):
        oldest = max(e["age_s"] for e in vehicle)
        for e in vehicle:
            e["note"] = f"FROZEN last verified snapshot, age {e['age_s']} s (kept as evidence, not scored)"
        sc.alert("operator", f"Vehicle feed outage on {raw['route_id']} (snapshot age {oldest} s); backup source requested")
        if raw.get("faults", {}).get("backup_available") and raw.get("backup_observation"):
            b = copy.deepcopy(raw["backup_observation"])
            b.update(status="accepted", note="Backup source after feed outage; re-verified", quality=policy["source_quality"]["trusted_external"])
            evidence.append(b)
            A["outage"] = "backup"
            sc.exception("FEED OUTAGE", "feed_outage_path", f"Snapshot frozen at age {oldest} s; backup {b['source_id']} found", "re-verified with backup")
            sc.step("Feed outage path", "backup check: YES → re-verify", b["source_id"])
        else:
            A["outage"] = "no_backup"
            sc.exception("FEED OUTAGE", "feed_outage_path", f"Snapshot frozen at age {oldest} s; no backup source", "monitoring only")
            sc.step("Feed outage path", "backup check: NO", "Freshness limit exceeded → monitoring only")

    # ---- Conflicting data path
    acc_delay = sorted([e for e in evidence if e["status"] == "accepted" and "delay_min" in e["fields"]],
                       key=lambda e: precedence_rank(policy, e["source_type"]))
    conflict_rows = []
    if len(acc_delay) >= 2:
        win = acc_delay[0]
        for e in acc_delay[1:]:
            dd = abs(e["fields"]["delay_min"] - win["fields"]["delay_min"])
            cc = ("cancelled" in e["fields"] and "cancelled" in win["fields"] and e["fields"]["cancelled"] != win["fields"]["cancelled"])
            if dd > policy["conflict_threshold_min"] or cc:
                conflict_rows.append((e, dd))
    unresolved_conflict = False
    if conflict_rows:
        for e, dd in conflict_rows:
            e["status"] = "quarantined"
            e["note"] = f"Conflicts with {acc_delay[0]['source_id']} by {dd:.0f} min; lower source precedence"
        desc = ", ".join(e["source_id"] for e, _ in conflict_rows)
        if operator_resolution == "accept_precedence":
            A["conflict"] = "resolved"
            sc.exception("CONFLICTING DATA", "conflicting_data_path", f"Quarantined {desc}; operator accepted precedence", "resolved")
            sc.step("Conflicting data path", "RESOLVED by operator", "Precedence winner kept; quarantined sources excluded")
        elif operator_resolution == "unresolved":
            A["conflict"] = "unresolved_final"
            sc.exception("CONFLICTING DATA", "conflicting_data_path", f"Quarantined {desc}; operator could not resolve", "monitoring")
        else:
            A["conflict"] = "pending"
            unresolved_conflict = True
            sc.exception("CONFLICTING DATA", "conflicting_data_path",
                         f"Quarantined {desc}; precedence vehicle > official > trusted external > corroborated public; C reduced",
                         "flagged for operator review")
            sc.alert("operator", "Conflicting data quarantined; operator review required")

    # ---- Resolve fields + context
    fields, origin, acc = resolve_fields(evidence, policy)
    for k, src in [("transit_dependence", "community"), ("income_disadvantage", "community"), ("disability", "community")]:
        if raw.get("community", {}).get(k) is not None:
            fields[k], origin[k] = raw["community"][k], "community (area-level)"
    acc_info = raw.get("access", {})
    if acc_info.get("essential_type") is not None:
        fields["essential_type"], origin["essential_type"] = acc_info["essential_type"], "access"
    if acc_info.get("alternatives") is not None:
        fields["alternatives"], origin["alternatives"] = acc_info["alternatives"], "access"
    conn = raw.get("connections", {})
    if conn.get("missed_transfers") is not None:
        fields["missed_transfers"], origin["missed_transfers"] = conn["missed_transfers"], "connecting services"
    if fields.get("cancelled") and "cancellation_risk" not in fields:
        fields["cancellation_risk"], origin["cancellation_risk"] = 1.0, origin.get("cancelled")
    if "event_severity" not in fields:
        fields["event_severity"], origin["event_severity"] = "none", "default: no event reported"
    elif origin.get("event_severity") not in ("official_alert", "trusted_external"):
        fields["event_severity"], origin["event_severity"] = "none", "unverified event ignored (E = 0)"
    A["fields"], A["field_origin"] = fields, origin
    A["missing"] = [f for f in REQUIRED_FIELDS if fields.get(f) is None]

    # ---- Data ready?
    feed_available = len([e for e in acc if not e["source_id"].startswith("VISION-")]) > 0
    ready = feed_available and not unresolved_conflict
    sc.step("Data ready?", "YES" if ready else "NO",
            f"feed available={feed_available}, agent healthy=True, unresolved conflict={unresolved_conflict}")

    # ---- Confidence C = 0.40Q + 0.25F + 0.20G + 0.15M
    def confidence(include_quarantined):
        pool = [e for e in evidence if e["status"] == "accepted" or (include_quarantined and e["status"] == "quarantined")]
        pool_nv = [e for e in pool if not e["source_id"].startswith("VISION-")]
        if not pool_nv:
            return {"Q": 0.0, "F": 0.0, "G": 0.0, "M": 0.0, "penalty": 0.0, "C": 0.0}
        Q = statistics.mean(e["quality"] for e in pool_nv)
        top = min(pool_nv, key=lambda e: precedence_rank(policy, e["source_type"]))
        F = n_cost(top["age_s"], *policy["bounds"]["age_s"])
        dl = [e["fields"]["delay_min"] for e in pool_nv if "delay_min" in e["fields"]]
        G = n_cost(max(dl) - min(dl), *policy["bounds"]["agreement_spread_min"]) if len(dl) >= 2 else policy["single_source_agreement"]
        M = 1 - len(A["missing"]) / len(REQUIRED_FIELDS)
        w = policy["confidence_weights"]
        pen = policy["conflict_penalty"] if unresolved_conflict else 0.0
        C = max(0.0, w["Q"] * Q + w["F"] * F + w["G"] * G + w["M"] * M - pen)
        return {"Q": r3(Q), "F": r3(F), "G": r3(G), "M": r3(M), "penalty": pen, "C": r3(C)}

    conf = sc.invoke("confidence_calculation", confidence, unresolved_conflict)
    A["confidence"] = conf

    # ---- Hard triggers (from accepted evidence only)
    ht = policy["hard_trigger"]
    lk = policy["lookups"]
    trig = []
    if fields.get("cancelled"):
        trig.append(("cancellation", origin.get("cancelled")))
    if fields.get("delay_min") is not None and fields["delay_min"] >= ht["delay_min"]:
        trig.append((f"delay ≥ {ht['delay_min']} min", origin.get("delay_min")))
    gap_measured = not raw.get("spec", {}).get("headway_from_delay")   # a gap derived from the delay is not a second trigger
    if gap_measured and fields.get("headway_ratio") is not None and fields["headway_ratio"] >= ht["gap_headway_ratio"]:
        trig.append((f"gap ≥ {ht['gap_headway_ratio']} × headway", origin.get("headway_ratio")))
    if fields.get("event_severity") == "emergency" and origin.get("event_severity") in ("official_alert",):
        trig.append(("verified emergency", "official_alert"))
    if (fields.get("alternatives") == 0 and lk["essential_severity"].get(fields.get("essential_type", "none"), 0) >= ht["essential_severity_min"]
            and (fields.get("cancelled") or (fields.get("delay_min") or 0) >= ht["delay_min"])):
        trig.append(("essential route with no alternative disrupted", origin.get("cancelled") or origin.get("delay_min")))
    if (fields.get("load_factor") or 0) >= ht["unboardable_load_factor"] and (fields.get("passups") or 0) >= 1:
        trig.append(("unboardable crowding", "vehicle_apc"))
    A["hard_triggers"] = [{"trigger": t, "source": s, "official": s in policy["official_sources"]} for t, s in trig]
    A["official_hard_trigger"] = any(t["official"] for t in A["hard_triggers"])

    # ---- Impact analyzer (always computed so monitoring cases record priority)
    if feed_available:
        A["scores"], A["normalization"] = sc.invoke("impact_analyzer", impact_scores, fields, raw, policy)
    sig = bool(A["hard_triggers"]) or (A["scores"] is not None and A["scores"]["R"] >= policy["significance_R"])
    A["significant"] = sig

    # ---- Gates (flowchart order)
    if A["outage"] == "no_backup" and not feed_available:
        A.update(gate="MONITORING", gate_reason="Feed outage with no backup; freshness limit exceeded → monitoring only")
    elif not feed_available:
        A.update(gate="MONITORING", gate_reason="No verified evidence (e.g. uncorroborated public report only) → continue monitoring")
    elif unresolved_conflict:
        A.update(gate="CONFLICT_REVIEW", gate_reason="Conflict quarantined; operator review required before analysis can proceed")
    elif A["conflict"] == "unresolved_final":
        A.update(gate="MONITORING", gate_reason="Conflict could not be resolved by operator → monitoring")
    elif conf["C"] < policy["confidence_threshold"] and not A["official_hard_trigger"]:
        A.update(gate="MONITORING", gate_reason=f"C = {conf['C']:.2f} < {policy['confidence_threshold']} and no official hard trigger → monitoring")
    else:
        bypass = conf["C"] < policy["confidence_threshold"]
        sc.step("Confidence gate", "PROCEED" + (" (official hard trigger bypass)" if bypass else ""),
                f"C = {conf['C']:.3f} (threshold {policy['confidence_threshold']})")
        if not sig:
            A.update(gate="BELOW_THRESHOLD", gate_reason=f"No hard trigger and R = {A['scores']['R']:.2f} < {policy['significance_R']} → safe default (rider info + monitor)")
        else:
            A.update(gate="SIGNIFICANT", gate_reason=("Hard trigger: " + ", ".join(t["trigger"] for t in A["hard_triggers"])) if A["hard_triggers"]
                     else f"R = {A['scores']['R']:.2f} ≥ {policy['significance_R']}")
    if A["gate"] in ("MONITORING", "CONFLICT_REVIEW"):
        sc.step("Confidence gate", A["gate"], A["gate_reason"])
    if A["scores"]:
        sc.step("Deterministic impact analyzer", "scored", f"R={A['scores']['R']:.3f} V={A['scores']['V']:.3f} A={A['scores']['A']:.3f} P={A['scores']['P']:.3f}")
    sc.step("Significance rules", "SIGNIFICANT" if sig else "not significant", A["gate_reason"] if A["gate"] in ("SIGNIFICANT", "BELOW_THRESHOLD") else "")

    # ---- Severity tier for deadline control
    if A["scores"]:
        P = A["scores"]["P"]
        ts = policy["severity_tiers"]
        if any(t["trigger"] == "verified emergency" for t in A["hard_triggers"]) or P >= ts["severe_P"]:
            A["tier"] = "severe"
        elif P >= ts["significant_P"] or A["hard_triggers"]:
            A["tier"] = "significant"
        else:
            A["tier"] = "moderate"

    # ---- Planner Copilot + Scenario Evaluator (significant cases only)
    if A["gate"] == "SIGNIFICANT":
        cons = {**policy["planner_defaults"], **(constraints or {})}
        opts = sc.invoke("planner_copilot", planner_copilot, fields, raw, A["scores"], policy, cons)
        feas = [o for o in opts if o["feasible"]]
        sc.step("Planner Copilot", f"{len(feas)} feasible / {len(opts)} considered",
                "; ".join(f"{o['label']}: {'feasible' if o['feasible'] else o['infeasible_reason']}" for o in opts))
        ranked = sc.invoke("scenario_evaluator", scenario_evaluator, feas, conf["C"], A["scores"]["P"], policy)
        A["options"], A["ranking"], A["constraints_used"] = opts, ranked, cons
        if not feas:
            sc.step("Planner Copilot", "NO FEASIBLE OPTION", "Escalate to human review")
            sc.alert("planner", "No feasible option; escalated to human review")
        else:
            sc.step("Scenario evaluator", "ranked", " > ".join(f"{r['label']} ({r['Si']:.3f})" for r in ranked))
    A.update(evidence=evidence, trace=sc.trace, exceptions=sc.exceptions, alerts=sc.alerts)
    return A


def impact_scores(fields, raw, policy):
    b, lk = policy["bounds"], policy["lookups"]
    comp, norm = {}, []

    def add(code, group, label, raw_val, val, L=None, U=None, method="n(x)"):
        comp[code] = val
        norm.append({"group": group, "component": code, "feature": label, "raw": raw_val, "L": L, "U": U, "method": method,
                     "normalized": r3(val) if val is not None else None,
                     "note": "MISSING — not imputed; contributes 0 and lowers completeness M" if val is None else ""})

    d = fields.get("delay_min")
    add("D", "R", "delay (min)", d, n_benefit(d, *b["delay_min"]), *b["delay_min"])
    h = fields.get("headway_ratio")
    add("H", "R", "gap / scheduled headway", h, n_benefit(h, *b["headway_ratio"]), *b["headway_ratio"])
    k = fields.get("cancellation_risk")
    add("K", "R", "cancellation risk", k, None if k is None else min(1.0, max(0.0, float(k))), 0, 1, "clip 0–1")
    ev = fields.get("event_severity", "none")
    add("E", "R", f"verified event ({fields.get('event_type', 'none')})", ev, lk["event_severity"].get(ev, 0.0), method="lookup table")
    add("N", "V", "transit dependence (area share)", fields.get("transit_dependence"),
        n_benefit(fields.get("transit_dependence"), *b["transit_dependence"]), *b["transit_dependence"])
    add("I", "V", "income disadvantage (area share)", fields.get("income_disadvantage"),
        n_benefit(fields.get("income_disadvantage"), *b["income_disadvantage"]), *b["income_disadvantage"])
    add("B", "V", "disability barrier (area share)", fields.get("disability"),
        n_benefit(fields.get("disability"), *b["disability"]), *b["disability"])
    tp = raw.get("time_period", "off_peak")
    add("T", "V", "time sensitivity", tp, lk["time_sensitivity"].get(tp, 0.3), method="lookup table")
    es = fields.get("essential_type")
    add("S", "A", "essential service severity", es, None if es is None else lk["essential_severity"].get(es, 0.0), method="lookup table")
    al = fields.get("alternatives")
    add("L", "A", "loss of alternatives (routes nearby)", al, n_cost(al, *b["alternatives"]), *b["alternatives"], "ninv(x)")
    mt = fields.get("missed_transfers")
    add("X", "A", "connection loss (missed transfers)", mt, n_benefit(mt, *b["missed_transfers"]), *b["missed_transfers"])
    pu = fields.get("passups")
    add("U", "A", "unable to board (pass-ups)", pu, n_benefit(pu, *b["passups"]), *b["passups"])

    def wsum(group):
        w = policy[f"{group}_weights"]
        return sum(w[c] * (comp[c] or 0.0) for c in w)

    R, V, A_ = wsum("R"), wsum("V"), wsum("A")
    pw = policy["P_weights"]
    P = pw["R"] * R + pw["V"] * V + pw["A"] * A_
    for row in norm:
        w = policy[f"{row['group']}_weights"][row["component"]]
        row["weight"] = w
        row["contribution"] = r3(w * (comp[row["component"]] or 0.0))
    return {"R": r3(R), "V": r3(V), "A": r3(A_), "P": r3(P), "components": {k: r3(v) for k, v in comp.items()}}, norm


def planner_copilot(fields, raw, scores, policy, cons):
    """Planner Copilot: generates feasible options only, applying resource, road, policy and safety constraints. Assigns no scores."""
    delay = fields.get("delay_min")
    delay_assumed = delay is None
    if delay is None:
        delay = raw["headway_min"]  # documented assumption when delay is missing
    if fields.get("cancelled"):
        delay = max(delay, raw["headway_min"])  # E-6: riders of a cancelled trip wait at least one headway
    riders = (fields.get("boardings") or 0) + (fields.get("passups") or 0)
    V = scores["V"]
    vul = riders * V
    conn = raw.get("connections", {})
    roads = raw.get("roads", {})
    alts = fields.get("alternatives") or 0
    cancelled = bool(fields.get("cancelled"))
    excluded = set(cons.get("excluded_options", []))
    opts = []

    def opt(oid, feasible, reason, dr, vul_helped, ap, of, cost, notes, extra=None, parties=None):
        o = {"option_id": oid, "type": oid.split("@")[0], "label": OPTION_LABELS[oid.split("@")[0]] + (f" — {extra}" if extra else ""),
             "feasible": feasible, "infeasible_reason": reason, "pred_delay_reduction_min": round(dr, 2),
             "pred_vulnerable_riders_helped": round(vul_helped, 1), "pred_access_protection": round(ap, 3),
             "feasibility": round(of, 3), "cost_usd": round(cost, 0), "notes": notes,
             "assumptions": "Delay missing: headway used as delay estimate" if delay_assumed else "",
             "parties": parties or ["Agency (in-house)"]}
        if o["type"] in excluded:
            o["feasible"], o["infeasible_reason"] = False, "Excluded by planner revision"
        opts.append(o)

    opt("do_nothing", True, "", 0, 0, 0.0, 1.0, 0, "Baseline: service recovers on its own")
    opt("rider_alerts", True, "", min(delay, 2) * 0.5, 0.25 * vul, 0.10 + 0.10 * min(alts, 3) / 3, 0.98, 50,
        "Push verified info to riders, stop displays and paratransit desk")
    if conn.get("connecting_route"):
        hold = max(0.0, delay - conn.get("transfer_window_min", 4))
        ok = hold <= cons["max_hold_min"]
        xn = n_benefit(conn.get("missed_transfers", 0), *policy["bounds"]["missed_transfers"]) or 0
        opt("protect_connection", ok, "" if ok else f"Hold needed {hold:.0f} min > max hold {cons['max_hold_min']} min",
            0.35 * delay, (conn.get("missed_transfers", 0)) * V, min(1.0, 0.30 + 0.50 * xn + (0.2 if conn.get("last_service") else 0)),
            0.85 - 0.15 * (hold / max(cons["max_hold_min"], 1)), 40 * hold, f"Hold {conn['connecting_route']} at {conn['transfer_stop_id']} up to {hold:.0f} min",
            extra=f"hold {conn['connecting_route']}", parties=[conn.get("partner") or "Agency (in-house)"])
    else:
        opt("protect_connection", False, "No connecting service at affected stops", 0, 0, 0, 0, 0, "")
    blockage = raw.get("event_type") in ("road_closure", "obstruction") or bool(fields.get("obstruction"))
    if blockage:
        reasons = []
        if not roads.get("detour_open", True):
            reasons.append("detour road closed")
        if not roads.get("reroute_allowed", True):
            reasons.append("reroute not permitted by policy")
        if cancelled:
            reasons.append("trip cancelled; nothing to reroute")
        ex = roads.get("detour_extra_min", 6)
        opt("reroute", not reasons, "; ".join(reasons), 0.6 * delay + (0.5 * raw["headway_min"] if delay_assumed else 0), 0.6 * vul,
            0.55, max(0.2, 0.75 - 0.02 * ex), 100 + 25 * ex, f"Detour adds ~{ex} min; skipped stops get rider alerts")
    else:
        opt("reroute", False, "No blockage or closure on the segment", 0, 0, 0, 0, 0, "")
    # Dispatch second bus from nearest depot (alternate depot only if the planner allows it in a revision)
    stops = [NET["stops"][s] for s in raw["affected_stop_ids"]]
    mid = (statistics.mean(s["lat"] for s in stops), statistics.mean(s["lon"] for s in stops))
    deps = sorted(raw["resources"]["depots"], key=lambda d: km_between((d["lat"], d["lon"]), mid))
    use = deps if cons.get("allow_alternate_depot") else deps[:1]
    for i, d in enumerate(use):
        dh = d.get("deadhead_min_override") or km_between((d["lat"], d["lon"]), mid) / 25 * 60
        cost = d.get("dispatch_cost_override") or raw["resources"]["bus_hour_cost_usd"] * (dh + 60) / 60
        reasons = []
        if d["spare_buses"] < 1:
            reasons.append("no spare bus")
        if d["operators"] < 1:
            reasons.append("no available operator")
        if dh > cons["max_deadhead_min"]:
            reasons.append(f"deadhead {dh:.0f} min > {cons['max_deadhead_min']} min")
        if cost > cons["budget_cap_usd"]:
            reasons.append(f"cost ${cost:.0f} > budget cap ${cons['budget_cap_usd']}")
        if not roads.get("segment_open", True):
            reasons.append("segment closed to buses")
        passup_relief = 0.5 * raw["headway_min"] * min(1.0, (fields.get("passups") or 0) / 20)
        dr = max(0.0, delay - 0.5 * dh) * 0.85 + passup_relief
        opt(f"dispatch_second_bus@{d['id']}", not reasons, "; ".join(reasons), dr, 0.85 * vul, 0.80,
            0.10 + 0.85 * (n_cost(dh, 0, 40) or 0), cost, f"Spare bus from {d['name']}, deadhead {dh:.0f} min",
            extra=d["name"] + (" (alternate)" if i > 0 else ""), parties=[d.get("operated_by", "Agency (in-house)")])
    return opts


def scenario_weights(P, policy):
    base, shift = policy["scenario_base"], policy["scenario_shift"]
    return {c: base[c] + shift.get(c, 0) * P for c in base}


def scenario_evaluator(feasible_opts, C, P, policy):
    """Si = Ci × Σ w_c(P)·component; tie-break: higher AP, then higher OF, then lower cost. No LLM judgment."""
    b, lk = policy["bounds"], policy["lookups"]
    w = scenario_weights(P, policy)
    out = []
    for o in feasible_opts:
        comp = {"DR": n_benefit(o["pred_delay_reduction_min"], *b["dr_min"]),
                "EB": n_benefit(o["pred_vulnerable_riders_helped"], *b["eb_riders"]),
                "AP": min(1.0, max(0.0, o["pred_access_protection"])), "OF": min(1.0, max(0.0, o["feasibility"])),
                "CE": n_cost(o["cost_usd"], *b["cost_usd"])}
        Ci = C * lk["option_certainty"][o["type"]]
        inner = sum(w[c] * comp[c] for c in w)
        out.append({"option_id": o["option_id"], "label": o["label"], "type": o["type"], **{c: r3(v) for c, v in comp.items()},
                    "Ci": r3(Ci), "Si": round(Ci * inner, 4), "cost_usd": o["cost_usd"],
                    "weights": {c: r3(v) for c, v in w.items()},
                    "weighted": {c: r3(Ci * w[c] * comp[c]) for c in w}})
    out.sort(key=lambda r: (-r["Si"], -r["AP"], -r["OF"], r["cost_usd"]))
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out

# ====================================================================================================
# notebook cell 22: C · Structured outputs (schemas)
# ====================================================================================================
#@title C · Structured outputs (schemas)
class Official(BaseModel):
    name: str
    department: str = ""
    completed: List[str] = []
    last_training: Optional[str] = Field(default=None, description="ISO date YYYY-MM-DD, or null if never trained")

    @field_validator("last_training")
    @classmethod
    def _iso_date(cls, v):                               # reject malformed dates with a clear message
        if v is not None:
            try:
                date.fromisoformat(v)
            except ValueError:
                raise ValueError(f"last_training must be YYYY-MM-DD, got {v!r}")
        return v


class ActionItem(BaseModel):
    official: str
    status: Literal["COMPLIANT", "AT_RISK", "NON_COMPLIANT"]
    missing_courses: List[str] = []
    action: str
    priority: Literal["high", "medium", "low"]


class ComplianceReport(BaseModel):                  # the agent's final answer
    answer: str = Field(description="Direct answer to the user's question, max 80 words")
    compliance_rate_pct: float = Field(ge=0, le=100)
    action_items: List[ActionItem]
    ethics_notes: str = Field(description="Fairness / privacy / due-process remarks, max 60 words")


class AgentStep(BaseModel):                         # what the agent says on every turn
    thought: str = Field(description="One short sentence of reasoning")
    action: Literal["tool", "final"]
    tool: Optional[str] = None
    args: Dict[str, Any] = {}
    report: Optional[ComplianceReport] = None       # only when action == "final"


print("✓ Section C schemas defined")

# ====================================================================================================
# notebook cell 23: C · The tools (plain Python rule engine)
# ====================================================================================================
#@title C · The tools (plain Python rule engine)
ETH_REQUIRED = ["conflicts", "transparency", "procurement", "data_protection"]
ETH_REFRESH_DAYS = 365
ETH_WARNING_DAYS = 30
ETH_TODAY = local_now().date()          # change to date(2026, 1, 1) etc. to test a specific reference date

ETH_ROSTER: List[Official] = []   # filled with load_eth_roster(); never rebound (tools read this list)


def _status_for(o: Official):
    missing = [c for c in ETH_REQUIRED if c not in o.completed]
    if o.last_training:
        days_since = (ETH_TODAY - date.fromisoformat(o.last_training)).days
        days_until_due = ETH_REFRESH_DAYS - days_since
    else:
        days_since, days_until_due = None, None
    overdue = days_until_due is not None and days_until_due < 0
    if missing or overdue or o.last_training is None:
        status = "NON_COMPLIANT"
    elif days_until_due <= ETH_WARNING_DAYS:
        status = "AT_RISK"
    else:
        status = "COMPLIANT"
    return {"official": o.name, "department": o.department, "status": status, "missing_courses": missing,
            "days_since_training": days_since, "days_until_refresher_due": days_until_due}


def eth_list_officials():
    return [o.name for o in ETH_ROSTER]


def eth_check_official(name):
    for o in ETH_ROSTER:
        if o.name.lower() == str(name).strip().lower():
            return _status_for(o)
    return {"error": f"unknown official '{name}'", "known": eth_list_officials()}


def eth_check_all():
    return [_status_for(o) for o in ETH_ROSTER]


def eth_team_summary():
    rows = eth_check_all()
    counts = {s: sum(r["status"] == s for r in rows) for s in ["COMPLIANT", "AT_RISK", "NON_COMPLIANT"]}
    total = len(rows)
    # AT_RISK people are compliant (refresher due soon), so they count toward the compliance rate.
    rate = round(100 * (counts["COMPLIANT"] + counts["AT_RISK"]) / total, 1) if total else 0.0
    current = round(100 * counts["COMPLIANT"] / total, 1) if total else 0.0
    return {"total": total, **counts, "compliance_rate_pct": rate, "fully_current_pct": current}


ETH_TOOLS = {
    "list_officials": (eth_list_officials, "no args. Returns the names of all officials."),
    "check_official": (eth_check_official, 'args: {"name": "<official name>"}. Returns status, missing courses, days until refresher.'),
    "check_all":      (eth_check_all,      "no args. Returns status for every official (use this first for team questions)."),
    "team_summary":   (eth_team_summary,   "no args. Returns counts per status and the compliance rate."),
}


def load_eth_roster(rows):
    """Replace the roster contents in place."""
    new = [Official(**d) for d in rows]
    if not new:
        raise ValueError("The roster is empty: add at least one person.")
    ETH_ROSTER.clear()
    ETH_ROSTER.extend(new)
    return len(ETH_ROSTER)


print("✓ Section C tools ready:", ", ".join(ETH_TOOLS))

# ====================================================================================================
# notebook cell 24: C · The single agent (think → act → observe loop) and guardrail
# ====================================================================================================
#@title C · The single agent (think → act → observe loop) and guardrail
ETH_TRACE = []
ETH_MAX_STEPS = 8


class ComplianceAgent:
    def __init__(self):
        tool_docs = "\n".join(f"- {name}: {doc}" for name, (_, doc) in ETH_TOOLS.items())
        self.system = (
            "You are the Ethics Training Compliance Agent for a public-transit agency. The people on the roster review and "
            "approve service-disruption actions, so their ethics training must be current.\n"
            "You do NOT calculate dates or statuses yourself: you call tools and report what they return.\n\n"
            f"TOOLS:\n{tool_docs}\n\n"
            "Each turn reply with ONE JSON object:\n"
            '  {"thought": "...", "action": "tool", "tool": "<tool name>", "args": {...}}   to call a tool, or\n'
            '  {"thought": "...", "action": "final", "report": {...ComplianceReport...}}    when you have what you need.\n'
            "Rules: call check_all (or team_summary) before answering team-level questions; include EVERY official in action_items; "
            "priority high = NON_COMPLIANT, medium = AT_RISK, low = COMPLIANT; keep tone neutral and fair (same standards for everyone); "
            "never speculate about reasons for non-compliance. Copy status and missing_courses EXACTLY from the tool output.\n"
            "Example of a final turn:\n"
            '{"thought": "done", "action": "final", "report": {"answer": "...", "compliance_rate_pct": 40.0, '
            '"action_items": [{"official": "A", "status": "NON_COMPLIANT", "missing_courses": ["procurement"], '
            '"action": "Complete procurement course", "priority": "high"}], "ethics_notes": "..."}}'
        )

    def run(self, question, verbose=True):
        if not ETH_ROSTER:                                   # never answer about an empty roster
            raise ValueError("The roster is empty — load it with load_eth_roster() first.")
        ETH_TRACE.clear()
        seen_calls = set()
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": f"Question: {question}\nToday's date: {ETH_TODAY.isoformat()}"}]
        for step in range(1, ETH_MAX_STEPS + 1):
            t0 = time.time()
            raw = call_llm(messages, schema=AgentStep)
            try:
                s = AgentStep.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as e:
                if verbose:
                    print(f"Step {step}: ⚠ invalid output, asking the agent to fix it")
                ETH_TRACE.append({"step": step, "type": "invalid", "detail": str(e)[:80], "seconds": round(time.time() - t0, 1)})
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"Invalid output: {str(e)[:400]}. Reply with corrected JSON only."}]
                continue
            if s.action == "tool":
                fn = ETH_TOOLS.get(s.tool or "")
                if fn is None:
                    result = {"error": f"unknown tool '{s.tool}'", "available": list(ETH_TOOLS)}
                elif (s.tool, json.dumps(s.args, sort_keys=True)) in seen_calls:
                    # a repeated call adds nothing: point the model back to the result it already has
                    result = {"note": f"You already called {s.tool} with these arguments; its result is above. Do not call "
                                      "tools again. Reply now with action \"final\" and the full report."}
                else:
                    seen_calls.add((s.tool, json.dumps(s.args, sort_keys=True)))
                    try:
                        result = fn[0](**s.args)
                    except TypeError as e:
                        result = {"error": f"bad arguments: {e}"}
                if verbose:
                    print(f"Step {step}: 🧠 {s.thought}\n         🔧 {s.tool}({json.dumps(s.args)}) ✓")
                ETH_TRACE.append({"step": step, "type": "tool", "detail": f"{s.tool}({json.dumps(s.args)})", "seconds": round(time.time() - t0, 1)})
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": "TOOL RESULT:\n" + json.dumps(result)}]
            elif s.report is not None:
                ETH_TRACE.append({"step": step, "type": "final", "detail": "ComplianceReport", "seconds": round(time.time() - t0, 1)})
                return s.report
            else:
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": 'action "final" needs a "report" object. Try again.'}]
        raise RuntimeError(f"Agent did not finish within {ETH_MAX_STEPS} steps. Try a stronger model.")


def verify_report(report):
    """GUARDRAIL: compare the LLM's report with the deterministic rule engine."""
    truth = {r["official"].lower(): r for r in eth_check_all()}
    issues = []
    if not truth:
        issues.append("the roster is empty — nothing can be verified")
    seen = set()
    for item in report.action_items:
        t = truth.get(item.official.lower())
        if t is None:
            issues.append(f"'{item.official}' is not on the roster (possible hallucination)")
            continue
        seen.add(item.official.lower())
        if item.status != t["status"]:
            issues.append(f"{item.official}: report says {item.status}, rules say {t['status']}")
        if set(item.missing_courses) != set(t["missing_courses"]):
            issues.append(f"{item.official}: missing courses differ (report {item.missing_courses} vs rules {t['missing_courses']})")
    for name in truth.keys() - seen:
        issues.append(f"{truth[name]['official']} is missing from the report")
    true_rate = eth_team_summary()["compliance_rate_pct"]
    if abs(report.compliance_rate_pct - true_rate) > 1:
        issues.append(f"compliance rate: report {report.compliance_rate_pct}% vs rules {true_rate}%")
    return issues


def reconcile_report(report):
    """GUARDRAIL: the rule engine owns the facts (status, missing courses, priority, rate). Overwrite them; keep the LLM's wording."""
    truth = {r["official"].lower(): r for r in eth_check_all()}
    prio = {"NON_COMPLIANT": "high", "AT_RISK": "medium", "COMPLIANT": "low"}
    fixed, seen = [], set()
    for item in report.action_items:
        t = truth.get(item.official.lower())
        if t is None:
            continue                                   # drop officials the LLM made up
        seen.add(item.official.lower())
        fixed.append(item.model_copy(update={"status": t["status"], "missing_courses": t["missing_courses"],
                                             "priority": prio[t["status"]]}))
    for name, t in truth.items():                      # add officials the LLM forgot
        if name not in seen:
            fixed.append(ActionItem(official=t["official"], status=t["status"], missing_courses=t["missing_courses"],
                                    action="(added by guardrail) see rule-engine status", priority=prio[t["status"]]))
    s = eth_team_summary()
    answer, notes = report.answer, []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", report.answer):
        v = float(m.group(1))
        if abs(v - s["compliance_rate_pct"]) <= 1:
            continue
        if abs(v - s["fully_current_pct"]) <= 1:
            notes.append(f"{m.group(0)} is the fully-current share (COMPLIANT only), not the compliance rate")
        else:
            notes.append(f"{m.group(0)} is not supported by the rule engine")
    verified = (f"[Rule engine] Compliance rate {s['compliance_rate_pct']}% (COMPLIANT + AT_RISK, {s['COMPLIANT'] + s['AT_RISK']} of "
                f"{s['total']}); fully current {s['fully_current_pct']}%; NON_COMPLIANT {s['NON_COMPLIANT']}.")
    if notes:
        answer = verified + " [Guardrail correction: " + "; ".join(notes) + ".] LLM explanation: " + report.answer
    return report.model_copy(update={"action_items": fixed, "compliance_rate_pct": s["compliance_rate_pct"], "answer": answer})


eth_agent = ComplianceAgent()
ETH_WORKFLOW = [("in", "Roster +\nQuestion", "input"), ("agent", "Compliance\nAgent (think)", "llm"),
                ("tools", "Tools\n(act / observe)", "tool"), ("final", "Structured\nReport", "llm"),
                ("guard", "Guardrail\nverify", "tool"), ("out", "Verified\nOutput", "output")]
print("✓ Section C agent + guardrail ready")

# ====================================================================================================
# notebook cell 25: C · Rosters: transit review team (used by the governance gate) + the Section C sample rosters
# ====================================================================================================
#@title C · Rosters: transit review team (used by the governance gate) + the Section C sample rosters
def days_ago(n):
    return (ETH_TODAY - timedelta(days=n)).isoformat()


ALL4 = ["conflicts", "transparency", "procurement", "data_protection"]

ETH_SAMPLES = {
    # INTEGRATION: the people who act in this system (fictional). Statuses cover every case for testing the gate.
    "Transit review team (this system)": [
        {"name": "Ana Lopez",    "department": "Operations Supervisor (first reviewer)",  "completed": ALL4, "last_training": days_ago(90)},
        {"name": "Ben Carter",   "department": "Service Planner",                          "completed": ALL4, "last_training": days_ago(345)},
        {"name": "Chen Wei",     "department": "Operations Supervisor (second reviewer)", "completed": ALL4, "last_training": days_ago(30)},
        {"name": "Dana Brooks",  "department": "Policy Owner",                             "completed": ALL4, "last_training": days_ago(120)},
        {"name": "Eli Turner",   "department": "Operations Supervisor",                    "completed": ["conflicts", "transparency"], "last_training": days_ago(60)},
        {"name": "Farah Haddad", "department": "Operations Staff",                         "completed": ALL4, "last_training": days_ago(400)},
        {"name": "Gabe Ortiz",   "department": "Service Planner",                          "completed": ALL4, "last_training": days_ago(200)},
    ],
    # Section C sample rosters (fictional)
    "Mixed team (all statuses)": [
        {"name": "Alice Morgan",  "department": "Legal",       "completed": ALL4,                         "last_training": days_ago(90)},
        {"name": "Bob Rivera",    "department": "Procurement", "completed": ["conflicts", "transparency"], "last_training": days_ago(120)},
        {"name": "Carla Singh",   "department": "Finance",     "completed": ALL4,                         "last_training": days_ago(400)},
        {"name": "Dev Patel",     "department": "IT",          "completed": ALL4,                         "last_training": days_ago(345)},
        {"name": "Erin Walsh",    "department": "Operations",  "completed": [],                           "last_training": None},
    ],
    "Fully compliant office": [
        {"name": "Frank Osei",    "department": "Legal",   "completed": ALL4, "last_training": days_ago(30)},
        {"name": "Grace Kim",     "department": "Finance", "completed": ALL4, "last_training": days_ago(150)},
        {"name": "Hector Diaz",   "department": "IT",      "completed": ALL4, "last_training": days_ago(200)},
    ],
    "Overdue refreshers (new hires vs. veterans)": [
        {"name": "Irene Novak",   "department": "Procurement", "completed": ALL4,                     "last_training": days_ago(500)},
        {"name": "Jamal Brooks",  "department": "Procurement", "completed": ALL4,                     "last_training": days_ago(700)},
        {"name": "Keiko Tanaka",  "department": "Legal",       "completed": ["conflicts"],            "last_training": days_ago(20)},
        {"name": "Liam O'Brien",  "department": "Finance",     "completed": ["conflicts", "procurement", "transparency"], "last_training": days_ago(60)},
    ],
    "Approaching deadlines (early intervention)": [
        {"name": "Mona Haddad",   "department": "Operations", "completed": ALL4, "last_training": days_ago(350)},
        {"name": "Noah Fischer",  "department": "IT",         "completed": ALL4, "last_training": days_ago(340)},
        {"name": "Olivia Grant",  "department": "Legal",      "completed": ALL4, "last_training": days_ago(100)},
    ],
}
ETH_DEFAULT_QUESTION = "Give me the compliance status of the team and tell me who needs action first and why."

load_eth_roster(ETH_SAMPLES["Transit review team (this system)"])
print(f"✓ Roster loaded: {len(ETH_ROSTER)} people · summary {eth_team_summary()}")
display_df = pd.DataFrame(eth_check_all())
print(display_df[["official", "department", "status", "missing_courses", "days_until_refresher_due"]].to_string(index=False))

# ====================================================================================================
# notebook cell 26: C · Run the agent (optional; needs the LLM) — the guardrail corrects any disagreement with the rules
# ====================================================================================================
#@title C · Run the agent (optional; needs the LLM) — the guardrail corrects any disagreement with the rules
RUN_SECTION_C_AGENT = True  #@param {type:"boolean"}
if RUN_SECTION_C_AGENT and LLM_READY:
    try:
        eth_report = eth_agent.run(ETH_DEFAULT_QUESTION)
        eth_issues = verify_report(eth_report)
        eth_final = reconcile_report(eth_report)
        print(f"LLM: {llm_source()} · structured output used: {mode_used()}")
        print("Guardrail:", "; ".join(eth_issues) if eth_issues else "✅ matched the rule engine")
        print("VERIFIED ANSWER:", eth_final.answer)
        print(pd.DataFrame([a.model_dump() for a in eth_final.action_items]).to_string(index=False))
    except Exception as e:   # a live-service failure must not stop Run all; the rule engine above is unaffected
        print(f"Section C agent could not finish ({type(e).__name__}: {str(e)[:200]}). The gate uses the rule engine above.")
else:
    print("Section C agent not run: it needs RUN_SECTION_C_AGENT and an LLM key in Colab Secrets. The rule engine above is what the gate uses.")

# ====================================================================================================
# notebook cell 28: D · Structured outputs (the "contract" between agents)
# ====================================================================================================
#@title D · Structured outputs (the "contract" between agents)
HOLDING_KINDS = ["direct_stock", "business_ownership", "family_business", "donation_received",
                 "blind_trust", "index_fund", "other"]


class Holding(BaseModel):
    entity: str = Field(description="Company / organisation name")
    kind: Literal["direct_stock", "business_ownership", "family_business", "donation_received",
                  "blind_trust", "index_fund", "other"]
    owner: Literal["official", "spouse", "family", "donor"] = "official"
    value_usd: Optional[float] = None
    note: str = ""


class Decision(BaseModel):
    bill: str = Field(description="Name of the bill / vote / decision")
    subject: str = Field(description="What it is about, in a few words")
    action: str = Field(default="", description="e.g. voted yes, signed contract, chaired hearing")


class ExtractedFacts(BaseModel):                       # Agent 1 output
    official_name: str
    role: str = ""
    holdings: List[Holding] = []
    decisions: List[Decision] = []
    relationships: List[str] = []


class Flag(BaseModel):
    type: Literal["financial_interest", "family_interest", "donor_influence", "other"]
    severity: Literal["low", "medium", "high"]
    entity: str
    decision: str
    reasoning: str
    evidence: List[str] = Field(default=[], description="Short quotes / facts from the case file")


class AnalystOutput(BaseModel):                        # Agent 3 output
    flags: List[Flag] = []
    overall_risk_score: float = Field(ge=0, le=1)
    dismissed_rule_hits: List[str] = Field(default=[], description="Rule-screener hits judged not to be conflicts, with reason")


class ReviewerOutput(BaseModel):                       # Agent 4 output
    verdict_on_analysis: Literal["agree", "partially_agree", "disagree"]
    innocent_explanations: List[str] = Field(default=[], description="Hypothetical explanations, each starting with 'Hypothetical:'")
    mitigating_factors: List[str] = Field(default=[], description="ONLY facts stated in the case file (e.g. recusal, blind trust, competitive bidding). Empty list if none. Never speculate.")
    questions_for_official: List[str] = []


class FinalReport(BaseModel):                          # Agent 5 output
    official: str
    recommendation: Literal["CLEAR", "MONITOR", "REVIEW", "ESCALATE"]
    risk_score: float = Field(ge=0, le=1)
    summary: str = Field(description="2-3 sentences (40-80 words): who, which holding, which decision, why it matters")
    key_flags: List[str] = []
    required_actions: List[str] = []
    due_process_note: str


print("✓ Section D schemas defined:", ", ".join(c.__name__ for c in [ExtractedFacts, AnalystOutput, ReviewerOutput, FinalReport]))

# ====================================================================================================
# notebook cell 29: D · The agents, the rule-screener tool, and the orchestrator
# ====================================================================================================
#@title D · The agents, the rule-screener tool, and the orchestrator
COI_TRACE = []
COI_TOTAL_STEPS = 5


def coi_traced(name, kind, fn, verbose=True):
    """Run one workflow step, print progress, record timing."""
    i = len(COI_TRACE) + 1
    if verbose:
        print(f"▶ Step {i}/{COI_TOTAL_STEPS}  {name:<22} [{kind}] ...", end=" ", flush=True)
    t0 = time.time()
    try:
        out = fn()
    except Exception:
        COI_TRACE.append({"step": i, "name": name, "kind": kind, "seconds": round(time.time() - t0, 1), "status": "failed"})
        if verbose:
            print("✗ failed")
        raise
    COI_TRACE.append({"step": i, "name": name, "kind": kind, "seconds": round(time.time() - t0, 1), "status": "ok"})
    if verbose:
        print(f"✓ {COI_TRACE[-1]['seconds']}s")
    return out


class COIAgent:                                         # renamed from `Agent` (clashed with Section B)
    def __init__(self, name, system, schema):
        self.name, self.system, self.schema = name, system, schema

    def run(self, user_message, verbose=True):
        def _call():
            obj, retries = run_structured(self.schema, self.system, user_message)
            return obj
        return coi_traced(self.name, "LLM agent", _call, verbose)


coi_extractor = COIAgent("Extractor Agent",
    "You are the Extractor Agent in a government-ethics review pipeline. Read the case file and extract ONLY "
    "facts that are explicitly stated: the official, their holdings/relationships (own, spouse, family, donors), "
    "and the votes/decisions they made or will make. Do not infer or invent. Use kind 'blind_trust' for blind trusts, "
    "'index_fund' for broad diversified funds, 'family_business' for a relative's company, 'donation_received' for donors.",
    ExtractedFacts)

coi_analyst = COIAgent("Conflict Analyst Agent",
    "You are the Conflict Analyst Agent. You receive extracted facts and the output of a deterministic rule screener. "
    "Decide which potential conflicts are REAL. For each flag give type, severity, the entity, the decision, your reasoning, "
    "and evidence taken from the facts. Severity guide: high = direct or family financial stake in the decision; "
    "medium = donor influence or sector-level exposure; low = remote exposure (blind trust, diversified fund). "
    "Put rule hits you judge to be false alarms in dismissed_rule_hits with a reason. If there is no conflict, return an empty flags list. "
    "overall_risk_score is 0 (none) to 1 (severe).",
    AnalystOutput)

coi_reviewer = COIAgent("Reviewer Agent",
    "You are the Reviewer Agent, a devil's advocate protecting due process. Challenge the analyst's flags. Keep FACTS and HYPOTHESES separate: "
    "mitigating_factors may contain ONLY things stated in the extracted facts (e.g. recusal, blind trust, competitive bidding, diversified fund); "
    "if the case states none, return an empty list. Never invent facts. Speculative alternatives go in innocent_explanations and must start with "
    "'Hypothetical:'. Write concrete questions the official should be allowed to answer. Say whether you agree, partially agree, or disagree "
    "with the analysis. Be fair, not lenient.",
    ReviewerOutput)

coi_decider = COIAgent("Decision Agent",
    "You are the Decision Agent. Combine the facts, rule screen, analysis and review into a final report. Recommendation guide: "
    "CLEAR = no material flags; MONITOR = only low-severity flags; REVIEW = medium/high flags with a stated mitigation; "
    "ESCALATE = high-severity flag with no stated mitigation. Only the reviewer's mitigating_factors count as mitigation; hypothetical explanations "
    "and unanswered questions do NOT lower the recommendation. The summary must be 2-3 sentences (40-80 words) naming the official, the holding, "
    "the decision and why it matters. required_actions must be concrete (e.g. 'request recusal statement'). due_process_note must state that the "
    "official may respond before any action and that a human decides.",
    FinalReport)


# ---------- the deterministic tool ----------
COI_SECTORS = {
    "technology": ["tech", "software", "semiconductor", "cloud", "cyber", "antitrust", "artificial intelligence", "data privacy"],
    "healthcare_pharma": ["pharma", "drug", "health", "medical", "biotech", "prescription", "medicare"],
    "energy": ["oil", "gas", "energy", "solar", "pipeline", "coal", "utility", "drilling"],
    "defense": ["defense", "weapon", "aerospace", "military"],           # "procurement" removed (sector-neutral)
    "finance": ["bank", "financ", "insur", "lending", "crypto"],
    # transportation / construction was missing, so transit vendors and road contracts were never screened
    "transportation_infrastructure": ["transit", "bus", "coach", "rail", "paratransit", "shuttle", "fleet", "depot",
                                      "highway", "road", "paving", "construction", "detour", "transportation"],
}
COI_WEIGHTS = {"direct_stock": 1.0, "business_ownership": 1.0, "family_business": 0.8, "donation_received": 0.7,
               "blind_trust": 0.4, "index_fund": 0.2, "other": 0.5}
COI_THRESHOLD = 0.7

_LEGAL_SUFFIX = r"\b(incorporated|inc|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|plc|group|holdings)\b\.?"
_GENERIC_WORDS = {"construction", "services", "service", "sons", "and", "the", "of", "international", "systems",
                  "solutions", "partners", "company", "group", "transport", "transit", "coach", "&"}


def norm_entity(text):
    """normalize names so 'TechCorp Inc.' matches "TechCorp's cloud division"."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"'s\b", "", t)
    t = re.sub(_LEGAL_SUFFIX, " ", t)
    t = re.sub(r"[^a-z0-9& ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def name_matches(entity, text):
    e, t = norm_entity(entity), " " + norm_entity(text) + " "
    if e and f" {e} " in t:
        return True
    core = " ".join(w for w in e.split() if w not in _GENERIC_WORDS)    # e.g. "nair" from "Nair & Sons Construction LLC"
    return len(core) >= 3 and f" {core} " in t


def sectors_of(text):
    t = text.lower()
    return {s for s, kws in COI_SECTORS.items() if any(re.search(r"\b" + re.escape(k), t) for k in kws)}


def rule_screen(facts, decisions=None, threshold=None):
    """Deterministic screen: match each holding against each decision by name or by sector.
    INTEGRATION: `decisions` can be supplied (the options a reviewer is approving) instead of facts.decisions."""
    threshold = COI_THRESHOLD if threshold is None else threshold
    hits = []
    for h in facts.holdings:
        for d in (decisions if decisions is not None else facts.decisions):
            decision_text = f"{d.bill} {d.subject} {d.action}"
            name_match = name_matches(h.entity, decision_text)
            shared = sectors_of(f"{h.entity} {h.note}") & sectors_of(decision_text)
            if name_match or shared:
                score = round(COI_WEIGHTS[h.kind] * (1.0 if name_match else 0.75), 2)
                hits.append({"holding": h.entity, "kind": h.kind, "owner": h.owner, "decision": d.bill,
                             "match": "name" if name_match else f"sector:{'/'.join(sorted(shared))}", "score": score})
    top = max([x["score"] for x in hits], default=0.0)
    return {"hits": hits, "rule_score": top, "threshold": threshold, "over_threshold": top >= threshold}


# ---------- the orchestrator ----------
def coi_run_pipeline(case_text, verbose=True):
    COI_TRACE.clear()
    facts = coi_extractor.run(f"CASE FILE:\n{case_text}", verbose)
    screen = coi_traced("Rule Screener", "python tool", lambda: rule_screen(facts), verbose)
    analysis = coi_analyst.run("EXTRACTED FACTS:\n" + facts.model_dump_json(indent=1) +
                               "\n\nRULE SCREENER OUTPUT:\n" + json.dumps(screen, indent=1), verbose)
    review = coi_reviewer.run("EXTRACTED FACTS:\n" + facts.model_dump_json(indent=1) +
                              "\n\nANALYST OUTPUT:\n" + analysis.model_dump_json(indent=1), verbose)
    final = coi_decider.run("EXTRACTED FACTS:\n" + facts.model_dump_json(indent=1) +
                            "\n\nRULE SCREEN:\n" + json.dumps(screen) +
                            "\n\nANALYST:\n" + analysis.model_dump_json() +
                            "\n\nREVIEWER:\n" + review.model_dump_json(), verbose)
    return {"facts": facts, "screen": screen, "analysis": analysis, "review": review, "final": final}


def coi_guardrail(result):
    """Consistency check on the final recommendation (two consistency checks + recommendation check)."""
    final, analysis, review, screen = result["final"], result["analysis"], result["review"], result["screen"]
    issues = []
    if any(f.severity == "high" for f in analysis.flags) and not review.mitigating_factors and final.recommendation != "ESCALATE":
        issues.append("high-severity flag with no stated mitigation, but the recommendation is not ESCALATE")
    if final.recommendation == "CLEAR" and analysis.flags:
        issues.append("recommendation is CLEAR although the analyst raised flags")
    if final.recommendation == "CLEAR" and screen["over_threshold"]:
        issues.append(f"recommendation is CLEAR although the deterministic rule score {screen['rule_score']} ≥ threshold {screen['threshold']}")
    return issues


def coi_enforce(result):
    """GUARDRAIL: when the consistency check fails, the stricter recommendation wins (a person still decides)."""
    issues, final = coi_guardrail(result), result["final"]
    rule = float(result["screen"].get("rule_score") or 0)
    if final.risk_score < rule:                     # the stated risk is never below the deterministic rule score
        final = final.model_copy(update={"risk_score": rule})
        issues = issues + [f"risk raised to the rule screener's {rule}"]
    if not any("ESCALATE" in i or "CLEAR" in i for i in issues):
        return final, issues
    new = "ESCALATE" if any("ESCALATE" in i for i in issues) else "REVIEW"
    return final.model_copy(update={"recommendation": new,
                                    "summary": final.summary + f" [Guardrail: changed {final.recommendation} → {new}.]"}), issues


COI_WORKFLOW = [("case", "Case file", "input"), ("Extractor Agent", "1 Extractor", "llm"),
                ("Rule Screener", "2 Rule\nScreener", "tool"), ("Conflict Analyst Agent", "3 Conflict\nAnalyst", "llm"),
                ("Reviewer Agent", "4 Reviewer", "llm"), ("Decision Agent", "5 Decision", "llm"),
                ("report", "Report", "output")]
print("✓ Section D: 4 agents + 1 tool + orchestrator ready")

# ====================================================================================================
# notebook cell 30: D · Disclosures: transit staff (used by the gate) + the Section D sample cases
# ====================================================================================================
#@title D · Disclosures: transit staff (used by the gate) + the Section D sample cases
COI_SAMPLES = {
    "Transit · planner reviews a dispatch to his spouse's depot contractor": """
Service planner Gabe Ortiz is the first reviewer on a bus disruption case. The top-ranked option dispatches a spare bus
from Depot B. Depot B is run under contract by Metro Coach Services LLC, which is paid per spare-bus dispatch. His spouse
is the majority owner of Metro Coach Services LLC. He is about to approve the Depot B dispatch and has not recused.
""",
    "Direct stock + related vote (clear conflict)": """
Senator Maria Alvarez sits on the Technology Oversight Committee. Her latest financial disclosure lists 12,000 shares of
TechCorp Inc. (value about $480,000) held directly in her own brokerage account. Last week she voted YES on the
Tech Antitrust Reform Act, which would exempt TechCorp's cloud division from a pending federal investigation.
She did not recuse herself and made no public statement about her holdings.
""",
    "Blind trust + broad regulation (ambiguous)": """
Senator David Chen owns a diversified portfolio inside a qualified blind trust managed by an independent trustee.
He knows the trust was originally funded with a mix of technology and healthcare stocks but cannot see current holdings.
He voted on the Digital Platform Regulation Act, which imposes new data privacy rules on large technology companies.
He has publicly disclosed the trust and stated that he does not direct any trades.
""",
    "Spouse's firm wins a contract (family interest)": """
Commissioner Priya Nair oversees the state Department of Transportation budget. Her spouse is the majority owner of
Nair & Sons Construction LLC. In March the department awarded Nair & Sons a $2.3 million road-repair contract; Priya
participated in the budget meeting that approved the appropriation for the Highway Repair Program and did not recuse.
The contract was competitively bid with three other bidders.
""",
    "Major donor, pharma vote (donor influence)": """
Representative James O'Connor received $250,000 in campaign contributions last cycle from executives and the PAC of
HealthFirst Pharmaceuticals, his single largest donor. This month he cast the deciding committee vote against the
Prescription Drug Price Cap Act, which HealthFirst lobbied heavily against. He owns only a small index fund
and has no direct holdings in any pharmaceutical company.
""",
    "Low-risk baseline (nothing suspicious)": """
Councilmember Lena Park holds a broad-market index fund in her retirement account (about $40,000) and owns no individual
stocks. She voted on a city budget amendment to repaint public libraries. No relatives work for the vendor. She received
no campaign donations from the vendor.
""",
}

# INTEGRATION: annual disclosures of the people who decide in this system (fictional). The structured facts are what
# the Extractor Agent would produce; they are stored so the gate works without a language model.
STAFF_DISCLOSURES = {
    "Gabe Ortiz": {
        "text": "Service planner Gabe Ortiz reviews disruption responses. His spouse is the majority owner of Metro Coach "
                "Services LLC, the contract operator of Depot B, which is paid per spare-bus dispatch.",
        "facts": ExtractedFacts(official_name="Gabe Ortiz", role="Service Planner", holdings=[
            Holding(entity="Metro Coach Services LLC", kind="family_business", owner="spouse",
                    note="contract operator of Depot B; paid per spare bus dispatch")],
            relationships=["spouse owns Metro Coach Services LLC"]),
    },
    "Chen Wei": {
        "text": "Operations supervisor Chen Wei holds a broad-market index fund (about $25,000) in a retirement account "
                "and no individual stocks.",
        "facts": ExtractedFacts(official_name="Chen Wei", role="Operations Supervisor",
                                holdings=[Holding(entity="Total Market Index Fund", kind="index_fund", value_usd=25000)]),
    },
    "Ana Lopez":   {"text": "Ana Lopez reports no holdings, family businesses or gifts.",   "facts": ExtractedFacts(official_name="Ana Lopez")},
    "Dana Brooks": {"text": "Dana Brooks reports no holdings, family businesses or gifts.", "facts": ExtractedFacts(official_name="Dana Brooks")},
    "Ben Carter":  {"text": "Ben Carter reports no holdings, family businesses or gifts.",  "facts": ExtractedFacts(official_name="Ben Carter")},
}

# Deterministic check of two Section D samples — no LLM needed
_alvarez = ExtractedFacts(official_name="Maria Alvarez", holdings=[Holding(entity="TechCorp Inc.", kind="direct_stock", value_usd=480000)],
                          decisions=[Decision(bill="Tech Antitrust Reform Act", subject="exempting TechCorp's cloud division from a pending federal investigation", action="voted yes")])
_nair = ExtractedFacts(official_name="Priya Nair", holdings=[Holding(entity="Nair & Sons Construction LLC", kind="family_business", owner="spouse")],
                       decisions=[Decision(bill="Highway Repair Program appropriation", subject="road-repair contract awarded to Nair & Sons", action="approved budget")])
_depot_b = [Decision(bill="Dispatch a second bus — Depot B", action="approve operational action",
                     subject="dispatch second bus for transit bus service; parties: Metro Coach Services LLC")]
for _who in ["Gabe Ortiz", "Chen Wei", "Ana Lopez"]:
    _s = rule_screen(STAFF_DISCLOSURES[_who]["facts"], _depot_b)
    print(f"{_who:10s} approving a Depot B dispatch → rule score {_s['rule_score']} "
          f"({'RECUSE' if _s['over_threshold'] else 'clear'}) {[h['holding'] for h in _s['hits']] or ''}")
print("Section D samples (stock vote, spouse contract) flagged by the screener:", rule_screen(_alvarez)["over_threshold"], rule_screen(_nair)["over_threshold"])

# ====================================================================================================
# notebook cell 31: D · Run the multi-agent pipeline (optional; needs the LLM)
# ====================================================================================================
#@title D · Run the multi-agent pipeline (optional; needs the LLM)
RUN_SECTION_D_PIPELINE = True  #@param {type:"boolean"}
COI_CASE = "Transit · planner reviews a dispatch to his spouse's depot contractor"  #@param ["Transit · planner reviews a dispatch to his spouse's depot contractor", "Direct stock + related vote (clear conflict)", "Spouse's firm wins a contract (family interest)", "Low-risk baseline (nothing suspicious)"]
if RUN_SECTION_D_PIPELINE and LLM_READY:
    try:
        coi_result = coi_run_pipeline(COI_SAMPLES[COI_CASE].strip())
        f, g_issues = coi_enforce(coi_result)
        print(f"LLM: {llm_source()} · case: {COI_CASE}")
        print(f"Rule screener: score {coi_result['screen']['rule_score']} (threshold {coi_result['screen']['threshold']})")
        print("Guardrail:", "; ".join(g_issues) if g_issues else "✅ consistent")
        print(f"{f.official} → {f.recommendation} (risk {f.risk_score:.2f})\n{f.summary}")
    except Exception as e:   # a live-service failure must not stop Run all; the rule screener above is unaffected
        print(f"Section D pipeline could not finish ({type(e).__name__}: {str(e)[:200]}). The gate uses the rule screener above.")
else:
    print("Section D pipeline not run: it needs RUN_SECTION_D_PIPELINE and an LLM key in Colab Secrets. The rule screener above is what the gate uses.")

# ====================================================================================================
# notebook cell 33: INTEGRATION · governance_check(): ethics-training gate (C) + conflict-of-interest gate (D)
# ====================================================================================================
#@title INTEGRATION · governance_check(): ethics-training gate (C) + conflict-of-interest gate (D)
def decisions_for_option(option):
    """Turn an option under review into Section D `Decision`s (its outside parties are what gets screened)."""
    parties = ", ".join(option.get("parties", []))
    return [Decision(bill=option["label"], subject=f"{option['type'].replace('_', ' ')} for transit bus service; parties: {parties}",
                     action="approve operational action")]


def decisions_for_proposal(proposal):
    changes = "; ".join(f"{c['path']} {c['old']}→{c['new']}" for c in proposal["changes"])
    return [Decision(bill=proposal["title"], subject=f"transit scoring policy change: {changes}", action="approve rule change")]


def governance_check(name, decision, option=None, proposal=None, policy=None):
    """Return (problems, notes). Problems block the submission; notes are logged in the audit record."""
    policy = policy or APP.PM.current
    g = policy.get("governance_gates", {})
    problems, notes = [], []
    if not g.get("enabled", True) or not name.strip():
        return problems, notes
    # ---- Section C: ethics training
    st = eth_check_official(name)
    if "error" in st:
        msg = f"{name} is not on the ethics-training roster"
        (problems if g.get("unknown_person") == "block" else notes).append(msg)
    elif st["status"] in g.get("training_block_statuses", []) and decision in g.get("training_gate_decisions", []):
        why = f"missing {', '.join(st['missing_courses'])}" if st["missing_courses"] else (
            "refresher overdue" if st["days_until_refresher_due"] is not None else "never trained")
        problems.append(f"{name} is NON_COMPLIANT on ethics training ({why}) and cannot {decision.lower()} — another reviewer must act")
    elif st["status"] == "AT_RISK":
        notes.append(f"{name}: ethics refresher due in {st['days_until_refresher_due']} days")
    else:
        notes.append(f"{name}: ethics training {st['status']}")
    # ---- Section D: conflict of interest
    if decision in g.get("coi_gate_decisions", []) and (option or proposal):
        disc = STAFF_DISCLOSURES.get(name.strip())
        if disc is None:
            notes.append(f"{name}: no conflict-of-interest disclosure on file")
        else:
            decs = decisions_for_option(option) if option else decisions_for_proposal(proposal)
            screen = rule_screen(disc["facts"], decs, g.get("coi_threshold", COI_THRESHOLD))
            name_hits = [h for h in screen["hits"] if h["match"] == "name" and h["score"] >= screen["threshold"]]
            if name_hits:
                h = name_hits[0]
                problems.append(f"possible conflict of interest: {name}'s {h['owner']} holding '{h['holding']}' ({h['kind']}) is a party "
                                f"to this option (score {h['score']} ≥ {screen['threshold']}) — recuse; another reviewer must approve")
            elif screen["hits"]:
                notes.append(f"{name}: conflict screen hits below the recusal rule — " +
                             "; ".join(f"{h['holding']} ({h['match']}, {h['score']})" for h in screen["hits"]))
            else:
                notes.append(f"{name}: conflict screen clear")
    return problems, notes


print("✓ Governance gate ready (training: Section C rules · independence: Section D rule screener)")

# ====================================================================================================
# notebook cell 35: E3 · Shared case state, Explanation Agent (live LLM), human review, deadlines, safe default, implementation
# ====================================================================================================
#@title E3 · Shared case state, Explanation Agent (live LLM), human review, deadlines, safe default, implementation
REVIEW_STATUSES = ("AWAITING_REVIEW", "AWAITING_SECOND_REVIEW")
STATUS_HELP = {
    "MONITORING": "Continue monitoring (unresolved or low-confidence data)",
    "CONFLICT_REVIEW": "Conflicting data quarantined — operator review needed",
    "SAFE_DEFAULT": "Safe default: verified rider information only, keep monitoring",
    "AWAITING_REVIEW": "Waiting for human reviewer",
    "AWAITING_SECOND_REVIEW": "High impact — waiting for independent second reviewer",
    "AWAITING_IMPLEMENTATION": "Approved — operations staff to carry out the action outside the prototype",
    "IMPLEMENTED": "Action implemented — ready for outcome evaluation",
    "EVALUATED": "Outcome evaluated",
    "CLOSED_FALSE_ALARM": "Closed as false alarm",
}


def agent_llm_generate(prompt, system, max_tokens=320, timeout=None):
    """Agent text through the shared live LLM client (OpenAI → OpenRouter → Hugging Face → Ollama if enabled)."""
    try:
        return llm_generate(prompt, system, max_tokens=max_tokens)
    except LLMUnavailable as e:
        raise AgentFailure(f"LLM call failed: {e}")


def rider_alert_text(case):
    """Deterministic rider message built from verified evidence only."""
    A, raw = case["analysis"], case["raw"]
    f = A.get("fields", {})
    acc = [e for e in A.get("evidence", []) if e["status"] == "accepted" and not e["source_id"].startswith("VISION-")]
    if not acc:
        return f"Route {raw['route_id']} {raw['route_name']}: we are checking reports of service problems. Updates will follow."
    src = min(acc, key=lambda e: precedence_rank(get_case_policy(case), e["source_type"]))
    stops = raw["affected_stop_ids"]
    parts = [f"Route {raw['route_id']} {raw['route_name']}"]
    if f.get("cancelled"):
        parts.append("a scheduled trip is cancelled")
    elif f.get("delay_min") is not None:
        parts.append(f"buses are running about {f['delay_min']:.0f} min late")
    if f.get("event_severity", "none") != "none":
        parts.append(f"due to {raw.get('event_type', 'an incident').replace('_', ' ')}")
    msg = ": ".join([parts[0], ", ".join(parts[1:]) or "service disruption"]) + f" between {NET['stops'][stops[0]]['name']} and {NET['stops'][stops[-1]]['name']}."
    nearby = sorted({r for s in stops for r in NET["stops"][s]["nearby_routes"]})
    if nearby:
        msg += f" Alternatives: route(s) {', '.join(nearby)}."
    msg += f" Verified by {src['source_id']} at {src['timestamp'][11:16]} ET. Accessible-service riders can call the paratransit desk."
    return msg


class CaseStore:
    """Single shared case record that every stage reads and writes (thread-safe; shared by all reviewers via the Gradio link)."""

    def __init__(self, policy_manager):
        self.PM = policy_manager
        self.cases, self.history, self.audit = {}, [], []
        self.active_id = None
        self.lock = threading.RLock()
        self.counter = 0
        self.last_message = "Ready. Pick a scenario and press ▶ Run pipeline."
        self.proposals, self.proposal_log, self.backtest = [], [], None
        self.insight_narrative = ""

    # ---------- audit
    def log(self, case, actor, role, stage, event, detail=""):
        e = {"ts": iso(utcnow()), "case_id": case["case_id"] if case else "—", "actor": actor, "role": role, "stage": stage,
             "event": event, "detail": detail, "policy_version": case["policy_version"] if case else self.PM.current["version"]}
        self.audit.append(e)
        if case is not None:
            case["audit"].append(e)

    @property
    def active(self):
        return self.cases.get(self.active_id)

    # ---------- intake
    def new_case(self, raw, use_llm=True):
        with self.lock:
            self.counter += 1
            policy = self.PM.current
            case = {"case_id": f"CASE-{self.counter:04d}", "created_at": iso(utcnow()), "scenario_name": raw["scenario_name"],
                    "policy_version": policy["version"], "raw": raw, "status": None, "analysis": None,
                    "constraints": {}, "revisions": 0, "operator_checks": 0, "operator_resolution": None,
                    "review": None, "decisions": [], "implementation": [], "outcome": None, "alerts": [], "audit": [],
                    "explanation": None, "rider_alert": None, "use_llm": use_llm, "history": False}
            self.cases[case["case_id"]] = case
            self.active_id = case["case_id"]
            self.log(case, "Signal Scout Agent", "agent", "Intake", "Case created", raw["scenario_name"])
            self._analyze(case)
            return case

    def _analyze(self, case):
        policy = get_case_policy(case)
        A = run_analysis(case["raw"], policy, case["operator_resolution"], case["constraints"])
        case["analysis"] = A
        for t in A["trace"]:
            self.log(case, "pipeline", "agent/rule", t["stage"], t["result"], t["detail"])
        for a in A["alerts"]:
            case["alerts"].append({**a, "ts": iso(utcnow())})
            self.log(case, "Signal Scout Agent", "agent", "Alert", f"Alert → {a['to']}", a["message"])
        g = A["gate"]
        if g == "SAFE_MODE":
            self.safe_default(case, "Agent error path: deterministic safe mode")
        elif g == "BELOW_THRESHOLD":
            self.safe_default(case, "Below significance threshold")
        elif g == "MONITORING":
            case["status"] = "MONITORING"
            self.log(case, "Signal Scout Agent", "agent", "Monitoring", "Continue monitoring", A["gate_reason"])
        elif g == "CONFLICT_REVIEW":
            case["status"] = "CONFLICT_REVIEW"
        elif g == "SIGNIFICANT":
            self.log(case, "Signal Scout Agent", "agent", "Create case + log", "Significant case", A["gate_reason"])
            self.explain(case)
            self.start_review(case)
        self.last_message = f"{case['case_id']}: {STATUS_HELP.get(case['status'], case['status'])}"

    # ---------- exception-path human actions
    def operator_resolve(self, case, operator, resolution):
        if case["status"] != "CONFLICT_REVIEW":
            raise ValueError("This case has no conflict awaiting operator review.")
        if not operator.strip():
            raise ValueError("Enter the operator's name.")
        case["operator_checks"] += 1
        self.log(case, operator, "operator", "Operator review", resolution, f"check {case['operator_checks']} of max {get_case_policy(case)['max_operator_checks']}")
        if resolution == "unresolved" and case["operator_checks"] < get_case_policy(case)["max_operator_checks"]:
            self.last_message = "Conflict still unresolved; one more operator check allowed before monitoring."
            return
        case["operator_resolution"] = "accept_precedence" if resolution == "resolved" else "unresolved"
        self._analyze(case)

    def monitor_action(self, case, action, actor="planner"):
        if case["status"] not in ("MONITORING", "SAFE_DEFAULT", "CONFLICT_REVIEW"):
            raise ValueError("Only monitoring / safe-default cases can be rechecked or re-escalated.")
        raw = copy.deepcopy(case["raw"])
        if action == "recheck":
            for o in raw["observations"]:
                o["age_s"] = min(o["age_s"], 30)
                o["timestamp"] = iso(utcnow() - timedelta(seconds=o["age_s"]))
            note = "Feed changed: fresh snapshot re-verified"
        else:
            s = {**raw["spec"], "delay_min": (raw["spec"]["delay_min"] or 0) + 8,
                 "headway_ratio": raw["spec"]["headway_ratio"] + 0.6, "passups": raw["spec"]["passups"] + 8,
                 "feed_age_s": 20, "sources": sorted(set(raw["spec"]["sources"]) | {"vehicle_apc"}),
                 "vision_agent_failures": 0, "public_corroborated": raw["spec"]["public_corroborated"]}
            raw = make_raw(s, raw["scenario_name"] + " (worsened)")
            note = "Conditions worsened: delay +8 min, gap +0.6×, pass-ups +8"
        case["raw"], case["operator_resolution"] = raw, None
        case["policy_version"] = self.PM.current["version"]
        self.log(case, actor, "planner", "Monitoring", "Re-escalate" if action != "recheck" else "Recheck", note)
        self._analyze(case)

    def close_false_alarm(self, case, actor, reason):
        if case["status"] not in ("MONITORING", "SAFE_DEFAULT"):
            raise ValueError("Only monitoring or safe-default cases can be closed as false alarms.")
        if len(reason.strip()) < 5:
            raise ValueError("Enter a reason for closing as false alarm.")
        case["status"] = "CLOSED_FALSE_ALARM"
        case["outcome"] = {"evaluated_at": iso(utcnow()), "false_alarm": True, "note": reason, "action_type": "none"}
        self.log(case, actor, "planner", "Monitoring", "Closed as false alarm", reason)
        self.last_message = f"{case['case_id']} closed as false alarm."

    # ---------- explanation agent
    def explain(self, case):
        case["explanation"] = {"brief_md": explanation_brief(case), "narrative": "", "narrative_source": "template",
                               "generated_at": iso(utcnow())}
        case["rider_alert"] = rider_alert_text(case)
        if case.get("use_llm") and LLM_READY:
            sc = SignalScout()
            try:
                txt = sc.invoke("explanation_agent", llm_narrative, case, retry=True)
                case["explanation"].update(narrative=txt, narrative_source=f"live LLM · {llm_source()}")
            except AgentFailure as e:
                case["explanation"]["narrative_source"] = "template (LLM narrative failed twice; template used)"
                self.log(case, "Explanation Agent", "agent", "Agent error path", "Fallback to template", str(e))
                sc.alert("technical owner", f"Explanation Agent failed twice: {e}")
            for x in sc.exceptions:
                self.log(case, "Explanation Agent", "agent", "Agent error path", x["outcome"], x["detail"])
        if not case["explanation"]["narrative"]:
            case["explanation"]["narrative"] = template_narrative(case)
        self.log(case, "Explanation Agent", "agent", "Explanation", "Options ranking brief published",
                 case["explanation"]["narrative_source"])

    # ---------- human review
    def start_review(self, case, note="Options presented to reviewer"):
        policy = get_case_policy(case)
        tier = case["analysis"]["tier"] or "moderate"
        mins = policy["deadlines_min"][tier]
        now = utcnow()
        case["review"] = {"tier": tier, "deadline_min": mins, "started_at": iso(now),
                          "deadline_at": iso(now + timedelta(seconds=mins * 60 * DEMO_DEADLINE_SCALE)),
                          "first": None, "second": None, "required_second": None,
                          "no_feasible_option": not case["analysis"]["ranking"]}
        case["status"] = "AWAITING_REVIEW"
        self.log(case, "Deadline control", "rule", "Human review", f"Timer started ({tier}: {mins} min, covers both reviewers)", note)

    def seconds_left(self, case):
        if not case or case["status"] not in REVIEW_STATUSES or not case.get("review"):
            return None
        return (parse_iso(case["review"]["deadline_at"]) - utcnow()).total_seconds()

    def enforce_deadline(self, case):
        left = self.seconds_left(case)
        if left is not None and left <= 0:
            self.safe_default(case, "Deadline expired with no complete response")
            return True
        return False

    def _validate_submission(self, case, name, reason, checklist, viewed, decision=None, option_id=None):
        p = get_case_policy(case)["review"]
        errs = []
        # INTEGRATION (Sections C + D): ethics-training and conflict-of-interest gates
        if decision and name.strip():
            opt = next((o for o in case["analysis"].get("options", []) if o["option_id"] == option_id), None)
            gp, gn = governance_check(name, decision, option=opt, policy=get_case_policy(case))
            errs += gp
            for n_ in gn:
                self.log(case, "Governance gate", "rule", "Reviewer gate (C+D)", "note", n_)
        if not name.strip():
            errs.append("reviewer name required")
        if len(reason.strip()) < p["min_reason_chars"]:
            errs.append(f"typed reason of at least {p['min_reason_chars']} characters required")
        if set(checklist or []) != set(REVIEW_CHECKLIST):
            errs.append("every checklist item must be ticked")
        if not viewed:
            errs.append("confirm you viewed the alternatives (rubber-stamp protection)")
        return errs

    def is_high_impact(self, case, option_id):
        p = get_case_policy(case)["high_impact"]
        o = next((o for o in case["analysis"]["options"] if o["option_id"] == option_id), None)
        P = case["analysis"]["scores"]["P"]
        return bool(o and (o["type"] in p["actions"] or o["cost_usd"] >= p["cost_min"] or P >= p["P_min"]))

    def submit_first(self, case, name, decision, option_id, reason, checklist, viewed, revision=None):
        if case["status"] != "AWAITING_REVIEW":
            raise ValueError(f"Case is {case['status']}; not awaiting first review.")
        if self.enforce_deadline(case):
            raise ValueError("Deadline had already expired — case routed to safe default.")
        errs = self._validate_submission(case, name, reason, checklist, viewed, decision, option_id)
        if errs:
            self.log(case, name or "?", "reviewer", "Human review", "Submission blocked", "; ".join(errs))
            raise ValueError("Cannot submit: " + "; ".join(errs))
        now = utcnow()
        rt = (now - parse_iso(case["review"]["started_at"])).total_seconds()
        ranking = case["analysis"]["ranking"]
        rec = ranking[0]["option_id"] if ranking else None
        d = {"stage": "first review", "reviewer": name.strip(), "decision": decision, "option_id": option_id if decision == "Approve" else None,
             "recommended_option_id": rec, "override": decision == "Approve" and option_id != rec, "reason": reason.strip(),
             "checklist_complete": True, "viewed_alternatives": True, "response_s": round(rt, 1), "ts": iso(now),
             "deadline_at": case["review"]["deadline_at"], "tier": case["review"]["tier"], "revision_no": case["revisions"]}
        if decision == "Approve":
            if not ranking:
                raise ValueError("No feasible option: revise constraints or reject.")
            if option_id not in [r["option_id"] for r in ranking]:
                raise ValueError("Pick one of the ranked feasible options.")
            case["decisions"].append(d)
            case["review"]["first"] = d
            hi = self.is_high_impact(case, option_id)
            case["review"]["required_second"] = hi
            self.log(case, name, "reviewer", "Human review", "APPROVE" + (" (override of recommendation)" if d["override"] else ""),
                     f"{option_id}; {reason}")
            if hi:
                case["status"] = "AWAITING_SECOND_REVIEW"
                case["review"]["second_requested_at"] = iso(now)
                self.last_message = "High-impact action — sent to an independent second reviewer (same deadline)."
            else:
                self._approved(case, option_id)
        elif decision == "Reject":
            case["decisions"].append(d)
            self.log(case, name, "reviewer", "Human review", "REJECT", reason)
            self.safe_default(case, f"Rejected by {name}")
        elif decision == "Revise":
            pmax = get_case_policy(case)["max_revisions"]
            case["decisions"].append(d)
            if case["revisions"] >= pmax:
                self.log(case, name, "reviewer", "Revision loop", "Third revision request", "Max two revisions → safe default")
                self.safe_default(case, "Third revision request")
                return
            case["revisions"] += 1
            case["constraints"] = {**case["constraints"], **(revision or {})}
            self.log(case, name, "planner", "Revision loop", f"Revision {case['revisions']} of {pmax}", json.dumps(revision or {}))
            A = run_analysis(case["raw"], get_case_policy(case), case["operator_resolution"], case["constraints"])
            case["analysis"] = A
            self.log(case, "Scenario evaluator", "rule", "Revision loop", "Scenarios rebuilt and rescored with unchanged rules",
                     " > ".join(f"{r['label']} ({r['Si']:.3f})" for r in A["ranking"]) or "no feasible option")
            self.explain(case)
            case["status"] = "AWAITING_REVIEW"
            case["review"]["started_at"] = iso(utcnow())  # response time measured per round
            if get_case_policy(case).get("deadline_resets_on_revision"):   # policy flag
                mins = case["review"]["deadline_min"]
                case["review"]["deadline_at"] = iso(utcnow() + timedelta(seconds=mins * 60 * DEMO_DEADLINE_SCALE))
            case["review"]["no_feasible_option"] = not A["ranking"]
            self.last_message = f"Revision {case['revisions']} rescored. " + (
                "New deadline started." if get_case_policy(case).get("deadline_resets_on_revision") else "Deadline unchanged.")
        else:
            raise ValueError("Decision must be Approve, Revise or Reject.")

    def submit_second(self, case, name, decision, reason, checklist, viewed):
        if case["status"] != "AWAITING_SECOND_REVIEW":
            raise ValueError("Case is not awaiting a second reviewer.")
        if self.enforce_deadline(case):
            raise ValueError("Deadline had already expired — case routed to safe default.")
        errs = self._validate_submission(case, name, reason, checklist, viewed, decision, case["review"]["first"]["option_id"])
        if name.strip() and case["review"]["first"] and name.strip().lower() == case["review"]["first"]["reviewer"].lower():
            errs.append("second reviewer must be a different, independent person")
        if errs:
            self.log(case, name or "?", "reviewer", "Second reviewer", "Submission blocked", "; ".join(errs))
            raise ValueError("Cannot submit: " + "; ".join(errs))
        now = utcnow()
        rt = (now - parse_iso(case["review"]["second_requested_at"])).total_seconds()
        oid = case["review"]["first"]["option_id"]
        d = {"stage": "second review", "reviewer": name.strip(), "decision": decision, "option_id": oid if decision == "Approve" else None,
             "recommended_option_id": case["review"]["first"]["recommended_option_id"], "override": False, "reason": reason.strip(),
             "checklist_complete": True, "viewed_alternatives": True, "response_s": round(rt, 1), "ts": iso(now),
             "deadline_at": case["review"]["deadline_at"], "tier": case["review"]["tier"], "revision_no": case["revisions"]}
        case["decisions"].append(d)
        case["review"]["second"] = d
        self.log(case, name, "second reviewer", "Second reviewer", decision.upper(), reason)
        if decision == "Approve":
            self._approved(case, oid)
        else:
            self.safe_default(case, f"Rejected by second reviewer {name}")

    def _approved(self, case, option_id):
        case["approved_option_id"] = option_id
        case["status"] = "AWAITING_IMPLEMENTATION"
        self.log(case, "system", "rule", "Implement action", "Approved — handed to operations staff", option_id)
        self.last_message = f"Approved {option_id}. Operations staff carry it out outside the prototype and send rider alerts."

    def safe_default(self, case, reason):
        case["status"] = "SAFE_DEFAULT"
        case["approved_option_id"] = None
        case["rider_alert"] = rider_alert_text(case)
        case["safe_default_reason"] = reason
        self.log(case, "Safe default", "rule", "Safe default", reason,
                 "Verified rider information sent; monitoring continues; no dispatch, rerouting or live control")
        self.last_message = f"{case['case_id']} → SAFE DEFAULT ({reason})."

    # ---------- implementation
    def record_implementation(self, case, staff, result, notes):
        if case["status"] != "AWAITING_IMPLEMENTATION":
            raise ValueError("No approved action is awaiting implementation.")
        if not staff.strip():
            raise ValueError("Enter the operations staff name.")
        if result in ("Partial", "Failed") and not notes.strip():          # notes required so the reviewer can decide
            raise ValueError("Describe what went wrong so the reviewer can decide (notes required for partial or failed).")
        rec = {"ts": iso(utcnow()), "staff": staff, "result": result, "notes": notes, "option_id": case["approved_option_id"]}
        case["implementation"].append(rec)
        self.log(case, staff, "operations", "Implement action", result, notes)
        if result == "Completed":
            case["status"] = "IMPLEMENTED"
            self.last_message = "Implementation recorded. Run the outcome evaluator next."
        else:
            self.log(case, "system", "rule", "Implement action", "FAILED OR PARTIAL: return to review", result)
            self.start_review(case, "Returned to review after failed/partial implementation")
            self.last_message = "Failed/partial action returned to human review with a new timer."

    # ---------- outcome evaluator (separate from the recommender; never changes policy)
    def predicted_for(self, case):
        A = case["analysis"]
        oid = case.get("approved_option_id")
        if not oid and A.get("options"):
            oid = next((o["option_id"] for o in A["options"] if o["type"] == "rider_alerts"), None)
        o = next((o for o in A.get("options", []) if o["option_id"] == oid), None)
        if o is None:
            return {"option_id": "rider information only", "type": "rider_alerts", "delay_reduction_min": 0.0,
                    "access_protection": 0.1, "vulnerable_riders_helped": 0.0, "cost_usd": 50.0}
        return {"option_id": o["option_id"], "type": o["type"], "delay_reduction_min": o["pred_delay_reduction_min"],
                "access_protection": o["pred_access_protection"], "vulnerable_riders_helped": o["pred_vulnerable_riders_helped"],
                "cost_usd": o["cost_usd"]}

    def simulate_observed(self, case):
        return simulated_observation(self.predicted_for(case), case["case_id"])

    def evaluate_outcome(self, case, obs, actor="Outcome Evaluator"):
        if case["status"] not in ("IMPLEMENTED", "SAFE_DEFAULT", "MONITORING", "EVALUATED"):
            raise ValueError("Evaluate after implementation, or for safe-default / monitoring cases.")
        case["outcome"] = outcome_metrics(self.predicted_for(case), obs)
        if case["status"] == "IMPLEMENTED":
            case["status"] = "EVALUATED"
        self.log(case, actor, "agent", "Outcome evaluator", "Predicted vs observed compared",
                 f"delay err {case['outcome']['delay_error_min']:+.1f} min; false alarm={case['outcome']['false_alarm']}. Policy weights unchanged.")
        self.last_message = f"Outcome evaluated for {case['case_id']}. Policy weights are never changed automatically."

    # ---------- reporting
    def all_cases(self, include_history=True):
        return (self.history if include_history else []) + list(self.cases.values())

    def rubber_stamp_report(self):
        rows = []
        p = self.PM.current["review"]
        decs = [d for c in self.all_cases() for d in c["decisions"]]
        by = {}
        for d in decs:
            by.setdefault(d["reviewer"], []).append(d)
        for r, ds in sorted(by.items()):
            n = len(ds)
            appr = sum(d["decision"] == "Approve" for d in ds) / n
            med = statistics.median(d["response_s"] for d in ds)
            fast = sum(d["response_s"] < p["fast_response_flag_s"] for d in ds)
            over = sum(bool(d.get("override")) for d in ds)
            flags = []
            if n >= p["min_n_for_pattern"] and appr >= p["approval_rate_flag"]:
                flags.append(f"approval rate ≥ {p['approval_rate_flag']:.0%}")
            if fast:
                flags.append(f"{fast} response(s) < {p['fast_response_flag_s']} s")
            rows.append({"reviewer": r, "decisions": n, "approval_rate": round(appr, 2), "median_response_s": round(med, 1),
                         "fast_responses": fast, "overrides": over, "audit_flag": "; ".join(flags) or "—"})
        return pd.DataFrame(rows)


def simulated_observation(pr, key):
    """Deterministic synthetic 'observed' outcome for a predicted action (replace with real post-incident data)."""
    rng = stable_rng("observed", key)
    k = {"dispatch_second_bus": 0.75, "reroute": 0.9}.get(pr["type"], 0.95)
    return {"observed_delay_reduction_min": round(pr["delay_reduction_min"] * k * rng.uniform(0.85, 1.1), 1),
            "observed_access_protection": round(min(1, pr["access_protection"] * rng.uniform(0.85, 1.05)), 2),
            "observed_vulnerable_riders_served": round(pr["vulnerable_riders_helped"] * rng.uniform(0.75, 1.0)),
            "actual_cost_usd": round(pr["cost_usd"] * (1.2 if pr["type"] == "dispatch_second_bus" else 1.0) * rng.uniform(0.95, 1.1)),
            "rider_reported_impact": rng.choice([1, 2, 2, 3, 3, 4]), "false_alarm": rng.random() < 0.15}


def get_case_policy(case):
    return APP.PM.get(case["policy_version"])


def outcome_metrics(pr, obs):
    return {"evaluated_at": iso(utcnow()), "action": pr["option_id"], "action_type": pr["type"],
            "pred_delay_reduction_min": pr["delay_reduction_min"], "obs_delay_reduction_min": obs["observed_delay_reduction_min"],
            "delay_error_min": round(pr["delay_reduction_min"] - obs["observed_delay_reduction_min"], 2),
            "pred_access": pr["access_protection"], "obs_access": obs["observed_access_protection"],
            "pred_equity_riders": pr["vulnerable_riders_helped"], "obs_equity_riders": obs["observed_vulnerable_riders_served"],
            "pred_cost_usd": pr["cost_usd"], "actual_cost_usd": obs["actual_cost_usd"],
            "cost_ratio": round(obs["actual_cost_usd"] / pr["cost_usd"], 2) if pr["cost_usd"] else None,
            "rider_reported_impact": obs["rider_reported_impact"], "false_alarm": bool(obs["false_alarm"])}


def explanation_brief(case):
    """Deterministic Options Ranking Brief: every option, formula, component score, rank, assumption, uncertainty, source, timestamp.
    Alternatives are displayed BEFORE the recommendation."""
    A = case["analysis"]
    pol = get_case_policy(case)
    P = A["scores"]["P"]
    w = scenario_weights(P, pol)
    L = [f"#### Options ranking brief — {case['case_id']} · rules v{case['policy_version']}"]
    L.append(f"**Priority P = {P:.3f}** · tier **{A['tier']}** · confidence **C = {A['confidence']['C']:.3f}**"
             + (" (official hard-trigger bypass of C gate)" if A["confidence"]["C"] < pol["confidence_threshold"] else ""))
    L.append(f"\n**Formula** Si = Ci × [{w['DR']:.3f}·DR + {w['EB']:.3f}·EB + {w['AP']:.3f}·AP + {w['OF']:.3f}·OF + {w['CE']:.3f}·CE]"
             f" — weights sum to {sum(w.values()):.2f}; as P rises, weight shifts from OF/CE to EB/AP. Ci = C × option certainty.")
    L.append("\n**All alternatives (shown before the recommendation)**\n")
    L.append("| Rank | Option | DR | EB | AP | OF | CE | Ci | **Si** | Cost | Assumptions |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    opts = {o["option_id"]: o for o in A["options"]}
    for r in A["ranking"]:
        o = opts[r["option_id"]]
        L.append(f"| {r['rank']} | {r['label']} | {r['DR']:.2f} | {r['EB']:.2f} | {r['AP']:.2f} | {r['OF']:.2f} | {r['CE']:.2f} | {r['Ci']:.2f} | **{r['Si']:.3f}** | ${r['cost_usd']:.0f} | {o['notes']}; parties: {', '.join(o.get('parties', []))}{'; ' + o['assumptions'] if o['assumptions'] else ''} |")
    inf = [o for o in A["options"] if not o["feasible"]]
    if inf:
        L.append("\n**Considered but infeasible:** " + "; ".join(f"{o['label']} ({o['infeasible_reason']})" for o in inf))
    L.append("\n**Uncertainty:** " + (f"missing fields not imputed: {', '.join(A['missing'])}; " if A["missing"] else "no missing fields; ")
             + f"Q={A['confidence']['Q']}, F={A['confidence']['F']}, G={A['confidence']['G']}, M={A['confidence']['M']}.")
    raw = case["raw"]
    if raw.get("incident_history"):
        L.append("**Incident history (input):** " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in raw["incident_history"].items() if k != "observed_at"))
    if raw.get("slm_interpretation"):
        L.append(f"**Section A SLM reading of the event report (supporting only, not scored):** {_html.escape(str(raw['slm_interpretation']))}")
    srcs = [f"{e['source_id']} ({e['source_type']}, {e['timestamp'][11:19]} ET, age {e['age_s']} s)" for e in A["evidence"] if e["status"] == "accepted"]
    L.append("**Sources used:** " + "; ".join(srcs))
    if A["ranking"]:
        top = A["ranking"][0]
        tie = len(A["ranking"]) > 1 and abs(A["ranking"][1]["Si"] - top["Si"]) < 0.01
        L.append(f"\n**Recommendation (rank 1):** {top['label']} — Si = {top['Si']:.3f}"
                 + (". ⚠️ Close call (< 0.01 from rank 2) — tie-break order: higher AP, higher OF, lower cost." if tie else "."))
    else:
        L.append("\n**No feasible option — escalated to human review.** Revise constraints or reject (safe default).")
    return "\n".join(L)


def template_narrative(case):
    A = case["analysis"]
    rk = A["ranking"]
    if not rk:
        return "No option satisfies the current resource, road, policy and safety constraints. The reviewer should revise constraints or reject, which routes the case to the safe default."
    alt = "; ".join(f"{r['label']} scores {r['Si']:.3f}" for r in rk[1:]) or "there are no other feasible options"
    trig = ", ".join(t["trigger"] for t in A["hard_triggers"]) or f"R = {A['scores']['R']:.2f}"
    return (f"This case is significant because of {trig}. Priority P = {A['scores']['P']:.2f} (tier {A['tier']}) shifts scenario weight toward equity "
            f"and access. Alternatives: {alt}. The deterministic evaluator ranks **{rk[0]['label']}** first with Si = {rk[0]['Si']:.3f}. "
            f"This is a recommendation only; the human reviewer decides.")


def narrative_names_rank1(text, label):
    """True when the narrative names the rank-1 option's action, e.g. 'dispatch a second bus' (any capitalisation)."""
    return label.split(" —")[0].strip().lower() in (text or "").lower()


def llm_narrative(case, attempt=1):
    A = case["analysis"]
    rk = A["ranking"]
    facts = {"route": f"{case['raw']['route_id']} {case['raw']['route_name']}", "triggers": [t["trigger"] for t in A["hard_triggers"]],
             "R": A["scores"]["R"], "V": A["scores"]["V"], "A": A["scores"]["A"], "P": A["scores"]["P"], "tier": A["tier"],
             "confidence": A["confidence"]["C"], "missing_fields": A["missing"],
             "options_ranked": [{"rank": r["rank"], "option": r["label"], "Si": r["Si"], "DR": r["DR"], "EB": r["EB"], "AP": r["AP"],
                                 "OF": r["OF"], "CE": r["CE"]} for r in rk],
             "essential_destinations": case["raw"]["access"]["essential_names"]}
    system = ("You are the Explanation Agent for a transit operations prototype. Explain deterministic results to a human reviewer in plain language. "
              "Rules: use ONLY the numbers in the facts; never invent or change numbers; describe the alternatives BEFORE the recommendation; "
              "the recommendation is always the rank-1 option; say the human reviewer makes the final decision; 5 sentences maximum; no headings.")
    if not rk:
        return template_narrative(case)
    txt = agent_llm_generate("Facts (JSON):\n" + json.dumps(facts, indent=1), system)
    if not narrative_names_rank1(txt, rk[0]["label"]):
        raise AgentFailure("Narrative did not name the rank-1 option; discarded by guard")
    return txt

# ====================================================================================================
# notebook cell 36: E3 · Incident history, Outcome Evaluator metrics, Insight Agent, policy-owner approval, backtest and versioning
# ====================================================================================================
#@title E3 · Incident history, Outcome Evaluator metrics, Insight Agent, policy-owner approval, backtest and versioning
def random_spec(rng):
    rid = rng.choice(["R1", "R2", "R3", "R4", "R5"])
    n = len(NET["routes"][rid]["stops"])
    i0 = rng.randint(0, n - 5)
    canc = rng.random() < 0.12
    ev = rng.choice(["none", "none", "none", "weather", "obstruction", "road_closure", "mechanical"])
    sev = "none" if ev == "none" else rng.choice(["minor", "major", "major", "emergency" if ev == "road_closure" else "major"])
    load = round(rng.uniform(0.4, 1.12), 2)
    return dict(route_id=rid, stop_range=(i0, i0 + rng.randint(3, 5)), time_period=rng.choice(["peak", "midday", "off_peak", "shift_change"]),
                delay_min=round(rng.uniform(2, 28), 1), headway_ratio=round(rng.uniform(1.0, 2.8), 2), cancelled=canc,
                cancellation_risk=round(rng.uniform(0.05, 0.7), 2), boardings=rng.randint(40, 220), load_factor=load,
                passups=rng.randint(1, 25) if load >= 1 else rng.randint(0, 3), event_type=ev, event_severity=sev,
                event_verified=ev != "none", sources=rng.choice([["vehicle_apc"], ["vehicle_apc", "official_alert"],
                                                                 ["vehicle_apc", "trusted_external"], ["vehicle_apc", "official_alert", "trusted_external"]]),
                feed_age_s=rng.randint(10, 120), detour_extra_min=rng.randint(4, 10))


def seed_history(store, n=30):
    """Simulated past cases: disruptions, responses and recovery outcomes (incident history input)."""
    rng = stable_rng("history")
    policy = store.PM.current
    base = utcnow() - timedelta(days=90)
    for i in range(n):
        spec = random_spec(rng)
        raw = make_raw(spec, f"Historical incident {i + 1:02d}")
        when = base + timedelta(days=i * 3, hours=rng.randint(6, 20))
        raw["generated_at"] = iso(when)
        A = run_analysis(raw, policy)
        case = {"case_id": f"HIST-{i + 1:03d}", "created_at": iso(when), "scenario_name": raw["scenario_name"], "policy_version": policy["version"],
                "raw": raw, "analysis": A, "constraints": {}, "operator_resolution": None, "decisions": [], "implementation": [],
                "outcome": None, "audit": [], "history": True, "revisions": 0, "approved_option_id": None}
        if A["gate"] == "SIGNIFICANT" and A["ranking"]:
            top = A["ranking"][0]["option_id"]
            rev = rng.choice(["Supervisor Lee", "Supervisor Ortiz", "Planner Chen"])
            case["decisions"].append({"stage": "first review", "reviewer": rev, "decision": "Approve", "option_id": top, "recommended_option_id": top,
                                      "override": False, "reason": "historical approval", "response_s": round(rng.uniform(20, 400), 1),
                                      "ts": iso(when + timedelta(minutes=4)), "tier": A["tier"]})
            case["approved_option_id"] = top
            case["status"] = "EVALUATED"
            pr = store.predicted_for(case)
            hard = bool(A["hard_triggers"])
            fa = (rng.random() < 0.6) if (not hard and A["scores"]["R"] < 0.62) else (rng.random() < 0.08)
            k = 0.72 if pr["type"] == "dispatch_second_bus" else 0.95
            eq = 0.78 if A["scores"]["P"] >= 0.5 else 0.95
            obs = {"observed_delay_reduction_min": round(pr["delay_reduction_min"] * k * rng.uniform(0.9, 1.1), 1),
                   "observed_access_protection": round(pr["access_protection"] * rng.uniform(0.85, 1.0), 2),
                   "observed_vulnerable_riders_served": round(pr["vulnerable_riders_helped"] * eq * rng.uniform(0.92, 1.05)),
                   "actual_cost_usd": round(pr["cost_usd"] * (1.28 if pr["type"] == "dispatch_second_bus" else 1.03)),
                   "rider_reported_impact": rng.randint(1, 4), "false_alarm": fa}
            case["outcome"] = outcome_metrics(pr, obs)
            case["recovery_min"] = round(max(5, (A["fields"].get("delay_min") or 10) * rng.uniform(1.2, 2.5)))
        else:
            case["status"] = "SAFE_DEFAULT" if A["gate"] in ("BELOW_THRESHOLD", "SAFE_MODE") else "MONITORING"
            real = rng.random() < 0.15
            case["outcome"] = {"evaluated_at": iso(when), "action": "rider information only", "action_type": "none",
                               "false_alarm": not real, "missed_real_disruption": real}
            case["recovery_min"] = round(rng.uniform(5, 25))
        store.history.append(case)


def incident_history_df(store):
    rows = []
    for c in store.history:
        A = c["analysis"]
        rows.append({"case_id": c["case_id"], "date": c["created_at"][:10], "route": c["raw"]["route_id"],
                     "event": c["raw"]["event_type"], "delay_min": A["fields"].get("delay_min") if A.get("fields") else None,
                     "significant": A["significant"], "response": c.get("approved_option_id") or "rider information only",
                     "recovery_min": c.get("recovery_min"), "false_alarm": (c.get("outcome") or {}).get("false_alarm")})
    return pd.DataFrame(rows)


def insight_agent(store):
    """Insight Agent: proposes threshold, bound or weight changes from outcome results. It cannot apply any change."""
    pol = store.PM.current
    ev = [c for c in store.all_cases() if c.get("outcome")]
    sig = [c for c in ev if c["analysis"] and c["analysis"].get("significant") and "pred_delay_reduction_min" in c["outcome"]]
    props = []

    rejected = {(c["path"], json.dumps(c["new"])) for p in store.proposal_log if p["status"] == "rejected" for c in p["changes"]}

    def add(title, changes, rationale, evidence):
        if all((c["path"], json.dumps(c["new"])) in rejected for c in changes):   # a rejected change is not re-proposed
            store.log(None, "Insight Agent", "agent", "Insight agent", "Skipped previously rejected proposal", title)
            return
        props.append({"id": f"PR-{len(store.proposal_log) + len(props) + 1:03d}", "title": title, "changes": changes,
                      "rationale": rationale, "evidence": evidence, "status": "proposed", "proposed_at": iso(utcnow())})

    if sig:
        nohard = [c for c in sig if not c["analysis"]["hard_triggers"]]
        if len(nohard) >= 3:
            fa = sum(c["outcome"]["false_alarm"] for c in nohard) / len(nohard)
            if fa > 0.25 and pol["significance_R"] < 0.70:
                add("Raise R significance threshold", [{"path": "significance_R", "old": pol["significance_R"], "new": round(pol["significance_R"] + 0.05, 2)}],
                    "Too many R-only significant cases turned out to be false alarms.", f"false-alarm rate {fa:.0%} over {len(nohard)} R-only cases (limit 25%)")
        for t in ["dispatch_second_bus", "reroute", "protect_connection"]:
            cs = [c for c in sig if c["outcome"]["action_type"] == t and c["outcome"]["pred_delay_reduction_min"] > 0]
            if len(cs) >= 3:
                ratio = sum(c["outcome"]["obs_delay_reduction_min"] for c in cs) / sum(c["outcome"]["pred_delay_reduction_min"] for c in cs)
                cur = pol["lookups"]["option_certainty"][t]
                if ratio < 0.85:
                    add(f"Lower option certainty for {t}", [{"path": f"lookups.option_certainty.{t}", "old": cur, "new": round(max(0.5, cur - 0.05), 2)}],
                        "Observed delay reduction is consistently below prediction for this action.", f"observed/predicted = {ratio:.2f} over {len(cs)} cases")
            cc = [c for c in cs if c["outcome"].get("cost_ratio")]
            if len(cc) >= 3:
                cr = statistics.mean(c["outcome"]["cost_ratio"] for c in cc)
                if cr > 1.2:
                    U = pol["bounds"]["cost_usd"][1]
                    add("Tighten cost normalization bound", [{"path": "bounds.cost_usd", "old": list(pol["bounds"]["cost_usd"]), "new": [0, int(U * 0.85)]}],
                        f"Actual costs for {t} exceed predictions, so cost efficiency is overstated.", f"mean actual/predicted cost {cr:.2f} over {len(cc)} cases")
        hp = [c for c in sig if c["analysis"]["scores"]["P"] >= 0.5 and c["outcome"]["pred_equity_riders"] > 0]
        if len(hp) >= 3:
            er = sum(c["outcome"]["obs_equity_riders"] for c in hp) / sum(c["outcome"]["pred_equity_riders"] for c in hp)
            sh = pol["scenario_shift"]["EB"]
            if er < 0.85 and sh < 0.15:
                ns = round(sh + 0.02, 2)
                add("Increase priority shift toward equity and access",
                    [{"path": "scenario_shift.EB", "old": sh, "new": ns}, {"path": "scenario_shift.AP", "old": pol["scenario_shift"]["AP"], "new": ns},
                     {"path": "scenario_shift.OF", "old": pol["scenario_shift"]["OF"], "new": -ns}, {"path": "scenario_shift.CE", "old": pol["scenario_shift"]["CE"], "new": -ns}],
                    "High-priority cases deliver less equity benefit than predicted; weight equity/access more as P rises (weights still sum to 1).",
                    f"observed/predicted vulnerable riders served = {er:.2f} over {len(hp)} cases with P ≥ 0.5")
    mon = [c for c in ev if c["outcome"].get("missed_real_disruption") is not None]
    if len(mon) >= 5:
        miss = sum(c["outcome"]["missed_real_disruption"] for c in mon) / len(mon)
        if miss > 0.30 and pol["confidence_threshold"] > 0.6:
            add("Lower confidence threshold", [{"path": "confidence_threshold", "old": pol["confidence_threshold"], "new": round(pol["confidence_threshold"] - 0.05, 2)}],
                "Monitored cases later proved to be real disruptions too often.", f"missed-disruption rate {miss:.0%} over {len(mon)} cases")
    store.proposals = props
    store.log(None, "Insight Agent", "agent", "Insight agent", f"{len(props)} proposal(s) generated",
              "; ".join(p["title"] for p in props) or "No change proposed — current rules kept")
    store.insight_narrative = insight_narrative_text(props, len(ev))
    return props


def insight_narrative_text(props, n):
    if not props:
        return f"Reviewed {n} evaluated cases. No change is warranted; current rules are kept."
    base = f"Reviewed {n} evaluated cases and drafted {len(props)} proposal(s). None is applied until the policy owner approves it and it passes a backtest."
    if LLM_READY:
        try:
            return agent_llm_generate(json.dumps([{k: p[k] for k in ("title", "changes", "evidence")} for p in props]),
                                   "Summarize these proposed transit policy changes for a policy owner in 3 sentences. Use only the given numbers. "
                                   "State clearly that nothing has been applied.", max_tokens=200, timeout=90)
        except AgentFailure:
            pass
    return base


def apply_changes(policy, changes):
    newp = copy.deepcopy(policy)
    for ch in changes:
        set_path(newp, ch["path"], copy.deepcopy(ch["new"]))
    return newp


def run_backtest(store, new_policy):
    """Re-run the deterministic pipeline on past cases under current vs proposed rules."""
    cur = store.PM.current
    rows = []
    for c in store.all_cases():
        a = run_analysis(c["raw"], cur, c.get("operator_resolution"), c.get("constraints"))
        b = run_analysis(c["raw"], new_policy, c.get("operator_resolution"), c.get("constraints"))
        out = c.get("outcome") or {}
        rows.append({"case_id": c["case_id"], "gate_current": a["gate"], "gate_proposed": b["gate"],
                     "top_current": a["ranking"][0]["option_id"] if a["ranking"] else "—",
                     "top_proposed": b["ranking"][0]["option_id"] if b["ranking"] else "—",
                     "Si_current": a["ranking"][0]["Si"] if a["ranking"] else None, "Si_proposed": b["ranking"][0]["Si"] if b["ranking"] else None,
                     "known_false_alarm": out.get("false_alarm")})
    df = pd.DataFrame(rows)
    sig_change = int((df.gate_current != df.gate_proposed).sum())
    top_change = int((df.top_current != df.top_proposed).sum())
    fa_avoided = int(((df.gate_current == "SIGNIFICANT") & (df.gate_proposed != "SIGNIFICANT") & (df.known_false_alarm == True)).sum())
    real_lost = int(((df.gate_current == "SIGNIFICANT") & (df.gate_proposed != "SIGNIFICANT") & (df.known_false_alarm == False)).sum())
    summary = (f"Backtest on {len(df)} past cases: {sig_change} gate change(s), {top_change} top-option change(s), "
               f"{fa_avoided} known false alarm(s) avoided, {real_lost} real disruption(s) that would no longer be escalated.")
    store.backtest = {"df": df, "summary": summary}
    return df, summary


def decide_proposal(store, pid, owner, decision, reason):
    p = next((p for p in store.proposals if p["id"] == pid), None)
    if p is None:
        raise ValueError("Pick a proposal (run the Insight Agent first).")
    if not owner.strip() or len(reason.strip()) < 10:
        raise ValueError("Policy owner name and a reason (≥ 10 characters) are required.")
    gp, gn = governance_check(owner, decision, proposal=p)                # INTEGRATION (Sections C + D)
    for n_ in gn:
        store.log(None, "Governance gate", "rule", "Policy owner gate (C+D)", "note", n_)
    if gp:
        store.log(None, owner, "policy owner", "Policy owner approval", "Submission blocked", "; ".join(gp))
        raise ValueError("Cannot submit: " + "; ".join(gp))
    if decision == "Approve":
        newp = apply_changes(store.PM.current, p["changes"])
        errs = validate_policy(newp)
        if errs:
            raise ValueError("Proposal breaks policy invariants: " + "; ".join(errs))
        _, summary = run_backtest(store, newp)
        v = store.PM.release(newp, owner, reason, p["changes"], summary)
        p.update(status=f"approved → v{v}", decided_by=owner, reason=reason, backtest=summary)
        store.log(None, owner, "policy owner", "Policy owner approval", f"APPROVED {pid} → new rule version {v}", summary)
        store.last_message = f"{pid} approved, backtested and released as rules v{v} (rollback available)."
    else:
        p.update(status="rejected", decided_by=owner, reason=reason)
        store.log(None, owner, "policy owner", "Policy owner approval", f"REJECTED {pid}", f"Current rules kept. Reason: {reason}")
        store.last_message = f"{pid} rejected; reason returned to the Insight Agent and current rules kept."
    store.proposal_log.append(copy.deepcopy(p))
    store.proposals = [x for x in store.proposals if x["id"] != pid]


PM = PolicyManager(DEFAULT_POLICY)
APP = CaseStore(PM)
seed_history(APP)
print(f"Seeded {len(APP.history)} historical cases; rules v{PM.current['version']} active.")

# ====================================================================================================
# notebook cell 39: B · 1. Setup and policy configuration (rules come from the shared PolicyManager)
# ====================================================================================================
#@title B · 1. Setup and policy configuration (rules come from the shared PolicyManager)
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable

import pandas as pd
import ipywidgets as widgets
from _ipython_shim import display, HTML

class PolicyConfigView(dict):
    """Read-only view of the shared versioned policy (APP.PM). Rules change only through Policy Owner Approval."""
    KEYS = {"rule_version": lambda p: "v" + p["version"], "confidence_threshold": lambda p: p["confidence_threshold"],
            "significance_threshold_R": lambda p: p["significance_R"], "max_reverify_attempts": lambda p: p["max_operator_checks"],
            "max_revisions": lambda p: p["max_revisions"], "min_reason_chars": lambda p: p["review"]["min_reason_chars"],
            "review_deadline_minutes": lambda p: p["deadlines_min"], "high_impact_actions": lambda p: p["high_impact"]["actions"],
            "deadline_resets_on_revision": lambda p: p.get("deadline_resets_on_revision", False),
            "delay_hard_trigger_min": lambda p: p["hard_trigger"]["delay_min"],
            "unboardable_load_factor": lambda p: p["hard_trigger"]["unboardable_load_factor"]}

    def __getitem__(self, k):
        return self.KEYS[k](APP.PM.current)

    def get(self, k, default=None):
        return self[k] if k in self.KEYS else default

    def __contains__(self, k):
        return k in self.KEYS

    def __setitem__(self, k, v):
        raise TypeError("CONFIG is read-only: change rules through Policy Owner Approval (APP.PM.release).")

    def __repr__(self):
        return repr({k: self[k] for k in self.KEYS})


CONFIG = PolicyConfigView()
RULE_VERSION_HISTORY = [{"version": CONFIG["rule_version"], "changed": "initial", "rollback_to": None}]


def now() -> datetime:
    return datetime.now(timezone.utc).astimezone(LOCAL_TZ)


def stamp(dt: Optional[datetime] = None) -> str:
    return (dt or now()).isoformat(timespec="seconds")


print("Setup complete. Rule version:", CONFIG["rule_version"])

# ====================================================================================================
# notebook cell 41: B · 2. Data structures (the Shared Case State contracts)
# ====================================================================================================
#@title B · 2. Data structures (the Shared Case State contracts)
@dataclass
class LoadSignal:
    """Vision Agent output. Counts only: no images or identities are stored."""
    stop_id: str
    route: str
    timestamp: str
    passenger_count: int
    capacity: int
    load_factor: float
    pass_ups: int
    crowding_level: str          # "normal" | "busy" | "crowded" | "unboardable"
    source: str = "vision_agent"


@dataclass
class VerificationResult:
    """Data Verification output."""
    status: str                  # "ready" | "feed_outage" | "agent_error" | "conflict"
    issues: list = field(default_factory=list)


@dataclass
class ImpactScores:
    """Deterministic Impact Analyzer output. All components normalized 0 to 1."""
    R: float                     # reliability impact
    V: float                     # rider vulnerability
    A: float                     # access impact
    P: float                     # priority
    hard_trigger: Optional[str] = None
    severity: str = "moderate"   # "severe" | "significant" | "moderate"


@dataclass
class Option:
    """Planner Copilot output. Options carry NO scores: the planner cannot score."""
    option_id: str
    action: str                  # "do_nothing" | "rider_alert" | "protect_connection" | "reroute" | "dispatch_second_bus"
    description: str
    resources_needed: dict
    constraints_checked: list


@dataclass
class ScoredOption:
    """Scenario Evaluator output: fixed published formula, same inputs give the same rank."""
    option: Option
    DR: float
    EB: float
    AP: float
    OF: float
    CE: float
    Ci: float
    score: float
    rank: int = 0


@dataclass
class Brief:
    """Explanation Agent output: what the human reviewer sees."""
    case_id: str
    summary: str
    options_table: list          # one dict per option, alternatives shown before the recommendation
    recommendation: str
    uncertainty_notes: list
    sources: list


@dataclass
class ReviewDecision:
    """Human Review / Second Reviewer output."""
    reviewer_role: str
    reviewer_name: str
    decision: str               # "approve" | "revise" | "reject" | "timeout"
    selected_option_id: Optional[str]
    reason: str
    new_constraints: str
    checklist_confirmed: bool
    seconds_to_decide: float
    timestamp: str


@dataclass
class ActionResult:
    """Implement Action output, entered by operations staff."""
    option_id: str
    status: str                  # "completed" | "partial" | "failed"
    rider_alert_sent: bool
    notes: str
    timestamp: str


@dataclass
class OutcomeReport:
    """Outcome Evaluator output. Never changes policy weights by itself."""
    case_id: str
    path: str                    # "action" | "safe_default" | "monitoring"
    predicted: dict
    observed: dict
    metrics: dict
    false_alarm: Optional[bool]


@dataclass
class PolicyProposal:
    """Insight Agent output. A proposal only: it cannot be applied without approval.
    parameter is a path to the rule it would change, e.g. "CONFIG.significance_threshold_R"
    or "COMPONENTS.dispatch_second_bus.DR"."""
    proposal_id: str
    parameter: str
    current_value: object
    proposed_value: object
    rationale: str
    evidence_cases: list


@dataclass
class PolicyDecision:
    """Policy Owner Approval output."""
    proposal_id: str
    decision: str                # "approve" | "reject"
    reason: str
    timestamp: str


@dataclass
class BacktestResult:
    """Backtest output: run before an approved rule change is released."""
    proposal_id: str
    passed: bool
    summary: str


@dataclass
class Case:
    """Shared Case State: the single record every stage reads and writes."""
    case_id: str
    route: str
    disruption_type: str
    description: str
    opened_at: str
    rule_version: str
    parent_case_id: Optional[str] = None
    status: str = "open"
    severity: str = "moderate"
    load_signals: list = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    reverify_attempts: int = 0
    confidence: Optional[float] = None
    impact: Optional[ImpactScores] = None
    options: list = field(default_factory=list)
    ranked_options: list = field(default_factory=list)
    brief: Optional[Brief] = None
    revision_count: int = 0
    constraints: list = field(default_factory=list)
    review_deadline: Optional[datetime] = None
    decisions: list = field(default_factory=list)
    selected_option_id: Optional[str] = None
    action_result: Optional[ActionResult] = None
    outcome: Optional[OutcomeReport] = None
    audit: list = field(default_factory=list)

    def log(self, stage: str, actor: str, actor_type: str, output: str, note: str = ""):
        self.audit.append({
            "time": stamp(), "case_id": self.case_id, "stage": stage, "actor": actor,
            "actor_type": actor_type, "output": output, "note": note,
            "rule_version": self.rule_version,   # the case's own rule version, not the current global one
        })

    def option(self, option_id: str) -> Optional[ScoredOption]:
        return next((s for s in self.ranked_options if s.option.option_id == option_id), None)


ALL_CASES: list = []          # every case, for the audit trail
OUTCOME_HISTORY: list = []    # every OutcomeReport, read by the Insight Agent
REJECTED_PROPOSALS: dict = {} # (parameter, proposed value) -> policy owner's reason; never resubmitted
# These persist across reruns of the workflow cell, like a real system would. Rerun this cell to clear them.
print("Data structures defined.")

# ====================================================================================================
# notebook cell 43: B · Sample scenario (Route 12 cancellation, Main St crowding, last rail connection)
# ====================================================================================================
#@title B · Sample scenario (Route 12 cancellation, Main St crowding, last rail connection)
SAMPLE_INPUTS = {
    "service_feeds": {
        "route": "12", "trip_id": "12-1740", "status": "cancelled",
        "scheduled_departure": "17:40", "next_trip": "17:55", "headway_min": 15, "delay_min": 0,
    },
    "connecting_services": {
        "stop_id": "MAIN_ST", "rail_line": "Northeast Line", "last_departure": "18:05", "transfer_window_min": 8,
    },
    "incident_history": [
        {"route": "12", "type": "cancellation", "avg_recovery_min": 22, "past_response": "rider_alert"},
    ],
    "community_access": {
        "area": "Downtown East (area level aggregate)", "transit_dependence_pct": 62,
        "disability_share_pct": 14, "low_income_share_pct": 41,
        "essential_destinations": ["Regional Hospital", "Community College"],
    },
    "operating_resources": {
        "spare_buses": 1, "spare_operators": 1, "cost_per_bus_hour": 145, "deadhead_min": 12,
    },
    "event_signals": [
        {"type": "official_alert", "source": "agency", "text": "Route 12 17:40 trip cancelled (operator shortage)"},
    ],
    "rider_feedback": [
        {"channel": "app", "text": "Bus never came at Main St", "corroborated": False},
    ],
}

# Anonymized counts from station and onboard sensors (no images, no identities)
RAW_SENSOR_FEED = [
    {"stop_id": "MAIN_ST", "route": "12", "timestamp": "17:46", "person_count": 58, "capacity": 45, "pass_ups": 0},
    {"stop_id": "ELM_AVE", "route": "12", "timestamp": "17:44", "person_count": 41, "capacity": 45, "pass_ups": 6},
]

# Optional demo controls: force a path for a presentation or test. None = use the normal result.
# Used only by the Section B runner (B · 8); the orchestrator and dashboard never apply them.
DEMO_OVERRIDES = {
    "verification_status": None,   # "feed_outage" | "agent_error" | "conflict" on the first check only
    "exception_resolved": None,    # True | False
    "confidence": None,            # e.g. 0.55 to test the confidence gate
    "R": None,                     # e.g. 0.30 to test the significance rule
    "remove_hard_trigger": False,  # True to test a case with no hard trigger
}
print("Sample scenario loaded.")

# ====================================================================================================
# notebook cell 45: B · 3. Validation and metrics functions (deterministic; numbers from the E2 engine)
# ====================================================================================================
#@title B · 3. Validation and metrics functions (deterministic; numbers from the E2 engine)
# Every function keeps the typed contract from B · 2 (inputs → VerificationResult, ImpactScores, ScoredOption, ...).
# The numbers come from the E2 engine, so Section B, the orchestrator and the dashboard give identical results.
_B_ENGINE = {}                 # case_id -> {"raw", "analysis", "resolution", "constraints"}
_B_LAST = {}                   # last verification run (inputs, signals, raw, analysis)
_B_RESOLVED_CONFLICTS = set()  # inputs whose conflict was resolved by applying source precedence
B_ACTION_FROM_ENGINE = {"do_nothing": "do_nothing", "rider_alerts": "rider_alert", "protect_connection": "protect_connection",
                        "reroute": "reroute", "dispatch_second_bus": "dispatch_second_bus"}
RULE_PATH_ALIASES = {"CONFIG.significance_threshold_R": "significance_R", "CONFIG.confidence_threshold": "confidence_threshold"}


def _inputs_key(inputs):
    return zlib.crc32(json.dumps(inputs, sort_keys=True, default=str).encode())


def _b_policy(case=None):
    return APP.PM.get(case.rule_version.lstrip("v")) if case is not None else APP.PM.current


def workflow_inputs_to_raw(inputs, load_signals, name="Section B sample · Route 12 cancellation"):
    """Section B's SAMPLE_INPUTS + Vision Agent load signals → shared raw bundle."""
    svc = inputs["service_feeds"]
    conn = inputs.get("connecting_services", {})
    comm = inputs.get("community_access", {})
    res = inputs.get("operating_resources", {})
    ext = str(svc["route"])
    rid = EXTERNAL_ROUTE_MAP.get(ext, "R3")
    stop_range = (2, 6)
    cancelled = svc.get("status") == "cancelled"
    headway = svc.get("headway_min") or NET["routes"][rid]["headway_min"]
    delay = svc.get("delay_min", 0) or 0
    gap = 2.0 if cancelled else round((headway + delay) / headway, 2)
    sigs = load_signals or []
    boardings = sum(s.passenger_count for s in sigs)
    load = max([s.load_factor for s in sigs], default=0.0)
    passups = sum(s.pass_ups for s in sigs)
    official = [e for e in inputs.get("event_signals", []) if e.get("type") == "official_alert"]
    feedback = inputs.get("rider_feedback", [])
    sources = ["vehicle_apc"] + (["official_alert"] if official else []) + (["public_report"] if feedback else [])
    cause = classify_cause(" ".join(e.get("text", "") for e in official))
    dep = nearest_depot_id(rid, stop_range)
    spec = dict(route_id=rid, stop_range=stop_range, time_period="peak", delay_min=delay, headway_ratio=gap,
                cancelled=cancelled, cancellation_risk=1.0 if cancelled else 0.2, boardings=boardings, load_factor=load,
                passups=passups, event_type=cause, event_severity="none" if cause == "none" else SEVERITY_BY_CAUSE.get(cause, "minor"),
                event_verified=bool(official), sources=sources, public_corroborated=any(f.get("corroborated") for f in feedback),
                last_service=bool(conn.get("last_departure")),
                depot_overrides={dep: {"spare_buses": res.get("spare_buses", 0), "operators": res.get("spare_operators", 0),
                                       "deadhead_min": res.get("deadhead_min")}})
    raw = make_raw(spec, name)
    raw["external_route_id"] = ext
    raw["route_name"] = f"Route {ext} (drawn on synthetic {rid})"
    raw["community"].update(transit_dependence=comm.get("transit_dependence_pct", 0) / 100 if comm.get("transit_dependence_pct") is not None else None,
                            income_disadvantage=comm.get("low_income_share_pct", 0) / 100 if comm.get("low_income_share_pct") is not None else None,
                            disability=comm.get("disability_share_pct", 0) / 100 if comm.get("disability_share_pct") is not None else None,
                            areas=[comm.get("area", "area aggregate")])
    names = comm.get("essential_destinations", [])
    types = [essential_type_of(n) for n in names]
    sev = DEFAULT_POLICY["lookups"]["essential_severity"]
    raw["access"].update(essential_type=max(types, key=lambda t: sev[t]) if types else "none", essential_names=names)
    raw["connections"].update(external_connecting_route=conn.get("rail_line"), transfer_window_min=conn.get("transfer_window_min", 4))
    if res.get("cost_per_bus_hour"):
        raw["resources"]["bus_hour_cost_usd"] = res["cost_per_bus_hour"]
    if inputs.get("incident_history"):
        h = inputs["incident_history"][0]
        raw["incident_history"] = {"past_type": h.get("type"), "average_recovery_min": h.get("avg_recovery_min"),
                                   "previous_response": h.get("past_response")}
    raw["raw_sensor_feed"] = [asdict(s) for s in sigs]
    raw["handoff_from"] = "Section B — transit_agents_workflow.ipynb"
    return raw


def _b_analyze(inputs, load_signals, constraints=None, policy=None):
    raw = workflow_inputs_to_raw(inputs, load_signals)
    res = "accept_precedence" if _inputs_key(inputs) in _B_RESOLVED_CONFLICTS else None
    return raw, run_analysis(raw, policy or APP.PM.current, res, constraints), res


# ---------------- data verification and exception paths ----------------
def verify_data(inputs: dict, load_signals: list) -> VerificationResult:
    """Schema, timestamps, refresh, source identity, completeness and agreement → "ready" or the exception path to take."""
    raw, A, _ = _b_analyze(inputs, load_signals)
    _B_LAST.update(inputs=inputs, signals=load_signals, raw=raw, analysis=A)
    issues = [f"{e['path']}: {e['detail']} → {e['outcome']}" for e in A["exceptions"]]   # incl. uncorroborated public reports
    if A["gate"] == "SAFE_MODE":
        status = "agent_error"
    elif A["conflict"] == "pending":
        status = "conflict"
    elif A["outage"] == "no_backup":
        status = "feed_outage"
    else:
        status = "ready"
    return VerificationResult(status=status, issues=issues)


def handle_feed_outage(case: Case, verification: VerificationResult) -> bool:
    """Freeze the last verified snapshot, request a backup, alert the operator. Returns True if resolved."""
    case.log("Feed outage path", "Exception handler", "Rule / code", "snapshot frozen; backup requested; operator alerted",
             "No backup source available → monitoring only")
    return False


def handle_agent_error(case: Case, verification: VerificationResult) -> bool:
    """Retry once with the same inputs; if it fails again use deterministic safe mode. Returns True if resolved."""
    case.log("Agent error path", "Exception handler", "Rule / code", "retried once with same inputs; failed again",
             "Technical owner and planner alerted; evidence preserved")
    return False


def handle_conflict(case: Case, verification: VerificationResult) -> bool:
    """Quarantine the conflict and apply source precedence (vehicle > official > trusted external > public)."""
    _B_RESOLVED_CONFLICTS.add(_inputs_key(_B_LAST["inputs"]))
    case.log("Conflicting data path", "Operator review", "Person", "source precedence applied",
             "Notebook path assumes the operator accepted precedence; the dashboard (tab 1) records an explicit operator decision")
    return True


# ---------------- confidence, hard triggers, impact and scenario ranking ----------------
def calculate_confidence(inputs: dict, load_signals: list, verification: VerificationResult) -> float:
    """C = 0.40Q + 0.25F + 0.20G + 0.15M (source quality, freshness, agreement, completeness)."""
    _, A, _ = _b_analyze(inputs, load_signals)
    return A["confidence"]["C"]


def detect_official_hard_trigger(inputs: dict) -> Optional[str]:
    """Name of the first hard trigger an OFFICIAL source verifies (lets the case bypass the C gate), else None."""
    _, A, _ = _b_analyze(inputs, _B_LAST.get("signals", []))
    return next((t["trigger"] for t in A["hard_triggers"] if t["official"]), None)


def analyze_impact(case: Case, inputs: dict) -> ImpactScores:
    """R = 0.30D + 0.25H + 0.25K + 0.20E · V = 0.35N + 0.25I + 0.20B + 0.20T · A = 0.35S + 0.30L + 0.20X + 0.15U ·
    P = 0.40R + 0.35V + 0.25A, plus hard trigger and severity tier."""
    raw, A, res = _b_analyze(inputs, case.load_signals, policy=_b_policy(case))
    _B_ENGINE[case.case_id] = {"raw": raw, "analysis": A, "resolution": res, "constraints": {}}
    s = A["scores"]
    trig = A["hard_triggers"][0]["trigger"] if A["hard_triggers"] else None
    return ImpactScores(R=s["R"], V=s["V"], A=s["A"], P=s["P"], hard_trigger=trig, severity=A["tier"] or "moderate")


def evaluate_scenarios(case: Case, options: list) -> list:
    """Si = Ci × [0.30DR + (0.15+0.10P)EB + (0.15+0.10P)AP + (0.25−0.10P)OF + (0.15−0.10P)CE]; tie break AP, OF, cost."""
    A = _B_ENGINE[case.case_id]["analysis"]
    by_id = {r["option_id"]: r for r in A["ranking"]}
    scored = []
    for o in options:
        r = by_id[o.option_id.split(f"{case.case_id}-", 1)[1]]
        scored.append(ScoredOption(option=o, DR=r["DR"], EB=r["EB"], AP=r["AP"], OF=r["OF"], CE=r["CE"], Ci=r["Ci"],
                                   score=r["Si"], rank=r["rank"]))
    scored.sort(key=lambda x: x.rank)
    for i, s in enumerate(scored, 1):
        s.rank = i
    return scored


# ---------------- outcome evaluation ----------------
def workflow_option_as_engine_option(case, option_id):
    eng = _B_ENGINE.get(case.case_id)
    if not eng or not option_id:
        return None
    eid = option_id.split(f"{case.case_id}-", 1)[-1]
    return next((o for o in eng["analysis"]["options"] if o["option_id"] == eid), None)


def evaluate_outcome(case: Case, action_result: Optional[ActionResult]) -> OutcomeReport:
    """Predicted vs observed delay, access, equity, cost and false alarm (separate from the recommender)."""
    o = workflow_option_as_engine_option(case, case.selected_option_id) if action_result else None
    if o is None:
        return OutcomeReport(case_id=case.case_id, path=case.status, predicted={}, observed={},
                             metrics={"note": "no action taken; false-alarm status is set when the case is closed"}, false_alarm=None)
    pr = {"option_id": o["option_id"], "type": o["type"], "delay_reduction_min": o["pred_delay_reduction_min"],
          "access_protection": o["pred_access_protection"], "vulnerable_riders_helped": o["pred_vulnerable_riders_helped"],
          "cost_usd": o["cost_usd"]}
    m = outcome_metrics(pr, simulated_observation(pr, case.case_id))
    gap = round((m["pred_equity_riders"] - m["obs_equity_riders"]) / m["pred_equity_riders"], 2) if m["pred_equity_riders"] else 0.0
    return OutcomeReport(case_id=case.case_id, path="action",
                         predicted={"delay_reduction_min": m["pred_delay_reduction_min"], "vulnerable_riders": m["pred_equity_riders"],
                                    "access": m["pred_access"], "cost_usd": m["pred_cost_usd"]},
                         observed={"delay_reduction_min": m["obs_delay_reduction_min"], "vulnerable_riders": m["obs_equity_riders"],
                                   "access": m["obs_access"], "cost_usd": m["actual_cost_usd"], "rider_reported_impact": m["rider_reported_impact"]},
                         metrics={"delay_reduction_error_min": m["delay_error_min"], "equity_benefit_gap": gap,
                                  "cost_usd": m["actual_cost_usd"], "cost_ratio": m["cost_ratio"], "action_type": o["type"]},
                         false_alarm=m["false_alarm"])


# ---------------- versioned rule changes (only reached after Policy Owner approval + backtest) ----------------
def get_rule(path: str):
    return copy.deepcopy(get_path(APP.PM.current, RULE_PATH_ALIASES.get(path, path)))


def _proposal_change(proposal):
    path = RULE_PATH_ALIASES.get(proposal.parameter, proposal.parameter)
    return {"path": path, "old": get_rule(path), "new": proposal.proposed_value}


def backtest_rule_change(proposal: PolicyProposal, past_cases: list) -> BacktestResult:
    """Re-run every past case under current vs proposed rules; fails if a real disruption would stop being escalated."""
    newp = apply_changes(APP.PM.current, [_proposal_change(proposal)])
    errs = validate_policy(newp)
    if errs:
        return BacktestResult(proposal_id=proposal.proposal_id, passed=False, summary="; ".join(errs))
    df, summary = run_backtest(APP, newp)
    lost = int(((df.gate_current == "SIGNIFICANT") & (df.gate_proposed != "SIGNIFICANT") & (df.known_false_alarm == False)).sum())
    return BacktestResult(proposal_id=proposal.proposal_id, passed=lost == 0,
                          summary=summary + ("" if lost == 0 else " → FAILED: real disruptions would no longer be escalated."))


def apply_rule_change(proposal: PolicyProposal) -> str:
    """Release the approved change as a new immutable version in the shared PolicyManager. Returns the version."""
    change = _proposal_change(proposal)
    old_version = CONFIG["rule_version"]
    v = APP.PM.release(apply_changes(APP.PM.current, [change]), "Policy owner (Section B form)", proposal.rationale, [change])
    RULE_VERSION_HISTORY.append({"version": "v" + v, "parameter": change["path"], "old_value": change["old"],
                                 "new_value": change["new"], "rollback_to": old_version, "proposal_id": proposal.proposal_id})
    APP.log(None, "Policy owner (Section B form)", "policy owner", "Rule release", f"v{v}", f"{change['path']}: {change['old']} → {change['new']}")
    return "v" + v


def rollback_last_rule_change() -> Optional[str]:
    """Re-release the previous rules as a new version; the history entry is kept, not deleted."""
    releases = [r for r in RULE_VERSION_HISTORY if r.get("rollback_to")]
    if not releases:
        print("Nothing to roll back.")
        return None
    last = releases[-1]
    v = APP.PM.rollback(last["rollback_to"].lstrip("v"), "Policy owner (Section B)", f"rollback of {last['version']}")
    RULE_VERSION_HISTORY.append({"version": "v" + v, "parameter": "*", "old_value": last["version"],
                                 "new_value": f"rules of {last['rollback_to']}", "rollback_to": None})
    print(f"Rolled back {last['version']} → rules of {last['rollback_to']} re-released as v{v}")
    return "v" + v


print("Validation and metrics functions ready (deterministic E2 engine) · rules", CONFIG["rule_version"])


# ====================================================================================================
# notebook cell 47: B · 4. Agent base class (live LLM notes through the shared client)
# ====================================================================================================
#@title B · 4. Agent base class (live LLM notes through the shared client)
class Agent:
    name = "Agent"
    actor_type = "AI agent"
    SYSTEM = ("You are an assistant inside a public transit operations prototype. Use only the facts given. Never invent "
              "numbers, never score or rank options, never approve or recommend actions beyond what the facts state. Be brief.")

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.last_note = None
        self.last_error = None

    def call_llm(self, prompt: str, max_tokens: int = 220) -> str:
        return llm_generate(prompt, self.SYSTEM, max_tokens=max_tokens)

    def maybe_llm(self, task: str, facts=None):
        """Live, non-authoritative LLM note. Returns None when the LLM is off, no facts are given, or the call fails.
        An agent's typed output never depends on this note."""
        self.last_note = None
        if not (self.use_llm and LLM_READY) or facts is None:
            return None
        try:
            self.last_note = self.call_llm(f"Task: {task}\nFacts (JSON): {json.dumps(facts, default=str)}")
        except LLMUnavailable as e:
            self.last_error = str(e)
        return self.last_note


# ====================================================================================================
# notebook cell 49: B · Vision Agent
# ====================================================================================================
#@title B · Vision Agent
class VisionAgent(Agent):
    name = "Vision Agent"

    def run(self, raw_feed: list) -> list:
        signals = []
        for r in raw_feed:
            lf = round(r["person_count"] / r["capacity"], 2)
            # same rule as the policy hard trigger (load ≥ 1.0 with at least one pass-up)
            if lf >= CONFIG["unboardable_load_factor"] and r["pass_ups"] >= 1:
                level = "unboardable"      # riders left behind at or over capacity
            elif lf >= 1.0 or r["pass_ups"] >= 1:
                level = "crowded"
            elif lf >= 0.8:
                level = "busy"
            else:
                level = "normal"
            signals.append(LoadSignal(stop_id=r["stop_id"], route=r["route"], timestamp=r["timestamp"],
                                      passenger_count=r["person_count"], capacity=r["capacity"],
                                      load_factor=lf, pass_ups=r["pass_ups"], crowding_level=level))
        return signals

# ====================================================================================================
# notebook cell 51: B · Signal Scout Agent
# ====================================================================================================
#@title B · Signal Scout Agent
class SignalScoutAgent(Agent):
    name = "Signal Scout Agent"

    def run(self, inputs: dict, load_signals: list, parent_case_id: Optional[str] = None) -> Optional[Case]:
        svc = inputs["service_feeds"]
        reasons = []
        if svc.get("status") == "cancelled":
            reasons.append(f"trip {svc['trip_id']} cancelled")
        if svc.get("delay_min", 0) >= CONFIG["delay_hard_trigger_min"]:      # policy value, not hard-coded 15
            reasons.append(f"delay {svc['delay_min']} min")
        crowded = [s for s in load_signals if s.crowding_level in ("crowded", "unboardable")]
        if crowded:
            reasons.append("crowding at " + ", ".join(f"{s.stop_id} ({s.crowding_level})" for s in crowded))
        if any(e.get("type") == "official_alert" for e in inputs.get("event_signals", [])):
            reasons.append("official alert received")
        if not reasons:
            return None
        case = Case(
            case_id="CASE-" + uuid.uuid4().hex[:6].upper(), route=svc["route"],
            disruption_type="cancellation" if svc.get("status") == "cancelled" else "delay_or_crowding",
            description=f"Route {svc['route']}: " + "; ".join(reasons),
            opened_at=stamp(), rule_version=CONFIG["rule_version"], parent_case_id=parent_case_id,
            load_signals=load_signals,
        )
        self.maybe_llm("Write a two-sentence case note describing the possible disruption for an operator.",
                       {"route": svc["route"], "reasons": reasons, "rule_version": CONFIG["rule_version"]})
        return case

# ====================================================================================================
# notebook cell 53: B · Planner Copilot Agent
# ====================================================================================================
#@title B · Planner Copilot Agent
class PlannerCopilotAgent(Agent):
    """Reads the reviewer's constraints ("no dispatch", "budget $300", "alternate depot", ...) and returns feasible options.
    Feasibility, resources and costs come from the E2 planner (resource, road, policy and safety checks). Assigns no scores."""
    name = "Planner Copilot Agent"

    def run(self, case: Case, inputs: dict, constraints: list) -> list:
        text = " ".join(constraints).lower()
        excluded = set()
        if any(k in text for k in ("no dispatch", "no extra bus", "no second bus")):
            excluded.add("dispatch_second_bus")
        if "no reroute" in text:
            excluded.add("reroute")
        if "no connection" in text:
            excluded.add("protect_connection")
        if "no alert" in text or "no rider alert" in text:
            excluded.add("rider_alerts")
        cons = {"excluded_options": sorted(excluded)}
        m = re.search(r"budget\s*\$?\s*(\d+)", text)
        if m:
            cons["budget_cap_usd"] = float(m.group(1))
        if "alternate depot" in text:
            cons["allow_alternate_depot"] = True
        eng = _B_ENGINE[case.case_id]
        A = run_analysis(eng["raw"], _b_policy(case), eng["resolution"], cons)
        eng.update(analysis=A, constraints=cons)
        options = []
        for o in A["options"]:
            if not o["feasible"]:
                case.log("Planner Copilot Agent", "Planner Copilot Agent", "AI agent", f"infeasible: {o['label']}", o["infeasible_reason"])
                continue
            options.append(Option(option_id=f"{case.case_id}-{o['option_id']}", action=B_ACTION_FROM_ENGINE[o["type"]],
                                  description=f"{o['label']} — {o['notes']}",
                                  resources_needed={"est_cost": o["cost_usd"], "parties": o["parties"]},
                                  constraints_checked=["resources", "roads", "policy", "safety"] +
                                                      [f"planner excluded {x}" for x in sorted(excluded)]))
        return options



# ====================================================================================================
# notebook cell 55: B · Explanation Agent
# ====================================================================================================
#@title B · Explanation Agent
class ExplanationAgent(Agent):
    name = "Explanation Agent"

    def run(self, case: Case, ranked: list) -> Brief:
        table =[{"rank": s.rank, "option_id": s.option.option_id, "action": s.option.action,
                  "description": s.option.description, "score_Si": s.score, "DR": s.DR, "EB": s.EB,
                  "AP": s.AP, "OF": s.OF, "CE": s.CE, "Ci": s.Ci,
                  "est_cost": s.option.resources_needed.get("est_cost", 0)} for s in ranked]
        feasible = [s for s in ranked if s.option.action != "do_nothing"]
        notes = [f"Data confidence C = {case.confidence:.2f}"]
        notes += [f"{s.option.action}: option confidence {s.Ci:.2f}" for s in ranked if s.Ci < 0.8]
        if not feasible:
            notes.append("No feasible option besides doing nothing: escalated for a human decision.")
        top = ranked[0]
        narrative = self.maybe_llm(
            "Explain these deterministic results to a human reviewer in at most 4 sentences. Describe the alternatives "
            "before the recommendation, which is always the rank-1 option, and say the human reviewer decides.",
            {"options_ranked": table, "confidence_C": case.confidence,
             "impact": {"R": case.impact.R, "V": case.impact.V, "A": case.impact.A, "P": case.impact.P}})
        if narrative and top.option.action.replace("_", " ") not in narrative.lower().replace("_", " ") \
                and top.option.description.lower()[:20] not in narrative.lower():
            narrative = None                      # guard: a narrative that does not name the rank-1 option is discarded
        if narrative:
            notes.append(f"Narrative ({llm_source()}, non-authoritative): {narrative}")
        return Brief(
            case_id=case.case_id,
            summary=f"{case.description}. Impact R={case.impact.R:.2f}, V={case.impact.V:.2f}, "
                    f"A={case.impact.A:.2f}, priority P={case.impact.P:.2f}. Severity: {case.severity}.",
            options_table=table,
            recommendation=f"Highest ranked: #{top.rank} {top.option.description} (Si = {top.score})",
            uncertainty_notes=notes,
            sources=["service feed", "vision load signals", "official alert", "community access (area level)"],
        )

# ====================================================================================================
# notebook cell 57: B · Insight Agent
# ====================================================================================================
#@title B · Insight Agent
class InsightAgent(Agent):
    name = "Insight Agent"

    def receive_rejection(self, proposal: PolicyProposal, reason: str):
        """Input from Policy Owner Approval when a proposal is rejected. The same change is not resubmitted."""
        REJECTED_PROPOSALS[(proposal.parameter, proposal.proposed_value)] = reason

    def run(self, outcome_history: list) -> list:
        candidates = []
        actions = [o for o in outcome_history if o.path == "action"]
        # only dispatch outcomes can justify a dispatch change, and the target is the dispatch option certainty
        dispatch = [o for o in actions if o.metrics.get("action_type") == "dispatch_second_bus"]
        errors = [o.metrics.get("delay_reduction_error_min", 0) for o in dispatch]
        if errors and max(errors) >= 3:
            cur = get_rule("lookups.option_certainty.dispatch_second_bus")
            candidates.append(("lookups.option_certainty.dispatch_second_bus", round(max(0.5, cur - 0.05), 2),
                               f"Observed delay reduction was {max(errors)} min below prediction.",
                               [o.case_id for o in dispatch]))
        known = [o for o in outcome_history if o.false_alarm is not None]
        if known and sum(o.false_alarm for o in known) / len(known) > 0.3:
            candidates.append(("significance_R", round(CONFIG["significance_threshold_R"] + 0.05, 2),   # policy path
                               "False alarm rate above 30 percent.", [o.case_id for o in known]))
        proposals = []
        for param, value, why, evidence in candidates:
            current = get_rule(param)
            if current == value or (param, value) in REJECTED_PROPOSALS:
                continue   # already in force, or the policy owner already said no
            note = self.maybe_llm("Explain this proposed rule change to a policy owner in 2 sentences and state that "
                                  "nothing is applied until they approve it and it passes a backtest.",
                                  {"parameter": param, "current": current, "proposed": value, "reason": why,
                                   "evidence_cases": len(evidence)})
            proposals.append(PolicyProposal(proposal_id="PROP-" + uuid.uuid4().hex[:5].upper(), parameter=param,
                                            current_value=current, proposed_value=value,
                                            rationale=why + (f" — LLM summary ({llm_source()}): {note}" if note else ""),
                                            evidence_cases=evidence))
        return proposals


print("Agents defined.")

# ====================================================================================================
# notebook cell 59: B · 5. Rule steps
# ====================================================================================================
#@title B · 5. Rule steps
def passes_confidence_gate(C: float, official_trigger: Optional[str]) -> bool:
    """Proceed when C >= threshold, or when an official source verifies a hard trigger."""
    return C >= CONFIG["confidence_threshold"] or official_trigger is not None


def is_significant(impact: ImpactScores) -> bool:
    """Significant if a hard trigger applies OR R >= threshold."""
    return impact.hard_trigger is not None or impact.R >= CONFIG["significance_threshold_R"]


def is_high_impact(option: Option) -> bool:
    return option.action in CONFIG["high_impact_actions"]


def review_deadline_for(severity: str) -> datetime:
    return now() + timedelta(minutes=CONFIG["review_deadline_minutes"].get(severity, 20))


def safe_default(case: Case, reason: str) -> dict:
    """Send verified rider information and keep monitoring. No dispatch, rerouting or live control."""
    case.status = "safe_default"
    msg = {"rider_message": f"Route {case.route}: service disruption. Check verified service updates before travelling.",
           "monitoring": True, "reason": reason}
    case.log("Safe Default", "Safe Default", "Rule / code", "verified rider information + monitoring", reason)
    return msg


def continue_monitoring(case: Case, reason: str) -> dict:
    """Record confidence, priority and reason. Recheck when a feed changes. No action taken."""
    case.status = "monitoring"
    rec = {"confidence": case.confidence, "priority": case.impact.P if case.impact else None, "reason": reason}
    case.log("Continue Monitoring", "Continue Monitoring", "Rule / code", "monitoring record", reason)
    return rec


EXCEPTION_HANDLERS = {"feed_outage": handle_feed_outage, "agent_error": handle_agent_error, "conflict": handle_conflict}


# ---------- Demo controls for the Section B runner only (see DEMO_OVERRIDES; off by default) ----------
def demo_verification(v: VerificationResult, attempt: int) -> VerificationResult:
    forced = DEMO_OVERRIDES["verification_status"]
    if forced and attempt == 0:
        return VerificationResult(status=forced, issues=[f"demo: forced {forced}"])
    return v


def demo_resolved(resolved: bool) -> bool:
    return resolved if DEMO_OVERRIDES["exception_resolved"] is None else bool(DEMO_OVERRIDES["exception_resolved"])


def demo_confidence(C: float) -> float:
    return C if DEMO_OVERRIDES["confidence"] is None else DEMO_OVERRIDES["confidence"]


def demo_impact(i: ImpactScores) -> ImpactScores:
    if DEMO_OVERRIDES["R"] is not None:
        i.R = DEMO_OVERRIDES["R"]
    if DEMO_OVERRIDES["remove_hard_trigger"]:
        i.hard_trigger = None
    return i


print("Rules defined.")

# ====================================================================================================
# notebook cell 61: B · 6. Human in the loop forms (ipywidgets)
# ====================================================================================================
#@title B · 6. Human in the loop forms (ipywidgets)
def run_safely(fn, msg_widget):
    """Widget callbacks in Colab hide errors. Show them in the form instead."""
    def handler(_):
        try:
            fn()
        except Exception as e:
            msg_widget.value = f"<span style='color:#C4496B'><b>Error:</b> {type(e).__name__}: {e}</span>"
            raise
    return handler


class ReviewForm:
    """Human Review form. Output: ReviewDecision passed to on_submit."""
    role = "Human Review"
    choices = [("Approve", "approve"), ("Revise", "revise"), ("Reject", "reject")]

    def __init__(self, case: Case, on_submit: Callable, preselected: Optional[str] = None,
                 exclude_reviewer: Optional[str] = None):
        self.case, self.on_submit, self.opened_at, self.done = case, on_submit, now(), False
        self.exclude_reviewer = exclude_reviewer
        labels = [(f"#{s.rank}  {s.option.description}  (Si {s.score})", s.option.option_id) for s in case.ranked_options]
        self.name_w = widgets.Text(placeholder="Your name (required)", description="Reviewer",
                                   layout=widgets.Layout(width="60%"))
        self.option_w = widgets.Dropdown(options=labels, value=preselected or labels[0][1],
                                         description="Option", layout=widgets.Layout(width="95%"))
        self.decision_w = widgets.ToggleButtons(options=self.choices, description="Decision")
        self.reason_w = widgets.Textarea(placeholder="Typed reason (required)", description="Reason",
                                         layout=widgets.Layout(width="95%", height="60px"))
        self.constraints_w = widgets.Text(placeholder="e.g. no dispatch (revise only)", description="Constraints",
                                          layout=widgets.Layout(width="95%"))
        self.check_w = widgets.Checkbox(description="I reviewed every alternative, score and uncertainty note",
                                        layout=widgets.Layout(width="95%"))
        self.submit_w = widgets.Button(description="Submit decision", button_style="primary")
        self.timeout_w = widgets.Button(description="Simulate timeout", button_style="warning")
        self.msg_w = widgets.HTML()
        self.submit_w.on_click(run_safely(lambda: self.submit(
            self.decision_w.value, self.option_w.value, self.reason_w.value, self.constraints_w.value,
            self.check_w.value, self.name_w.value), self.msg_w))
        self.timeout_w.on_click(run_safely(lambda: self.submit("timeout"), self.msg_w))

    def widget(self):
        deadline = self.case.review_deadline.astimezone(LOCAL_TZ).strftime("%H:%M:%S %Z") if self.case.review_deadline else "none"
        head = widgets.HTML(f"<h4 style='margin:6px 0'>🟦 {self.role} | case {self.case.case_id} | deadline {deadline}</h4>")
        return widgets.VBox([head, self.name_w, self.option_w, self.decision_w, self.reason_w, self.constraints_w,
                             self.check_w, widgets.HBox([self.submit_w, self.timeout_w]), self.msg_w],
                            layout=widgets.Layout(border="2px solid #0E2E4F", padding="8px", margin="8px 0"))

    def submit(self, decision: str, option_id: Optional[str] = None, reason: str = "",
               constraints: str = "", checklist: bool = False, reviewer: str = "") -> Optional[ReviewDecision]:
        if self.done:
            return None
        if decision != "timeout" and self.case.review_deadline and now() > self.case.review_deadline:
            decision, reason = "timeout", "Deadline passed before submit"
        if decision != "timeout":
            problems = []
            if not reviewer.strip():
                problems.append("enter your name")
            elif self.exclude_reviewer and reviewer.strip().lower() == self.exclude_reviewer.strip().lower():
                problems.append("the second reviewer must be a different person from the first")
            if decision == "approve" and self.case.option(option_id) is None:
                problems.append("choose an option to approve")
            if not checklist:
                problems.append("tick the checklist")
            if len(reason.strip()) < CONFIG["min_reason_chars"]:
                problems.append(f"type a reason of at least {CONFIG['min_reason_chars']} characters")
            if decision == "revise" and not constraints.strip():
                problems.append("enter the new constraints for the planner")
            # INTEGRATION (Sections C + D): ethics-training and conflict-of-interest gate
            if reviewer.strip():
                opt = workflow_option_as_engine_option(self.case, option_id) if decision == "approve" else None
                gp, gn = governance_check(reviewer, decision.capitalize(), option=opt)
                problems += gp
                for n_ in gn:
                    self.case.log("Reviewer gate (C+D)", "Governance gate", "Rule / code", "note", n_)
            if problems:
                self.msg_w.value = "<span style='color:#C4496B'>Cannot submit: " + "; ".join(problems) + ".</span>"
                return None
        self.done = True
        for w in (self.name_w, self.option_w, self.decision_w, self.reason_w, self.constraints_w, self.check_w,
                  self.submit_w, self.timeout_w):
            w.disabled = True
        rd = ReviewDecision(reviewer_role=self.role, reviewer_name=reviewer.strip() or "(no response)",
                            decision=decision, selected_option_id=option_id if decision == "approve" else None,
                            reason=reason or "No decision before deadline", new_constraints=constraints,
                            checklist_confirmed=checklist,
                            seconds_to_decide=round((now() - self.opened_at).total_seconds(), 1), timestamp=stamp())
        self.case.decisions.append(asdict(rd))
        self.case.log(self.role, rd.reviewer_name, "Person", rd.decision,
                      f"{rd.reason} (decided in {rd.seconds_to_decide}s)")
        self.msg_w.value = f"<b>Submitted:</b> {decision}"
        self.on_submit(rd)
        return rd


class SecondReviewerForm(ReviewForm):
    """Second Reviewer for high impact actions. Same deadline. Can approve or reject only."""
    role = "Second Reviewer"
    choices = [("Approve", "approve"), ("Reject", "reject")]

    def __init__(self, case: Case, on_submit: Callable, preselected: str, first_reviewer: str):
        super().__init__(case, on_submit, preselected, exclude_reviewer=first_reviewer)
        self.option_w.disabled = True
        self.constraints_w.layout.display = "none"


class ImplementActionForm:
    """Operations staff confirm what happened. Output: ActionResult passed to on_submit."""
    role = "Implement Action"

    def __init__(self, case: Case, on_submit: Callable):
        self.case, self.on_submit, self.done = case, on_submit, False
        opt = case.option(case.selected_option_id)
        self.head = widgets.HTML(f"<h4 style='margin:6px 0'>🟦 Implement Action | {opt.option.description}</h4>"
                                 "<i>Carried out by operations staff outside this prototype.</i>")
        self.status_w = widgets.ToggleButtons(options=[("Completed", "completed"), ("Partial", "partial"), ("Failed", "failed")],
                                              description="Result")
        self.alert_w = widgets.Checkbox(value=True, description="Rider alert sent")
        self.notes_w = widgets.Textarea(placeholder="Notes (required if partial or failed)", description="Notes",
                                        layout=widgets.Layout(width="95%", height="50px"))
        self.submit_w = widgets.Button(description="Confirm result", button_style="primary")
        self.msg_w = widgets.HTML()
        self.submit_w.on_click(run_safely(lambda: self.submit(self.status_w.value, self.alert_w.value,
                                                              self.notes_w.value), self.msg_w))

    def widget(self):
        return widgets.VBox([self.head, self.status_w, self.alert_w, self.notes_w, self.submit_w, self.msg_w],
                            layout=widgets.Layout(border="2px solid #0E2E4F", padding="8px", margin="8px 0"))

    def submit(self, status: str, rider_alert_sent: bool = True, notes: str = "") -> Optional[ActionResult]:
        if self.done:
            return None
        if status in ("partial", "failed") and not notes.strip():
            self.msg_w.value = "<span style='color:#C4496B'>Describe what went wrong so the reviewer can decide.</span>"
            return None
        self.done = True
        for w in (self.status_w, self.alert_w, self.notes_w, self.submit_w):
            w.disabled = True
        ar = ActionResult(option_id=self.case.selected_option_id, status=status,
                          rider_alert_sent=rider_alert_sent, notes=notes, timestamp=stamp())
        self.case.action_result = ar
        self.case.log(self.role, "Operations staff", "Person", status, notes)
        self.on_submit(ar)
        return ar


class PolicyOwnerForm:
    """Policy owner approves or rejects an Insight Agent proposal. Output: PolicyDecision."""
    role = "Policy Owner Approval"

    def __init__(self, proposal: PolicyProposal, case: Case, on_submit: Callable):
        self.proposal, self.case, self.on_submit, self.done = proposal, case, on_submit, False
        p = proposal
        self.head = widgets.HTML(
            f"<h4 style='margin:6px 0'>🟦 Policy Owner Approval | {p.proposal_id}</h4>"
            f"<b>{p.parameter}</b>: {p.current_value} → {p.proposed_value}<br><i>{p.rationale}</i><br>"
            "Approved changes are backtested on past cases, then released as a new rule version with rollback.")
        self.name_w = widgets.Text(placeholder="Your name (required)", description="Policy owner",
                                   layout=widgets.Layout(width="60%"))
        self.decision_w = widgets.ToggleButtons(options=[("Approve", "approve"), ("Reject", "reject")], description="Decision")
        self.reason_w = widgets.Textarea(placeholder="Typed reason (required)", description="Reason",
                                         layout=widgets.Layout(width="95%", height="50px"))
        self.submit_w = widgets.Button(description="Submit decision", button_style="primary")
        self.msg_w = widgets.HTML()
        self.submit_w.on_click(run_safely(lambda: self.submit(self.decision_w.value, self.reason_w.value,
                                                              self.name_w.value), self.msg_w))

    def widget(self):
        return widgets.VBox([self.head, self.name_w, self.decision_w, self.reason_w, self.submit_w, self.msg_w],
                            layout=widgets.Layout(border="2px solid #0E2E4F", padding="8px", margin="8px 0"))

    def submit(self, decision: str, reason: str = "", owner: str = "") -> Optional[PolicyDecision]:
        if self.done:
            return None
        problems = []
        if not owner.strip():
            problems.append("enter your name")
        if len(reason.strip()) < CONFIG["min_reason_chars"]:
            problems.append(f"type a reason of at least {CONFIG['min_reason_chars']} characters")
        if owner.strip():   # INTEGRATION (Sections C + D)
            gp, _ = governance_check(owner, decision.capitalize(), proposal={"title": self.proposal.parameter,
                                     "changes": [{"path": self.proposal.parameter, "old": self.proposal.current_value, "new": self.proposal.proposed_value}]})
            problems += gp
        if problems:
            self.msg_w.value = "<span style='color:#C4496B'>Cannot submit: " + "; ".join(problems) + ".</span>"
            return None
        self.done = True
        for w in (self.name_w, self.decision_w, self.reason_w, self.submit_w):
            w.disabled = True
        pd_ = PolicyDecision(proposal_id=self.proposal.proposal_id, decision=decision, reason=reason, timestamp=stamp())
        self.case.log(self.role, owner.strip(), "Person", decision, f"{self.proposal.parameter}: {reason}")
        self.on_submit(pd_)
        return pd_


print("Human in the loop forms defined.")

# ====================================================================================================
# notebook cell 63: B · 7. Agent-level runner (Section B's typed agents, Signal Scout coordination)
# ====================================================================================================
#@title B · 7. Agent-level runner (Section B's typed agents, Signal Scout coordination)
class Workflow:
    def __init__(self, inputs: dict, raw_feed: list, use_llm: bool = False):
        self.inputs, self.raw_feed = inputs, raw_feed
        self.vision, self.scout = VisionAgent(use_llm), SignalScoutAgent(use_llm)
        self.planner, self.explainer, self.insight = PlannerCopilotAgent(use_llm), ExplanationAgent(use_llm), InsightAgent(use_llm)
        self.case: Optional[Case] = None
        self.active_form = None
        self.out = widgets.Output()

    # ---------- display helpers ----------
    def say(self, kind: str, title: str, body: str = ""):
        colors = {"ai": "#DCEFEC", "code": "#FFF3DC", "human": "#DCE3EC", "data": "#EFE9F8", "decision": "#FDE6EC"}
        with self.out:
            display(HTML(f"<div style='background:{colors[kind]};padding:6px 10px;margin:4px 0;border-radius:6px'>"
                         f"<b>{title}</b><br>{body}</div>"))

    def show(self, obj):
        with self.out:
            display(obj)

    # ---------- Section 1 and 2: detect and check ----------
    def start(self, parent_case_id: Optional[str] = None):
        display(self.out)
        self.run_detection(parent_case_id)

    def run_detection(self, parent_case_id: Optional[str] = None):
        signals = self.vision.run(self.raw_feed)
        self.say("ai", "Vision Agent → load signals", "<br>".join(
            f"{s.stop_id}: load {s.load_factor}, pass ups {s.pass_ups}, {s.crowding_level}" for s in signals))

        case = self.scout.run(self.inputs, signals, parent_case_id)
        if case is None:
            self.say("ai", "Signal Scout Agent → no disruption detected")
            return
        self.case = case
        ALL_CASES.append(case)
        case.log("Vision Agent", "Vision Agent", "AI agent", f"{len(signals)} load signals")
        case.log("Signal Scout Agent", "Signal Scout Agent", "AI agent", "case created", case.description)
        self.say("ai", f"Signal Scout Agent → case {case.case_id} created", case.description)
        if self.scout.last_note:
            case.log("Signal Scout Agent", "Signal Scout Agent", "AI agent", "LLM case note (non-authoritative)",
                     f"{llm_source()}: {self.scout.last_note}")
            self.say("ai", f"Signal Scout note ({llm_source()}, non-authoritative)", self.scout.last_note)
        self.verify_and_assess()

    def verify_and_assess(self):
        case = self.case
        while True:
            v = demo_verification(verify_data(self.inputs, case.load_signals), case.reverify_attempts)
            case.verification = v
            case.log("Data Verification", "Data Verification", "Rule / code", v.status, "; ".join(v.issues))
            self.say("code", f"Data Verification → {v.status}", "; ".join(v.issues))
            if v.status == "ready":
                break
            resolved = demo_resolved(EXCEPTION_HANDLERS[v.status](case, v))
            case.log(f"Exception path: {v.status}", "Exception handler", "Rule / code", f"resolved={resolved}")
            self.say("code", f"Exception path ({v.status}) → resolved: {resolved}")
            if not resolved and v.status == "agent_error":
                # the flowchart sends a repeated agent failure to deterministic safe mode → Safe Default
                return self.to_safe_default("Agent error: deterministic safe mode")
            if not resolved:
                return self.to_monitoring("Unresolved data problem")
            case.reverify_attempts += 1
            if case.reverify_attempts > CONFIG["max_reverify_attempts"]:
                return self.to_monitoring("Reverify limit reached")

        case.confidence = demo_confidence(calculate_confidence(self.inputs, case.load_signals, case.verification))
        official = detect_official_hard_trigger(self.inputs)
        gate = passes_confidence_gate(case.confidence, official)
        case.log("Confidence Calculation", "Confidence Calculation", "Rule / code",
                 f"C={case.confidence:.2f}", f"official hard trigger: {official}")
        self.say("code", f"Confidence Calculation → C = {case.confidence:.2f}",
                 f"Official hard trigger: {official}. Gate passed: {gate}")
        if not gate:
            return self.to_monitoring("Confidence below threshold")

        case.impact = demo_impact(analyze_impact(case, self.inputs))
        case.severity = case.impact.severity
        i = case.impact
        case.log("Impact Analyzer", "Impact Analyzer", "Rule / code", f"R={i.R} V={i.V} A={i.A} P={i.P}",
                 f"hard trigger: {i.hard_trigger}")
        self.say("code", "Impact Analyzer → scores", f"R={i.R}, V={i.V}, A={i.A}, P={i.P}, hard trigger: {i.hard_trigger}")

        sig = is_significant(i)
        case.log("Significant?", "Significance rule", "Decision", "yes" if sig else "no")
        self.say("decision", f"Significant? → {'yes' if sig else 'no'}")
        if not sig:
            return self.to_safe_default("Below significance threshold")
        self.plan()

    # ---------- Section 3: plan and decide ----------
    def plan(self):
        case = self.case
        case.options = self.planner.run(case, self.inputs, case.constraints)
        case.log("Planner Copilot Agent", "Planner Copilot Agent", "AI agent", f"{len(case.options)} options (no scores)")
        self.say("ai", f"Planner Copilot Agent → {len(case.options)} options",
                 "<br>".join(o.description for o in case.options))
        if all(o.action == "do_nothing" for o in case.options):
            self.say("ai", "Planner Copilot Agent → no feasible option: escalating to review")
            case.log("Planner Copilot Agent", "Planner Copilot Agent", "AI agent", "no feasible option: escalate")

        case.ranked_options = evaluate_scenarios(case, case.options)
        case.log("Scenario Evaluator", "Scenario Evaluator", "Rule / code",
                 f"ranked {len(case.ranked_options)}; top {case.ranked_options[0].option.action}")

        case.brief = self.explainer.run(case, case.ranked_options)
        case.log("Explanation Agent", "Explanation Agent", "AI agent", "brief ready")
        b = case.brief
        self.say("ai", "Explanation Agent → brief for the reviewer", b.summary)
        self.show(pd.DataFrame(b.options_table).set_index("rank"))
        self.say("ai", "Recommendation", b.recommendation + "<br><i>" + "<br>".join(b.uncertainty_notes) + "</i>")
        self.open_review()

    def open_review(self):
        """Every new brief starts a new review deadline. The second reviewer shares it."""
        # a revision keeps the original deadline unless the policy flag says otherwise (policy flag)
        if self.case.review_deadline is None or self.case.revision_count == 0 or CONFIG["deadline_resets_on_revision"]:
            self.case.review_deadline = review_deadline_for(self.case.severity)
        self.active_form = ReviewForm(self.case, self.on_review)
        self.show(self.active_form.widget())

    def on_review(self, rd: ReviewDecision):
        case = self.case
        if rd.decision in ("reject", "timeout"):
            return self.to_safe_default(f"Human Review: {rd.decision}")
        if rd.decision == "revise":
            case.revision_count += 1
            if case.revision_count > CONFIG["max_revisions"]:
                return self.to_safe_default("Revision limit reached")
            case.constraints.append(rd.new_constraints)
            self.say("human", f"Revision {case.revision_count} of {CONFIG['max_revisions']}",
                     f"New constraints: {rd.new_constraints}. Rebuilding options; rules unchanged.")
            return self.plan()
        case.selected_option_id = rd.selected_option_id
        opt = case.option(rd.selected_option_id).option
        if is_high_impact(opt):
            self.say("human", "High impact action → second reviewer required")
            self.active_form = SecondReviewerForm(case, self.on_second_review, preselected=opt.option_id,
                                                  first_reviewer=rd.reviewer_name)
            return self.show(self.active_form.widget())
        self.open_implement()

    def on_second_review(self, rd: ReviewDecision):
        if rd.decision != "approve":
            return self.to_safe_default(f"Second Reviewer: {rd.decision}")
        self.open_implement()

    def open_implement(self):
        self.case.status = "approved"
        self.active_form = ImplementActionForm(self.case, self.on_action_result)
        self.show(self.active_form.widget())

    def on_action_result(self, ar: ActionResult):
        if ar.status in ("failed", "partial"):
            self.say("human", f"Action {ar.status} → returned to Human Review", ar.notes)
            self.case.review_deadline = review_deadline_for(self.case.severity)
            self.active_form = ReviewForm(self.case, self.on_review)
            return self.show(self.active_form.widget())
        self.case.status = "implemented"
        self.learn(ar)

    # ---------- Fallbacks ----------
    def to_safe_default(self, reason: str):
        msg = safe_default(self.case, reason)
        self.say("code", "Safe Default", f"{reason}<br>Rider message: {msg['rider_message']}")
        self.learn(None)

    def to_monitoring(self, reason: str):
        continue_monitoring(self.case, reason)
        self.say("code", "Continue Monitoring", reason)
        self.learn(None)

    def re_escalate(self, updated_inputs: Optional[dict] = None, updated_feed: Optional[list] = None):
        """Safe default / monitoring re-escalate to the Signal Scout when conditions change (connector A)."""
        if self.case is None or self.case.status not in ("safe_default", "monitoring"):
            print("Re-escalation applies only to cases in safe default or monitoring.")
            return
        self.case.log("Re-escalate", "Signal Scout Agent", "AI agent", "re-escalated")
        self.inputs = updated_inputs or self.inputs
        self.raw_feed = updated_feed or self.raw_feed
        self.run_detection(parent_case_id=self.case.case_id)

    # ---------- Learn and improve ----------
    def learn(self, ar: Optional[ActionResult]):
        case = self.case
        case.outcome = evaluate_outcome(case, ar)
        OUTCOME_HISTORY.append(case.outcome)
        case.log("Outcome Evaluator", "Outcome Evaluator", "Rule / code", case.outcome.path, json.dumps(case.outcome.metrics))
        self.say("code", "Outcome Evaluator → report", f"Path: {case.outcome.path}. Metrics: {case.outcome.metrics}")

        proposals = self.insight.run(OUTCOME_HISTORY)
        case.log("Insight Agent", "Insight Agent", "AI agent", f"{len(proposals)} proposal(s)")
        if not proposals:
            self.say("ai", "Insight Agent → no rule changes proposed")
            return self.finish()
        self.pending = list(proposals)
        self.next_proposal()

    def next_proposal(self):
        if not self.pending:
            return self.finish()
        p = self.pending.pop(0)
        self.say("ai", f"Insight Agent → proposal {p.proposal_id}", f"{p.parameter}: {p.current_value} → {p.proposed_value}")
        self.active_form = PolicyOwnerForm(p, self.case, lambda d, p=p: self.on_policy_decision(p, d))
        self.show(self.active_form.widget())

    def on_policy_decision(self, p: PolicyProposal, d: PolicyDecision):
        if d.decision == "reject":
            self.insight.receive_rejection(p, d.reason)
            self.say("human", "Policy owner rejected → reason returned to Insight Agent, current rules kept", d.reason)
            self.case.log("Insight Agent", "Insight Agent", "AI agent", "received rejection reason", d.reason)
            return self.next_proposal()
        bt = backtest_rule_change(p, ALL_CASES)
        self.case.log("Backtest", "Backtest", "Rule / code", f"passed={bt.passed}", bt.summary)
        if not bt.passed:
            self.say("code", "Backtest failed → current rules kept", bt.summary)
            return self.next_proposal()
        old = CONFIG["rule_version"]
        new = apply_rule_change(p)
        self.case.log("Rule release", "Scoring rules", "Rule / code", new,
                      f"{p.parameter}: {p.current_value} → {p.proposed_value}; rollback to {old}")
        self.say("code", f"Backtest passed → released rule version {new}",
                 f"{p.parameter}: {p.current_value} → {p.proposed_value}. Rollback available to {old}. {bt.summary}")
        self.next_proposal()

    def finish(self):
        self.active_form = None
        self.say("data", f"Case {self.case.case_id} complete → status: {self.case.status}",
                 "Run the <b>Audit trail and outputs</b> cell to see every step.")


print("Workflow defined.")

# ====================================================================================================
# notebook cell 65: B · 8. Run the agent-level runner (optional)
# ====================================================================================================
#@title B · 8. Run the agent-level runner (optional)
RUN_SECTION_B_FORMS = True  #@param {type:"boolean"}
wf = None
if RUN_SECTION_B_FORMS:
    wf = Workflow(SAMPLE_INPUTS, RAW_SENSOR_FEED, use_llm=LLM_READY)
    wf.start()

# ====================================================================================================
# notebook cell 67: B · 9. Audit trail and outputs
# ====================================================================================================
#@title B · 9. Audit trail and outputs
AUDIT_COLUMNS = ["time", "case_id", "stage", "actor_type", "actor", "output", "note", "rule_version"]


def audit_table(cases=None) -> pd.DataFrame:
    rows = [r for c in (cases if cases is not None else ALL_CASES) for r in c.audit]
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def decisions_table(cases=None) -> pd.DataFrame:
    rows = [dict(case_id=c.case_id, **d) for c in (cases if cases is not None else ALL_CASES) for d in c.decisions]
    return pd.DataFrame(rows)


def export_case(case: Case, folder: str = OUTPUT_DIR) -> str:
    path = f"{folder}/{case.case_id}.json"
    with open(path, "w") as f:
        json.dump(asdict(case), f, indent=2, default=str)
    return path


if wf is not None and wf.case:
    print(f"Case {wf.case.case_id}: status = {wf.case.status}")
    display(audit_table([wf.case]))
    if wf.case.decisions:
        display(decisions_table([wf.case])[["case_id", "reviewer_role", "reviewer_name", "decision",
                                            "seconds_to_decide", "checklist_confirmed", "reason"]])
    display(pd.DataFrame(RULE_VERSION_HISTORY))
    print("Saved:", export_case(wf.case))
else:
    print("No Section B runner case (B · 8 is off). Orchestrator and gallery cases are in Section O, F and the results summary.")

# ====================================================================================================
# notebook cell 69: B · 10. Scripted run (no clicking)
# ====================================================================================================
#@title B · 10. Scripted run (no clicking)
RUN_SCRIPTED_TEST = True  #@param {type:"boolean"}

if RUN_SCRIPTED_TEST:
    t = Workflow(SAMPLE_INPUTS, RAW_SENSOR_FEED)
    t.start()
    top = t.case.ranked_options[0].option.option_id
    t.active_form.submit("approve", top, "Highest score and protects the rail connection",
                         checklist=True, reviewer="Ana Lopez")
    if isinstance(t.active_form, SecondReviewerForm):
        t.active_form.submit("approve", top, "Agree, spare bus and operator confirmed",
                             checklist=True, reviewer="Chen Wei")
    t.active_form.submit("completed", True, "Bus arrived 17:58")
    while isinstance(t.active_form, PolicyOwnerForm):
        t.active_form.submit("approve", "Evidence supports recalibration", owner="Dana Brooks")
    display(audit_table([t.case]))

# ====================================================================================================
# notebook cell 71: O · Orchestrator: pick a case and run it end to end (agents → rules → people → outcome → learning loop)
# ====================================================================================================
#@title O · Orchestrator: pick a case and run it end to end (agents → rules → people → outcome → learning loop)
ORCH_STEP_DELAY_S = 0.2  #@param {type:"number"}
import ipywidgets as W
import html as _h
from _ipython_shim import display, HTML

# ---------------------------------------------------------------- case sources
def _orch_sources():
    src = {"LIVE · tonight's NJ Transit hand-off (Section A)": lambda: copy.deepcopy(_raw_a),
           "Section B sample · Route 12 cancellation + last rail connection":
               lambda: workflow_inputs_to_raw(SAMPLE_INPUTS, VisionAgent().run(RAW_SENSOR_FEED))}
    src.update({k: (lambda k=k: make_raw(PRESETS[k], k)) for k in PRESETS})
    return src


ORCH_SOURCES = _orch_sources()
ROSTER_NAMES = [r["name"] for r in ETH_SAMPLES["Transit review team (this system)"]]
ORCH_PEOPLE = {"first": "Ana Lopez", "second": "Chen Wei", "planner": "Ben Carter", "owner": "Dana Brooks"}   # pre-filled; edit freely

# ---------------------------------------------------------------- look and feel (same colour key as the workflow diagram)
KIND = {"data": ("#EDE9FE", "#5B21B6", "Data / output"), "agent": ("#DCFCE7", "#166534", "AI agent"),
        "rule": ("#DBEAFE", "#1E40AF", "Fixed rule"), "human": ("#FFEDD5", "#9A3412", "Person"),
        "safe": ("#FEE2E2", "#991B1B", "Safe default / monitoring")}
AGENT_STAGES = ("Signal Scout", "Vision Agent", "Planner Copilot", "Explanation", "Insight", "Outcome")
HUMAN_ROLES = ("reviewer", "second reviewer", "operator", "planner", "operations", "policy owner")


def _kind_of(entry):
    if entry["role"] in HUMAN_ROLES:
        return "human"
    if entry["stage"] in ("Safe default", "Monitoring"):
        return "safe"
    if any(entry["stage"].startswith(s) for s in AGENT_STAGES) or entry["role"] == "agent":
        return "agent"
    return "rule"


def _chip(name, kind, state, note=""):
    bg, fg, _ = KIND[kind]
    icon = {"done": "✓", "active": "●", "stop": "■", "fail": "✗"}[state]
    border = "3px solid #F97316" if state == "active" else f"1px solid {fg}"
    return (f"<div style='display:inline-block;margin:3px;padding:5px 9px;border-radius:8px;background:{bg};color:{fg};"
            f"border:{border};font:12px/1.3 sans-serif;vertical-align:top;max-width:190px'><b>{icon} {_h.escape(name)}</b>"
            + (f"<br><span style='font-size:11px'>{_h.escape(str(note))}</span>" if note else "") + "</div>")


def diagram_progress(case):
    """Which boxes of the workflow diagram this case has passed through, and where it is waiting now."""
    A, st = case["analysis"], case["status"]
    rv = case.get("review") or {}
    firsts = [d for d in case["decisions"] if d["stage"] == "first review"]
    seconds = [d for d in case["decisions"] if d["stage"] == "second review"]
    ch = [_chip("Data inputs", "data", "done", f"{len(case['raw'].get('observations', []))} feeds"),
          _chip("Vision Agent", "agent", "fail" if A["gate"] == "SAFE_MODE" else "done"),
          _chip("Signal Scout", "agent", "done", "case opened")]
    ev = A.get("evidence", [])
    ch.append(_chip("Data verification", "rule", "done", f"{sum(e['status'] == 'accepted' for e in ev)} accepted / {len(ev)}"))
    paths = list(dict.fromkeys(x["path"] for x in A.get("exceptions", [])))
    if paths:
        ch.append(_chip("Exception path", "rule", "done", ", ".join(paths).lower()))
    if st == "CONFLICT_REVIEW":
        ch.append(_chip("Operator review", "human", "active", f"check {case['operator_checks'] + 1} of 2"))
    if A.get("confidence"):
        C = A["confidence"]["C"]
        ok = A["gate"] in ("SIGNIFICANT", "BELOW_THRESHOLD")
        ch.append(_chip("Confidence gate", "rule", "done" if ok else "stop",
                        f"C = {C:.2f}" + (" (official bypass)" if ok and C < get_case_policy(case)["confidence_threshold"] else "")))
    if A.get("scores"):
        s = A["scores"]
        ch.append(_chip("Impact analyzer", "rule", "done", f"R {s['R']:.2f} · V {s['V']:.2f} · A {s['A']:.2f} · P {s['P']:.2f}"))
        ch.append(_chip("Significant?", "rule", "done" if A["significant"] else "stop",
                        ", ".join(t["trigger"] for t in A["hard_triggers"]) or ("R ≥ 0.50" if A["significant"] else "no")))
    if A["gate"] == "SIGNIFICANT":
        ch += [_chip("Planner Copilot", "agent", "done", f"{sum(o['feasible'] for o in A['options'])} feasible"),
               _chip("Scenario evaluator", "rule", "done", f"top Si {A['ranking'][0]['Si']:.3f}" if A["ranking"] else "none"),
               _chip("Explanation Agent", "agent", "done", (case.get("explanation") or {}).get("narrative_source", "")[:28])]
        if st == "AWAITING_REVIEW" or firsts:
            ch.append(_chip("Human review", "human", "active" if st == "AWAITING_REVIEW" else "done",
                            f"{rv.get('tier', '')} · {rv.get('deadline_min', '')} min" if st == "AWAITING_REVIEW"
                            else f"{firsts[-1]['reviewer']}: {firsts[-1]['decision']}"))
        if st == "AWAITING_SECOND_REVIEW" or seconds:
            ch.append(_chip("Second reviewer", "human", "active" if st == "AWAITING_SECOND_REVIEW" else "done",
                            "high-impact action" if st == "AWAITING_SECOND_REVIEW" else f"{seconds[-1]['reviewer']}: {seconds[-1]['decision']}"))
        if st == "AWAITING_IMPLEMENTATION" or case["implementation"]:
            ch.append(_chip("Implement action", "human", "active" if st == "AWAITING_IMPLEMENTATION" else "done",
                            "operations staff" if st == "AWAITING_IMPLEMENTATION" else case["implementation"][-1]["result"]))
    if st == "SAFE_DEFAULT":
        ch.append(_chip("Safe default", "safe", "done", case.get("safe_default_reason", "")[:40]))
    if st == "MONITORING":
        ch.append(_chip("Continue monitoring", "safe", "active", "recheck / re-escalate / close"))
    if st == "CLOSED_FALSE_ALARM":
        ch.append(_chip("Closed: false alarm", "safe", "done"))
    if case.get("outcome") and "delay_error_min" in case["outcome"]:
        ch.append(_chip("Outcome evaluator", "agent", "done", f"delay error {case['outcome']['delay_error_min']:+.1f} min"))
    elif case.get("outcome"):
        ch.append(_chip("Outcome evaluator", "agent", "done", "rider information only"))
    if case.get("_policy_step"):
        ch.append(_chip("Insight → Policy owner", "human", "done", case["_policy_step"]))
    elif st in ("EVALUATED", "CLOSED_FALSE_ALARM") or (st == "SAFE_DEFAULT" and case.get("outcome")):
        ch.append(_chip("Insight → Policy owner", "human", "active", "learning loop (optional)"))
    legend = " ".join(f"<span style='background:{b};color:{f};padding:2px 6px;border-radius:5px;font:11px sans-serif'>{t}</span>"
                      for b, f, t in KIND.values())
    return "<div>" + "".join(ch) + f"</div><div style='margin-top:4px'>{legend} · ● = waiting here</div>"


def audit_html(case, last=None):
    rows = case["audit"][-last:] if last else case["audit"]
    out = ["<table style='border-collapse:collapse;font:12px sans-serif;width:100%'>"]
    for e in rows:
        bg, fg, _ = KIND[_kind_of(e)]
        out.append(f"<tr style='background:{bg};color:{fg}'><td style='padding:3px 6px;white-space:nowrap'>{e['ts'][11:19]} ET</td>"
                   f"<td style='padding:3px 6px'><b>{_h.escape(e['stage'])}</b></td><td style='padding:3px 6px'>{_h.escape(str(e['actor']))}</td>"
                   f"<td style='padding:3px 6px'>{_h.escape(str(e['event']))}</td>"
                   f"<td style='padding:3px 6px;color:#334155'>{_h.escape(str(e['detail']))[:220]}</td></tr>")
    return "".join(out) + "</table>"


def _banner_html(text, kind="human"):
    bg, fg, _ = KIND[kind]
    return (f"<div style='margin:8px 0;padding:10px 12px;border-left:6px solid {fg};background:{bg};color:{fg};"
            f"font:14px sans-serif'>{text}</div>")


def _md_html(text):
    """Markdown → HTML for display inside widgets (tables included)."""
    try:
        import markdown as _mdlib
        out = _mdlib.markdown(text, extensions=["tables"])
    except Exception:
        out = "<pre style='white-space:pre-wrap'>" + _h.escape(text) + "</pre>"
    return (out.replace("<table>", "<table style='border-collapse:collapse;font-size:12px;margin:6px 0'>")
               .replace("<th>", "<th style='border:1px solid #d1d5db;padding:3px 6px;background:#f3f4f6'>")
               .replace("<td>", "<td style='border:1px solid #e5e7eb;padding:3px 6px'>"))


def _w_html(html):
    return W.HTML(value=f"<div style='font:13px sans-serif;max-width:1100px'>{html}</div>")


# ---------------------------------------------------------------- the orchestrator
class Orchestrator:
    """One place to run a case end to end. AI agents and fixed rules run automatically; the run stops at every box in the
    diagram where a person decides, shows that person a form, and continues from their answer.
    The screen is built from widgets only (no Output-capture contexts), so it redraws reliably in Colab however many cases
    are processed."""

    def __init__(self, store):
        self.S, self.case, self.msg = store, None, ""
        self.src = W.Dropdown(options=list(ORCH_SOURCES), description="Case", layout=W.Layout(width="640px"))
        self.use_llm = W.Checkbox(value=LLM_READY, description="Live LLM narrative", indent=False)
        self.go = W.Button(description="▶ Process case", button_style="primary", layout=W.Layout(width="160px"))
        self.refresh = W.Button(description="↻ Refresh / check deadline", layout=W.Layout(width="220px"))
        self.go.on_click(lambda _: self.start())
        self.refresh.on_click(lambda _: self.render())
        self.top = W.HTML()
        self.form = W.VBox()
        self.log = W.HTML()
        self.ui = W.VBox([W.HTML("<h3 style='margin:4px 0'>Orchestrator · select a case and process it end to end</h3>"),
                          W.HBox([self.src, self.use_llm]), W.HBox([self.go, self.refresh]), self.top, self.form,
                          W.HTML("<b>Audit trail (every step, who did it, and why · times in ET)</b>"), self.log])

    # ---- run the automatic part, showing each agent / rule step as it happens
    def start(self):
        self.msg = ""
        self.top.value = _banner_html(f"Processing <b>{_h.escape(self.src.value)}</b> …", "agent")
        self.form.children = ()
        self.log.value = ""
        try:
            raw = ORCH_SOURCES[self.src.value]()
            self.case = self.S.new_case(raw, use_llm=self.use_llm.value)
            self.case["orchestrator"] = True
        except Exception as e:
            self.top.value = _banner_html(f"Could not open the case: {_h.escape(str(e))}", "safe")
            return
        for i in range(1, len(self.case["audit"]) + 1):           # replay the automatic steps one by one
            self.log.value = audit_html({"audit": self.case["audit"][:i]})
            time.sleep(max(0.0, ORCH_STEP_DELAY_S))
        self.render()

    def _act(self, fn, ok_msg=None):
        """Run a person's submission; refusals (gates, rubber-stamp checks, deadlines) are shown, not raised."""
        try:
            fn()
            self.msg = ok_msg or self.S.last_message
            self._after()
            self.render()
        except ValueError as e:              # refused: keep the form as typed so the person can correct it
            self.msg = "⛔ " + str(e)
            self.render(rebuild_form=False)

    def _after(self):
        c = self.case
        if c["status"] == "IMPLEMENTED":                           # Outcome Evaluator runs automatically after a completed action
            self.S.evaluate_outcome(c, self.S.simulate_observed(c))

    # ---- screen
    def render(self, rebuild_form=True):
        c = self.case
        if c is None:
            return
        if c["status"] in REVIEW_STATUSES:
            self.S.enforce_deadline(c)
        A = c["analysis"]
        top = (f"<div style='font:13px sans-serif;margin:4px 0'><b>{c['case_id']}</b> · {_h.escape(c['scenario_name'])} · "
               f"rules v{c['policy_version']} · gate <b>{A['gate']}</b> — {_h.escape(A.get('gate_reason', ''))}</div>"
               + diagram_progress(c))
        if self.msg:
            top += _banner_html(_h.escape(self.msg), "safe" if self.msg.startswith("⛔") else "rule")
        self.top.value = top
        if rebuild_form:
            try:
                self.form.children = tuple(self._form_for(c))
            except Exception as e:                                  # never leave the person without a screen
                self.form.children = (_w_html(_banner_html(f"Could not draw the form: {_h.escape(type(e).__name__)}: "
                                                           f"{_h.escape(str(e))}", "safe")),)
        self.log.value = audit_html(c)

    def _who(self, label, default=""):
        return W.Combobox(options=ROSTER_NAMES, value=default, placeholder="name", description=label,
                          ensure_option=False, layout=W.Layout(width="360px"))

    def _review_inputs(self):
        reason = W.Textarea(placeholder="Reason (at least 15 characters)", layout=W.Layout(width="640px", height="60px"))
        checks = [W.Checkbox(value=False, description=t, indent=False, layout=W.Layout(width="640px")) for t in REVIEW_CHECKLIST]
        viewed = W.Checkbox(value=False, description="I viewed every alternative before deciding (rubber-stamp protection)",
                            indent=False, layout=W.Layout(width="640px"))
        return reason, checks, viewed

    def _form_for(self, c):
        """Return the widgets for whoever must act next on this case."""
        st, A = c["status"], c["analysis"]
        left = self.S.seconds_left(c)
        clock = (f" · ⏱ {int(left // 60)} min {int(left % 60)} s left (covers both reviewers)" if left is not None else "")
        out = []

        if st == "AWAITING_REVIEW":
            out.append(_w_html(_banner_html(f"🔔 <b>Human Review needed</b> — an operations supervisor or planner decides{clock}")))
            out.append(_w_html(_md_html(c["explanation"]["brief_md"])))
            if c["explanation"].get("narrative"):
                out.append(_w_html(_md_html(f"**Explanation Agent** ({c['explanation']['narrative_source']}, non-authoritative): "
                                            f"{c['explanation']['narrative']}")))
            who = self._who("Reviewer", ORCH_PEOPLE["first"])
            dec = W.ToggleButtons(options=["Approve", "Revise", "Reject"], description="Decision")
            opts = [(f"{r['rank']}. {r['label']} (Si {r['Si']:.3f})", r["option_id"]) for r in A["ranking"]]
            opt = W.Dropdown(options=opts or [("no feasible option", None)], description="Option", layout=W.Layout(width="640px"))
            reason, checks, viewed = self._review_inputs()
            budget = W.IntText(value=get_case_policy(c)["planner_defaults"]["budget_cap_usd"], description="Budget cap $")
            hold = W.IntText(value=get_case_policy(c)["planner_defaults"]["max_hold_min"], description="Max hold min")
            alt = W.Checkbox(value=False, description="Allow alternate depot", indent=False)
            excl = W.SelectMultiple(options=["dispatch_second_bus", "reroute", "protect_connection", "rider_alerts"],
                                    description="Exclude", rows=4)
            rev_box = W.VBox([W.HTML("<i>Revision constraints (used only for Revise; max two revisions)</i>"),
                              W.HBox([budget, hold, alt]), excl])
            submit = W.Button(description="Submit review", button_style="warning")
            expire = W.Button(description="Simulate: deadline passes", button_style="danger")

            def do(_):
                rev = {"budget_cap_usd": budget.value, "max_hold_min": hold.value, "allow_alternate_depot": alt.value,
                       "excluded_options": list(excl.value)} if dec.value == "Revise" else None
                self._act(lambda: self.S.submit_first(c, who.value, dec.value, opt.value, reason.value,
                                                      [cb.description for cb in checks if cb.value], viewed.value, rev))
            submit.on_click(do)
            expire.on_click(lambda _: self._expire(c))
            out.append(W.VBox([who, dec, opt, reason, *checks, viewed, rev_box, W.HBox([submit, expire])]))

        elif st == "AWAITING_SECOND_REVIEW":
            f = c["review"]["first"]
            out.append(_w_html(_banner_html(
                f"🔔 <b>Independent Second Reviewer needed</b> — {f['reviewer']} approved a high-impact action: "
                f"<b>{_h.escape(next(o['label'] for o in A['options'] if o['option_id'] == f['option_id']))}</b>{clock}. "
                "Must be a different person.")))
            out.append(_w_html(_md_html(c["explanation"]["brief_md"])))
            who = self._who("Reviewer 2", ORCH_PEOPLE["second"])
            dec = W.ToggleButtons(options=["Approve", "Reject"], description="Decision")
            reason, checks, viewed = self._review_inputs()
            submit = W.Button(description="Submit second review", button_style="warning")
            expire = W.Button(description="Simulate: deadline passes", button_style="danger")
            submit.on_click(lambda _: self._act(lambda: self.S.submit_second(c, who.value, dec.value, reason.value,
                                                                             [cb.description for cb in checks if cb.value], viewed.value)))
            expire.on_click(lambda _: self._expire(c))
            out.append(W.VBox([who, dec, reason, *checks, viewed, W.HBox([submit, expire])]))

        elif st == "AWAITING_IMPLEMENTATION":
            label = next(o["label"] for o in A["options"] if o["option_id"] == c["approved_option_id"])
            out.append(_w_html(_banner_html(f"🔔 <b>Implement Action</b> — operations staff carry out “{_h.escape(label)}” outside "
                                            "the prototype and record the result. Partial or failed actions go back to human review.")))
            staff = W.Text(value="Operations staff (depot dispatcher)", description="Staff", layout=W.Layout(width="420px"))
            res = W.ToggleButtons(options=["Completed", "Partial", "Failed"], description="Result")
            notes = W.Textarea(placeholder="Notes (required for Partial or Failed)", layout=W.Layout(width="640px", height="50px"))
            submit = W.Button(description="Record result", button_style="warning")
            submit.on_click(lambda _: self._act(lambda: self.S.record_implementation(c, staff.value, res.value, notes.value)))
            out.append(W.VBox([staff, res, notes, submit]))

        elif st == "CONFLICT_REVIEW":
            q = [e for e in A["evidence"] if e["status"] == "quarantined"]
            out.append(_w_html(_banner_html("🔔 <b>Operator review needed</b> — sources disagree. Quarantined: "
                                            + "; ".join(f"{e['source_id']} ({_h.escape(e['note'])})" for e in q))))
            who = W.Text(value="Operator on duty (control centre)", description="Operator", layout=W.Layout(width="420px"))
            res = W.ToggleButtons(options=["resolved", "unresolved"], description="Result",
                                  tooltips=["Accept source precedence (vehicle data wins)", "Cannot tell which source is right"])
            submit = W.Button(description="Submit operator check", button_style="warning")
            submit.on_click(lambda _: self._act(lambda: self.S.operator_resolve(c, who.value, res.value)))
            out.append(W.VBox([who, res, submit]))

        elif st in ("MONITORING", "SAFE_DEFAULT"):
            kind = "Continue monitoring" if st == "MONITORING" else "Safe default"
            out.append(_w_html(_banner_html(f"<b>{kind}</b> — {_h.escape(c.get('safe_default_reason') or A.get('gate_reason', ''))}. "
                                            "Verified rider information only; no dispatch, rerouting or live control.", "safe")))
            if c.get("rider_alert"):
                out.append(_w_html(_md_html(f"**Rider alert:** {c['rider_alert']}")))
            b1 = W.Button(description="Recheck (feed changed)")
            b2 = W.Button(description="Re-escalate (conditions worsen)", button_style="danger")
            b3 = W.Button(description="Evaluate outcome")
            who = self._who("Planner", ORCH_PEOPLE["planner"])
            why = W.Text(placeholder="reason for closing as a false alarm", layout=W.Layout(width="420px"))
            b4 = W.Button(description="Close as false alarm")
            b1.on_click(lambda _: self._act(lambda: self.S.monitor_action(c, "recheck")))
            b2.on_click(lambda _: self._act(lambda: self.S.monitor_action(c, "re-escalate")))
            b3.on_click(lambda _: self._act(lambda: self.S.evaluate_outcome(c, self.S.simulate_observed(c))))
            b4.on_click(lambda _: self._act(lambda: self.S.close_false_alarm(c, who.value, why.value)))
            out.append(W.VBox([W.HBox([b1, b2, b3]), W.HBox([who, why, b4])]))
            if c.get("outcome"):
                out += self._learning_loop(c)

        elif st in ("EVALUATED", "CLOSED_FALSE_ALARM"):
            o = c.get("outcome") or {}
            if "delay_error_min" in o:
                out.append(_w_html(_banner_html(
                    f"<b>Outcome Evaluator</b> (separate from the recommender): predicted vs observed delay error "
                    f"{o['delay_error_min']:+.1f} min · false alarm {o['false_alarm']} · policy weights unchanged.", "agent")))
                out.append(_w_html(pd.DataFrame([{k: v for k, v in o.items() if not isinstance(v, (dict, list))}])
                                   .to_html(index=False, border=0)))
            out += self._learning_loop(c)
        return out

    def _expire(self, c):
        c["review"]["deadline_at"] = iso(utcnow() - timedelta(seconds=1))
        self.S.enforce_deadline(c)
        self.msg = "Deadline passed with no complete response → Safe Default."
        self.render()

    # ---- learning loop: Insight Agent → Policy Owner → backtest → new rule version
    def _learning_loop(self, c):
        """Widgets for the learning loop. Proposal details and the backtest are shown in an HTML widget, not an Output."""
        head = W.HTML("<h4 style='margin:10px 0 4px'>Learning loop · Insight Agent → Policy Owner → backtest → rule version</h4>")
        run = W.Button(description="Run Insight Agent")

        def run_insight(_):
            props = insight_agent(self.S)
            self.msg = f"Insight Agent proposed {len(props)} change(s). Nothing changes until the policy owner approves."
            self.render()
        run.on_click(run_insight)
        if not self.S.proposals:
            return [head, run, _w_html(f"<i>No open proposals. Current rules v{self.S.PM.current['version']}.</i>")]
        pick = W.Dropdown(options=[(f"{p['id']} · {p['title']}", p["id"]) for p in self.S.proposals],
                          description="Proposal", layout=W.Layout(width="640px"))
        info = W.HTML()
        bt = W.Button(description="Preview backtest")

        def detail(_=None):
            p = next(x for x in self.S.proposals if x["id"] == pick.value)
            info.value = _md_html(f"**{p['title']}** — {p['rationale']} _(evidence: {p['evidence']})_\n\n"
                                  + "\n".join(f"- `{x['path']}`: {x['old']} → {x['new']}" for x in p["changes"]))

        def preview(_):
            p = next(x for x in self.S.proposals if x["id"] == pick.value)
            _, summary = run_backtest(self.S, apply_changes(self.S.PM.current, p["changes"]))
            detail()
            info.value += _md_html(f"**Backtest:** {summary}")
        pick.observe(detail, "value")
        bt.on_click(preview)
        who = self._who("Owner", ORCH_PEOPLE["owner"])
        dec = W.ToggleButtons(options=["Approve", "Reject"], description="Decision")
        why = W.Text(placeholder="reason (≥ 10 characters)", layout=W.Layout(width="520px"))
        go = W.Button(description="Submit policy decision", button_style="warning")

        def decide(_):
            pid = pick.value
            try:
                decide_proposal(self.S, pid, who.value, dec.value, why.value)
                c["_policy_step"] = f"{pid} {dec.value.lower()}d → rules v{self.S.PM.current['version']}"
                self.msg = self.S.last_message
                self.render()
            except ValueError as e:
                self.msg = "⛔ " + str(e)
                self.render(rebuild_form=False)
        go.on_click(decide)
        detail()
        return [head, run, W.VBox([pick, bt, info, who, dec, why, go])]


ORCH = Orchestrator(APP)
display(ORCH.ui)
print("Pick a case and press ▶ Process case. AI agents and rules run automatically; the run stops at each box where a person decides.")


# ====================================================================================================
# notebook cell 72: O · Record of orchestrator cases (re-run after finishing a case so the record is saved in the notebook)
# ====================================================================================================
#@title O · Record of orchestrator cases (re-run after finishing a case so the record is saved in the notebook)
_orch_cases = [c for c in APP.cases.values() if c.get("orchestrator")]
if not _orch_cases:
    print("No orchestrator case yet. Process a case above, finish it, then run this cell again.")
for _c in _orch_cases:
    display(HTML(f"<h4 style='margin:12px 0 4px'>{_c['case_id']} · {_h.escape(_c['scenario_name'])} · final status "
                 f"<b>{_c['status']}</b> · rules v{_c['policy_version']}</h4>"))
    display(HTML(diagram_progress(_c)))
    display(HTML(audit_html(_c)))


# ====================================================================================================
# notebook cell 74: F · Scenario gallery: every path through the workflow, with scripted reviewers
# ====================================================================================================
#@title F · Scenario gallery: every path through the workflow, with scripted reviewers
RUN_SCENARIO_GALLERY = True  #@param {type:"boolean"}
GALLERY_USE_LLM = False  #@param {type:"boolean"}
# GALLERY_USE_LLM = False keeps the gallery fast and free: the live case still uses the LLM, the synthetic cases use
# template narratives. Scores, gates and decisions are identical either way (the LLM never enters them).
from datetime import timedelta

GALLERY = []                 # one summary row per scenario
GALLERY_CASES = {}           # scenario id → case record
PEOPLE = {"first": "Ana Lopez", "second": "Chen Wei", "planner": "Ben Carter", "owner": "Dana Brooks",
          "untrained": "Eli Turner", "conflicted": "Gabe Ortiz"}
OPS_STAFF = "Operations staff (depot dispatcher)"
OPERATOR = "Operator on duty (control centre)"
GOOD_REASON = "Checked the verified evidence and every alternative; this option best protects access for affected riders."


def _think(case, seconds, key="started_at"):
    """Scripted reviewers answer instantly; back-date the review start so response times look like real ones."""
    case["review"][key] = iso(utcnow() - timedelta(seconds=seconds))


def _blocked(fn, *a, **k):
    """Run a submission that should be refused; return the refusal text (or None if it was accepted)."""
    try:
        fn(*a, **k)
        return None
    except ValueError as e:
        return str(e)


def _top(case):
    r = case["analysis"]["ranking"]
    return r[0]["option_id"] if r else None


def _label(case, oid):
    return next((o["label"] for o in case["analysis"]["options"] if o["option_id"] == oid), oid)


def first_review(case, name, decision="Approve", option=None, reason=GOOD_REASON, checklist=None, viewed=True,
                 think=95, revision=None):
    _think(case, think)
    APP.submit_first(case, name, decision, option or _top(case), reason,
                     REVIEW_CHECKLIST if checklist is None else checklist, viewed, revision)


def second_review(case, name, decision="Approve", reason="Independent check: evidence, cost and equity impact confirmed.",
                  think=80):
    _think(case, think, "second_requested_at")
    APP.submit_second(case, name, decision, reason, REVIEW_CHECKLIST, True)


def implement(case, result="Completed", notes=""):
    APP.record_implementation(case, OPS_STAFF, result, notes)


def evaluate(case):
    APP.evaluate_outcome(case, APP.simulate_observed(case))
    o = case["outcome"]
    if "delay_error_min" in o:
        return (f"Outcome Evaluator: predicted vs observed delay error {o['delay_error_min']:+.1f} min, "
                f"false alarm = {o['false_alarm']} (policy weights unchanged)")
    return f"Outcome Evaluator: rider information only; false alarm = {o.get('false_alarm')}"


def approve_to_outcome(case, steps, first=None, second=None, option=None, think=95):
    """Approve → (second reviewer if high impact) → implement → evaluate. Appends a line per step."""
    first, second = first or PEOPLE["first"], second or PEOPLE["second"]
    oid = option or _top(case)
    first_review(case, first, option=oid, think=think)
    steps.append(f"Human Review: {first} approved “{_label(case, oid)}”"
                 + (" (override of the rank-1 recommendation)" if oid != _top(case) else " (rank-1 option)"))
    if case["status"] == "AWAITING_SECOND_REVIEW":
        second_review(case, second)
        steps.append(f"High-impact action → independent Second Reviewer {second} approved")
    else:
        steps.append("Not high impact → no second reviewer required")
    implement(case)
    steps.append(f"Implement Action: {OPS_STAFF} recorded Completed")
    steps.append(evaluate(case))


def open_case(raw, use_llm=None):
    return APP.new_case(raw, use_llm=GALLERY_USE_LLM if use_llm is None else use_llm)


def gate_line(case):
    A = case["analysis"]
    C = (A.get("confidence") or {}).get("C")
    s = A.get("scores") or {}
    trig = ", ".join(t["trigger"] for t in A.get("hard_triggers", [])) or "none"
    return (f"Rules: C = {C} · R = {s.get('R')} · P = {s.get('P')} · hard triggers: {trig} → gate {A['gate']}"
            + (f" ({A['gate_reason']})" if A.get("gate_reason") else ""))


def record(sid, title, path, case, steps):
    GALLERY_CASES[sid] = case
    GALLERY.append({"#": sid, "scenario": title, "diagram path shown": path, "case": case["case_id"],
                    "gate": case["analysis"]["gate"], "final status": case["status"],
                    "people": ", ".join(dict.fromkeys(d["reviewer"] for d in case["decisions"])) or "—",
                    "rules": case["policy_version"]})
    case["demo_script"] = sid
    print(f"\n━━ {sid} · {title}  [{case['case_id']}]")
    print(f"   Diagram path: {path}")
    for s in steps:
        print(f"   • {s}")
    print(f"   ⇒ Final status: {case['status']} — {STATUS_HELP.get(case['status'], '')}")


def scenario(fn):
    """Run one scenario; a failure is printed and the gallery continues."""
    try:
        fn()
    except Exception as e:
        print(f"\n━━ {fn.__name__}: could not finish ({type(e).__name__}: {str(e)[:300]})")
    return fn


# ------------------------------------------------------------------ the scenarios
def s01_live():
    raw = copy.deepcopy(_raw_a)
    case = open_case(raw, use_llm=LLM_READY)
    f = case["analysis"].get("fields", {})
    steps = [f"Live hand-off: route {raw.get('external_route_id')} · delay {f.get('delay_min')} min · time period "
             f"{raw.get('time_period')} · weather {raw['weather_context'].get('conditions')} · "
             f"camera count {raw['vision'].get('waiting_at_stop')} waiting",
             gate_line(case)]
    st = case["status"]
    if st == "AWAITING_REVIEW":
        steps.append(f"Explanation brief ready ({case['explanation']['narrative_source']}); review timer "
                     f"{case['review']['deadline_min']} min ({case['review']['tier']})")
        approve_to_outcome(case, steps)
        path = "Live inputs → verification → confidence → impact → significant → planner → evaluator → review → implement → outcome"
    elif st == "SAFE_DEFAULT":
        steps.append("Not significant tonight → Safe Default: verified rider information, no dispatch, monitoring continues")
        steps.append(evaluate(case))
        path = "Live inputs → verification → confidence → impact → below threshold → Safe Default → outcome"
    elif st == "MONITORING":
        steps.append("Continue Monitoring: the case is kept and re-checked when the feed changes (see scenario 11)")
        path = "Live inputs → verification → monitoring"
    else:
        path = "Live inputs → " + st
    record("01", "LIVE · tonight's NJ Transit hand-off", path, case, steps)


def s02_two_person():
    case = open_case(make_raw(PRESETS["Severe · R2 Hospital Line trip cancelled (verified)"],
                              "Gallery · severe cancellation on the Hospital Line"))
    steps = [gate_line(case), f"Severe tier → {case['review']['deadline_min']}-minute deadline covering both reviewers"]
    first_review(case, PEOPLE["first"])
    steps.append(f"Human Review: {PEOPLE['first']} approved “{_label(case, _top(case))}” (rank 1)")
    if case["status"] == "AWAITING_SECOND_REVIEW":
        msg = _blocked(second_review, case, PEOPLE["first"])
        steps.append(f"Same person tries to act as second reviewer → BLOCKED: {msg}")
        second_review(case, PEOPLE["second"])
        steps.append(f"Independent Second Reviewer {PEOPLE['second']} approved")
    implement(case)
    steps.append(f"Implement Action: Completed")
    steps.append(evaluate(case))
    record("02", "Severe cancellation · two-person approval", "Hard trigger → review → second reviewer (independence check) → implement → outcome",
           case, steps)


def s03_safeguards():
    case = open_case(make_raw(PRESETS["Significant · R1 Crosstown unboardable crowding at peak"],
                              "Gallery · unboardable crowding on the Crosstown"))
    steps = [gate_line(case)]
    msg = _blocked(first_review, case, PEOPLE["first"], reason="ok", checklist=[], viewed=False, think=4)
    steps.append(f"Rubber-stamp attempt (4 s, no checklist, 2-letter reason, alternatives not viewed) → BLOCKED: {msg}")
    msg = _blocked(first_review, case, PEOPLE["untrained"])
    steps.append(f"Ethics-training gate (Section C): {PEOPLE['untrained']} → BLOCKED: {msg}")
    depot_b = [o for o in case["analysis"]["ranking"] if any("Metro Coach" in p for p in
               next(x for x in case["analysis"]["options"] if x["option_id"] == o["option_id"])["parties"])]
    if not depot_b:
        first_review(case, PEOPLE["planner"], decision="Revise", revision={"allow_alternate_depot": True})
        steps.append(f"Revision 1 of 2: {PEOPLE['planner']} allowed the alternate depot (Depot B, a contract operator); "
                     "options rebuilt and rescored with unchanged rules")
        depot_b = [o for o in case["analysis"]["ranking"] if any("Metro Coach" in p for p in
                   next(x for x in case["analysis"]["options"] if x["option_id"] == o["option_id"])["parties"])]
    if depot_b:
        msg = _blocked(first_review, case, PEOPLE["conflicted"], option=depot_b[0]["option_id"])
        steps.append(f"Conflict-of-interest gate (Section D): {PEOPLE['conflicted']} approving “{depot_b[0]['label']}” "
                     f"→ BLOCKED: {msg}")
    approve_to_outcome(case, steps, first=PEOPLE["planner"])
    steps.append(f"{PEOPLE['planner']} is AT_RISK (refresher due soon): allowed, and the note is kept in the audit record")
    record("03", "Rubber-stamp protection + ethics and conflict-of-interest gates",
           "Review blocked ×3 (rubber stamp, training, conflict) → valid review → implement → outcome", case, steps)


def s04_override_failed():
    case = open_case(make_raw(PRESETS["Hard-trigger bypass · Verified emergency closure on R3, sparse data"],
                              "Gallery · emergency closure with sparse data"))
    C = case["analysis"]["confidence"]["C"]
    steps = [gate_line(case),
             f"C = {C} is {'below' if C < APP.PM.current['confidence_threshold'] else 'above'} 0.70, but a verified official "
             "hard trigger bypasses the confidence gate"]
    ranking = case["analysis"]["ranking"]
    alt = ranking[1]["option_id"] if len(ranking) > 1 else _top(case)
    first_review(case, PEOPLE["first"], option=alt)
    steps.append(f"Human Review: {PEOPLE['first']} chose “{_label(case, alt)}” instead of the rank-1 option "
                 "(override logged for the audit)")
    if case["status"] == "AWAITING_SECOND_REVIEW":
        second_review(case, PEOPLE["second"])
        steps.append(f"Second Reviewer {PEOPLE['second']} approved")
    implement(case, "Failed", "Detour signage crew could not reach the closure; buses still on the blocked segment.")
    steps.append("Implement Action: FAILED → case returned to Human Review with a new timer")
    alerts = next((o["option_id"] for o in case["analysis"]["ranking"] if o["option_id"] == "rider_alerts"), _top(case))
    first_review(case, PEOPLE["first"], option=alerts, reason="Detour failed; push verified rider alerts while crews reposition.")
    steps.append(f"Second round: {PEOPLE['first']} approved “{_label(case, alerts)}”")
    if case["status"] == "AWAITING_SECOND_REVIEW":
        second_review(case, PEOPLE["second"])
    implement(case)
    steps.append("Implement Action: Completed")
    steps.append(evaluate(case))
    record("04", "Confidence bypass · reviewer override · failed action returns to review",
           "Official hard trigger bypasses C → override → implement FAILED → back to review → implement → outcome", case, steps)


def s05_revisions():
    case = open_case(make_raw(PRESETS["Moderate · R4 Northside winter-weather delays"], "Gallery · winter-weather delays"))
    steps = [gate_line(case)]
    if case["status"] != "AWAITING_REVIEW":
        record("05", "Revision loop (max two, then safe default)", "not significant in this run", case, steps)
        return
    for i, rev in enumerate([{"budget_cap_usd": 300}, {"excluded_options": ["dispatch_second_bus"]}], 1):
        first_review(case, PEOPLE["planner"], decision="Revise", revision=rev,
                     reason=f"Revision {i}: try a cheaper plan before committing resources.")
        steps.append(f"Revision {i} of 2 ({json.dumps(rev)}): rescored → "
                     + " > ".join(f"{r['label']} ({r['Si']:.3f})" for r in case["analysis"]["ranking"]))
    first_review(case, PEOPLE["planner"], decision="Revise", revision={"max_hold_min": 2},
                 reason="Third revision request to test the loop limit.")
    steps.append("Third revision request → Safe Default (the loop is capped at two)")
    record("05", "Revision loop (max two, then safe default)", "Review → revise → revise → third revise → Safe Default", case, steps)


def s06_reject():
    spec = dict(route_id="R4", stop_range=(1, 6), time_period="peak", delay_min=19, headway_ratio=2.2, boardings=95,
                load_factor=0.8, sources=["vehicle_apc", "official_alert"], event_type="mechanical", event_severity="minor",
                event_verified=True, official_delay=19)
    case = open_case(make_raw(spec, "Gallery · 19-minute mechanical delay, reviewer rejects"))
    steps = [gate_line(case)]
    first_review(case, PEOPLE["first"], decision="Reject", think=12,
                 reason="Operator reports the bus is moving again; resources are not needed.")
    steps.append(f"Human Review: {PEOPLE['first']} REJECTED after 12 s (a fast response the rubber-stamp audit flags)")
    steps.append("Reject → Safe Default: rider information only, monitoring continues")
    record("06", "Reviewer rejects", "Review → reject → Safe Default", case, steps)


def s07_deadline():
    spec = dict(route_id="R2", stop_range=(3, 8), time_period="peak", delay_min=22, headway_ratio=2.4, cancelled=True,
                boardings=120, load_factor=0.9, sources=["vehicle_apc", "official_alert"], event_type="mechanical",
                event_severity="major", event_verified=True, official_delay=22)
    case = open_case(make_raw(spec, "Gallery · nobody responds before the deadline"))
    steps = [gate_line(case), f"Timer: {case['review']['deadline_min']} min ({case['review']['tier']} tier)"]
    case["review"]["deadline_at"] = iso(utcnow() - timedelta(seconds=1))     # simulate the deadline passing
    APP.enforce_deadline(case)
    steps.append("No complete response before the deadline → Safe Default (no action is taken by default)")
    msg = _blocked(first_review, case, PEOPLE["first"])
    steps.append(f"Late approval attempt → BLOCKED: {msg}")
    record("07", "Deadline expires", "Review timer expires → Safe Default", case, steps)


def s08_below():
    case = open_case(make_raw(PRESETS["Below threshold · R5 Southside 4-minute delay"], "Gallery · 4-minute delay"))
    steps = [gate_line(case), "No hard trigger and R < 0.50 → Safe Default automatically (no person needed)",
             evaluate(case)]
    record("08", "Below significance threshold", "Impact → not significant → Safe Default → outcome", case, steps)


def s09_conflict_resolved():
    case = open_case(make_raw(PRESETS["Conflicting data · R3 vehicle feed vs. public reports"], "Gallery · conflicting data, resolved"))
    steps = [gate_line(case), "Public reports disagree with the vehicle feed by > 8 min → quarantined, C reduced"]
    APP.operator_resolve(case, OPERATOR, "resolved")
    steps.append(f"Operator review: accepted source precedence (vehicle data wins) → re-analysed")
    steps.append(gate_line(case))
    if case["status"] == "SAFE_DEFAULT":
        steps.append(evaluate(case))
    record("09", "Conflicting data · operator resolves", "Verification → conflict → operator review → resolved → re-analysis",
           case, steps)


def s10_conflict_unresolved():
    case = open_case(make_raw(PRESETS["Conflicting data · R3 vehicle feed vs. public reports"], "Gallery · conflicting data, unresolved"))
    steps = [gate_line(case)]
    APP.operator_resolve(case, OPERATOR, "unresolved")
    steps.append("Operator check 1 of 2: unresolved")
    APP.operator_resolve(case, OPERATOR, "unresolved")
    steps.append("Operator check 2 of 2: unresolved → Continue Monitoring")
    record("10", "Conflicting data · unresolved after two checks", "Conflict → operator ×2 → Continue Monitoring", case, steps)


def s11_outage_reescalate():
    case = open_case(make_raw(PRESETS["Feed outage · R5 vehicle feed stale (no backup)"], "Gallery · feed outage, then worsens"))
    steps = [gate_line(case), "Vehicle feed older than the 300 s freshness limit and no backup → snapshot frozen, Continue Monitoring"]
    APP.monitor_action(case, "re-escalate")
    steps.append("Conditions worsened (fresh feed: delay +8 min, gap +0.6×, pass-ups +8) → RE-ESCALATE → re-analysed")
    steps.append(gate_line(case))
    if case["status"] == "AWAITING_REVIEW":
        steps.append("Left open on purpose: finish this case live in the dashboard (tab 3) during the demo")
    record("11", "Feed outage → monitoring → re-escalation", "Feed outage (no backup) → monitoring → re-escalate → review", case, steps)


def s12_outage_backup():
    case = open_case(make_raw(PRESETS["Feed outage · R1 vehicle feed stale (backup available)"], "Gallery · feed outage with backup"))
    steps = [gate_line(case), "Vehicle feed stale → backup source found → re-verified and analysis continues"]
    if case["status"] == "AWAITING_REVIEW":
        approve_to_outcome(case, steps)
    record("12", "Feed outage with a backup source", "Feed outage → backup check YES → re-verify → review → implement → outcome",
           case, steps)


def s13_social():
    case = open_case(make_raw(PRESETS["Uncorroborated social report only · R4"], "Gallery · social media report only"))
    steps = [gate_line(case), "An uncorroborated public report cannot trigger action → Continue Monitoring"]
    APP.close_false_alarm(case, PEOPLE["planner"], "No vehicle or official data confirms the reported crash.")
    steps.append(f"{PEOPLE['planner']} closed the case as a false alarm (reason logged)")
    record("13", "Uncorroborated social report", "Verification excludes report → monitoring → closed as false alarm", case, steps)


def s14_agent_error():
    case = open_case(make_raw(PRESETS["Agent error · Vision Agent fails twice on R2"], "Gallery · Vision Agent fails twice"))
    steps = [f"Rules: gate {case['analysis']['gate']} ({case['analysis']['gate_reason']})",
             "Retry once with the same inputs; second failure → deterministic safe mode, evidence kept, "
             "technical owner and planner alerted → Safe Default"]
    record("14", "Agent error", "Agent fails twice → safe mode → Safe Default", case, steps)


def s15_policy_loop():
    print("\n━━ 15 · Learning loop: Insight Agent → Policy Owner → backtest → new rule version")
    props = insight_agent(APP)
    print(f"   • Insight Agent read {sum(1 for c in APP.all_cases() if c.get('outcome'))} evaluated cases and proposed "
          f"{len(props)} change(s): " + ("; ".join(p["title"] for p in props) or "none"))
    if not props:
        print("   • Nothing proposed → current rules kept")
        return
    p0 = props[0]
    msg = _blocked(decide_proposal, APP, p0["id"], PEOPLE["untrained"], "Approve", "Looks reasonable to me, approve it.")
    print(f"   • {PEOPLE['untrained']} tries to approve {p0['id']} → BLOCKED: {msg}")
    newp = apply_changes(APP.PM.current, p0["changes"])
    _, bt = run_backtest(APP, newp)
    print(f"   • Backtest preview of {p0['id']} “{p0['title']}”: {bt}")
    decide_proposal(APP, p0["id"], PEOPLE["owner"], "Reject",
                    "Not enough recent evidence; revisit after the winter service change.")
    print(f"   • Policy Owner {PEOPLE['owner']} REJECTED {p0['id']} → reason returned to the Insight Agent, rules stay "
          f"v{APP.PM.current['version']}")
    again = insight_agent(APP)
    print(f"   • Insight Agent re-run: the rejected change is not re-proposed ({len(again)} proposal(s) now)")
    if again:
        p1 = again[0]
        decide_proposal(APP, p1["id"], PEOPLE["owner"], "Approve",
                        "Backtest shows fewer false alarms with no real disruption lost.")
        print(f"   • Policy Owner {PEOPLE['owner']} APPROVED {p1['id']} “{p1['title']}” → backtested and released as rules "
              f"v{APP.PM.current['version']} (rollback to v1.0.0 available in dashboard tab 7)")
    GALLERY.append({"#": "15", "scenario": "Learning loop · policy change rejected, then one approved",
                    "diagram path shown": "Outcome → Insight → Policy Owner (reject / approve) → backtest → new rule version",
                    "case": "—", "gate": "—", "final status": f"rules v{APP.PM.current['version']}",
                    "people": ", ".join([PEOPLE["untrained"], PEOPLE["owner"]]), "rules": APP.PM.current["version"]})


if RUN_SCENARIO_GALLERY:
    _old = [cid for cid, case in APP.cases.items() if case.get("demo_script")]
    for cid in _old:                                   # a re-run replaces the earlier gallery cases instead of adding to them
        APP.cases.pop(cid)
    if _old:
        print(f"Removed {len(_old)} cases from an earlier gallery run, so nothing is counted twice.")
    # Scenarios 1 to 14 run on the published rules (the first version), so each path behaves as documented even if a
    # policy change was approved earlier in this session (for example in the orchestrator). The rules in force are put back
    # before scenario 15, whose policy decision then builds on them.
    _rules_in_force = APP.PM.current_index
    APP.PM.current_index = 0
    try:
        print(f"Scenario gallery · published rules v{APP.PM.current['version']} (rules in force this session: "
              f"v{APP.PM.versions[_rules_in_force]['version']}) · scripted people: "
              + ", ".join(f"{k} = {v}" for k, v in PEOPLE.items()))
        for fn in [s01_live, s02_two_person, s03_safeguards, s04_override_failed, s05_revisions, s06_reject, s07_deadline,
                   s08_below, s09_conflict_resolved, s10_conflict_unresolved, s11_outage_reescalate, s12_outage_backup,
                   s13_social, s14_agent_error]:
            scenario(fn)
    finally:
        APP.PM.current_index = _rules_in_force
    scenario(s15_policy_loop)
    GALLERY_DF = pd.DataFrame(GALLERY)
    print("\n")
    display(GALLERY_DF)
    print("Rubber-stamp audit (all reviewers, history + this session):")
    display(APP.rubber_stamp_report())
    GALLERY_DF.to_csv(os.path.join(OUTPUT_DIR, "scenario_gallery.csv"), index=False)
    pd.DataFrame([e for c in GALLERY_CASES.values() for e in c["audit"]]).to_csv(
        os.path.join(OUTPUT_DIR, "scenario_gallery_audit.csv"), index=False)
    print("Saved:", os.path.join(OUTPUT_DIR, "scenario_gallery.csv"), "and scenario_gallery_audit.csv")
else:
    GALLERY_DF = pd.DataFrame()
    print("Scenario gallery not run (RUN_SCENARIO_GALLERY is off).")


# ====================================================================================================
# notebook cell 76: V1. Evaluation settings, helpers and readiness check
# ====================================================================================================
#@title V1. Evaluation settings, helpers and readiness check
EV_USE_LIVE = True                  #@param {type:"boolean"}
EV_SNAPSHOT_PATH = ""               #@param {type:"string"}
EV_MAX_REAL_CASES = 60              #@param {type:"integer"}
EV_NJ_BBOX = (38.8, 41.4, -75.6, -73.8)   # south, north, west, east: a position outside this box is an error
EV_BUSDV_RADIUS = 2000              #@param {type:"integer"}
EV_BUSDV_MODE = "ALL"               #@param {type:"string"}
EV_BUSDV_STOP = ""                  #@param {type:"string"}
EV_ALERT_LABELS_CSV = ""            #@param {type:"string"}
import os, re, json, copy, math, statistics
from datetime import datetime, timezone, timedelta
import pandas as pd
import requests

EV_DIR = os.path.join(OUTPUT_DIR, "evaluation")
os.makedirs(EV_DIR, exist_ok=True)

# Names this section reads from the rest of the notebook. If one is missing, run the notebook from the top first.
EV_NEEDS = ["APP", "PM", "run_analysis", "make_raw", "PRESETS", "SPEC_DEFAULTS", "scenario_weights", "LIVE_CONTEXT",
            "section_a_sample_inputs", "build_signal_scout_input", "signal_scout_input_to_raw", "extract_delay",
            "classify_cause", "CAUSE_KEYWORDS", "CV_RESULTS", "CV_SOURCES", "read_secret", "km_between",
            "SERVICE_AREA_LATLON", "explanation_brief", "utcnow", "iso", "parse_iso"]
EV_MISSING = [n for n in EV_NEEDS if n not in globals()]
if EV_MISSING:
    raise RuntimeError("Run the whole notebook from the top before Section V. Missing: " + ", ".join(EV_MISSING))

EV_RESULTS = []          # one row per check: part, check, expected, actual, result, data


def ev_check(part, check, expected, actual, passed, data="", note=""):
    """Record one check. passed is True, False, None (skipped, with the reason in note) or "info" (a number we only
    report, such as a system metric, which has no pass mark)."""
    result = "reported" if passed == "info" else ("skipped" if passed is None else ("pass" if passed else "FAIL"))
    EV_RESULTS.append({"part": part, "check": check, "expected": str(expected), "actual": str(actual),
                       "result": result, "data": data, "note": note})
    return passed


def ev_part_table(part):
    rows = [r for r in EV_RESULTS if r["part"] == part]
    if rows:
        display(pd.DataFrame(rows)[["check", "expected", "actual", "result", "data", "note"]])
    n_pass = sum(r["result"] == "pass" for r in rows)
    n_fail = sum(r["result"] == "FAIL" for r in rows)
    n_skip = sum(r["result"] == "skipped" for r in rows)
    n_rep = sum(r["result"] == "reported" for r in rows)
    print(f"{part}: {n_pass} passed, {n_fail} failed, {n_skip} skipped" + (f", {n_rep} reported" if n_rep else ""))


def ev_clear(part):
    EV_RESULTS[:] = [r for r in EV_RESULTS if r["part"] != part]


def ev_pct(x, n):
    return None if not n else round(100 * x / n, 1)


# Expected answers come from the published rules (the first version, from the v5 document). Section F's learning loop
# may have released a newer version during this run, which only the gallery cases (V8) use.
EV_POLICY = APP.PM.versions[0]["policy"] if getattr(APP.PM, "versions", None) else APP.PM.current
print(f"Evaluation ready. Expected answers use published rules v{EV_POLICY['version']} "
      f"(current rules: v{APP.PM.current['version']}). Output folder {EV_DIR}")
print(f"Live data allowed: {EV_USE_LIVE and globals().get('LIVE', False)} (A1 DATA_MODE = {globals().get('DATA_MODE')})")

# ====================================================================================================
# notebook cell 78: V2. Real feed snapshot and data quality checks
# ====================================================================================================
#@title V2. Real feed snapshot and data quality checks
ev_clear("V2")


def ev_vehicle_rows(vp, tu):
    """Every vehicle in the feed, joined to its trip update, in the same shape A1 uses for the selected bus."""
    lat0, lon0 = SERVICE_AREA_LATLON
    updates = {}
    for ent in (tu.entity if tu is not None else []):
        if ent.HasField("trip_update"):
            t = ent.trip_update
            delays = [s.arrival.delay if s.HasField("arrival") else s.departure.delay
                      for s in t.stop_time_update if s.HasField("arrival") or s.HasField("departure")]
            updates[t.trip.trip_id] = {"delay_s": delays[0] if delays else (t.delay if t.HasField("delay") else None),
                                       "cancelled": t.trip.schedule_relationship == t.trip.CANCELED,
                                       "route_id": t.trip.route_id}
    rows = []
    for ent in vp.entity:
        if not ent.HasField("vehicle"):
            continue
        v = ent.vehicle
        u = updates.get(v.trip.trip_id, {})
        lat, lon = v.position.latitude, v.position.longitude
        rows.append({"route_id": v.trip.route_id or u.get("route_id") or "", "trip_id": v.trip.trip_id,
                     "vehicle_id": v.vehicle.id, "lat": lat, "lon": lon,
                     "distance_km": round(km_between((lat, lon), (lat0, lon0)), 2),
                     "delay_s": u.get("delay_s"), "cancelled": bool(u.get("cancelled", False)),
                     "has_trip_update": v.trip.trip_id in updates,
                     "timestamp": int(v.timestamp or vp.header.timestamp or 0)})
    return rows, updates


def ev_alert_rows(al):
    out = []
    for ent in (al.entity if al is not None else []):
        if ent.HasField("alert"):
            a = ent.alert
            txt = lambda ts: ts.translation[0].text.strip() if len(ts.translation) else ""
            out.append({"alert_id": ent.id, "routes": sorted({ie.route_id for ie in a.informed_entity if ie.route_id}),
                        "stops": sorted({ie.stop_id for ie in a.informed_entity if ie.stop_id}),
                        "agency_wide": any(ie.agency_id and not ie.route_id and not ie.stop_id for ie in a.informed_entity),
                        "header": txt(a.header_text), "description": txt(a.description_text)[:600],
                        "effect": a.Effect.Name(a.effect) if a.effect else None})
    return out


def ev_take_snapshot():
    """Live GTFSG2 snapshot through A1's login and feed functions. Returns (snapshot, status)."""
    if not (EV_USE_LIVE and globals().get("LIVE")):
        return None, "live data switched off"
    if not _NJT.get("token"):
        status = njt_authenticate()
        if not _NJT.get("token"):
            return None, f"NJ Transit login: {status}"
    feeds, status = {}, {}
    for ep in ("getVehiclePositions", "getTripUpdates", "getAlerts"):
        try:
            feeds[ep] = njt_feed(ep)
            status[ep] = f"ok ({len(feeds[ep].entity)} entities)"
        except Exception as e:
            status[ep] = f"failed ({type(e).__name__})"
    vp = feeds.get("getVehiclePositions")
    if vp is None:
        return None, f"vehicle feed unavailable: {status}"
    vehicles, updates = ev_vehicle_rows(vp, feeds.get("getTripUpdates"))
    snap = {"taken_at": iso(utcnow()), "system": _NJT.get("system"), "feed_status": status,
            "header_timestamp": int(vp.header.timestamp or 0), "vehicles": vehicles,
            "trip_updates": len(updates), "alerts": ev_alert_rows(feeds.get("getAlerts")), "source": "live"}
    return snap, "live"


EV_SNAPSHOT, EV_SNAPSHOT_STATUS = ev_take_snapshot()
if EV_SNAPSHOT:
    path = os.path.join(EV_DIR, f"njt_snapshot_{EV_SNAPSHOT['taken_at'][:19].replace(':', '')}.json")
    for p in (path, os.path.join(EV_DIR, "njt_snapshot_latest.json")):
        with open(p, "w") as f:
            json.dump(EV_SNAPSHOT, f, indent=1)
    print("Live snapshot saved:", path)
else:
    replay = EV_SNAPSHOT_PATH or os.path.join(EV_DIR, "njt_snapshot_latest.json")
    if os.path.exists(replay):
        with open(replay) as f:
            EV_SNAPSHOT = json.load(f)
        EV_SNAPSHOT["source"] = f"replayed real snapshot from {EV_SNAPSHOT['taken_at']}"
        print(f"No live feed ({EV_SNAPSHOT_STATUS}). Replaying {replay}")
    else:
        print(f"No live feed ({EV_SNAPSHOT_STATUS}) and no saved snapshot. V2 to V4 will be marked skipped.")

if EV_SNAPSHOT:
    V = EV_SNAPSHOT["vehicles"]
    n = len(V)
    src = EV_SNAPSHOT["source"]
    taken = parse_iso(EV_SNAPSHOT["taken_at"])
    head_age = (taken - datetime.fromtimestamp(EV_SNAPSHOT["header_timestamp"], timezone.utc)).total_seconds() \
        if EV_SNAPSHOT.get("header_timestamp") else None
    s, nb, w, e = EV_NJ_BBOX
    in_box = sum(s <= v["lat"] <= nb and w <= v["lon"] <= e for v in V)
    with_route = sum(bool(v["route_id"]) for v in V)
    with_trip = sum(bool(v["trip_id"]) for v in V)
    ids = [v["vehicle_id"] for v in V if v["vehicle_id"]]
    dups = len(ids) - len(set(ids))
    ages = [(taken - datetime.fromtimestamp(v["timestamp"], timezone.utc)).total_seconds() for v in V if v["timestamp"]]
    fresh = sum(a <= EV_POLICY["freshness_limit_s"] for a in ages)
    matched = sum(v["has_trip_update"] for v in V)
    delays = [v["delay_s"] / 60 for v in V if v["delay_s"] is not None]
    alerts = EV_SNAPSHOT["alerts"]

    ev_check("V2", "vehicles in the feed", "> 0", n, n > 0, src)
    ev_check("V2", "feed header age (s)", f"<= {EV_POLICY['freshness_limit_s']}", None if head_age is None else round(head_age),
             None if head_age is None else head_age <= EV_POLICY["freshness_limit_s"], src)
    ev_check("V2", "vehicles with a route id (%)", ">= 95", ev_pct(with_route, n), n and ev_pct(with_route, n) >= 95, src)
    ev_check("V2", "vehicles with a trip id (%)", ">= 95", ev_pct(with_trip, n), n and ev_pct(with_trip, n) >= 95, src)
    ev_check("V2", "positions inside New Jersey and New York (%)", ">= 99", ev_pct(in_box, n), n and ev_pct(in_box, n) >= 99, src)
    ev_check("V2", "vehicles listed twice", 0, dups, dups == 0, src)
    ev_check("V2", "vehicle reports fresher than 300 s (%)", ">= 80", ev_pct(fresh, len(ages)),
             None if not ages else ev_pct(fresh, len(ages)) >= 80, src)
    ev_check("V2", "vehicles matched to a trip update (%)", ">= 50", ev_pct(matched, n), n and ev_pct(matched, n) >= 50, src,
             "delay is unknown for unmatched vehicles, so they can only reach the engine with a missing delay")
    ev_check("V2", "delays that are believable (between minus 30 and 180 min) (%)", ">= 99",
             ev_pct(sum(-30 <= d <= 180 for d in delays), len(delays)),
             None if not delays else ev_pct(sum(-30 <= d <= 180 for d in delays), len(delays)) >= 99, src)
    alert_text = [a for a in alerts if a["header"] or a["description"]]
    # An alert can target a route, a stop, or the whole agency; only route alerts can reach a bus case, so their share is reported.
    targeted = sum(bool(a["routes"] or a.get("stops") or a.get("agency_wide")) for a in alert_text)
    ev_check("V2", "alerts that say what they affect: route, stop or whole agency (%)", ">= 95", ev_pct(targeted, len(alert_text)),
             None if not alert_text else ev_pct(targeted, len(alert_text)) >= 95, src,
             "" if alert_text else "no alerts in this snapshot")
    ev_check("V2", "alerts that name a bus route, so they can reach a case (%)", "reported",
             ev_pct(sum(bool(a["routes"]) for a in alert_text), len(alert_text)), "info", src,
             "stop and system-wide alerts are valid but never attach to a route case")

    EV_FEED_SUMMARY = {
        "source": src, "vehicles": n, "trip_updates": EV_SNAPSHOT["trip_updates"], "alerts": len(alerts),
        "routes": len({v['route_id'] for v in V if v['route_id']}),
        "median_delay_min": round(statistics.median(delays), 1) if delays else None,
        "p90_delay_min": round(sorted(delays)[int(0.9 * (len(delays) - 1))], 1) if delays else None,
        "late_15_min_or_more": sum(d >= 15 for d in delays), "cancelled_trips": sum(v["cancelled"] for v in V),
        "within_2_km_of_newark_penn": sum(v["distance_km"] <= 2 for v in V)}
    print("Feed summary:", EV_FEED_SUMMARY)
else:
    EV_FEED_SUMMARY = {}
    ev_check("V2", "real feed available", "live or saved snapshot", EV_SNAPSHOT_STATUS, None, "none",
             "add the NJ Transit secrets, or set EV_SNAPSHOT_PATH to a saved snapshot")
ev_part_table("V2")

# ====================================================================================================
# notebook cell 80: V3. BUSDV2 login, vehicle locations, departures and cross checks against GTFSG2
# ====================================================================================================
#@title V3. BUSDV2 login, vehicle locations, departures and cross checks against GTFSG2
ev_clear("V3")
EV_BUSDV_BASE = "https://pcsdata.njtransit.com/api/BUSDV2/"
_EV_BUS = {"token": None, "status": "not attempted"}


def ev_busdv_login():
    user = read_secret("NJT_BUS_USERNAME") or read_secret("NJ_TRANSIT_USERNAME")
    pw = read_secret("NJT_BUS_PASSWORD") or read_secret("NJ_TRANSIT_PASSWORD")
    if not (user and pw):
        return "skipped (no NJ Transit login in Colab Secrets)"
    last = "rejected"
    for mode in ("form", "multipart"):
        try:
            if mode == "form":
                r = requests.post(EV_BUSDV_BASE + "authenticateUser", data={"username": user, "password": pw}, timeout=20)
            else:
                r = requests.post(EV_BUSDV_BASE + "authenticateUser", files={"username": (None, user), "password": (None, pw)},
                                  timeout=20)
            r.raise_for_status()
            body = r.json()
        except (requests.RequestException, ValueError) as e:
            last = f"request failed ({type(e).__name__})"
            continue
        token = body.get("UserToken") if isinstance(body, dict) else None
        if str(body.get("Authenticated", "")).lower() == "true" and token:
            _EV_BUS["token"] = token
            return "authenticated (BUSDV2)"
        last = "rejected (no token returned)"
    return last


def ev_busdv_post(endpoint, **fields):
    """POST with the token as a multipart field, the same pattern NJ Transit uses for GTFSG2."""
    files = {"token": (None, _EV_BUS["token"])}
    files.update({k: (None, str(v)) for k, v in fields.items()})
    r = requests.post(EV_BUSDV_BASE + endpoint, files=files, timeout=30)
    r.raise_for_status()
    return r.json()


def ev_records(body, must_have):
    """Find the list of records in a response, whatever it is wrapped in."""
    if isinstance(body, list):
        return [x for x in body if isinstance(x, dict)]
    if isinstance(body, dict):
        for v in body.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and any(k in v[0] for k in must_have):
                return v
        for v in body.values():
            found = ev_records(v, must_have) if isinstance(v, (dict, list)) else []
            if found:
                return found
    return []


def ev_num(x):
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def ev_vid(x):
    return str(x or "").strip().lstrip("0")


def ev_clock(text, ref):
    """Read a DepartureVision time such as '10:35 PM', '22:35' or an ISO time, on the service day of ref."""
    t = str(text or "").strip()
    if not t:
        return None
    try:
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=ref.tzinfo)
    except ValueError:
        pass
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"):
        try:
            p = datetime.strptime(t.upper(), fmt)
            if p.year == 1900:
                p = ref.replace(hour=p.hour, minute=p.minute, second=0, microsecond=0)
                if (p - ref).total_seconds() < -12 * 3600:      # a time just after midnight belongs to the next day
                    p += timedelta(days=1)
                return p
            return p.replace(tzinfo=ref.tzinfo)
        except ValueError:
            continue
    return None


EV_BUS = {"status": "not run", "vehicles": [], "departures": []}
if EV_USE_LIVE and globals().get("LIVE"):
    EV_BUS["status"] = ev_busdv_login()
    if _EV_BUS["token"]:
        lat0, lon0 = SERVICE_AREA_LATLON
        try:
            body = ev_busdv_post("getVehicleLocations", lat=lat0, lon=lon0, radius=EV_BUSDV_RADIUS, mode=EV_BUSDV_MODE)
            EV_BUS["vehicles"] = ev_records(body, ("VehicleID", "VehicleRoute"))
            EV_BUS["vehicle_keys"] = sorted(EV_BUS["vehicles"][0].keys()) if EV_BUS["vehicles"] else []
        except Exception as e:
            EV_BUS["vehicle_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        if EV_BUSDV_STOP:
            try:
                body = ev_busdv_post("getBusDV", stop=EV_BUSDV_STOP, direction="", route="", ip="")
                EV_BUS["departures"] = ev_records(body, ("public_route", "departuretime"))
            except Exception as e:
                EV_BUS["departure_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        EV_BUS["taken_at"] = iso(utcnow())
        with open(os.path.join(EV_DIR, "busdv2_snapshot_latest.json"), "w") as f:
            json.dump(EV_BUS, f, indent=1, default=str)
else:
    EV_BUS["status"] = "live data switched off"
print("BUSDV2:", EV_BUS["status"], "| vehicles:", len(EV_BUS["vehicles"]), "| departures:", len(EV_BUS["departures"]),
      "| fields seen:", EV_BUS.get("vehicle_keys", "-"))

bus_src = "NJ Transit BUSDV2 (live) vs GTFSG2"
if EV_BUS["vehicles"] and EV_SNAPSHOT:
    gt = {ev_vid(v["vehicle_id"]): v for v in EV_SNAPSHOT["vehicles"] if v["vehicle_id"]}
    pairs = []
    for b in EV_BUS["vehicles"]:
        g = gt.get(ev_vid(b.get("VehicleID")))
        blat, blon = ev_num(b.get("VehicleLat")), ev_num(b.get("VehicleLong"))
        if g and blat is not None and blon is not None:
            pairs.append({"vehicle": g["vehicle_id"], "route_gtfs": g["route_id"], "route_busdv": str(b.get("VehicleRoute", "")).strip(),
                          "gap_km": round(km_between((g["lat"], g["lon"]), (blat, blon)), 3),
                          "passenger_load_busdv": b.get("VehiclePassengerLoad")})
    EV_BUS["pairs"] = pairs
    nb = len(EV_BUS["vehicles"])
    small = f" Only {nb} BUSDV2 buses near Newark Penn at this hour: repeat at a busier time." if nb < 10 else ""
    ev_check("V3", "BUSDV2 buses also found in GTFSG2 (%)", ">= 80", f"{ev_pct(len(pairs), nb)} ({len(pairs)} of {nb})",
             ev_pct(len(pairs), nb) >= 80, bus_src,
             "both feeds are snapshots a few seconds apart, so a few buses can enter or leave the area in between." + small)
    if pairs:
        same_route = sum(p["route_gtfs"].lstrip("0") == p["route_busdv"].lstrip("0") for p in pairs)
        gaps = sorted(p["gap_km"] for p in pairs)
        ev_check("V3", "matched buses on the same route in both feeds (%)", ">= 95",
                 f"{ev_pct(same_route, len(pairs))} ({same_route} of {len(pairs)})",
                 ev_pct(same_route, len(pairs)) >= 95, bus_src)
        ev_check("V3", "median distance between the two GPS positions (km)", "<= 0.5", round(statistics.median(gaps), 3),
                 statistics.median(gaps) <= 0.5, bus_src, "a bus moves about 0.2 km in 30 s of city traffic")
        display(pd.DataFrame(pairs).head(15))
elif EV_BUS["vehicles"]:
    ev_check("V3", "GTFSG2 snapshot to compare with", "available", "none", None, bus_src, "V2 found no GTFSG2 data")
else:
    ev_check("V3", "BUSDV2 vehicle locations", "available", EV_BUS["status"] + (" " + EV_BUS.get("vehicle_error", "")), None,
             "none", "needs live NJ Transit access; see the note above about the request format")

if EV_BUS["departures"] and EV_SNAPSHOT:
    ref = utcnow().astimezone(__import__("zoneinfo").ZoneInfo("America/New_York"))
    by_route = {}
    for v in EV_SNAPSHOT["vehicles"]:
        if v["delay_s"] is not None and v["route_id"]:
            by_route.setdefault(v["route_id"].lstrip("0"), []).append(v["delay_s"] / 60)
    rows = []
    for d in EV_BUS["departures"]:
        pred, sched = ev_clock(d.get("departuretime"), ref), ev_clock(d.get("sched_dep_time"), ref)
        route = str(d.get("public_route", "")).strip().lstrip("0")
        if pred and sched and route in by_route:
            dv_delay = (pred - sched).total_seconds() / 60
            gt_delay = statistics.median(by_route[route])
            rows.append({"route": route, "busdv_delay_min": round(dv_delay, 1), "gtfs_median_delay_min": round(gt_delay, 1),
                         "difference_min": round(abs(dv_delay - gt_delay), 1)})
    EV_BUS["delay_pairs"] = rows
    if rows:
        diffs = [r["difference_min"] for r in rows]
        conflict = sum(x > EV_POLICY["conflict_threshold_min"] for x in diffs)
        ev_check("V3", "median delay difference between the feeds (min)", "<= 3", round(statistics.median(diffs), 1),
                 statistics.median(diffs) <= 3, bus_src)
        ev_check("V3", "route pairs E2 would treat as a conflict (> 8 min apart) (%)", "<= 10", ev_pct(conflict, len(rows)),
                 ev_pct(conflict, len(rows)) <= 10, bus_src, "this is how often the conflicting data path would fire on real data")
        display(pd.DataFrame(rows))
    else:
        ev_check("V3", "departures that could be compared", "> 0", 0, None, bus_src,
                 "no departure at this stop had a readable time and a route present in GTFSG2")
elif not EV_BUSDV_STOP:
    ev_check("V3", "departure delay comparison", "a stop code in EV_BUSDV_STOP", "not set", None, "none",
             "set EV_BUSDV_STOP to a bus stop number to compare predicted delays")
else:
    ev_check("V3", "BUSDV2 departures at stop " + EV_BUSDV_STOP, "available",
             EV_BUS.get("departure_error") or "none returned", None, "none", "check the stop number and the login")
ev_part_table("V3")

# ====================================================================================================
# notebook cell 82: V4. Run the pipeline on real buses and compare with the published rules
# ====================================================================================================
#@title V4. Run the pipeline on real buses and compare with the published rules
ev_clear("V4")
EV_REAL_CASES = []


def ev_expected_gate(v, taken_at):
    age = (parse_iso(taken_at) - datetime.fromtimestamp(v["timestamp"], timezone.utc)).total_seconds() if v["timestamp"] else 0
    delay = None if v["delay_s"] is None else v["delay_s"] / 60
    if age > EV_POLICY["freshness_limit_s"]:
        return {"MONITORING"}, f"report {age:.0f} s old"
    if v["cancelled"] or (delay is not None and delay >= EV_POLICY["hard_trigger"]["delay_min"]):
        return {"SIGNIFICANT"}, "cancelled" if v["cancelled"] else f"delay {delay:.1f} min"
    return {"BELOW_THRESHOLD", "MONITORING"}, "no hard trigger" if delay is not None else "delay unknown"


def ev_run_real_vehicle(v, alerts):
    """Section A hand off and E2 decision for one real vehicle. LIVE_CONTEXT and the switches are restored afterwards."""
    saved_ctx = copy.deepcopy(LIVE_CONTEXT["nj_transit"])
    saved = {k: globals().get(k) for k in ("USE_SLM", "USE_LLM_ORGANIZER")}
    alert = next((a for a in alerts if v["route_id"] and v["route_id"] in a["routes"]), None)
    try:
        globals().update(USE_SLM=False, USE_LLM_ORGANIZER=False)
        LIVE_CONTEXT["nj_transit"] = {**saved_ctx, "service": dict(v),
                                      "alert": {k: alert[k] for k in ("header", "description", "effect")} if alert else None}
        ssi = build_signal_scout_input(section_a_sample_inputs())
        raw = signal_scout_input_to_raw(ssi, f"Evaluation, real route {v['route_id']} vehicle {v['vehicle_id']}")
        return raw, run_analysis(raw, EV_POLICY), ssi
    finally:
        LIVE_CONTEXT["nj_transit"] = saved_ctx
        globals().update(saved)


if EV_SNAPSHOT:
    src = EV_SNAPSHOT["source"]
    taken = EV_SNAPSHOT["taken_at"]
    # Delayed and cancelled buses first, so the important cases are always inside the cap
    pool = sorted(EV_SNAPSHOT["vehicles"], key=lambda v: (not v["cancelled"], -(v["delay_s"] or 0)))[:EV_MAX_REAL_CASES]
    for v in pool:
        expected, why = ev_expected_gate(v, taken)
        try:
            raw, A, ssi = ev_run_real_vehicle(v, EV_SNAPSHOT["alerts"])
            s = A.get("scores") or {}
            trig = [t["trigger"] for t in A.get("hard_triggers", [])]
            row = {"route": v["route_id"], "vehicle": v["vehicle_id"],
                   "delay_raw_min": None if v["delay_s"] is None else round(v["delay_s"] / 60, 2),
                   "delay_in_engine_min": A.get("fields", {}).get("delay_min"), "cancelled": v["cancelled"],
                   "expected": " or ".join(sorted(expected)), "reason": why, "engine_gate": A["gate"],
                   "C": (A.get("confidence") or {}).get("C"), "R": s.get("R"), "P": s.get("P"),
                   "hard_triggers": ", ".join(trig) or "none",
                   "top_option": A["ranking"][0]["option_id"] if A.get("ranking") else "",
                   "agrees": A["gate"] in expected}
            if A["gate"] == "SIGNIFICANT" and "SIGNIFICANT" not in expected and not trig \
                    and (s.get("R") or 0) >= EV_POLICY["significance_R"]:
                row["agrees"], row["reason"] = True, f"R = {s.get('R')} >= {EV_POLICY['significance_R']} (significance rule)"
            if A["gate"] == "MONITORING" and "MONITORING" in expected and "BELOW_THRESHOLD" in expected:
                row["agrees"] = (A.get("confidence") or {}).get("C", 1) < EV_POLICY["confidence_threshold"] \
                    or any(e["status"] == "stale" for e in A.get("evidence", []))
        except Exception as e:
            row = {"route": v["route_id"], "vehicle": v["vehicle_id"],
                   "delay_raw_min": None if v["delay_s"] is None else round(v["delay_s"] / 60, 2), "delay_in_engine_min": None,
                   "cancelled": v["cancelled"], "expected": " or ".join(sorted(expected)), "reason": why,
                   "engine_gate": f"error: {type(e).__name__}: {str(e)[:80]}", "agrees": False}
        EV_REAL_CASES.append(row)
    df = pd.DataFrame(EV_REAL_CASES)
    df.to_csv(os.path.join(EV_DIR, "real_cases_pipeline_vs_rules.csv"), index=False)
    n, ok = len(df), int(df["agrees"].sum())
    ev_check("V4", "real buses where the pipeline reached the rule's answer (%)", 100, ev_pct(ok, n), ok == n, src,
             f"{n} real buses; table saved to real_cases_pipeline_vs_rules.csv")
    must = df[df["expected"] == "SIGNIFICANT"]
    ev_check("V4", "real cancellations and 15+ minute delays escalated (recall)", "100%",
             f"{int(must['agrees'].sum())} of {len(must)}", None if must.empty else bool(must["agrees"].all()), src,
             "" if len(must) else "no bus was cancelled or 15+ minutes late in this snapshot")
    not_sig = df[df["expected"] != "SIGNIFICANT"]
    false_esc = not_sig[(not_sig["engine_gate"] == "SIGNIFICANT") & (not_sig["R"].fillna(0) < EV_POLICY["significance_R"])]
    ev_check("V4", "real buses escalated with no hard trigger and R below 0.50 (false escalations)", 0, len(false_esc),
             len(false_esc) == 0, src, "a case with R of 0.50 or more is significant by the rules, so it is not counted here")
    if "C" in df and df["C"].nunique() == 1:
        ev_check("V4", "confidence C on real buses", "reported", df["C"].iloc[0], "info", src,
                 "the same for every bus: one real-time source, so quality, agreement and completeness do not vary")
    rounding = df[df["delay_raw_min"].notna() & (df["delay_raw_min"] < 15) & (df["delay_in_engine_min"].fillna(0) >= 15)] \
        if "delay_in_engine_min" in df else df.iloc[0:0]
    ev_check("V4", "delays pushed over 15 min by rounding", 0, len(rounding), len(rounding) == 0, src)
    ev_check("V4", "cases that crashed", 0, int(df["engine_gate"].astype(str).str.startswith("error").sum()),
             int(df["engine_gate"].astype(str).str.startswith("error").sum()) == 0, src)
    print("Gate counts on real data:", df["engine_gate"].value_counts().to_dict())
    display(df.head(20))
else:
    ev_check("V4", "real cases", "snapshot from V2", "none", None, "none", "needs a live or saved snapshot")
ev_part_table("V4")

# ====================================================================================================
# notebook cell 84: V5. Hand labeled test set for delay, cause and the small language model
# ====================================================================================================
#@title V5. Hand labeled test set for delay, cause and the small language model
ev_clear("V5")
# text, true delay in minutes (None if not stated), true cause (one of the team's categories), route named in the text
EV_TEXT_GOLD = [
    ("Route 62 buses are running about 20 minutes late due to heavy snow.", 20, "weather", "62"),
    ("Expect delays of up to 30 minutes on the 39 because of a disabled vehicle on Market St.", 30, "obstruction", "39"),
    ("Bus 13 is delayed 12 minutes due to a mechanical problem.", 12, "mechanical", "13"),
    ("Route 1 detoured from Broad St due to a road closure for construction.", None, "road_closure", "1"),
    ("Route 25 running 15 min late following a crash on McCarter Hwy.", 15, "obstruction", "25"),
    ("No delays reported on Route 11. Service is operating normally.", None, "none", "11"),
    ("Route 108 service restored after an earlier medical emergency on board.", None, "medical", "108"),
    ("Route 30 delays of 10 to 15 minutes because of flooding near Penn Station.", 15, "weather", "30"),
    ("Gate 5 change: Route 70 now departs from Gate 7 at Newark Penn Station.", None, "none", "70"),
    ("Route 5 is about 8 minutes behind schedule.", 8, "none", "5"),
    ("Customer service hours change this weekend for Route 21.", None, "none", "21"),
    ("Route 34 buses delayed approximately 25 minutes, police activity blocked the road.", 25, "obstruction", "34"),
    ("Icy roads: Route 59 is running 18 minutes late.", 18, "weather", "59"),
    ("Route 72 detour in effect: Washington St closed to traffic.", None, "road_closure", "72"),
    ("Route 41 trips are 22 mins late because of a vehicle breakdown.", 22, "mechanical", "41"),
    ("Route 99 is on time. No service changes.", None, "none", "99"),
    ("Route 28 has a 6 minute delay due to a sick passenger.", 6, "medical", "28"),
    ("Storm warning in effect. Route 94 may run up to 40 minutes late.", 40, "weather", "94"),
    ("Route 43 buses delayed by 9 minutes due to a collision at Broad and Market.", 9, "obstruction", "43"),
    ("Route 76 delayed 35 minutes. Road closed at Raymond Blvd.", 35, "road_closure", "76"),
]
EV_CAUSES = set(CAUSE_KEYWORDS) | {"none"}
assert all(c in EV_CAUSES for _, _, c, _ in EV_TEXT_GOLD)

rows = []
for text, d, c, route in EV_TEXT_GOLD:
    got_d, got_c = extract_delay(text), classify_cause(text)
    rows.append({"text": text, "true_delay": d, "extract_delay": got_d, "delay_ok": (got_d == d) if d is not None else got_d is None,
                 "true_cause": c, "classify_cause": got_c, "cause_ok": got_c == c, "route": route})
gold = pd.DataFrame(rows)
n = len(gold)
src = "hand labeled test set (20 alerts)"
ev_check("V5", "delay read correctly (%)", ">= 90", ev_pct(int(gold["delay_ok"].sum()), n), ev_pct(int(gold["delay_ok"].sum()), n) >= 90, src)
stated = gold[gold["true_delay"].notna()]
ev_check("V5", "a delay invented where none is stated", 0, int(gold[gold["true_delay"].isna()]["extract_delay"].notna().sum()),
         int(gold[gold["true_delay"].isna()]["extract_delay"].notna().sum()) == 0, src)
ev_check("V5", "stated delays missed", 0, int(stated["extract_delay"].isna().sum()), int(stated["extract_delay"].isna().sum()) == 0, src)
ev_check("V5", "cause classified correctly (%)", ">= 85", ev_pct(int(gold["cause_ok"].sum()), n), ev_pct(int(gold["cause_ok"].sum()), n) >= 85, src)
per_cause = gold.groupby("true_cause").apply(lambda g: f"{int(g['cause_ok'].sum())}/{len(g)}", include_groups=False).to_dict()
print("Cause accuracy by category:", per_cause)

# Small language model: only if Section A loaded it
slm_ready = bool(globals().get("_SLM", {}).get("model"))
if slm_ready and globals().get("USE_SLM"):
    hits_route = hits_delay = 0
    for text, d, c, route in EV_TEXT_GOLD:
        a = slm_answers(text) or {}
        hits_route += route in str(a.get("route", ""))
        hits_delay += (str(d) in str(a.get("delay", ""))) if d is not None else not re.search(r"\d", str(a.get("delay", "")))
    ev_check("V5", "small language model: route correct (%)", "reported", ev_pct(hits_route, n), "info", src,
             "supporting text only, never scored, so there is no pass mark")
    ev_check("V5", "small language model: delay correct (%)", "reported", ev_pct(hits_delay, n), "info", src,
             "supporting text only, never scored, so there is no pass mark")
else:
    ev_check("V5", "small language model accuracy", "model loaded in A2", "not loaded", None, src,
             (globals().get("_SLM", {}).get("error") or "USE_SLM is off")[:120])

# Real alerts: export for labeling, and score them if labels exist
if EV_SNAPSHOT and EV_SNAPSHOT["alerts"]:
    real = pd.DataFrame([{"alert_id": a["alert_id"], "routes": ", ".join(a["routes"]),
                          "text": f"{a['header']}. {a['description']}".strip(". "),
                          "system_delay_min": extract_delay(f"{a['header']}. {a['description']}"),
                          "system_cause": classify_cause(f"{a['header']}. {a['description']}"),
                          "true_delay_min": "", "true_cause": ""} for a in EV_SNAPSHOT["alerts"]])
    real.to_csv(os.path.join(EV_DIR, "real_alerts_to_label.csv"), index=False)
    print(f"{len(real)} real alerts exported to real_alerts_to_label.csv for labeling")
if EV_ALERT_LABELS_CSV and os.path.exists(EV_ALERT_LABELS_CSV):
    lab = pd.read_csv(EV_ALERT_LABELS_CSV).fillna("")
    lab = lab[lab["true_cause"].astype(str).str.len() > 0]
    if len(lab):
        dl_ok = sum((str(r["true_delay_min"]).strip() in ("", "nan") and extract_delay(r["text"]) is None)
                    or (str(r["true_delay_min"]).strip() not in ("", "nan") and extract_delay(r["text"]) == float(r["true_delay_min"]))
                    for _, r in lab.iterrows())
        c_ok = sum(classify_cause(r["text"]) == str(r["true_cause"]).strip() for _, r in lab.iterrows())
        ev_check("V5", "real alerts: delay read correctly (%)", ">= 90", ev_pct(dl_ok, len(lab)), ev_pct(dl_ok, len(lab)) >= 90,
                 f"{len(lab)} labeled real alerts")
        ev_check("V5", "real alerts: cause classified correctly (%)", ">= 85", ev_pct(c_ok, len(lab)), ev_pct(c_ok, len(lab)) >= 85,
                 f"{len(lab)} labeled real alerts")
display(gold[~(gold["delay_ok"] & gold["cause_ok"])][["text", "true_delay", "extract_delay", "true_cause", "classify_cause"]])
ev_part_table("V5")

# ====================================================================================================
# notebook cell 86: V6. Detector counts against hand counts
# ====================================================================================================
#@title V6. Detector counts against hand counts
ev_clear("V6")
# camera id: (expected source file, frames used, {frame index: (fully visible people, partly visible people)})
EV_CV_TRUTH = {
    "VISION-STOP-NEWARK-PENN": ("bus.jpg", 1, {0: (3, 1)}),
    "VISION-APPROACH-NEWARK-PENN": ("vtest.avi", 8, {0: (3, 0), 113: (7, 1), 226: (6, 1), 340: (6, 0),
                                                       453: (3, 0), 567: (6, 0), 680: (7, 0), 794: (7, 1)}),
}
src = "CV sample media, hand counted"
cams = {c["camera_id"]: c for c in CV_RESULTS.get("cameras", [])}
urls = {s["camera_id"]: s["url"] for s in CV_SOURCES}
rows = []
for cam_id, (fname, nframes, truth) in EV_CV_TRUTH.items():
    cam = cams.get(cam_id)
    if cam is None or not urls.get(cam_id, "").endswith(fname) or len(cam["people_per_frame"]) != nframes:
        ev_check("V6", f"{cam_id} counts", "same file and frames as the hand counts", "camera not run or settings changed", None, src,
                 "; ".join(CV_RESULTS.get("errors", []))[:120])
        continue
    for (idx, (vis, part)), got in zip(sorted(truth.items()), cam["people_per_frame"]):
        rows.append({"camera": cam_id, "frame": idx, "visible": vis, "partly_visible": part, "detector": got,
                     "error_vs_visible": got - vis, "within_range": vis <= got <= vis + part})
if rows:
    cv = pd.DataFrame(rows)
    EV_CV_ROWS = cv
    mae = cv["error_vs_visible"].abs().mean()
    ev_check("V6", "frames counted within the hand count range (%)", ">= 75", ev_pct(int(cv["within_range"].sum()), len(cv)),
             ev_pct(int(cv["within_range"].sum()), len(cv)) >= 75, src, f"detector: {CV_RESULTS.get('detector')}")
    ev_check("V6", "mean counting error (people per frame)", "<= 1.0", round(mae, 2), mae <= 1.0, src)
    ev_check("V6", "counting bias (average over or under count)", "between -1 and 1", round(cv["error_vs_visible"].mean(), 2),
             abs(cv["error_vs_visible"].mean()) <= 1, src, "negative means people are missed")
    display(cv)
ev_check("V6", "no identities or images kept", "identities_stored and images_stored are False for every camera",
         [(c["identities_stored"], c["images_stored"]) for c in CV_RESULTS.get("cameras", [])],
         all(not c["identities_stored"] and not c["images_stored"] for c in CV_RESULTS.get("cameras", [])) if cams else None, src)
ev_part_table("V6")

# ====================================================================================================
# notebook cell 88: V7. Golden presets, thresholds, formulas, determinism and monotonic checks
# ====================================================================================================
#@title V7. Golden presets, thresholds, formulas, determinism and monotonic checks
ev_clear("V7")
src = "E2 engine, synthetic cases"
EV_PRESET_GOLD = {
    "Severe · R2 Hospital Line trip cancelled (verified)": "SIGNIFICANT",
    "Significant · R1 Crosstown unboardable crowding at peak": "SIGNIFICANT",
    "Moderate · R4 Northside winter-weather delays": "SIGNIFICANT",
    "Below threshold · R5 Southside 4-minute delay": "BELOW_THRESHOLD",
    "Conflicting data · R3 vehicle feed vs. public reports": "CONFLICT_REVIEW",
    "Feed outage · R1 vehicle feed stale (backup available)": "SIGNIFICANT",
    "Feed outage · R5 vehicle feed stale (no backup)": "MONITORING",
    "Uncorroborated social report only · R4": "MONITORING",
    "Agent error · Vision Agent fails twice on R2": "SAFE_MODE",
    "Hard-trigger bypass · Verified emergency closure on R3, sparse data": "SIGNIFICANT",
}
for name, expected in EV_PRESET_GOLD.items():
    if name not in PRESETS:
        ev_check("V7", f"preset: {name}", expected, "preset not found", None, src, "preset renamed or removed")
        continue
    A = run_analysis(make_raw(PRESETS[name], name), EV_POLICY)
    ev_check("V7", f"preset: {name}", expected, A["gate"], A["gate"] == expected, src)


def ev_gate(**spec):
    return run_analysis(make_raw({**spec}, "Evaluation case"), EV_POLICY)


base = dict(route_id="R1", stop_range=(3, 7), time_period="midday", headway_ratio=1.2, cancellation_risk=0.1, boardings=60,
            load_factor=0.5, passups=0, sources=["vehicle_apc"], backup_available=False)
ht = EV_POLICY["hard_trigger"]["delay_min"]
a_under, a_at = ev_gate(**base, delay_min=ht - 0.1), ev_gate(**base, delay_min=ht)
trig = lambda A: any(t["trigger"].startswith("delay") for t in A["hard_triggers"])
ev_check("V7", f"delay {ht - 0.1} min: no delay trigger", False, trig(a_under), not trig(a_under), src)
ev_check("V7", f"delay {ht} min: delay trigger", True, trig(a_at), trig(a_at), src)
lim = EV_POLICY["freshness_limit_s"]
f_in, f_out = ev_gate(**base, delay_min=18, feed_age_s=lim - 1), ev_gate(**base, delay_min=18, feed_age_s=lim + 1)
ev_check("V7", f"feed {lim - 1} s old: analysed", "not MONITORING", f_in["gate"], f_in["gate"] != "MONITORING", src)
ev_check("V7", f"feed {lim + 1} s old, no backup: monitoring", "MONITORING", f_out["gate"], f_out["gate"] == "MONITORING", src)

# Formulas recomputed from the logged components with the v5 weights
v5 = {"R": {"D": .30, "H": .25, "K": .25, "E": .20}, "V": {"N": .35, "I": .25, "B": .20, "T": .20},
      "A": {"S": .35, "L": .30, "X": .20, "U": .15}, "P": {"R": .40, "V": .35, "A": .25}}
bad_formula, bad_si, checked = [], [], 0
for name in EV_PRESET_GOLD:
    if name not in PRESETS:
        continue
    A = run_analysis(make_raw(PRESETS[name], name), EV_POLICY)
    if not A.get("scores"):
        continue
    comp, sc = A["scores"]["components"], A["scores"]
    for g in ("R", "V", "A"):
        want = sum(w * (comp[c] or 0) for c, w in v5[g].items())
        if abs(want - sc[g]) > 0.002:
            bad_formula.append(f"{name}: {g} {sc[g]} vs {want:.3f}")
    want_p = sum(w * sc[k] for k, w in v5["P"].items())
    if abs(want_p - sc["P"]) > 0.002:
        bad_formula.append(f"{name}: P {sc['P']} vs {want_p:.3f}")
    w = scenario_weights(sc["P"], EV_POLICY)
    for r in A.get("ranking", []):
        checked += 1
        want_si = r["Ci"] * sum(w[c] * r[c] for c in w)
        if abs(want_si - r["Si"]) > 0.003:
            bad_si.append(f"{name}: {r['option_id']} Si {r['Si']} vs {want_si:.4f}")
ev_check("V7", "R, V, A and P match the v5 formulas", "0 differences", len(bad_formula), not bad_formula, src, "; ".join(bad_formula)[:200])
ev_check("V7", "every Si equals Ci times the weighted sum", "0 differences", f"{len(bad_si)} of {checked}", not bad_si, src,
         "; ".join(bad_si)[:200])
sums = [round(sum(scenario_weights(p / 10, EV_POLICY).values()), 6) for p in range(11)]
ev_check("V7", "scenario weights add up to 1 for P from 0 to 1", "all 1.0", sorted(set(sums)), set(sums) == {1.0}, src)

# Determinism
name = "Severe · R2 Hospital Line trip cancelled (verified)"
if name in PRESETS:
    r1 = [(r["option_id"], r["Si"]) for r in run_analysis(make_raw(PRESETS[name], name), EV_POLICY)["ranking"]]
    r2 = [(r["option_id"], r["Si"]) for r in run_analysis(make_raw(PRESETS[name], name), EV_POLICY)["ranking"]]
    ev_check("V7", "same case twice gives the same ranking", "identical", "identical" if r1 == r2 else "different", r1 == r2, src)

# More need never lowers a score
Rs = [ev_gate(**base, delay_min=d)["scores"]["R"] for d in range(0, 31, 3)]
ev_check("V7", "R never falls as the delay grows (0 to 30 min)", "non decreasing", Rs, all(b >= a for a, b in zip(Rs, Rs[1:])), src)
VP = []
for td in (0.05, 0.2, 0.35, 0.5, 0.65):
    raw = make_raw({**base, "delay_min": 10}, "Evaluation case")
    raw["community"]["transit_dependence"] = td
    s = run_analysis(raw, EV_POLICY)["scores"]
    VP.append((s["V"], s["P"]))
ev_check("V7", "V and P never fall as transit dependence grows", "non decreasing", VP,
         all(b[0] >= a[0] and b[1] >= a[1] for a, b in zip(VP, VP[1:])), src)
ev_check("V7", "the policy passes its own invariants", "no errors", validate_policy(EV_POLICY) or "no errors",
         not validate_policy(EV_POLICY), src)
ev_part_table("V7")

# ====================================================================================================
# notebook cell 90: V8. Scenario gallery cases end where the diagram says
# ====================================================================================================
#@title V8. Scenario gallery cases end where the diagram says
ev_clear("V8")
EV_GALLERY_GOLD = {
    "01": {"SAFE_DEFAULT", "MONITORING", "EVALUATED", "AWAITING_REVIEW", "CONFLICT_REVIEW"},
    "02": {"EVALUATED"}, "03": {"EVALUATED"}, "04": {"EVALUATED"}, "05": {"SAFE_DEFAULT"}, "06": {"SAFE_DEFAULT"},
    "07": {"SAFE_DEFAULT"}, "08": {"SAFE_DEFAULT"}, "09": {"SAFE_DEFAULT", "EVALUATED", "AWAITING_REVIEW"},
    "10": {"MONITORING"}, "11": {"AWAITING_REVIEW"}, "12": {"EVALUATED"}, "13": {"CLOSED_FALSE_ALARM"}, "14": {"SAFE_DEFAULT"},
}
EV_GALLERY_GATE = {"02": "SIGNIFICANT", "08": "BELOW_THRESHOLD", "10": "MONITORING", "13": "MONITORING", "14": "SAFE_MODE"}
gal = globals().get("GALLERY_CASES") or {}
src = "Section F scripted cases"
if not gal:
    ev_check("V8", "scenario gallery", "run in Section F", "not run", None, src, "set RUN_SCENARIO_GALLERY to True and rerun F")
for sid, allowed in EV_GALLERY_GOLD.items():
    case = gal.get(sid)
    if case is None:
        if gal:
            ev_check("V8", f"scenario {sid}", " or ".join(sorted(allowed)), "did not finish", False, src)
        continue
    ev_check("V8", f"scenario {sid} final status", " or ".join(sorted(allowed)), case["status"], case["status"] in allowed, src)
    if sid in EV_GALLERY_GATE:
        ev_check("V8", f"scenario {sid} gate", EV_GALLERY_GATE[sid], case["analysis"]["gate"],
                 case["analysis"]["gate"] == EV_GALLERY_GATE[sid], src)
if gal:
    # Safeguards: every approval must have a name, a reason of the minimum length and a full checklist
    p = EV_POLICY["review"]
    approvals = [d for c in gal.values() for d in c["decisions"] if d.get("decision") == "Approve"]
    weak = [d for d in approvals if len(str(d.get("reason", ""))) < p["min_reason_chars"] or not d.get("reviewer")]
    ev_check("V8", "approvals without a name or a real reason", 0, len(weak), not weak, src)
    pairs = [c for c in gal.values() if len({d["reviewer"] for d in c["decisions"] if d.get("stage") in ("first review", "second review")})
             < 2 and any(d.get("stage") == "second review" for d in c["decisions"])]
    ev_check("V8", "high impact approvals signed by one person only", 0, len(pairs), not pairs, src)
ev_part_table("V8")

# ====================================================================================================
# notebook cell 92: V9. Explanation narratives only use numbers the rules produced
# ====================================================================================================
#@title V9. Explanation narratives only use numbers the rules produced
ev_clear("V9")
src = "explanations of cases this session"
EV_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def ev_allowed_numbers(case):
    A = case["analysis"]
    vals = set()

    def add(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            for d in (0, 1, 2, 3):
                vals.add(round(float(x), d))
        elif isinstance(x, dict):
            for v in x.values():
                add(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                add(v)
    add(A.get("scores")); add(A.get("confidence")); add(A.get("fields")); add(A.get("ranking")); add(A.get("options"))
    add(case.get("review")); add(EV_POLICY.get("confidence_threshold")); add(EV_POLICY.get("significance_R"))
    for t in EV_NUM.findall(case["explanation"].get("brief_md", "") + " " + case.get("scenario_name", "")):
        add(float(t))
    return vals


rows = []
for cid, case in APP.cases.items():
    ex = case.get("explanation") or {}
    if not ex.get("narrative") or not case["analysis"].get("ranking"):
        continue
    allowed = ev_allowed_numbers(case)
    nums = [float(t) for t in EV_NUM.findall(ex["narrative"])]
    unsupported = [x for x in nums if not any(abs(x - a) < 0.0051 for a in allowed)]
    top = case["analysis"]["ranking"][0]["label"]
    brief = ex.get("brief_md", "")
    order_ok = brief.find("All alternatives") != -1 and brief.find("All alternatives") < brief.find("Recommendation")
    rows.append({"case": cid, "narrative_by": ex.get("narrative_source"), "numbers": len(nums), "unsupported": unsupported,
                 "names_rank_1": narrative_names_rank1(ex["narrative"], top), "alternatives_before_recommendation": order_ok})
if rows:
    ex_df = pd.DataFrame(rows)
    llm = ex_df[~ex_df["narrative_by"].astype(str).str.startswith("template")]
    tmpl = ex_df[ex_df["narrative_by"].astype(str).str.startswith("template")]
    for label, part in (("template", tmpl), ("live LLM", llm)):
        if len(part):
            n_num = int(part["numbers"].sum())
            n_bad = int(part["unsupported"].apply(len).sum())
            ev_check("V9", f"{label} narratives: unsupported numbers", 0, f"{n_bad} of {n_num}", n_bad == 0, f"{len(part)} {label} narrative(s)")
            ev_check("V9", f"{label} narratives naming the rank 1 option (%)", 100, ev_pct(int(part["names_rank_1"].sum()), len(part)),
                     bool(part["names_rank_1"].all()), f"{len(part)} {label} narrative(s)")
    ev_check("V9", "briefs listing all alternatives before the recommendation (%)", 100,
             ev_pct(int(ex_df["alternatives_before_recommendation"].sum()), len(ex_df)),
             bool(ex_df["alternatives_before_recommendation"].all()), src)
    if not len(llm):
        ev_check("V9", "live LLM narratives", "at least one", 0, None, src, "no live-LLM narrative among the checked cases (the gallery uses templates). Process a case in Section O with "
                 "'Live LLM narrative' ticked, then rerun V9")
    display(ex_df)
else:
    ev_check("V9", "explanations to check", "at least one significant case", 0, None, src, "run Section F or the orchestrator first")
ev_part_table("V9")

# ====================================================================================================
# notebook cell 94: V10. Ethics training gate and conflict of interest gate
# ====================================================================================================
#@title V10. Ethics training gate and conflict of interest gate
ev_clear("V10")
EV_GOV_NEEDS = ["ETH_ROSTER", "ETH_SAMPLES", "load_eth_roster", "eth_check_official", "eth_team_summary", "_status_for",
                "Official", "ComplianceReport", "ActionItem", "verify_report", "reconcile_report", "governance_check",
                "rule_screen", "STAFF_DISCLOSURES", "ExtractedFacts", "Holding", "Decision", "decisions_for_option",
                "coi_enforce", "Flag", "AnalystOutput", "ReviewerOutput", "FinalReport", "ETH_TODAY"]
EV_GOV_MISSING = [n for n in EV_GOV_NEEDS if n not in globals()]
if EV_GOV_MISSING:
    ev_check("V10", "Sections C and D loaded", "yes", "missing " + ", ".join(EV_GOV_MISSING), None, "", "run Sections C and D first")
else:
    _saved_roster = [o.model_copy() for o in ETH_ROSTER]
    try:
        load_eth_roster(ETH_SAMPLES["Transit review team (this system)"])
        src_c = "Section C transit review team roster"

        # 1. Training rules, expected answers worked out by hand from the published rules
        EV_ETH_GOLD = {"Ana Lopez": "COMPLIANT", "Ben Carter": "AT_RISK", "Chen Wei": "COMPLIANT", "Dana Brooks": "COMPLIANT",
                       "Eli Turner": "NON_COMPLIANT", "Farah Haddad": "NON_COMPLIANT", "Gabe Ortiz": "COMPLIANT"}
        for who, want in EV_ETH_GOLD.items():
            got = eth_check_official(who).get("status")
            ev_check("V10", f"training status of {who}", want, got, got == want, src_c)
        got = sorted(eth_check_official("Eli Turner").get("missing_courses", []))
        ev_check("V10", "courses Eli Turner is missing", "data_protection, procurement", ", ".join(got),
                 got == ["data_protection", "procurement"], src_c)
        rate = eth_team_summary()["compliance_rate_pct"]
        ev_check("V10", "team compliance rate (COMPLIANT plus AT_RISK)", 71.4, rate, abs(rate - 71.4) < 0.1, src_c, "5 of 7 people")

        ALL4_ = ["conflicts", "transparency", "procurement", "data_protection"]
        def _days(n):
            return (ETH_TODAY - timedelta(days=n)).isoformat()
        EV_ETH_EDGES = [(334, "COMPLIANT", "refresher due in 31 days"), (335, "AT_RISK", "due in exactly 30 days"),
                        (365, "AT_RISK", "due today"), (366, "NON_COMPLIANT", "one day overdue")]
        for n, want, note in EV_ETH_EDGES:
            got = _status_for(Official(name="Test", completed=ALL4_, last_training=_days(n)))["status"]
            ev_check("V10", f"training {n} days ago", want, got, got == want, "edge case", note)
        got = _status_for(Official(name="Test", completed=ALL4_, last_training=None))["status"]
        ev_check("V10", "all courses but never dated", "NON_COMPLIANT", got, got == "NON_COMPLIANT", "edge case")

        # 2. Training gate
        gate_cases = [("Eli Turner", "Approve", True), ("Eli Turner", "Revise", True), ("Eli Turner", "Reject", False),
                      ("Farah Haddad", "Approve", True), ("Ben Carter", "Approve", False), ("Ana Lopez", "Approve", False)]
        for who, dec, want_block in gate_cases:
            probs, notes = governance_check(who, dec, policy=EV_POLICY)
            blocked = any("ethics training" in p for p in probs)
            ev_check("V10", f"{who} tries to {dec}", "blocked" if want_block else "allowed",
                     "blocked" if blocked else "allowed", blocked == want_block, "governance_check")
        probs, notes = governance_check("Ben Carter", "Approve", policy=EV_POLICY)
        ev_check("V10", "AT_RISK reviewer gets a note in the audit log", "note about the refresher",
                 notes[0] if notes else "no note", any("refresher due" in n for n in notes), "governance_check")
        probs, notes = governance_check("Zoe Unknown", "Approve", policy=EV_POLICY)
        want = EV_POLICY["governance_gates"].get("unknown_person", "warn")
        got = "blocked" if probs else "note only"
        ev_check("V10", "person not on the roster", "blocked" if want == "block" else "note only", got,
                 got == ("blocked" if want == "block" else "note only"), "governance_check", f"policy says {want}")

        # 3. Conflict of interest screen
        src_d = "Section D staff disclosures"
        thr = EV_POLICY["governance_gates"]["coi_threshold"]
        depot_b = {"label": "Dispatch a second bus, Depot B", "type": "dispatch_second_bus",
                   "parties": ["Metro Coach Services LLC"]}
        in_house = {"label": "Hold the connecting train", "type": "hold_train", "parties": ["Agency (in-house)"]}
        for who, opt, want_score, want_block in [("Gabe Ortiz", depot_b, 0.8, True), ("Chen Wei", depot_b, 0.0, False),
                                                 ("Ana Lopez", depot_b, 0.0, False), ("Gabe Ortiz", in_house, 0.6, False)]:
            sc = rule_screen(STAFF_DISCLOSURES[who]["facts"], decisions_for_option(opt), thr)
            ev_check("V10", f"conflict score, {who} approving {opt['label']}", want_score, sc["rule_score"],
                     abs(sc["rule_score"] - want_score) < 0.001, src_d)
            probs, _ = governance_check(who, "Approve", option=opt, policy=EV_POLICY)
            blocked = any("conflict of interest" in p for p in probs)
            ev_check("V10", f"{who} approves {opt['label']}", "must recuse" if want_block else "allowed",
                     "must recuse" if blocked else "allowed", blocked == want_block, "governance_check")
        probs, _ = governance_check("Gabe Ortiz", "Reject", option=depot_b, policy=EV_POLICY)
        ev_check("V10", "Gabe Ortiz rejects the Depot B option", "allowed", "blocked" if probs else "allowed", not probs,
                 "governance_check", "the conflict gate only applies to Approve")

        # Section D sample cases, scores worked out by hand from the weights
        def _facts(entity, kind, owner="official"):
            return ExtractedFacts(official_name="Test", holdings=[Holding(entity=entity, kind=kind, owner=owner)])
        def _dec(bill, subject):
            return [Decision(bill=bill, subject=subject, action="voted")]
        EV_COI_GOLD = [
            ("own stock, named in the bill", _facts("TechCorp Inc.", "direct_stock"),
             _dec("Tech Antitrust Reform Act", "exempting TechCorp's cloud division"), 1.0, True),
            ("spouse's firm gets the contract", _facts("Nair & Sons Construction LLC", "family_business", "spouse"),
             _dec("Highway Repair Program", "road repair contract awarded to Nair & Sons"), 0.8, True),
            ("donor in the same sector only", _facts("HealthFirst Pharmaceuticals", "donation_received", "donor"),
             _dec("Prescription Drug Price Cap Act", "capping prescription drug prices"), 0.53, False),
            ("blind trust, same sector only", _facts("Tech growth portfolio", "blind_trust"),
             _dec("Digital Platform Regulation Act", "data privacy rules for technology companies"), 0.3, False),
            ("index fund, unrelated vote", _facts("Total Market Index Fund", "index_fund"),
             _dec("Library repainting", "repaint public libraries"), 0.0, False),
        ]
        for label, facts, decs, want_score, want_over in EV_COI_GOLD:
            sc = rule_screen(facts, decs, thr)
            ev_check("V10", f"conflict screen: {label}", f"{want_score}, " + ("recuse" if want_over else "below threshold"),
                     f"{sc['rule_score']}, " + ("recuse" if sc["over_threshold"] else "below threshold"),
                     abs(sc["rule_score"] - want_score) < 0.011 and sc["over_threshold"] == want_over, "edge case")

        # 4. Guardrails, using a made up language model answer so no model is needed
        fake = ComplianceReport(answer="Everyone is fine, 100% compliant.", compliance_rate_pct=100.0, ethics_notes="",
                                action_items=[ActionItem(official="Eli Turner", status="COMPLIANT", missing_courses=[],
                                                         action="none", priority="low"),
                                              ActionItem(official="Zara Invented", status="COMPLIANT", missing_courses=[],
                                                         action="none", priority="low")])
        issues = verify_report(fake)
        ev_check("V10", "Section C guardrail finds the made up facts", "4 or more problems", len(issues), len(issues) >= 4,
                 "made up report", "wrong status, missing courses, invented person, people left out, wrong rate")
        fixed = reconcile_report(fake)
        truth = {r["official"]: r["status"] for r in eth_check_all()}
        got = {a.official: a.status for a in fixed.action_items}
        ev_check("V10", "after the guardrail the report matches the rules", "same people and statuses as the rule engine",
                 "matches" if got == truth else f"differs: {got}", got == truth, "made up report")

        bad = {"final": FinalReport(official="Test", recommendation="CLEAR", risk_score=0.1, summary="No issue.",
                                    due_process_note="A person decides."),
               "analysis": AnalystOutput(flags=[Flag(type="family_interest", severity="high", entity="Metro Coach Services LLC",
                                                     decision="Depot B dispatch", reasoning="spouse owns the operator")],
                                         overall_risk_score=0.9),
               "review": ReviewerOutput(verdict_on_analysis="agree"),
               "screen": {"over_threshold": True, "rule_score": 0.8, "threshold": thr, "hits": []}}
        final, g_issues = coi_enforce(bad)
        ev_check("V10", "Section D guardrail raises CLEAR to ESCALATE", "ESCALATE", final.recommendation,
                 final.recommendation == "ESCALATE", "made up report", "high flag, no stated mitigation")

        # 5. Audit of every decision the case store accepted
        accepted, bad_training, bad_coi = 0, [], []
        for case in APP.cases.values():
            gates = get_case_policy(case).get("governance_gates", {})
            for d in case.get("decisions", []):
                if d.get("decision") not in ("Approve", "Revise") or d.get("stage") not in ("first review", "second review"):
                    continue
                accepted += 1
                st = eth_check_official(d["reviewer"])
                if st.get("status") in gates.get("training_block_statuses", []):
                    bad_training.append(f"{case['case_id']} {d['reviewer']}")
                opt = next((o for o in case["analysis"].get("options", []) if o["option_id"] == d.get("option_id")), None)
                disc = STAFF_DISCLOSURES.get(d["reviewer"])
                if d["decision"] == "Approve" and opt and disc:
                    sc = rule_screen(disc["facts"], decisions_for_option(opt), gates.get("coi_threshold", thr))
                    if any(h["match"] == "name" and h["score"] >= sc["threshold"] for h in sc["hits"]):
                        bad_coi.append(f"{case['case_id']} {d['reviewer']}")
        src_a = f"{accepted} accepted reviewer decisions in the case store"
        ev_check("V10", "accepted decisions by a NON_COMPLIANT reviewer", 0, len(bad_training), not bad_training if accepted else None,
                 src_a, ", ".join(bad_training) if bad_training else ("" if accepted else "no decisions in the case store yet"))
        ev_check("V10", "accepted approvals with a recusal level conflict", 0, len(bad_coi), not bad_coi if accepted else None,
                 src_a, ", ".join(bad_coi) if bad_coi else ("" if accepted else "no decisions in the case store yet"))
        blocked_t = sum(1 for e in APP.audit if e.get("event") == "Submission blocked" and "ethics training" in str(e.get("detail")))
        blocked_c = sum(1 for e in APP.audit if e.get("event") == "Submission blocked" and "conflict of interest" in str(e.get("detail")))
        ev_check("V10", "submissions the training gate blocked during this run", "", blocked_t, "info", "audit log")
        ev_check("V10", "submissions the conflict gate blocked during this run", "", blocked_c, "info", "audit log")
    finally:
        ETH_ROSTER.clear()
        ETH_ROSTER.extend(_saved_roster)
ev_part_table("V10")

# ====================================================================================================
# notebook cell 96: V11. Scorecard and saved results
# ====================================================================================================
#@title V11. Scorecard and saved results
EV_SCORECARD = pd.DataFrame(EV_RESULTS)
if len(EV_SCORECARD):
    summary = (EV_SCORECARD.assign(n=1).pivot_table(index="part", columns="result", values="n", aggfunc="sum", fill_value=0)
               .reindex(columns=["pass", "FAIL", "skipped", "reported"], fill_value=0))
    display(summary)
    fails = EV_SCORECARD[EV_SCORECARD["result"] == "FAIL"]
    if len(fails):
        print("Checks that failed (each one is something to fix or explain):")
        display(fails[["part", "check", "expected", "actual", "note"]])
    EV_SCORECARD.to_csv(os.path.join(EV_DIR, "evaluation_scorecard.csv"), index=False)
    with open(os.path.join(EV_DIR, "evaluation_summary.json"), "w") as f:
        json.dump({"evaluated_at": iso(utcnow()), "rules_version": EV_POLICY["version"],
                   "real_data": {"gtfsg2": EV_SNAPSHOT["source"] if EV_SNAPSHOT else EV_SNAPSHOT_STATUS,
                                 "busdv2": EV_BUS.get("status"), "feed_summary": EV_FEED_SUMMARY},
                   "counts": {k: int(v) for k, v in EV_SCORECARD["result"].value_counts().items()},
                   "checks": EV_RESULTS}, f, indent=1, default=str)
    passed = int((EV_SCORECARD["result"] == "pass").sum())
    failed = int((EV_SCORECARD["result"] == "FAIL").sum())
    skipped = int((EV_SCORECARD["result"] == "skipped").sum())
    reported = int((EV_SCORECARD["result"] == "reported").sum())
    print(f"{passed} of {passed + failed} checks passed. {skipped} skipped (no data for them in this run), "
          f"{reported} numbers reported without a pass mark. Saved to {EV_DIR}")
else:
    print("No checks recorded. Run V1 to V10 first.")

# ====================================================================================================
# notebook cell 98: V12. Evaluation dashboard (visual summary of V2 to V10)
# ====================================================================================================
#@title V12. Evaluation dashboard (visual summary of V2 to V10)
import html as _vh
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _ipython_shim import HTML, display

EV_PART_LABELS = {
    "V2": ("Real feed quality", "NJ Transit GTFSG2: freshness, ids, positions, delays, alerts"),
    "V3": ("Second feed cross-check", "BUSDV2 against GTFSG2: same buses, routes and positions"),
    "V4": ("Pipeline on real buses", "every real bus through Section A and E2 against the published rules"),
    "V5": ("Reading alert text", "delay and cause extraction on a hand-labeled test set"),
    "V6": ("Computer vision counts", "detector counts against hand counts"),
    "V7": ("Scoring rules", "presets, thresholds, formulas, determinism, monotonicity"),
    "V8": ("Workflow paths", "gallery cases end where the diagram says"),
    "V9": ("Explanation grounding", "narratives use only rule-produced numbers"),
    "V10": ("Governance gates", "ethics training and conflict of interest"),
}
# Status colors (reserved for pass / fail) always ship with an icon and a label; skipped and reported are neutral.
EV_STYLE = {"pass": ("#0ca30c", "✓", "passed"), "FAIL": ("#d03b3b", "✗", "failed"),
            "skipped": ("#a3a29c", "–", "skipped"), "reported": ("#2a78d6", "i", "reported")}
INK, INK2, SURFACE, RULE = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _ev_counts(rows):
    return {k: sum(r["result"] == k for r in rows) for k in EV_STYLE}


def _ev_tile(value, label, sub="", tone=None):
    accent = f"border-top:4px solid {EV_STYLE[tone][0]};" if tone else f"border-top:4px solid {RULE};"
    return (f"<div style='flex:1 1 150px;min-width:150px;background:#fff;border:1px solid {RULE};{accent}border-radius:8px;"
            f"padding:12px 14px'><div style='font-size:26px;font-weight:650;color:{INK};line-height:1.1'>{value}</div>"
            f"<div style='font-size:13px;color:{INK};margin-top:4px'>{label}</div>"
            f"<div style='font-size:11.5px;color:{INK2};margin-top:2px'>{sub}</div></div>")


def ev_dashboard_html():
    rows = EV_RESULTS
    tot = _ev_counts(rows)
    graded = tot["pass"] + tot["FAIL"]
    when = iso(utcnow())[:16].replace("T", " ")
    real = pd.DataFrame(globals().get("EV_REAL_CASES") or [])
    sig = real[real["expected"] == "SIGNIFICANT"] if len(real) else real
    by = {r["check"]: r for r in rows}
    tiles = [
        _ev_tile(f"{tot['pass']} / {graded}", "checks passed", f"{round(100 * tot['pass'] / graded) if graded else 0}% of graded checks", "pass"),
        _ev_tile(tot["FAIL"], "checks failed", "listed below with what to fix", "FAIL" if tot["FAIL"] else None),
        _ev_tile(tot["skipped"], "checks skipped", "no data in this run (not counted as passed)"),
        _ev_tile(len(real), "real buses run end to end",
                 f"{int(real['agrees'].sum())} reached the rule's answer" if len(real) else "no live snapshot"),
        _ev_tile(f"{int(sig['agrees'].sum())} / {len(sig)}" if len(sig) else "—", "real 15+ min delays escalated",
                 "recall on real NJ Transit buses", ("pass" if len(sig) and sig["agrees"].all() else "FAIL") if len(sig) else None),
    ]
    legend = " ".join(f"<span style='display:inline-flex;align-items:center;gap:5px;margin-right:14px;font-size:12px;color:{INK2}'>"
                      f"<span style='width:12px;height:12px;border-radius:3px;background:{c}'></span>{icon} {lab}</span>"
                      for c, icon, lab in EV_STYLE.values())
    bars = []
    for part, (name, desc) in EV_PART_LABELS.items():
        pr = [r for r in rows if r["part"] == part]
        if not pr:
            continue
        cnt = _ev_counts(pr)
        n = len(pr)
        segs = "".join(
            f"<div title='{name}: {cnt[k]} {EV_STYLE[k][2]}' style='width:{100 * cnt[k] / n:.2f}%;background:{EV_STYLE[k][0]};"
            f"height:100%;border-right:2px solid {SURFACE};box-sizing:border-box'></div>" for k in EV_STYLE if cnt[k])
        label = ", ".join(f"{cnt[k]} {EV_STYLE[k][2]}" for k in EV_STYLE if cnt[k])
        flag = (f"<span style='color:{EV_STYLE['FAIL'][0]};font-weight:600'>✗ {cnt['FAIL']} to fix</span>" if cnt["FAIL"]
                else f"<span style='color:#006300;font-weight:600'>✓ all graded checks pass</span>" if cnt["pass"] else "")
        bars.append(
            f"<div style='display:grid;grid-template-columns:minmax(150px,230px) 1fr minmax(120px,170px);gap:12px;align-items:center;"
            f"padding:8px 0;border-bottom:1px solid {RULE}'>"
            f"<div><div style='font-size:13px;font-weight:600;color:{INK}'>{part} · {name}</div>"
            f"<div style='font-size:11.5px;color:{INK2}'>{desc}</div></div>"
            f"<div><div style='display:flex;height:14px;border-radius:4px;overflow:hidden;background:{RULE}'>{segs}</div>"
            f"<div style='font-size:11.5px;color:{INK2};margin-top:3px'>{label}</div></div>"
            f"<div style='font-size:12px;text-align:right'>{flag}</div></div>")
    fails = [r for r in rows if r["result"] == "FAIL"]
    fail_html = "".join(
        f"<div style='border-left:4px solid {EV_STYLE['FAIL'][0]};background:#fff;border-radius:6px;padding:8px 12px;margin:6px 0;"
        f"border:1px solid {RULE};border-left-width:4px'><div style='font-size:13px;color:{INK}'><b>✗ {r['part']} · "
        f"{_vh.escape(r['check'])}</b></div><div style='font-size:12px;color:{INK2}'>expected {_vh.escape(r['expected'])} · "
        f"got <b style='color:{INK}'>{_vh.escape(r['actual'])}</b>{(' · ' + _vh.escape(r['note'])) if r['note'] else ''}</div></div>"
        for r in fails) or f"<div style='font-size:13px;color:#006300'>✓ No failed checks.</div>"
    skips = [r for r in rows if r["result"] == "skipped"]
    skip_html = "".join(f"<li>{r['part']} · {_vh.escape(r['check'])}: {_vh.escape(r['note'] or r['actual'])}</li>" for r in skips)
    src = (EV_SNAPSHOT or {}).get("source", EV_SNAPSHOT_STATUS) if "EV_SNAPSHOT" in globals() else "—"
    return (
        f"<div style='font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:{SURFACE};color:{INK};"
        f"border:1px solid {RULE};border-radius:12px;padding:18px 20px;max-width:1100px'>"
        f"<div style='font-size:20px;font-weight:650'>System evaluation · {tot['pass']} of {graded} graded checks passed</div>"
        f"<div style='font-size:12.5px;color:{INK2};margin:4px 0 14px'>Run {when} ET · expected answers from published rules "
        f"v{EV_POLICY['version']} · NJ Transit data: {_vh.escape(str(src))} · BUSDV2: {_vh.escape(str(EV_BUS.get('status')))}</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:10px;margin-bottom:18px'>{''.join(tiles)}</div>"
        f"<div style='font-size:15px;font-weight:650;margin-bottom:4px'>Results by part</div><div style='margin-bottom:6px'>{legend}</div>"
        f"{''.join(bars)}"
        f"<div style='font-size:15px;font-weight:650;margin:18px 0 4px'>Failed checks</div>{fail_html}"
        + (f"<div style='font-size:15px;font-weight:650;margin:18px 0 4px'>Skipped (no data this run)</div>"
           f"<ul style='font-size:12.5px;color:{INK2};margin:4px 0 0 18px;padding:0'>{skip_html}</ul>" if skips else "")
        + "</div>")


def ev_dashboard_figure():
    """Two evidence panels: real buses against the two significance rules, and detector vs hand counts."""
    real = pd.DataFrame(globals().get("EV_REAL_CASES") or [])
    cvr = globals().get("EV_CV_ROWS")
    panels = [p for p in ("real", "cv") if (p == "real" and len(real) and "R" in real) or (p == "cv" and cvr is not None and len(cvr))]
    if not panels:
        return None
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.4), facecolor="white")
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        ax.set_facecolor("white")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
        ax.set_axisbelow(True)
        if p == "real":
            d = real.dropna(subset=["delay_raw_min", "R"])
            gates = [("SIGNIFICANT", "#eb6834"), ("BELOW_THRESHOLD", "#2a78d6"), ("MONITORING", "#1baf7a")]
            for g, col in gates:
                s = d[d["engine_gate"] == g]
                if len(s):
                    ax.scatter(s["delay_raw_min"], s["R"], s=46, color=col, edgecolor="white", linewidth=1.2, zorder=3,
                               label=f"{g.replace('_', ' ').lower()} ({len(s)})")
            ht = EV_POLICY["hard_trigger"]["delay_min"]
            rs = EV_POLICY["significance_R"]
            ax.axvline(ht, color="#52514e", linestyle=(0, (4, 3)), linewidth=1)
            ax.axhline(rs, color="#52514e", linestyle=(0, (1, 3)), linewidth=1)
            ax.text(ht + 0.4, 0.97, f"{ht} min hard trigger", transform=ax.get_xaxis_transform(), fontsize=9, color=INK2, va="top")
            ax.text(0.01, rs + 0.015, f"R = {rs} significance line", transform=ax.get_yaxis_transform(), fontsize=9, color=INK2)
            ax.set_xlabel("Delay reported by NJ Transit (min)")
            ax.set_ylabel("Reliability impact R")
            ax.set_ylim(0, max(1.0, float(d["R"].max()) + 0.05) if len(d) else 1)
            ax.set_title(f"{int(real['agrees'].sum())} of {len(real)} real buses reached the rule's answer",
                         loc="left", fontsize=12, color=INK, fontweight="bold")
            ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
                      title="Engine gate", title_fontsize=9)
        else:
            c = cvr.reset_index(drop=True)
            x = np.arange(len(c))
            lab = [("photo" if "STOP" in r.camera else f"video {r.frame}") for r in c.itertuples()]
            ax.vlines(x, c["visible"], c["visible"] + c["partly_visible"], color="#a3a29c", linewidth=8, zorder=2)
            ax.scatter(x, c["visible"], marker="_", s=180, color="#52514e", zorder=3)
            ok = c["within_range"]
            ax.scatter(x[ok], c["detector"][ok], s=60, color="#2a78d6", edgecolor="white", linewidth=1.2, zorder=4)
            ax.scatter(x[~ok], c["detector"][~ok], s=60, marker="D", color="#2a78d6", edgecolor="white", linewidth=1.2, zorder=4)
            for xi, (g, k) in enumerate(zip(c["detector"], ok)):
                if not k:
                    ax.annotate("outside", (xi, g), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=INK2)
            ax.set_xticks(x, lab, rotation=35, ha="right", fontsize=8.5)
            ax.set_ylabel("People in the frame")
            ax.set_ylim(0, max(c["detector"].max(), (c["visible"] + c["partly_visible"]).max()) + 2)
            ax.set_title(f"Detector inside the hand-count range in {int(ok.sum())} of {len(c)} frames",
                         loc="left", fontsize=12, color=INK, fontweight="bold")
            ax.legend(handles=[Line2D([0], [0], color="#a3a29c", linewidth=8, label="hand count: visible to incl. partly visible"),
                               Line2D([0], [0], marker="o", color="w", markerfacecolor="#2a78d6", markersize=8, label="detector, inside range"),
                               Line2D([0], [0], marker="D", color="w", markerfacecolor="#2a78d6", markersize=7, label="detector, outside range")],
                      frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    return fig


if not EV_RESULTS:
    print("No evaluation results yet. Run V1 to V11 first.")
else:
    _html_out = ev_dashboard_html()
    display(HTML(_html_out))
    with open(os.path.join(EV_DIR, "evaluation_dashboard.html"), "w") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>System evaluation</title><body style='margin:24px;background:#fff'>"
                + _html_out + "</body>")
    _fig = ev_dashboard_figure()
    if _fig is not None:
        _fig.savefig(os.path.join(EV_DIR, "evaluation_evidence.png"), dpi=150, bbox_inches="tight")
        plt.show()
    print("Saved evaluation_dashboard.html and evaluation_evidence.png to", EV_DIR)


# ====================================================================================================
# notebook cell 100: E4 · Outputs: interactive map, tables and charts
# ====================================================================================================
#@title E4 · Outputs: interactive map, tables and charts
import folium
import plotly.graph_objects as go

STATUS_COLOR = {"MONITORING": "#6b7280", "CONFLICT_REVIEW": "#d97706", "SAFE_DEFAULT": "#2563eb", "AWAITING_REVIEW": "#dc2626",
                "AWAITING_SECOND_REVIEW": "#b91c1c", "AWAITING_IMPLEMENTATION": "#7c3aed", "IMPLEMENTED": "#059669",
                "EVALUATED": "#047857", "CLOSED_FALSE_ALARM": "#9ca3af"}
DEST_ICON = {"hospital": "plus-sign", "dialysis": "tint", "school": "book", "employment_center": "briefcase", "clinic": "heart", "retail": "shopping-cart"}


def _vcolor(v):
    # light → dark purple ramp for area vulnerability (area-level aggregate)
    ramp = ["#f2f0f7", "#dadaeb", "#bcbddc", "#9e9ac8", "#756bb1", "#54278f"]
    return ramp[min(len(ramp) - 1, int(v * len(ramp)))]


def area_vulnerability(a, policy):
    b, w = policy["bounds"], policy["V_weights"]
    return (w["N"] * n_benefit(a["transit_dependence"], *b["transit_dependence"]) + w["I"] * n_benefit(a["income_disadvantage"], *b["income_disadvantage"])
            + w["B"] * n_benefit(a["disability"], *b["disability"])) / (w["N"] + w["I"] + w["B"])


def build_map(case):
    pol = APP.PM.current
    m = folium.Map(location=list(CITY_CENTER), zoom_start=13, tiles="CartoDB positron", control_scale=True)
    fg_area = folium.FeatureGroup(name="Community areas (vulnerability, area-level)", show=True)
    for a in NET["areas"].values():
        v = area_vulnerability(a, pol)
        folium.Rectangle(bounds=a["bounds"], color="#ffffff", weight=1, fill=True, fill_color=_vcolor(v), fill_opacity=0.45,
                         tooltip=f"{a['name']}: transit dependence {a['transit_dependence']:.0%}, income disadvantage {a['income_disadvantage']:.0%}, "
                                 f"disability {a['disability']:.0%} · vulnerability index {v:.2f}").add_to(fg_area)
    fg_area.add_to(m)
    affected = set(case["raw"]["affected_stop_ids"]) if case else set()
    rid = case["raw"]["route_id"] if case else None
    fg_routes = folium.FeatureGroup(name="Routes and stops", show=True)
    for r in NET["routes"].values():
        hot = r["id"] == rid
        folium.PolyLine(r["path"], color=r["color"], weight=7 if hot else 4, opacity=0.95 if hot else 0.55,
                        dash_array="8 6" if r["mode"] == "rail" else None,
                        tooltip=f"{r['id']} {r['name']} ({r['mode']}, every {r['headway_min']} min)").add_to(fg_routes)
        for s in r["stops"]:
            on = s["stop_id"] in affected
            folium.CircleMarker((s["lat"], s["lon"]), radius=6 if on else 3, color="#b91c1c" if on else r["color"], fill=True,
                                fill_opacity=0.9, tooltip=f"{s['name']} · {s['stop_id']}" + (" · AFFECTED" if on else "")).add_to(fg_routes)
    fg_routes.add_to(m)
    fg_dest = folium.FeatureGroup(name="Essential destinations", show=True)
    for d in NET["destinations"].values():
        folium.Marker((d["lat"], d["lon"]), tooltip=f"{d['name']} ({d['type'].replace('_', ' ')})",
                      icon=folium.Icon(color="darkred" if d["type"] in ("hospital", "dialysis") else "cadetblue", icon=DEST_ICON[d["type"]])).add_to(fg_dest)
    for d in NET["depots"].values():
        folium.Marker((d["lat"], d["lon"]), tooltip=f"{d['name']}: {d['spare_buses']} spare bus(es), {d['operators']} operator(s)",
                      icon=folium.Icon(color="gray", icon="home")).add_to(fg_dest)
    fg_dest.add_to(m)
    if case:
        A = case["analysis"]
        stops = [NET["stops"][s] for s in case["raw"]["affected_stop_ids"]]
        mid = (statistics.mean(s["lat"] for s in stops), statistics.mean(s["lon"] for s in stops))
        fg_case = folium.FeatureGroup(name=f"{case['case_id']} incident and options", show=True)
        f = A.get("fields", {})
        sc = A.get("scores") or {}
        pop = (f"<b>{case['case_id']}</b><br>{_html.escape(case['scenario_name'])}<br>Status: <b>{case['status']}</b><br>"
               f"Delay: {f.get('delay_min', '—')} min · gap {f.get('headway_ratio', '—')}× · pass-ups {f.get('passups', '—')}<br>"
               f"C={A['confidence']['C'] if A.get('confidence') else '—'} · R={sc.get('R', '—')} V={sc.get('V', '—')} A={sc.get('A', '—')} P={sc.get('P', '—')}")
        folium.Marker(mid, popup=folium.Popup(pop, max_width=320), tooltip="Incident",
                      icon=folium.Icon(color="red" if case["status"] in REVIEW_STATUSES else "orange", icon="warning-sign")).add_to(fg_case)
        # Vehicle with delay
        v0 = stops[len(stops) // 2]
        folium.CircleMarker((v0["lat"] + 0.0012, v0["lon"]), radius=9, color="#111827", fill=True, fill_color="#facc15", fill_opacity=1,
                            tooltip=f"Vehicle on {case['raw']['route_id']}: {f.get('delay_min', '?')} min late, load {f.get('load_factor', '?')}").add_to(fg_case)
        chosen = case.get("approved_option_id")
        for o in A.get("options", []):
            if not o["feasible"]:
                continue
            style = {"weight": 5 if o["option_id"] == chosen else 3, "opacity": 0.95 if o["option_id"] == chosen else 0.6}
            if o["type"] == "dispatch_second_bus":
                dep = NET["depots"][o["option_id"].split("@")[1]]
                folium.PolyLine([(dep["lat"], dep["lon"]), mid], color="#16a34a", dash_array="6 6", tooltip=o["label"], **style).add_to(fg_case)
            elif o["type"] == "reroute":
                a, b = (stops[0]["lat"], stops[0]["lon"]), (stops[-1]["lat"], stops[-1]["lon"])
                off = 0.009
                folium.PolyLine([a, (a[0] + off, a[1] + off), (b[0] + off, b[1] + off), b], color="#ea580c", dash_array="4 6",
                                tooltip=o["label"], **style).add_to(fg_case)
            elif o["type"] == "protect_connection":
                ts = NET["stops"][case["raw"]["connections"]["transfer_stop_id"]]
                folium.CircleMarker((ts["lat"], ts["lon"]), radius=12, color="#0891b2", fill=False, weight=3, tooltip=o["label"]).add_to(fg_case)
        fg_case.add_to(m)
        m.fit_bounds([[min(s["lat"] for s in stops) - 0.02, min(s["lon"] for s in stops) - 0.03],
                      [max(s["lat"] for s in stops) + 0.02, max(s["lon"] for s in stops) + 0.03]])
    folium.LayerControl(collapsed=True).add_to(m)
    legend = ("<div style='position:fixed;bottom:18px;left:18px;z-index:9999;background:white;padding:8px 10px;border-radius:6px;"
              "font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)'><b>Synthetic network (fictional)</b><br>"
              "<span style='color:#b91c1c'>●</span> affected stops &nbsp;<span style='color:#16a34a'>- -</span> dispatch path<br>"
              "<span style='color:#ea580c'>- -</span> reroute &nbsp;<span style='color:#0891b2'>○</span> protected transfer<br>"
              "Shading = area vulnerability (darker = higher)</div>")
    m.get_root().html.add_child(folium.Element(legend))
    return m


def map_iframe(case, height=540):
    doc = build_map(case).get_root().render()
    return f'<iframe srcdoc="{_html.escape(doc, quote=True)}" style="width:100%;height:{height}px;border:0;border-radius:8px"></iframe>'


def ph(df, msg="—"):
    return df if (df is not None and len(df)) else pd.DataFrame({"info": [msg]})


def evidence_df(case):
    if not case:
        return ph(None)
    rows = [{"source_id": e["source_id"], "source_type": e["source_type"], "status": e["status"], "quality": e.get("quality"),
             "timestamp_utc": e["timestamp"][:19], "age_s": e["age_s"],
             "fields": ", ".join(f"{k}={v}" for k, v in e["fields"].items()), "note": e["note"]} for e in case["analysis"]["evidence"]]
    return ph(pd.DataFrame(rows))


def exceptions_df(case):
    if not case:
        return ph(None)
    return ph(pd.DataFrame(case["analysis"]["exceptions"]), "No exception paths taken")


def trace_md(case):
    if not case:
        return "_No case yet._"
    icon = lambda r: "🟥" if any(k in r.upper() for k in ("SAFE", "NO", "MONITOR", "CONFLICT", "FAIL")) else "🟩"
    lines = [f"### Pipeline trace — {case['case_id']}", f"**Gate:** `{case['analysis']['gate']}` — {case['analysis']['gate_reason']}", ""]
    lines += [f"{icon(t['result'])} **{t['stage']}** · {t['result']}" + (f"  \n<small>{_html.escape(t['detail'])}</small>" if t["detail"] else "")
              for t in case["analysis"]["trace"]]
    if case.get("safe_default_reason"):
        lines.append(f"\n🔵 **Safe default:** {case['safe_default_reason']}")
    return "\n\n".join(lines)


def conf_df(case):
    if not case or not case["analysis"].get("confidence"):
        return ph(None, "Confidence not computed (agent safe mode)")
    c = case["analysis"]["confidence"]
    w = get_case_policy(case)["confidence_weights"]
    rows = [{"component": k, "meaning": m, "value": c[k], "weight": w[k], "contribution": r3(w[k] * c[k])}
            for k, m in [("Q", "source quality"), ("F", "freshness"), ("G", "cross-source agreement"), ("M", "completeness")]]
    rows.append({"component": "penalty", "meaning": "unresolved conflict", "value": c["penalty"], "weight": -1, "contribution": -c["penalty"]})
    rows.append({"component": "C", "meaning": f"threshold {get_case_policy(case)['confidence_threshold']}", "value": c["C"], "weight": None, "contribution": c["C"]})
    return pd.DataFrame(rows)


def norm_df(case):
    if not case or not case["analysis"].get("normalization"):
        return ph(None, "Impact scores not computed")
    return pd.DataFrame(case["analysis"]["normalization"])[["group", "component", "feature", "raw", "L", "U", "method", "normalized", "weight", "contribution", "note"]]


def rvap_plot(case):
    fig = go.Figure()
    if case and case["analysis"].get("scores"):
        s = case["analysis"]["scores"]
        pol = get_case_policy(case)
        fig.add_bar(x=["R reliability", "V vulnerability", "A access", "P priority"], y=[s["R"], s["V"], s["A"], s["P"]],
                    marker_color=["#2563eb", "#7c3aed", "#0891b2", "#111827"], text=[f"{v:.3f}" for v in (s["R"], s["V"], s["A"], s["P"])],
                    textposition="outside")
        fig.add_hline(y=pol["significance_R"], line_dash="dash", line_color="#dc2626", annotation_text=f"R significance {pol['significance_R']}")
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), yaxis_range=[0, 1.1], title="Component scores (0–1)", plot_bgcolor="white")
    return fig


def sig_md(case):
    if not case:
        return ""
    A = case["analysis"]
    trig = "\n".join(f"- **{t['trigger']}** (source: {t['source']}{', official ✔' if t['official'] else ''})" for t in A["hard_triggers"]) or "- none"
    miss = ", ".join(A["missing"]) or "none"
    return (f"### Significance\n**Result:** {'SIGNIFICANT' if A['significant'] else 'not significant'} · tier **{A['tier'] or '—'}**\n\n"
            f"**Hard triggers (bypass R threshold):**\n{trig}\n\n**Missing values (not imputed):** {miss}")


def options_df(case):
    if not case or not case["analysis"].get("ranking"):
        return ph(None, "No ranked options (case not significant, or no feasible option)")
    return pd.DataFrame([{k: r[k] for k in ["rank", "option_id", "label", "DR", "EB", "AP", "OF", "CE", "Ci", "Si", "cost_usd"]} for r in case["analysis"]["ranking"]])


def infeasible_df(case):
    if not case:
        return ph(None)
    inf = [{"option": o["label"], "reason": o["infeasible_reason"]} for o in case["analysis"].get("options", []) if not o["feasible"]]
    return ph(pd.DataFrame(inf), "All generated options are feasible")


def options_plot(case):
    fig = go.Figure()
    if case and case["analysis"].get("ranking"):
        rk = case["analysis"]["ranking"]
        labels = [f"#{r['rank']} {r['label'][:34]}" for r in rk][::-1]
        colors = {"DR": "#2563eb", "EB": "#7c3aed", "AP": "#0891b2", "OF": "#16a34a", "CE": "#f59e0b"}
        for c in ["DR", "EB", "AP", "OF", "CE"]:
            fig.add_bar(y=labels, x=[r["weighted"][c] for r in rk][::-1], name=c, orientation="h", marker_color=colors[c])
        fig.update_layout(barmode="stack")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=20), title="Si = Ci × weighted components (stacked)", plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.15))
    return fig


def decisions_df(case):
    if not case or not case["decisions"]:
        return ph(None, "No human decisions yet")
    return pd.DataFrame(case["decisions"])[["ts", "stage", "reviewer", "decision", "option_id", "recommended_option_id", "override",
                                             "response_s", "tier", "deadline_at", "revision_no", "reason"]]


def deadline_md(case):
    if not case:
        return "_No active case._"
    st = case["status"]
    if st in REVIEW_STATUSES:
        left = APP.seconds_left(case)
        mm, ss = divmod(max(0, int(left)), 60)
        r = case["review"]
        who = "second reviewer" if st == "AWAITING_SECOND_REVIEW" else "first reviewer"
        bar = "🟥" if left < 60 * DEMO_DEADLINE_SCALE else "🟧" if left < 180 * DEMO_DEADLINE_SCALE else "🟩"
        extra = " · ⚠️ **No feasible option** — revise or reject" if r.get("no_feasible_option") else ""
        return (f"### {bar} {mm:02d}:{ss:02d} left — waiting for {who}\nTier **{r['tier']}** ({r['deadline_min']} min, covers both reviewers) · "
                f"revisions used {case['revisions']}/{get_case_policy(case)['max_revisions']}{extra}. No response → safe default.")
    return f"**Status:** `{st}` — {STATUS_HELP.get(st, '')}"


def outcome_df(case):
    if not case or not case.get("outcome"):
        return ph(None, "No outcome evaluated yet")
    o = case["outcome"]
    if "pred_delay_reduction_min" not in o:
        return pd.DataFrame([o])
    rows = [("Delay reduction (min)", o["pred_delay_reduction_min"], o["obs_delay_reduction_min"]),
            ("Access protection (0–1)", o["pred_access"], o["obs_access"]),
            ("Equity: vulnerable riders served", o["pred_equity_riders"], o["obs_equity_riders"]),
            ("Cost (USD)", o["pred_cost_usd"], o["actual_cost_usd"])]
    df = pd.DataFrame([{"metric": m, "predicted": p, "observed": ob, "difference": r3((ob or 0) - (p or 0))} for m, p, ob in rows])
    extra = pd.DataFrame([{"metric": "False alarm", "predicted": "—", "observed": o["false_alarm"], "difference": "—"},
                          {"metric": "Rider-reported impact (1 low – 5 high)", "predicted": "—", "observed": o["rider_reported_impact"], "difference": "—"}])
    return pd.concat([df.astype(object), extra], ignore_index=True)


def outcome_plot(case):
    fig = go.Figure()
    o = case.get("outcome") if case else None
    if o and "pred_delay_reduction_min" in o:
        m = ["Delay reduction", "Access", "Equity riders", "Cost"]
        pr = [o["pred_delay_reduction_min"], o["pred_access"], o["pred_equity_riders"], o["pred_cost_usd"]]
        ob = [o["obs_delay_reduction_min"], o["obs_access"], o["obs_equity_riders"], o["actual_cost_usd"]]
        fig.add_bar(x=m, y=[1.0 if p else 0 for p in pr], name="predicted (=1.0)", marker_color="#94a3b8")
        fig.add_bar(x=m, y=[(b / p) if p else 0 for p, b in zip(pr, ob)], name="observed / predicted", marker_color="#2563eb")
    fig.update_layout(height=300, barmode="group", margin=dict(l=20, r=20, t=30, b=20), title="Observed relative to predicted", plot_bgcolor="white")
    return fig


def versions_df():
    return pd.DataFrame([{"version": v["version"], "released_at": v["released_at"], "approved_by": v["approved_by"], "reason": v["reason"],
                          "changes": "; ".join(f"{c['path']}: {c['old']} → {c['new']}" for c in v["changes"]) or "—",
                          "backtest": v["backtest_summary"] or "—", "active": i == APP.PM.current_index}
                         for i, v in enumerate(APP.PM.versions)])


def proposals_df():
    return ph(pd.DataFrame([{"id": p["id"], "title": p["title"], "changes": "; ".join(f"{c['path']}: {c['old']} → {c['new']}" for c in p["changes"]),
                             "evidence": p["evidence"], "rationale": p["rationale"], "status": p["status"]} for p in APP.proposals]),
              "No open proposals — run the Insight Agent")


def proposal_log_df():
    return ph(pd.DataFrame([{"id": p["id"], "title": p["title"], "status": p["status"], "decided_by": p.get("decided_by"), "reason": p.get("reason"),
                             "backtest": p.get("backtest", "—")} for p in APP.proposal_log]), "No decisions yet")


def audit_df(case_only=False):
    rows = APP.audit if not case_only or not APP.active else APP.active["audit"]
    return ph(pd.DataFrame(rows[::-1][:500]), "Audit log empty")


def cases_df():
    rows = [{"case_id": c["case_id"], "created_at": c["created_at"][:19], "scenario": c["scenario_name"], "status": c["status"],
             "gate": c["analysis"]["gate"], "P": (c["analysis"].get("scores") or {}).get("P"), "tier": c["analysis"].get("tier"),
             "rules": c["policy_version"], "approved_option": c.get("approved_option_id")} for c in APP.cases.values()]
    return ph(pd.DataFrame(rows[::-1]), "No live cases yet")


def monitor_df():
    rows = [{"case_id": c["case_id"], "status": c["status"], "route": c["raw"]["route_id"],
             "confidence": (c["analysis"].get("confidence") or {}).get("C"), "priority_P": (c["analysis"].get("scores") or {}).get("P"),
             "reason": c.get("safe_default_reason") or c["analysis"]["gate_reason"]}
            for c in APP.cases.values() if c["status"] in ("MONITORING", "SAFE_DEFAULT", "CONFLICT_REVIEW")]
    return ph(pd.DataFrame(rows), "Monitoring queue empty")


def alerts_df(case):
    return ph(pd.DataFrame(case["alerts"]) if case else None, "No alerts")


def export_outputs():
    """Write every Colab output (shared case state, evidence, exceptions, scores, scenarios, decisions, outcomes, audit, rules, map)."""
    ts = local_now().strftime("%Y%m%dT%H%M%S%Z")
    path = os.path.join(OUTPUT_DIR, f"transit_dashboard_outputs_{ts}.zip")
    cases = list(APP.cases.values())
    ev, ex, sc, op, dc, oc = [], [], [], [], [], []
    for c in cases:
        A = c["analysis"]
        ev += [{"case_id": c["case_id"], **{k: v for k, v in e.items() if k != "fields"}, "fields": json.dumps(e["fields"])} for e in A["evidence"]]
        ex += [{"case_id": c["case_id"], **e} for e in A["exceptions"]]
        if A.get("scores"):
            sc.append({"case_id": c["case_id"], **{k: A["scores"][k] for k in "RVAP"}, **A["scores"]["components"], **(A["confidence"] or {}), "tier": A["tier"]})
        op += [{"case_id": c["case_id"], **{k: v for k, v in r.items() if k not in ("weights", "weighted")}} for r in A.get("ranking", [])]
        dc += [{"case_id": c["case_id"], **d} for d in c["decisions"]]
        if c.get("outcome"):
            oc.append({"case_id": c["case_id"], **c["outcome"]})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("shared_case_state.json", json.dumps(cases, indent=1, default=str))
        z.writestr("policy_versions.json", json.dumps(APP.PM.versions, indent=1, default=str))
        z.writestr("insight_proposals.json", json.dumps(APP.proposals + APP.proposal_log, indent=1, default=str))
        for name, rows in [("verified_evidence", ev), ("exception_log", ex), ("component_scores", sc), ("scenario_table_and_rank", op),
                           ("human_decisions_and_timing", dc), ("outcome_evaluation", oc), ("audit_record", APP.audit)]:
            z.writestr(f"{name}.csv", pd.DataFrame(rows).to_csv(index=False))
        z.writestr("reviewer_audit_rubber_stamp.csv", APP.rubber_stamp_report().to_csv(index=False))
        z.writestr("incident_history.csv", incident_history_df(APP).to_csv(index=False))
        z.writestr("section_c_ethics_training_status.csv", pd.DataFrame(eth_check_all()).to_csv(index=False))
        z.writestr("section_d_disclosures.csv", disclosures_df().to_csv(index=False))
        try:
            z.writestr("section_b_notebook_form_audit.csv", audit_table().to_csv(index=False))
        except NameError:
            pass
        if APP.active:
            z.writestr(f"map_{APP.active['case_id']}.html", build_map(APP.active).get_root().render())
    APP.log(None, "system", "system", "Export", "Outputs exported", path)
    return path

# ====================================================================================================
# helpers kept from notebook cell 101: E5 · Gradio dashboard (human-in-the-loop)
# ====================================================================================================
# ---- Sections C + D helpers for tab 8
def eth_summary_md():
    t = eth_team_summary()
    return (f"**Roster:** {t['total']} people · compliance rate **{t['compliance_rate_pct']}%** (COMPLIANT + AT_RISK) · "
            f"fully current {t['fully_current_pct']}% · 🟢 {t['COMPLIANT']} · 🟡 {t['AT_RISK']} · 🔴 {t['NON_COMPLIANT']}")


def eth_status_df():
    return ph(pd.DataFrame(eth_check_all()), "Roster empty")


def disclosures_df():
    rows = [{"person": n, "holdings": "; ".join(f"{h.owner}: {h.entity} ({h.kind})" for h in d["facts"].holdings) or "none",
             "disclosure": d["text"]} for n, d in STAFF_DISCLOSURES.items()]
    return pd.DataFrame(rows)


def coi_screen_df(case):
    if not case or not case["analysis"].get("options"):
        return ph(None, "Run a significant case to screen people against its options")
    rows = []
    for n, d in STAFF_DISCLOSURES.items():
        for o in case["analysis"]["options"]:
            if not o["feasible"]:
                continue
            sc = rule_screen(d["facts"], decisions_for_option(o), APP.PM.current["governance_gates"]["coi_threshold"])
            top = max(sc["hits"], key=lambda h: h["score"]) if sc["hits"] else None
            rows.append({"person": n, "option": o["label"], "parties": ", ".join(o["parties"]), "rule_score": sc["rule_score"],
                         "match": top["match"] if top else "—",
                         "gate": "RECUSE" if (top and top["match"] == "name" and sc["over_threshold"]) else ("note" if top else "clear")})
    return pd.DataFrame(rows)


def b_audit_df():
    try:
        df = audit_table()
    except NameError:
        df = None
    return ph(df, "No Section B notebook-form cases yet")

# ---- end of notebook cells ----
import matplotlib.pyplot as _plt
_plt.close("all")                          # free figures drawn during setup
