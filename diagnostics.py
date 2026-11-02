"""Connection check: shows whether each key and login actually works, without ever showing a key.

Used by both web interfaces (Streamlit and Gradio). Every check makes one small real request:
  - AI providers (OpenAI, OpenRouter, Hugging Face): one tiny chat request each ("Reply with the single word: ready")
  - NJ Transit: log in, then read the live vehicle, delay and alert feeds and count buses inside Newark
  - US Census and National Weather Service: one lookup each
"""
import time

SECRET_NAMES = [
    ("OPENAI_API_KEY", "AI agents (default provider)"),
    ("OPENROUTER_API_KEY", "AI agents (fallback 1)"),
    ("HF_API_KEY", "AI agents (fallback 2; HF_TOKEN also works)"),
    ("NJ_TRANSIT_USERNAME", "Live NJ Transit buses, routes and alerts"),
    ("NJ_TRANSIT_PASSWORD", "Live NJ Transit buses, routes and alerts"),
    ("CENSUS_API_KEY", "Census neighbourhood data (optional)"),
    ("NWS_CONTACT_EMAIL", "Weather service contact (optional)"),
    ("DASHBOARD_PASSWORD", "Password for the Gradio app (optional)"),
]

OK, WARN, FAIL, SKIP = "✅ working", "⚠️ check", "❌ not working", "➖ not set"


def _present(E, name):
    v = E.read_secret(name)
    if not v and name == "HF_API_KEY":
        v = E.read_secret("HF_TOKEN")
    return bool(v), len(v or "")


def secrets_table(E):
    """Which secrets the app can see (never their values)."""
    rows = []
    for name, use in SECRET_NAMES:
        found, n = _present(E, name)
        looks_odd = found and (n < 6 or name.endswith("_KEY") and n < 20)
        rows.append({"Secret": name, "Used for": use,
                     "Found by the app?": (f"yes ({n} characters)" + (" — looks too short" if looks_odd else "")) if found
                     else "no"})
    return rows


def reload_keys(E):
    """Pick up keys added after the app started: rebuild the AI provider list and log in to NJ Transit again."""
    chain = [E._provider_entry(n, E.LLM_CUSTOM_MODEL.strip() if n == E.LLM_PROVIDER else "") for n in E._chain_names]
    for n in ("openai", "openrouter", "huggingface"):          # include every provider that now has a key
        if n not in E._chain_names:
            chain.append(E._provider_entry(n))
    usable = [p for p in chain if p["usable"]]
    E.LLM_CHAIN[:] = usable
    E.LLM_READY = bool(usable)
    E.LLM.update(provider=usable[0]["provider"] if usable else None, model=usable[0]["model"] if usable else None)
    E._NJT.update(token=None, system=None)
    njt = E.njt_authenticate() if E.read_secret("NJ_TRANSIT_USERNAME") else "skipped (no NJ_TRANSIT_USERNAME / NJ_TRANSIT_PASSWORD)"
    return {"ai_providers": [f"{p['label']} · {p['model']}" for p in usable], "nj_transit": njt}


def check_ai(E):
    rows = []
    for name in ("openai", "openrouter", "huggingface"):
        cfg = E.CATALOG[name]
        p = E._provider_entry(name, E.LLM_CUSTOM_MODEL.strip() if name == E.LLM_PROVIDER else "")
        if not p["key"]:
            rows.append({"Check": f"AI · {cfg['label']}", "Result": SKIP, "Detail": f"no {cfg['key_name']} found",
                         "How to fix": f"Add {cfg['key_name']} to the secrets (optional if another AI provider works)."})
            continue
        t0 = time.time()
        try:
            reply = E._call_one(p, [{"role": "user", "content": "Reply with the single word: ready"}], 0.0, 16, None)
            rows.append({"Check": f"AI · {cfg['label']}", "Result": OK,
                         "Detail": f"{p['model']} answered {reply.strip()[:20]!r} in {time.time() - t0:.1f} s", "How to fix": ""})
        except Exception as e:  # LLMUnavailable carries a readable reason (auth failed, quota, model rejected)
            msg = E._scrub(e)
            fix = ("The key is wrong, expired or was pasted with extra characters. Create a new key and replace it."
                   if "auth" in msg else
                   "The account has no credit. Add billing or credit for this key." if "quota" in msg else
                   "The model name was refused; check LLM_CUSTOM_MODEL, or try again later." if "rejected" in msg else
                   "The provider did not answer; try again in a minute.")
            rows.append({"Check": f"AI · {cfg['label']}", "Result": FAIL, "Detail": msg, "How to fix": fix})
    in_use = ", ".join(f"{p['label']} · {p['model']}" for p in E.LLM_CHAIN) or "none"
    rows.append({"Check": "AI providers the agents are using now", "Result": OK if E.LLM_READY else WARN,
                 "Detail": in_use + f" · {E.LLM_STATS['calls']} AI calls this session"
                           + (f" ({', '.join(f'{k}: {v}' for k, v in E.LLM_STATS['by_provider'].items())})"
                              if E.LLM_STATS["by_provider"] else ""),
                 "How to fix": "" if E.LLM_READY else "If a provider above is ✅, press 'Apply new keys' so the agents use it."})
    return rows


def check_njt(E, live_cache=None):
    import newark_live as NL
    rows = []
    user, pw = E.read_secret("NJ_TRANSIT_USERNAME"), E.read_secret("NJ_TRANSIT_PASSWORD")
    if not (user and pw):
        missing = [n for n, v in (("NJ_TRANSIT_USERNAME", user), ("NJ_TRANSIT_PASSWORD", pw)) if not v]
        return [{"Check": "NJ Transit login", "Result": SKIP, "Detail": f"missing: {', '.join(missing)}",
                 "How to fix": "Add both secrets exactly as named (capital letters, underscores), then press 'Apply new keys'."}]
    if not E._NJT.get("token"):
        status = E.njt_authenticate()
    else:
        status = f"authenticated ({E._NJT.get('system')}), token reused"
    if not E._NJT.get("token"):
        fix = ("NJ Transit refused the username/password. Check them by logging in at developer.njtransit.com, and make "
               "sure your account has access to the BUS data API (rail-only access will not work)."
               if "rejected" in status else
               "Could not reach NJ Transit (network or their server). Try again in a few minutes. If it keeps failing, "
               "NJ Transit may have limited how many logins your account can make today; wait until after midnight ET."
               if "failed" in status else "See the detail.")
        return [{"Check": "NJ Transit login", "Result": FAIL, "Detail": status, "How to fix": fix}]
    rows.append({"Check": "NJ Transit login", "Result": OK, "Detail": status, "How to fix": ""})
    for ep, what in (("getVehiclePositions", "bus positions"), ("getTripUpdates", "delays"), ("getAlerts", "alerts")):
        try:
            msg = NL._feed(E, ep)
            n_ent = len(msg.entity)
            extra = ""
            if ep == "getVehiclePositions":
                inside = sum(1 for e in msg.entity if e.HasField("vehicle")
                             and NL.in_newark(e.vehicle.position.latitude, e.vehicle.position.longitude))
                extra = f", {inside} inside Newark"
            rows.append({"Check": f"NJ Transit {what}", "Result": OK if n_ent or ep == "getAlerts" else WARN,
                         "Detail": f"{ep}: {n_ent} records{extra}",
                         "How to fix": "" if n_ent or ep == "getAlerts" else "The feed is empty right now (normal late at night)."})
        except NL.LiveUnavailable as e:
            rows.append({"Check": f"NJ Transit {what}", "Result": FAIL, "Detail": str(e),
                         "How to fix": "Logged in, but this feed failed. Try again shortly; if it persists, your account "
                                       "may not include this feed."})
    if live_cache is not None:
        st_ = live_cache.static
        if st_:
            rows.append({"Check": "NJ Transit route map (getGTFS)", "Result": OK,
                         "Detail": f"{len(st_['routes'])} routes enter Newark ({st_['source']})", "How to fix": ""})
        elif live_cache.static_error:
            rows.append({"Check": "NJ Transit route map (getGTFS)", "Result": FAIL, "Detail": live_cache.static_error,
                         "How to fix": "The route download failed; it is retried automatically after 5 minutes."})
        else:
            rows.append({"Check": "NJ Transit route map (getGTFS)", "Result": WARN, "Detail": "not downloaded yet",
                         "How to fix": "Open the Live Newark tab once; the route map downloads then (up to a minute)."})
    return rows


def check_public(E):
    rows = []
    try:
        c = E.fetch_census(E.LAT, E.LON)
        rows.append({"Check": "US Census (neighbourhood data)", "Result": OK if c.get("tract") else FAIL,
                     "Detail": c.get("status", ""), "How to fix": "" if c.get("tract") else
                     "Census did not answer; the app uses the documented sample instead. A CENSUS_API_KEY can help."})
    except Exception as e:
        rows.append({"Check": "US Census (neighbourhood data)", "Result": FAIL, "Detail": type(e).__name__, "How to fix": ""})
    try:
        w = E.fetch_weather(E.LAT, E.LON)
        ok = sum(r["status"] == "ok" for r in w.get("connectors", []))
        rows.append({"Check": "National Weather Service", "Result": OK if ok else FAIL,
                     "Detail": f"{ok}/{len(w.get('connectors', []))} weather connectors answered · {w.get('current_conditions') or 'no current conditions'}",
                     "How to fix": "" if ok else "Weather service did not answer; the app uses 'Clear' as the sample."})
    except Exception as e:
        rows.append({"Check": "National Weather Service", "Result": FAIL, "Detail": type(e).__name__, "How to fix": ""})
    return rows


def run_all(E, live_cache=None):
    return check_ai(E) + check_njt(E, live_cache) + check_public(E)


AGENT_TABLE = [
    ("Explanation Agent", "Plain-language summary of the ranked options",
     "Every new incident when 'AI explanation' is ticked; look for the narrative source naming the provider, not 'template'"),
    ("Signal Scout note", "Two-sentence case note", "Every new incident (shown in the trace / audit)"),
    ("Input organizer (Section A)", "Bullet summary of live vs sample data", "Data inputs tab → Collate and hand off"),
    ("Insight Agent", "Summary of proposed rule changes", "Policy governance → Run Insight Agent"),
    ("Ethics agent (Section C)", "Compliance report, corrected by the rule engine", "Ethics tab → Run compliance agent"),
    ("Conflict-of-interest agents (Section D)", "5-step review of a disclosure", "Ethics tab → Run 5-step pipeline"),
]
