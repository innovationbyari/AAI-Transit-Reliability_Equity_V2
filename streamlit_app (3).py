"""Streamlit dashboard: Agentic AI for Public Transit Reliability and Equity.

AI agents sense, organize and explain; fixed, published rules score; people decide.
The engine (transit_ai/engine.py) is generated from the notebook in notebooks/, so the app and the notebook run the
same code. This app replaces the notebook's Gradio dashboard (Gradio apps do not run on Streamlit Community Cloud).

Run locally:  streamlit run streamlit_app.py
"""
import importlib
import io
import json
import os
import sys
import time
import types
from datetime import datetime, timezone

import streamlit as st

# The engine lives in the transit_ai/ folder. If the files were uploaded without that folder (engine.py and
# _ipython_shim.py next to this file), point the transit_ai package at this folder so the imports still work.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
if (not os.path.isdir(os.path.join(_APP_DIR, "transit_ai"))
        and os.path.exists(os.path.join(_APP_DIR, "engine.py")) and "transit_ai" not in sys.modules):
    _pkg = types.ModuleType("transit_ai")
    _pkg.__path__ = [_APP_DIR]
    sys.modules["transit_ai"] = _pkg

# The engine reads API keys from Colab Secrets or environment variables. On Streamlit Community Cloud the keys are in the
# app's secrets, so copy them into the environment before the engine starts (values are never printed).
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)) and str(_v).strip():
            os.environ.setdefault(str(_k), str(_v).strip())
except Exception:
    pass

import pandas as pd
import streamlit.components.v1 as components

st.set_page_config(page_title="Transit Reliability & Equity", page_icon="🚌", layout="wide")


# --------------------------------------------------------------------------------------------------------------------
# Engine: loaded once per server process and shared by every viewer (one demo state, like the notebook dashboard)
# --------------------------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_engine(generation: int):
    """Import (or re-run, on reset) the engine. Importing runs the notebook setup once: live connectors, computer
    vision, the scenario gallery and the evaluation."""
    if "transit_ai.engine" in sys.modules:
        return importlib.reload(sys.modules["transit_ai.engine"])
    return importlib.import_module("transit_ai.engine")


@st.cache_resource
def _generation_box():
    return {"n": 0}


GEN = _generation_box()
with st.spinner("Starting the engine: live data connectors, computer vision, scenario gallery and evaluation "
                "(about 30 seconds on first load)…"):
    E = load_engine(GEN["n"])
APP = E.APP

CHECKLIST = list(E.REVIEW_CHECKLIST)


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------
def show_df(df, height=None, key=None):
    """Display a DataFrame; nested values are shown as text so Arrow can serialize mixed columns."""
    if df is None:
        return
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            vals = df[col].map(lambda v: v if isinstance(v, (str, int, float, bool, type(None))) else json.dumps(v, default=str))
            kinds = {type(v) for v in vals if v is not None and v == v}
            if len(kinds) > 1:              # mixed text and numbers in one column: show everything as text
                vals = vals.map(lambda v: "" if v is None else str(v))
            df[col] = vals
    kw = {"hide_index": True, "width": "stretch"}
    if height:
        kw["height"] = height
    st.dataframe(df, key=key, **kw)


def show_map(html_doc, height):
    """Folium map (needs JavaScript, so it is embedded in an iframe). st.iframe replaces components.html in new Streamlit."""
    if hasattr(st, "iframe"):
        st.iframe(html_doc, height=height)
    else:
        components.html(html_doc, height=height)


def flash(msg, level="info"):
    st.session_state["flash"] = (msg, level)


def act(fn, *args):
    """Run one state-changing step under the shared lock, keep the engine's message, then redraw the page."""
    with APP.lock:
        try:
            msg = fn(*args)
            if isinstance(msg, str) and msg:
                APP.last_message = msg
            flash(APP.last_message, "warning" if str(APP.last_message).startswith(("⚠️", "⛔")) else "success")
        except (ValueError, KeyError) as e:
            APP.last_message = f"⚠️ {e}"
            flash(APP.last_message, "warning")
    st.rerun()


def raw_presets():
    """Section A / B hand-offs plus the synthetic presets, as in the notebook dashboard."""
    presets = {
        "Live hand-off · tonight's Section A inputs (live where available)":
            lambda: E.signal_scout_input_to_raw(E.signal_scout_input),
        "Section A · NJ-001 road closure (sample hand-off)":
            lambda: E.signal_scout_input_to_raw(E.build_signal_scout_input(E.section_a_sample_inputs())),
        "Section B · Route 12 cancellation + last rail connection":
            lambda: E.workflow_inputs_to_raw(E.SAMPLE_INPUTS, E.VisionAgent().run(E.RAW_SENSOR_FEED)),
    }
    for name, spec in E.PRESETS.items():
        presets[name] = (lambda s=spec, n=name: E.make_raw(s, n))
    return presets


PRESETS = raw_presets()


def here(text):
    """The engine's messages were written for Colab; on this app keys live in the app's secrets."""
    return str(text).replace("Colab Secrets", "the app's secrets")


def status_pill(status):
    col = E.STATUS_COLOR.get(status, "#111")
    return (f"<span style='background:{col};color:white;padding:2px 8px;border-radius:4px;font-size:0.85rem'>"
            f"{status}</span>")


# --------------------------------------------------------------------------------------------------------------------
# Plain-language text for people who do not know the technical side
# --------------------------------------------------------------------------------------------------------------------
FRIENDLY_SCENARIOS = {
    "Live hand-off · tonight's Section A inputs (live where available)":
        "Tonight's real situation near Newark Penn Station (live data when available)",
    "Section A · NJ-001 road closure (sample hand-off)": "Example: a road closure delays a bus near Newark Penn",
    "Section B · Route 12 cancellation + last rail connection":
        "Example: a bus trip is cancelled and riders may miss the last train",
    "Severe · R2 Hospital Line trip cancelled (verified)": "Example: a Hospital Line bus trip is cancelled",
    "Significant · R1 Crosstown unboardable crowding at peak": "Example: buses are too full to board at rush hour",
    "Moderate · R4 Northside winter-weather delays": "Example: winter weather slows buses down",
    "Below threshold · R5 Southside 4-minute delay": "Example: a small 4-minute delay",
    "Conflicting data · R3 vehicle feed vs. public reports": "Example: bus tracking and rider reports disagree",
    "Feed outage · R1 vehicle feed stale (backup available)": "Example: bus tracking stops working (a backup is available)",
    "Feed outage · R5 vehicle feed stale (no backup)": "Example: bus tracking stops working (no backup)",
    "Uncorroborated social report only · R4": "Example: an unconfirmed social-media report of a crash",
    "Agent error · Vision Agent fails twice on R2": "Example: the camera rider-counting system fails",
    "Hard-trigger bypass · Verified emergency closure on R3, sparse data": "Example: an official emergency road closure",
}

# status → (short label, what it means, who acts next, where to go)
PLAIN_STATUS = {
    "AWAITING_REVIEW": ("Needs a decision", "The system has ranked the possible responses. A supervisor or planner now "
                        "chooses what to do.", "Operations supervisor or service planner", "② Review & decide"),
    "AWAITING_SECOND_REVIEW": ("Needs a second approval", "The chosen action is costly or high-impact, so a second, "
                               "different person must also approve it.", "A second supervisor (not the first reviewer)",
                               "② Review & decide"),
    "AWAITING_IMPLEMENTATION": ("Approved: carry it out", "The action was approved. Staff carry it out in the real "
                                "world (this app does not dispatch buses) and record whether it worked.",
                                "Operations staff (for example the depot dispatcher)", "③ Carry out & record"),
    "IMPLEMENTED": ("Done: record the result", "Write down what actually happened so the system can learn from it.",
                    "Planner or supervisor", "③ Carry out & record"),
    "EVALUATED": ("Finished", "This case is complete. Start a new incident from the sidebar.", "No one", None),
    "SAFE_DEFAULT": ("Riders informed, no extra action", "No extra buses were sent. Riders get verified information and "
                     "staff keep watching. You can re-check it, escalate it, or record the result.",
                     "Service planner", "① What happened"),
    "MONITORING": ("Being watched", "The information is not reliable enough to act on yet. Re-check when new data "
                   "arrives, escalate if things get worse, or close it as a false alarm.", "Service planner",
                   "① What happened"),
    "CONFLICT_REVIEW": ("Sources disagree: needs a check", "Two data sources disagree about this incident. An operator "
                        "checks which one is right.", "Operator", "① What happened"),
    "CLOSED_FALSE_ALARM": ("Closed: false alarm", "Someone confirmed nothing was wrong. Nothing else to do.", "No one", None),
}

PLAIN_CHECKLIST = {
    "I checked the verified evidence, sources and timestamps":
        "I looked at what was reported, where it came from, and when",
    "I viewed every alternative and its component scores": "I looked at every possible response, not just the top one",
    "I considered equity and access impacts on affected areas":
        "I thought about riders who depend most on this bus (no car, low income, disability)",
    "I confirmed resource, road, policy and safety constraints":
        "I checked that buses, drivers and roads are available and the action is safe and allowed",
}

GLOSSARY = {
    "Incident / case": "One problem on one bus route, for example a late or cancelled trip. Each case gets an ID like CASE-0015.",
    "Significant": "Serious enough that a person should decide on a response. Small problems only get rider information.",
    "Recommended option": "The response the published rules rank first. You can choose another one, but you must explain why "
                          "(this is recorded as an override).",
    "Approve / Revise / Reject": "Approve = do the chosen response. Revise = ask for different options (for example a lower "
                                 "budget). Reject = do nothing extra; riders still get verified information (the safe default).",
    "Second reviewer": "For costly or high-impact actions, a second, different person must also approve.",
    "Safe default": "What happens when no one decides in time or the decision is Reject: no extra buses, riders get verified "
                    "information, and staff keep watching.",
    "Review deadline": "How long the reviewer has to decide (5, 10 or 20 minutes depending on how serious it is). When time runs "
                       "out, the safe default is applied.",
    "Ethics training gate": "People whose required ethics training is out of date cannot approve or revise.",
    "Conflict of interest": "People with a personal tie to a company involved in an option (for example a contractor) must step "
                            "aside for that decision.",
    "Confidence (C)": "How much the data can be trusted, from 0 to 1 (higher is better).",
    "Priority (P)": "How much this incident matters, from 0 to 1, combining how bad the disruption is (R), how vulnerable the "
                    "riders are (V) and whether they have other ways to travel (A).",
    "Score (Si)": "How good each response is, from 0 to 1, after weighing delay saved, riders helped, cost and certainty.",
    "AI agents": "Software that gathers information and writes explanations. They never make the decision: fixed rules "
                 "rank the options and people decide.",
}


def roster():
    """People on the ethics-training roster: (name, role)."""
    try:
        return [(r["official"], r.get("department", "")) for r in E.eth_check_all()]
    except Exception:
        return []


def person_picker(label, key, exclude=(), help_text=None):
    """Pick a name from the team list (avoids typos), or type another name."""
    people = [(n, d) for n, d in roster() if n not in exclude]
    choices = ["— choose your name —"] + [n for n, _ in people] + ["Someone else (type the name)"]
    roles = dict(people)
    pick = st.selectbox(label, choices, key=key, help=help_text,
                        format_func=lambda n: f"{n} · {roles[n]}" if n in roles else n)
    if pick == "Someone else (type the name)":
        return st.text_input("Full name", key=f"{key}_other").strip()
    return "" if pick.startswith("—") else pick


def scenario_label(name):
    return FRIENDLY_SCENARIOS.get(name, name)


# --------------------------------------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚌 Transit Reliability & Equity")
    st.caption("A practice tool for deciding how to respond to bus problems. **It does not control real buses.**")
    SIMPLE = st.toggle("Simple view", value=st.session_state.get("simple_view", True), key="simple_view",
                       help="Simple view shows only the steps a reviewer needs. Turn it off to see every score, "
                            "rule and audit table.")

    st.markdown("### 1. Start a new incident")
    names = list(PRESETS)
    preset = st.selectbox("What happened?", names, index=3 if len(names) > 3 else 0,
                          format_func=scenario_label if SIMPLE else (lambda n: n),
                          help="Pick a situation to practise with. The system checks the data, decides whether it is "
                               "serious, and ranks possible responses.")
    use_llm = st.checkbox("Write the explanation with AI", value=E.LLM_READY, disabled=not E.LLM_READY,
                          help="Needs an AI key in the app's secrets. The AI only writes the explanation; it never "
                               "changes a score or a decision.")
    if st.button("▶ Start incident", type="primary", width="stretch"):
        act(lambda: APP.new_case(PRESETS[preset](), use_llm=use_llm))
    st.caption("Or use a real NJ Transit bus from the **🔴 Live Newark** tab.")

    st.divider()
    st.markdown("### 2. Choose an incident to work on")
    ids = list(APP.cases)[::-1]
    if ids:
        cur = APP.active_id if APP.active_id in ids else ids[0]

        def _case_label(i):
            cs = APP.cases[i]
            stat = PLAIN_STATUS.get(cs["status"], (cs["status"],))[0] if SIMPLE else cs["status"]
            return f"{i} · {stat} · {cs['scenario_name'][:38]}"

        pick = st.selectbox("Incident", ids, index=ids.index(cur), format_func=_case_label)
        if pick != APP.active_id:
            with APP.lock:
                APP.active_id = pick
            st.rerun()
    else:
        st.info("No incidents yet. Start one above.")

    st.divider()
    if st.button("↻ Refresh", width="stretch", help="Show changes made by other people using the app."):
        st.rerun()
    with st.expander("System status (technical)"):
        llm = ", ".join(p["label"] for p in E.LLM_CHAIN) if E.LLM_READY else "off → template narratives"
        lc = E.LIVE_CONTEXT
        w = lc["weather"]
        st.markdown(
            f"**Rules** v{APP.PM.current['version']}  \n"
            f"**LLM:** {llm}  \n"
            f"**NJ Transit:** {here(lc['nj_transit']['status'])}  \n"
            f"**Census ACS:** {here(lc['census']['status'])}  \n"
            f"**Weather (NWS):** {w.get('source')} ({sum(r['status'] == 'ok' for r in w.get('connectors', []))}/"
            f"{len(w.get('connectors', []))} connectors)  \n"
            f"**Vision:** {E.CV_RESULTS.get('detector') or 'unavailable'}")
    with st.expander("Reset demo"):
        st.caption("Everyone using this app shares one practice space. Reset clears all incidents and decisions "
                   "(takes about 30 seconds).")
        if st.button("Reset everything", type="secondary"):
            GEN["n"] += 1
            st.session_state.clear()
            st.rerun()


# --------------------------------------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------------------------------------
if SIMPLE:
    st.title("🚌 Bus Disruption Response")
    st.caption("When a bus is late, cancelled or too full, this tool shows what happened, ranks the possible responses "
               "using fixed published rules, and asks the right people to decide. AI only gathers information and "
               "explains; **people make every decision.**")
    with st.expander("ℹ️ How this works (read me first)", expanded=not st.session_state.get("seen_intro")):
        st.session_state["seen_intro"] = True
        st.markdown(
            "1. **Start an incident**: pick a real late bus in **🔴 Live Newark buses**, or pick a practice example in "
            "the left sidebar and press **▶ Start incident**.\n"
            "2. **① What happened**: read the plain-language summary and look at the map.\n"
            "3. **② Review & decide**: if a decision is needed, choose your name, pick a response, explain why in a "
            "sentence, and tick the four checks. Costly actions need a second person to approve.\n"
            "4. **③ Carry out & record**: after approval, staff record whether the action worked, then what the "
            "result was.\n\n"
            "The yellow box below always tells you **what to do next and who should do it**. Words you don't know are "
            "explained in **❓ Help & glossary**. Turn off *Simple view* in the sidebar to see all the technical detail.")
else:
    st.title("Agentic AI for Public Transit Reliability and Equity")
    st.caption("AI agents sense, organize and explain · fixed, published rules score · people decide. "
               "A language model's text never enters a score, a routing decision or an approval.")

if "flash" in st.session_state:
    msg, level = st.session_state.pop("flash")
    if SIMPLE and APP.active:                       # show option names instead of internal codes
        for _o in APP.active["analysis"].get("options", []):
            msg = msg.replace(_o["option_id"], f"“{_o['label']}”")
    getattr(st, level if level in ("success", "warning", "info", "error") else "info")(here(msg))

c = APP.active


def _time_left(case):
    try:
        secs = (E.parse_iso(case["review"]["deadline_at"]) - E.utcnow()).total_seconds()
        return max(0, int(secs))
    except Exception:
        return None


@st.fragment(run_every="5s")
def deadline_banner():
    """Review timers keep running: check the deadline every few seconds (the dashboard's 2 s timer)."""
    with APP.lock:
        case = APP.active
        expired = bool(case and APP.enforce_deadline(case))
    if not case:
        if SIMPLE:
            st.info("👉 **Next step:** start an incident from the sidebar on the left.")
        return
    if SIMPLE:
        label, meaning, who, where = PLAIN_STATUS.get(case["status"], (case["status"], "", "", None))
        lines = [f"**{case['case_id']}** · {case['scenario_name']}", f"### {label}", meaning]
        if who and who != "No one":
            lines.append(f"**Who acts next:** {who}" + (f" · **Go to:** {where}" if where else ""))
        if case["status"] in E.REVIEW_STATUSES:
            left = _time_left(case)
            if left is not None:
                lines.append(f"⏱️ **Time left to decide: {left // 60} min {left % 60:02d} s.** If no one decides in time, "
                             "riders get verified information only (the safe default).")
        st.warning("\n\n".join(lines))
    else:
        st.markdown(f"{status_pill(case['status'])} **{case['case_id']}** · {case['scenario_name']}",
                    unsafe_allow_html=True)
        st.markdown(E.deadline_md(case), unsafe_allow_html=True)
    if expired:
        st.error("The time to decide ran out, so the safe default was applied. Press ↻ Refresh.")


deadline_banner()


# --------------------------------------------------------------------------------------------------------------------
# Live Newark: real NJ Transit routes and buses (newark_live.py)
# --------------------------------------------------------------------------------------------------------------------
import newark_live as NL  # noqa: E402

_LIVE_CACHE_DIR = os.path.join(getattr(E, "OUTPUT_DIR", "transit_outputs"), "njt_cache")


@st.cache_resource
def _live_backoff():
    return {"static_failed_at": 0.0, "static_error": ""}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _newark_static(generation: int):
    return NL.load_static(E, _LIVE_CACHE_DIR)


@st.cache_data(ttl=60, show_spinner=False)
def _newark_live(generation: int, refresh_key: int):
    return NL.load_live(E, _get_static())


def _get_static():
    """Real Newark routes; after a failure, wait 5 minutes before trying the big download again."""
    bo = _live_backoff()
    if time.time() - bo["static_failed_at"] < 300:
        return None
    try:
        return _newark_static(GEN["n"])
    except NL.LiveUnavailable as e:
        bo.update(static_failed_at=time.time(), static_error=str(e))
        return None


def _delay_text(b):
    if b["cancelled"]:
        return "cancelled"
    if b["delay_min"] is None:
        return "no delay info"
    if b["delay_min"] <= 0.5:
        return "on time" if b["delay_min"] >= -1 else f"{-b['delay_min']:g} min early"
    return f"{b['delay_min']:g} min late"


def _bus_colour(b):
    if b["cancelled"] or (b["delay_min"] or 0) >= NL.VERY_LATE_MIN:
        return "#dc2626"
    if (b["delay_min"] or 0) >= NL.LATE_MIN:
        return "#f59e0b"
    return "#16a34a"


def _live_map(static, live, highlight=None):
    import folium
    m = folium.Map(location=list(NL.NEWARK_CENTER), zoom_start=13, tiles="CartoDB positron", control_scale=True)
    b = NL.NEWARK_BBOX
    folium.Rectangle([(b["lat_min"], b["lon_min"]), (b["lat_max"], b["lon_max"])], color="#6b7280", weight=1,
                     dash_array="6 6", fill=False, tooltip="City of Newark (area shown)").add_to(m)
    palette = ["#2563eb", "#7c3aed", "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5", "#0d9488"]
    if static:
        fg = folium.FeatureGroup(name="NJ Transit bus routes in Newark")
        for i, ln in enumerate(static["lines"]):
            r = static["routes"].get(ln["route_id"], {})
            hot = highlight and ln["route_id"] == highlight
            folium.PolyLine(ln["path"], color=r.get("color") or palette[i % len(palette)], weight=6 if hot else 2.5,
                            opacity=0.95 if hot else 0.45,
                            tooltip=f"Route {r.get('short', ln['route_id'])} {r.get('long', '')}").add_to(fg)
        fg.add_to(m)
    if live:
        fb = folium.FeatureGroup(name="Buses right now")
        for bus in live["buses"]:
            folium.CircleMarker((bus["lat"], bus["lon"]), radius=7, color="#111827", weight=1, fill=True,
                                fill_color=_bus_colour(bus), fill_opacity=0.95,
                                tooltip=f"Route {bus['route']} · bus {bus['vehicle_id']} · {_delay_text(bus)}").add_to(fb)
        fb.add_to(m)
    folium.Marker(list(NL.NEWARK_PENN), tooltip="Newark Penn Station", icon=folium.Icon(color="gray", icon="train",
                                                                                         prefix="fa")).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    legend = ("<div style='position:fixed;bottom:18px;left:18px;z-index:9999;background:white;padding:6px 10px;"
              "border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)'>"
              "<b>Live NJ Transit buses</b><br>"
              f"<span style='color:#16a34a'>●</span> on time / under {NL.LATE_MIN} min late<br>"
              f"<span style='color:#f59e0b'>●</span> {NL.LATE_MIN}–{NL.VERY_LATE_MIN} min late<br>"
              f"<span style='color:#dc2626'>●</span> {NL.VERY_LATE_MIN}+ min late or cancelled<br>"
              "Lines: real bus routes that enter Newark</div>")
    m.get_root().html.add_child(folium.Element(legend))
    return m


def render_live_newark():
    st.markdown("#### Real NJ Transit buses in Newark, right now")
    if not E._NJT.get("token"):
        st.warning("**Not connected to NJ Transit yet.** " + here(str(E.LIVE_CONTEXT["nj_transit"].get("status", ""))))
        st.markdown(
            "To switch this on:\n"
            "1. On share.streamlit.io open your app → **⋮ → Settings → Secrets**.\n"
            "2. Add these two lines with your NJ Transit developer login, then **Save**:\n"
            "```toml\nNJ_TRANSIT_USERNAME = \"your-username\"\nNJ_TRANSIT_PASSWORD = \"your-password\"\n```\n"
            "3. The app restarts by itself. If it doesn't, use **Reset demo** in the sidebar.")
        if st.button("Try connecting again"):
            with st.spinner("Logging in to NJ Transit…"):
                status = E.njt_authenticate()
            flash(f"NJ Transit login: {status}", "success" if E._NJT.get("token") else "warning")
            st.rerun()
        return

    rk = st.session_state.setdefault("live_refresh", 0)
    c1, c2 = st.columns([1, 3])
    if c1.button("↻ Update buses", help="Fetch the latest positions (updates are also fetched automatically every minute)."):
        st.session_state["live_refresh"] = rk + 1
        st.rerun()
    with st.spinner("Loading Newark bus routes (first time each day can take a minute)…"):
        static = _get_static()
    try:
        with st.spinner("Fetching live buses…"):
            live = _newark_live(GEN["n"], st.session_state["live_refresh"])
    except NL.LiveUnavailable as e:
        st.error(f"Could not get live buses from NJ Transit: {e}")
        live = None
    if live:
        age = int(time.time() - live["fetched_at"])
        feed = (f" · feed time {datetime.fromtimestamp(live['feed_time'], timezone.utc).astimezone(E.LOCAL_TZ).strftime('%H:%M:%S %Z')}"
                if live.get("feed_time") and hasattr(E, "LOCAL_TZ") else "")
        n_late = sum((b["delay_min"] or 0) >= NL.LATE_MIN or b["cancelled"] for b in live["buses"])
        c2.markdown(f"**{len(live['buses'])} buses** in Newark now · **{n_late}** at least {NL.LATE_MIN} min late or cancelled · "
                    f"**{len(live['alerts'])}** service alerts · updated {age} s ago{feed}")
        for err in live.get("errors", []):
            st.caption(f"⚠️ {err}")
    if static:
        st.caption(f"{len(static['routes'])} real bus routes enter Newark ({static['source']}).")
    else:
        bo = _live_backoff()
        st.caption("Route lines are not available right now" + (f": {bo['static_error']}" if bo["static_error"] else "")
                   + ". Live buses are still shown.")

    sel_bus = None
    if live and live["buses"]:
        buses = live["buses"]
        opts = list(range(len(buses)))
        pick = st.selectbox("Choose a bus (cancelled and most-late buses are listed first)", opts,
                            format_func=lambda i: f"Route {buses[i]['route']}"
                                                  + (f" ({buses[i]['route_name'][:35]})" if buses[i]["route_name"] else "")
                                                  + f" · bus {buses[i]['vehicle_id']} · {_delay_text(buses[i])}",
                            key="live_bus_pick")
        sel_bus = buses[pick]
    show_map(_live_map(static, live, highlight=sel_bus["route_id"] if sel_bus else None).get_root().render(), 520)

    if live and live["buses"]:
        show_df(pd.DataFrame([{"Route": b["route"], "Route name": b["route_name"], "Bus": b["vehicle_id"],
                               "Status": _delay_text(b), "km from Newark Penn": b["km_from_penn"],
                               "Seconds since update": b["age_s"]} for b in live["buses"]]), height=260)
    elif live:
        st.info("No NJ Transit buses are reporting inside Newark right now (this is normal late at night).")

    if sel_bus:
        st.markdown("---\n#### Start an incident from this bus")
        route_alerts = [a for a in (live["alerts"] if live else []) if sel_bus["route_id"] in a["route_ids"]]
        st.write(f"**Route {sel_bus['route']}** {sel_bus['route_name']} · bus {sel_bus['vehicle_id']} · "
                 f"**{_delay_text(sel_bus)}** · {sel_bus['km_from_penn']} km from Newark Penn.")
        alert = None
        if route_alerts:
            labels = ["No alert"] + [a["header"][:90] or a["description"][:90] for a in route_alerts]
            i = st.radio("NJ Transit alert for this route to include", range(len(labels)), format_func=labels.__getitem__,
                         index=1, key="live_alert_pick")
            alert = route_alerts[i - 1] if i else None
        else:
            st.caption("No active NJ Transit alert for this route.")
        if (sel_bus["age_s"] or 0) > 300:
            st.caption("⚠️ This bus last reported more than 5 minutes ago, so the system will treat its data as stale.")
        if st.button("▶ Start incident from this real bus", type="primary"):
            def _start():
                inputs, name = NL.inputs_for_bus(E, sel_bus, alert)
                ssi = E.build_signal_scout_input(inputs)
                APP.new_case(E.signal_scout_input_to_raw(ssi, name=name), use_llm=use_llm)
            act(_start)
        st.caption("The system then checks this bus's data, decides whether it is serious, and ranks responses. "
                   "Note: rider and neighbourhood figures used in the scoring come from the Census connector or the "
                   "documented sample, and responses are planned on the practice network.")

    if live and live["alerts"]:
        with st.expander(f"NJ Transit service alerts for Newark routes ({len(live['alerts'])})"):
            for a in live["alerts"]:
                st.markdown(f"**Routes {', '.join(a['routes']) or '—'}:** {a['header']}  \n{a['description']}")


# --------------------------------------------------------------------------------------------------------------------
# Simple view: three steps plus help, in plain language
# --------------------------------------------------------------------------------------------------------------------
def _plain_summary(case):
    A = case["analysis"]
    f = A.get("fields", {})
    raw = case.get("raw", {})
    route = raw.get("external_route_id") or raw.get("route_id") or "the route"
    route_name = raw.get("route_name")
    bits = []
    if f.get("cancelled"):
        bits.append("a trip was **cancelled**")
    if f.get("delay_min") is not None:
        bits.append(f"buses are about **{f['delay_min']:g} minutes late**")
    if (f.get("load_factor") or 0) >= 1:
        bits.append("buses are **too full to board**")
    if f.get("event_type") and f.get("event_type") != "none":
        bits.append(f"cause reported: **{str(f['event_type']).replace('_', ' ')}**")
    what = "; ".join(bits) or "a possible service problem was reported"
    st.markdown(f"#### On route {route}{f' ({route_name})' if route_name else ''}: {what}.")

    gate = A.get("gate")
    gate_text = {
        "SIGNIFICANT": "✅ **Serious enough for a person to decide on a response.**",
        "BELOW_THRESHOLD": "ℹ️ **Minor.** Riders get verified information; no extra action is needed.",
        "MONITORING": "👀 **Not confirmed yet.** The data is not reliable enough to act on, so staff keep watching.",
        "CONFLICT_REVIEW": "⚠️ **Sources disagree.** An operator needs to check which one is right.",
        "SAFE_MODE": "🛟 **A system component failed**, so the tool fell back to the safe default.",
    }.get(gate, f"Result: {gate}")
    st.markdown(gate_text)
    why = A.get("gate_reason")
    if why:
        st.caption(f"Why: {why}")

    who = []
    if f.get("essential_type"):
        who.append(f"the route serves a **{str(f['essential_type']).replace('_', ' ')}**")
    if f.get("transit_dependence") is not None and f["transit_dependence"] >= 0.4:
        who.append("many nearby households **have no car**")
    if f.get("alternatives") == 0:
        who.append("riders have **no other bus or train** nearby")
    if f.get("missed_transfers"):
        who.append(f"about **{f['missed_transfers']} riders could miss a connection**")
    if who:
        st.markdown("**Who is affected:** " + "; ".join(who) + ".")
    rk = A.get("ranking") or []
    if rk:
        top = rk[0]
        st.markdown(f"**Recommended response:** {top['label']} (estimated cost ${top['cost_usd']:,.0f}). "
                    f"See all {len(rk)} options in **② Review & decide**.")


def _whole(x):
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return x


def _option_table(case):
    opts = {o["option_id"]: o for o in case["analysis"].get("options", [])}
    rows = []
    for r in case["analysis"].get("ranking", []):
        o = opts.get(r["option_id"], {})
        rows.append({"Rank": r["rank"], "Response": r["label"] + ("  ⭐ recommended" if r["rank"] == 1 else ""),
                     "Minutes of delay saved (est.)": _whole(o.get("pred_delay_reduction_min")),
                     "Vulnerable riders helped (est.)": _whole(o.get("pred_vulnerable_riders_helped")),
                     "Cost (USD)": _whole(r.get("cost_usd")), "Overall score (0–1)": round(r["Si"], 2)})
    show_df(pd.DataFrame(rows))
    st.caption("The overall score weighs delay saved, riders helped, access to essential places, cost and how "
               "certain the option is to work. Higher is better.")


def _missing(name, reason, ticks, viewed, need_option=False, option=None):
    gaps = []
    if not name:
        gaps.append("choose your name")
    if need_option and not option:
        gaps.append("pick a response")
    if len((reason or "").strip()) < 15:
        gaps.append("write a reason (at least 15 characters)")
    if len(ticks) < len(CHECKLIST):
        gaps.append(f"tick all four checks ({len(ticks)} of 4 done)")
    if not viewed:
        gaps.append("confirm you looked at every option")
    return gaps


def _checks(prefix):
    st.markdown("**Before you submit, confirm each of these:**")
    return [item for item in CHECKLIST if st.checkbox(PLAIN_CHECKLIST.get(item, item), key=f"{prefix}_{item}")]


def simple_step1(case):
    cid = case["case_id"] if case else ""
    if not case:
        st.info("Start an incident from the sidebar to see what happened.")
        return
    _plain_summary(case)
    show_map(E.build_map(case).get_root().render(), 480)
    st.caption("Map: red dots are the affected stops; shading shows areas with more riders who depend on transit "
               "(darker = more). The network is fictional, drawn over a real Newark map.")
    if case.get("rider_alert"):
        st.markdown(f"**Message for riders:** {case['rider_alert']}")

    if case["status"] == "CONFLICT_REVIEW":
        st.markdown("---\n#### Operator check: the data sources disagree")
        st.write("Look at the sources listed under *Details*, then say whether you resolved which one is right.")
        op = person_picker("Your name (operator)", f"{cid}_s_op_name")
        o1, o2 = st.columns(2)
        if o1.button("✔ Resolved: use the more reliable source", type="primary"):
            act(APP.operator_resolve, case, op, "resolved")
        if o2.button("Still unclear"):
            act(APP.operator_resolve, case, op, "unresolved")

    if case["status"] in ("MONITORING", "SAFE_DEFAULT"):
        st.markdown("---\n#### Planner actions")
        st.write("Use these when new information comes in.")
        who = person_picker("Your name (planner)", f"{cid}_s_mon_name")

        def _mon(kind, reason=""):
            if kind == "false_alarm":
                APP.close_false_alarm(case, who or "planner", reason)
            else:
                APP.monitor_action(case, kind, who or "planner")

        b1, b2 = st.columns(2)
        if b1.button("🔄 Re-check with new data", help="Run the checks again, for example after the bus feed is back."):
            act(_mon, "recheck")
        if b2.button("⬆ Escalate: it got worse", help="Send it to a reviewer to decide on a response."):
            act(_mon, "worsen")
        why_fa = st.text_input("If nothing was actually wrong, say why", key=f"{cid}_s_fa_reason",
                               placeholder="e.g. Driver confirmed the bus was on time; the report was a mistake.")
        if st.button("✖ Close as false alarm"):
            act(_mon, "false_alarm", why_fa)

    with st.expander("Details: data sources and every step the system took"):
        st.markdown(E.trace_md(case), unsafe_allow_html=True)
        show_df(E.evidence_df(case))


def simple_step2(case):
    cid = case["case_id"] if case else ""
    if not case:
        st.info("Start an incident from the sidebar first.")
        return
    if case["status"] not in E.REVIEW_STATUSES:
        st.success("No decision is needed on this incident right now. Check the yellow box above for the next step.")
        if case["decisions"]:
            st.markdown("**Decisions already made:**")
            for d in case["decisions"]:
                st.markdown(f"- {d.get('ts', '')[:16].replace('T', ' ')} · **{d.get('reviewer')}** "
                            f"({d.get('stage')}): **{d.get('decision')}** — “{d.get('reason')}”")
        return

    rk = case["analysis"].get("ranking", [])
    st.markdown("#### The possible responses")
    if rk:
        _option_table(case)
        expl = case.get("explanation")
        if expl:
            with st.expander("Full explanation (how each option was scored)"):
                st.markdown(expl["brief_md"], unsafe_allow_html=True)
                if expl.get("narrative"):
                    st.markdown(f"**Plain-language summary ({expl['narrative_source']}):** {expl['narrative']}")
    else:
        st.info("No workable response was found. You can Revise (for example raise the budget) or Reject.")

    if case["status"] == "AWAITING_REVIEW":
        st.markdown("---\n#### Your decision")
        name = person_picker("1. Who are you?", f"{cid}_s_r1_name",
                             help_text="Only people on the team list can approve. People whose ethics training has "
                                       "lapsed, or who have a conflict of interest, are stopped automatically.")
        decision = st.radio("2. What do you decide?", ["Approve", "Revise", "Reject"], horizontal=True, key=f"{cid}_s_r1_dec",
                            captions=["Carry out a response", "Ask for different options", "No extra action (safe default)"])
        option = None
        rev = None
        if decision == "Approve" and rk:
            labels = {r["option_id"]: f"{r['label']}" + ("  ⭐ recommended" if r["rank"] == 1 else "") for r in rk}
            option = st.radio("3. Which response?", list(labels), format_func=labels.get, key=f"{cid}_s_r1_opt")
            if option != rk[0]["option_id"]:
                st.info("You picked something other than the recommendation. That's allowed. Please explain why "
                        "in your reason; it is recorded as an override.")
        elif decision == "Revise":
            st.markdown("**3. What should change?** (you can revise at most twice)")
            excl = st.multiselect("Rule out these kinds of response", list(E.OPTION_LABELS),
                                  format_func=lambda k: E.OPTION_LABELS[k], key=f"{cid}_s_rv_excl")
            budget = st.slider("Maximum budget (USD)", 100, 2000, 1000, 50, key=f"{cid}_s_rv_budget")
            hold = st.slider("Longest a connecting bus may wait for riders (minutes)", 0, 15, 5, key=f"{cid}_s_rv_hold")
            alt = st.checkbox("Allow sending a bus from the other depot", key=f"{cid}_s_rv_alt")
            rev = {"excluded_options": list(excl), "budget_cap_usd": float(budget), "max_hold_min": float(hold),
                   "allow_alternate_depot": bool(alt)}
        else:
            st.caption("Riders will get verified information only, and staff will keep watching.")
        reason = st.text_area("4. Why? (one or two sentences, at least 15 characters)", key=f"{cid}_s_r1_reason",
                              placeholder="e.g. 18-minute delay on a hospital route with no other bus nearby; "
                                          "a second bus from Depot A is the fastest fix.")
        ticks = _checks(f"{cid}_s_r1")
        viewed = st.checkbox("I looked at every option in the table above before deciding", key=f"{cid}_s_r1_viewed")
        gaps = _missing(name, reason, ticks, viewed, need_option=(decision == "Approve"), option=option)
        if gaps:
            st.caption("Still needed: " + "; ".join(gaps) + ".")
        if st.button("Submit decision", type="primary", disabled=bool(gaps)):
            act(APP.submit_first, case, name, decision, option, reason, ticks, viewed, rev)

    elif case["status"] == "AWAITING_SECOND_REVIEW":
        f1 = case["review"]["first"]
        chosen = next((r["label"] for r in rk if r["option_id"] == f1["option_id"]), f1["option_id"])
        st.markdown("---\n#### Second approval needed")
        st.write(f"**{f1['reviewer']}** approved **{chosen}** because: “{f1['reason']}”. Because this action is costly or "
                 "high-impact, a **different** person must also approve it. Either person can reject it.")
        name = person_picker("1. Who are you? (must be a different person)", f"{cid}_s_r2_name", exclude=(f1["reviewer"],))
        decision = st.radio("2. Do you agree?", ["Approve", "Reject"], horizontal=True, key=f"{cid}_s_r2_dec",
                            captions=["Go ahead with this response", "Stop it (safe default)"])
        reason = st.text_area("3. Why? (at least 15 characters)", key=f"{cid}_s_r2_reason",
                              placeholder="e.g. I checked the delay and the depot has a spare bus and driver; agreed.")
        ticks = _checks(f"{cid}_s_r2")
        viewed = st.checkbox("I looked at every option in the table above", key=f"{cid}_s_r2_viewed")
        gaps = _missing(name, reason, ticks, viewed)
        if gaps:
            st.caption("Still needed: " + "; ".join(gaps) + ".")
        if st.button("Submit second approval", type="primary", disabled=bool(gaps)):
            act(APP.submit_second, case, name, decision, reason, ticks, viewed)


def simple_step3(case):
    cid = case["case_id"] if case else ""
    if not case:
        st.info("Start an incident from the sidebar first.")
        return
    st_ = case["status"]
    if st_ == "AWAITING_IMPLEMENTATION":
        label = next((r["label"] for r in case["analysis"].get("ranking", []) if r["option_id"] == case["approved_option_id"]),
                     case["approved_option_id"])
        st.markdown(f"#### Carry out: {label}")
        st.write("Do this in the real world (this app does not send buses), then record how it went.")
        staff = person_picker("Who carried it out?", f"{cid}_s_im_staff")
        res = st.radio("How did it go?", ["Completed", "Partial", "Failed"], horizontal=True, key=f"{cid}_s_im_res",
                       captions=["Done as planned", "Only partly done", "Could not be done"])
        notes = st.text_input("Notes (optional)", key=f"{cid}_s_im_notes", placeholder="e.g. Bus left Depot A at 6:12 pm.")
        if st.button("Record what happened", type="primary", disabled=not staff):
            act(APP.record_implementation, case, staff, res, notes)
        if not staff:
            st.caption("Still needed: choose who carried it out.")
    elif st_ in ("IMPLEMENTED", "SAFE_DEFAULT", "MONITORING"):
        st.markdown("#### Record the result")
        st.write("After the incident is over, write down what actually happened. This helps the system learn which "
                 "responses work. Nothing changes automatically: rule changes always need a policy owner's approval.")
        key = f"prefill_{case['case_id']}"
        if st.button("Fill in example values", help="Uses simulated results so you can practise this step."):
            try:
                st.session_state[key] = APP.simulate_observed(case)
                st.rerun()
            except (ValueError, KeyError) as ex:
                st.warning(str(ex))
        pre = st.session_state.get(key, {})
        a, b = st.columns(2)
        dr = a.number_input("Minutes of delay actually saved", min_value=0.0,
                            value=float(pre.get("observed_delay_reduction_min", 0)))
        eq = b.number_input("Vulnerable riders actually helped (number)", min_value=0.0,
                            value=float(pre.get("observed_vulnerable_riders_served", 0)))
        cost = a.number_input("Actual cost (USD)", min_value=0.0, value=float(pre.get("actual_cost_usd", 0)))
        acc = b.slider("How well were essential trips protected? (0 = not at all, 1 = fully)", 0.0, 1.0,
                       float(min(1.0, pre.get("observed_access_protection", 0))), 0.05)
        imp = a.slider("How much were riders affected? (1 = barely, 5 = badly)", 1, 5, int(pre.get("rider_reported_impact", 2)))
        fa = b.checkbox("It turned out to be a false alarm", value=bool(pre.get("false_alarm", False)))
        if st.button("Save the result", type="primary"):
            act(APP.evaluate_outcome, case, {"observed_delay_reduction_min": float(dr), "observed_access_protection": float(acc),
                                             "observed_vulnerable_riders_served": float(eq), "actual_cost_usd": float(cost),
                                             "rider_reported_impact": int(imp), "false_alarm": bool(fa)})
    elif st_ == "EVALUATED":
        st.success("This incident is finished and its result is saved.")
        with st.expander("Predicted vs actual"):
            show_df(E.outcome_df(case))
    else:
        st.info("Nothing to record yet. A response has to be approved first (see the yellow box above).")


def simple_help():
    st.markdown("#### Words used in this tool")
    for term, meaning in GLOSSARY.items():
        st.markdown(f"**{term}:** {meaning}")
    st.markdown("#### Who does what")
    st.markdown("- **Operations supervisor or service planner:** makes the first decision.\n"
                "- **Second supervisor:** approves costly or high-impact actions (must be a different person).\n"
                "- **Operations staff:** carry out the action and record how it went.\n"
                "- **Operator:** checks data when two sources disagree.\n"
                "- **Policy owner:** approves any change to the rules (in the technical view).")
    st.markdown("#### The team list")
    status_word = {"COMPLIANT": "✅ can approve", "AT_RISK": "🟡 can approve (training due soon)",
                   "NON_COMPLIANT": "⛔ cannot approve (training out of date)"}
    show_df(pd.DataFrame([{"Name": r["official"], "Role": r.get("department", ""),
                           "Ethics training": status_word.get(r["status"], r["status"])} for r in E.eth_check_all()]))


if SIMPLE:
    t0, t1, t2, t3, t4 = st.tabs(["🔴 Live Newark buses", "① What happened", "② Review & decide", "③ Carry out & record",
                                  "❓ Help & glossary"])
    with t0:
        render_live_newark()
    with t1:
        simple_step1(c)
    with t2:
        simple_step2(c)
    with t3:
        simple_step3(c)
    with t4:
        simple_help()
    st.stop()

tabs = st.tabs(["1 · Intake & verification", "2 · Component scores", "3 · Options & ranking", "4 · Human review",
                "5 · Implementation & outcome", "6 · Policy governance", "7 · Ethics & independence",
                "8 · Gallery & evaluation", "9 · Audit & export", "0 · Data inputs (Section A)", "🔴 Live Newark"])

# 1 · Intake & verification ------------------------------------------------------------------------------------------
with tabs[0]:
    with st.expander("Custom scenario builder (all data inputs)"):
        with st.form("custom"):
            a, b, d, e = st.columns(4)
            route = a.selectbox("Route", ["R1", "R2", "R3", "R4", "R5"], index=1)
            i0 = b.slider("First affected stop", 0, 8, 1)
            n_stops = d.slider("Affected stops", 2, 6, 5)
            period = e.selectbox("Time period", list(E.DEFAULT_POLICY["lookups"]["time_sensitivity"]), index=2)
            a, b, d, e = st.columns(4)
            delay = a.slider("Delay (min)", 0.0, 40.0, 16.0, 0.5)
            gap = b.slider("Gap / headway", 1.0, 3.5, 2.1, 0.05)
            cancelled = d.checkbox("Trip cancelled")
            risk = e.slider("Cancellation risk", 0.0, 1.0, 0.3, 0.05)
            a, b, d, e, f = st.columns(5)
            boardings = a.slider("Boardings", 0, 300, 150, 5)
            load = b.slider("Load factor", 0.2, 1.3, 0.9, 0.01)
            passups = d.slider("Pass-ups", 0, 40, 5)
            ev_type = e.selectbox("Event type", ["none", "weather", "obstruction", "road_closure", "mechanical", "medical"])
            ev_sev = f.selectbox("Event severity", ["none", "minor", "major", "emergency"])
            a, b, d, e = st.columns(4)
            sources = a.multiselect("Sources", ["vehicle_apc", "official_alert", "trusted_external", "public_report"],
                                    default=["vehicle_apc", "official_alert"])
            feed_age = b.slider("Vehicle feed age (s)", 0, 1500, 20, 10)
            backup = d.checkbox("Backup feed available", value=True)
            vfail = e.slider("Vision Agent failures", 0, 2, 0)
            a, b, d, e = st.columns(4)
            detour = a.checkbox("Detour road open", value=True)
            pdelay = b.slider("Public-report delay (min)", 0, 40, 16)
            pcorr = d.checkbox("Public report corroborated")
            missing = e.multiselect("Simulate missing fields", ["cancellation_risk", "headway_ratio", "passups", "disability"])
            if st.form_submit_button("▶ Run custom scenario", type="primary"):
                spec = dict(route_id=route, stop_range=(int(i0), int(i0) + int(n_stops) - 1), time_period=period,
                            delay_min=float(delay), headway_ratio=float(gap), cancelled=bool(cancelled),
                            cancellation_risk=float(risk), boardings=int(boardings), load_factor=float(load),
                            passups=int(passups), event_type=ev_type, event_severity=ev_sev, event_verified=ev_sev != "none",
                            sources=list(sources) or ["vehicle_apc"], feed_age_s=int(feed_age), backup_available=bool(backup),
                            vision_agent_failures=int(vfail), detour_open=bool(detour), public_delay=float(pdelay),
                            public_corroborated=bool(pcorr), missing_fields=list(missing))
                act(lambda: APP.new_case(E.make_raw(spec, f"Custom · {route} · delay {delay} min"), use_llm=use_llm))

    left, right = st.columns([3, 2])
    with left:
        show_map(E.build_map(c).get_root().render(), 560)
    with right:
        if c:
            st.markdown(E.trace_md(c), unsafe_allow_html=True)
        else:
            st.info("Run a scenario from the sidebar to see the incident trace.")
    if c:
        st.markdown("**Verified evidence** (accepted / stale / quarantined / excluded)")
        show_df(E.evidence_df(c))
        x, y = st.columns(2)
        with x:
            st.markdown("**Exception log**")
            show_df(E.exceptions_df(c))
        with y:
            st.markdown("**Alerts** (operator, technical owner, planner)")
            show_df(E.alerts_df(c))

        if c["status"] == "CONFLICT_REVIEW":
            st.markdown("#### Operator review — conflicting data (max two checks, then monitor)")
            op = st.text_input("Operator name", key="op_name")
            o1, o2 = st.columns(2)
            if o1.button("Resolved: apply source precedence and re-verify"):
                act(APP.operator_resolve, c, op, "resolved")
            if o2.button("Still unresolved"):
                act(APP.operator_resolve, c, op, "unresolved")

    st.markdown("#### Monitoring queue (monitoring, safe default, conflict review)")
    show_df(E.monitor_df())
    mon_ids = [x["case_id"] for x in APP.cases.values() if x["status"] in ("MONITORING", "SAFE_DEFAULT", "CONFLICT_REVIEW")]
    if mon_ids:
        m1, m2, m3 = st.columns(3)
        mon = m1.selectbox("Case", mon_ids, index=mon_ids.index(APP.active_id) if APP.active_id in mon_ids else 0)
        mon_name = m2.text_input("Planner name", key="mon_name")
        mon_reason = m3.text_input("Reason (for false alarm)", key="mon_reason")
        b1, b2, b3 = st.columns(3)

        def _monitor(cid, kind, name, reason=""):
            APP.active_id = cid
            if kind == "false_alarm":
                APP.close_false_alarm(APP.cases[cid], name or "planner", reason)
            else:
                APP.monitor_action(APP.cases[cid], kind, name or "planner")

        if b1.button("Recheck (feed changed)"):
            act(_monitor, mon, "recheck", mon_name)
        if b2.button("Re-escalate (conditions worsen)"):
            act(_monitor, mon, "worsen", mon_name)
        if b3.button("Close as false alarm"):
            act(_monitor, mon, "false_alarm", mon_name, mon_reason)
    with st.expander("Incident history (past disruptions, responses, recovery)"):
        show_df(E.incident_history_df(APP))

# 2 · Component scores -----------------------------------------------------------------------------------------------
with tabs[1]:
    if not c:
        st.info("Run a scenario first.")
    else:
        x, y = st.columns(2)
        with x:
            st.markdown("**Confidence** C = 0.40Q + 0.25F + 0.20G + 0.15M "
                        "(proceed when C ≥ 0.70 or an official source verifies a hard trigger)")
            show_df(E.conf_df(c))
            st.markdown(E.sig_md(c), unsafe_allow_html=True)
        with y:
            st.plotly_chart(E.rvap_plot(c), width="stretch")
        st.markdown("**Normalization and impact** — n(x) = clip((x−L)/(U−L), 0, 1); ninv(x) = 1 − n(x); "
                    "R = 0.30D + 0.25H + 0.25K + 0.20E · V = 0.35N + 0.25I + 0.20B + 0.20T · "
                    "A = 0.35S + 0.30L + 0.20X + 0.15U · P = 0.40R + 0.35V + 0.25A")
        show_df(E.norm_df(c))

# 3 · Options & ranking ----------------------------------------------------------------------------------------------
with tabs[2]:
    if not c or not c["analysis"].get("ranking"):
        st.info("Options appear for significant cases.")
    else:
        x, y = st.columns([3, 2])
        with x:
            st.markdown("**Scenario table and rank**")
            show_df(E.options_df(c))
            st.markdown("**Considered but infeasible**")
            show_df(E.infeasible_df(c))
        with y:
            st.plotly_chart(E.options_plot(c), width="stretch")
        expl = c.get("explanation")
        if expl:
            st.markdown(expl["brief_md"], unsafe_allow_html=True)   # deterministic text built by the rules
            st.markdown(f"**Narrative ({expl['narrative_source']}, non-authoritative):** {expl['narrative']}")

        def _regen():
            c["use_llm"] = use_llm
            APP.explain(c)
            return f"Narrative regenerated ({c['explanation']['narrative_source']})."

        if st.button("↻ Regenerate narrative"):
            act(_regen)

# 4 · Human review ---------------------------------------------------------------------------------------------------
with tabs[3]:
    if not c:
        st.info("Run a scenario first.")
    else:
        st.caption("The review timer is shown above the tabs and keeps counting down.")
        if c["analysis"].get("ranking"):
            st.markdown("**All alternatives (review before deciding)**")
            show_df(E.options_df(c))
        feas = [(f"#{r['rank']} {r['label']} (Si {r['Si']:.3f})", r["option_id"]) for r in c["analysis"].get("ranking", [])]
        left, right = st.columns(2)
        with left:
            st.markdown("### First reviewer")
            with st.form("first_review"):
                r1_name = st.text_input("Reviewer name", placeholder="e.g. Ana Lopez — must be on the tab 7 roster")
                r1_dec = st.radio("Decision", ["Approve", "Revise", "Reject"], horizontal=True)
                r1_opt = st.selectbox("Option to approve (defaults to rank 1; other choices are logged as overrides)",
                                      [o for _, o in feas] or [None],
                                      format_func=lambda o: dict((v, k) for k, v in feas).get(o, "—"))
                r1_reason = st.text_area("Typed reason (required)", height=80)
                r1_check = [item for item in CHECKLIST if st.checkbox(item, key=f"r1_{item}")]
                r1_viewed = st.checkbox("I viewed all alternatives before the recommendation", key="r1_viewed")
                with st.expander("Revision: change constraints / request another option (max two)"):
                    rv_excl = st.multiselect("Exclude option types", list(E.OPTION_LABELS),
                                             format_func=lambda k: E.OPTION_LABELS[k])
                    rv_budget = st.slider("Budget cap (USD)", 100, 2000, 1000, 50)
                    rv_hold = st.slider("Max connection hold (min)", 0, 15, 5)
                    rv_alt = st.checkbox("Request option: dispatch from alternate depot")
                if st.form_submit_button("Submit first review", type="primary"):
                    rev = None
                    if r1_dec == "Revise":
                        rev = {"excluded_options": list(rv_excl), "budget_cap_usd": float(rv_budget),
                               "max_hold_min": float(rv_hold), "allow_alternate_depot": bool(rv_alt)}
                    act(APP.submit_first, c, r1_name, r1_dec, r1_opt, r1_reason, r1_check, r1_viewed, rev)
        with right:
            st.markdown("### Second reviewer (high-impact actions)")
            if c["status"] == "AWAITING_SECOND_REVIEW":
                f1 = c["review"]["first"]
                st.markdown(f"**First reviewer {f1['reviewer']}** approved `{f1['option_id']}`: “{f1['reason']}”. "
                            "An independent reviewer must also approve; either can reject.")
            with st.form("second_review"):
                r2_name = st.text_input("Independent reviewer name", placeholder="e.g. Chen Wei")
                r2_dec = st.radio("Decision", ["Approve", "Reject"], horizontal=True)
                r2_reason = st.text_area("Typed reason (required)", height=80)
                r2_check = [item for item in CHECKLIST if st.checkbox(item, key=f"r2_{item}")]
                r2_viewed = st.checkbox("I viewed all alternatives before the recommendation", key="r2_viewed")
                if st.form_submit_button("Submit second review", type="primary"):
                    act(APP.submit_second, c, r2_name, r2_dec, r2_reason, r2_check, r2_viewed)

            def _expire():
                if c["status"] not in E.REVIEW_STATUSES:
                    raise ValueError("No review timer is running.")
                c["review"]["deadline_at"] = E.iso(E.utcnow() - E.timedelta(seconds=1))
                APP.log(c, "demo", "tester", "Deadline control", "Deadline expiry simulated")
                APP.enforce_deadline(c)

            if st.button("⏱ Simulate deadline expiry (demo)"):
                act(_expire)
        if c.get("rider_alert"):
            st.markdown(f"**Rider alert (verified info only):** {c['rider_alert']}")
        if c["decisions"]:
            st.markdown("**Human decision and timing**")
            show_df(E.decisions_df(c))

# 5 · Implementation & outcome ---------------------------------------------------------------------------------------
with tabs[4]:
    if not c:
        st.info("Run a scenario first.")
    else:
        if c["status"] == "AWAITING_IMPLEMENTATION":
            st.markdown(f"### ✅ Approved: `{c['approved_option_id']}`\nOperations staff carry out the action **outside the "
                        "prototype** and send rider alerts. Record the result below.")
        elif c["implementation"]:
            st.markdown("\n".join(f"- {i['ts'][:19]} · {i['staff']} · **{i['result']}** · {i['notes']}"
                                  for i in c["implementation"]))
        else:
            st.markdown("_No approved action awaiting implementation._")
        with st.form("impl"):
            a, b, d = st.columns(3)
            staff = a.text_input("Operations staff")
            res = b.radio("Result", ["Completed", "Partial", "Failed"], horizontal=True)
            notes = d.text_input("Notes")
            if st.form_submit_button("Record implementation", type="primary"):
                act(APP.record_implementation, c, staff, res, notes)

        st.markdown("### Outcome evaluator (separate from the recommender; never silently changes policy weights)")
        pre = st.session_state.get(f"prefill_{c['case_id']}", {})
        if st.button("Prefill simulated observations"):
            try:
                st.session_state[f"prefill_{c['case_id']}"] = APP.simulate_observed(c)
                st.rerun()
            except (ValueError, KeyError) as e:
                st.warning(str(e))
        with st.form("outcome"):
            a, b, d, e_, f, g = st.columns(6)
            o_dr = a.number_input("Observed delay reduction (min)", value=float(pre.get("observed_delay_reduction_min", 0)))
            o_acc = b.number_input("Observed access protection (0–1)", value=float(pre.get("observed_access_protection", 0)))
            o_eq = d.number_input("Vulnerable riders served", value=float(pre.get("observed_vulnerable_riders_served", 0)))
            o_cost = e_.number_input("Actual cost (USD)", value=float(pre.get("actual_cost_usd", 0)))
            o_imp = f.slider("Rider-reported impact", 1, 5, int(pre.get("rider_reported_impact", 2)))
            o_fa = g.checkbox("False alarm", value=bool(pre.get("false_alarm", False)))
            if st.form_submit_button("Evaluate outcome", type="primary"):
                act(APP.evaluate_outcome, c, {"observed_delay_reduction_min": float(o_dr),
                                              "observed_access_protection": float(o_acc),
                                              "observed_vulnerable_riders_served": float(o_eq),
                                              "actual_cost_usd": float(o_cost), "rider_reported_impact": int(o_imp),
                                              "false_alarm": bool(o_fa)})
        if c.get("outcome"):
            x, y = st.columns(2)
            with x:
                show_df(E.outcome_df(c))
            with y:
                st.plotly_chart(E.outcome_plot(c), width="stretch")

# 6 · Policy governance ----------------------------------------------------------------------------------------------
with tabs[5]:
    def _insight():
        props = E.insight_agent(APP)
        return f"Insight Agent drafted {len(props)} proposal(s). Nothing is applied without policy-owner approval."

    if st.button("🔍 Run Insight Agent (proposes only)", type="primary"):
        act(_insight)
    st.markdown(APP.insight_narrative or "_Run the Insight Agent to analyse outcomes._")
    st.markdown("**Proposed changes**")
    show_df(E.proposals_df())
    prop_ids = [p["id"] for p in APP.proposals]
    a, b, d = st.columns(3)
    pid = a.selectbox("Proposal", prop_ids or [None])
    owner = b.text_input("Policy owner name", placeholder="e.g. Dana Brooks")
    owner_reason = d.text_input("Reason (required)")
    b1, b2, b3 = st.columns(3)

    def _preview(p_id):
        p = next((p for p in APP.proposals if p["id"] == p_id), None)
        if not p:
            raise ValueError("Pick a proposal.")
        _, s = E.run_backtest(APP, E.apply_changes(APP.PM.current, p["changes"]))
        return "Preview only — " + s

    if b1.button("Preview backtest"):
        act(_preview, pid)
    if b2.button("Approve → backtest → release new version", type="primary"):
        act(E.decide_proposal, APP, pid, owner, "Approve", owner_reason)
    if b3.button("Reject (keep current rules)"):
        act(E.decide_proposal, APP, pid, owner, "Reject", owner_reason)
    bt = APP.backtest
    st.markdown(bt["summary"] if bt else "_No backtest yet._")
    if bt is not None and bt.get("df") is not None:
        show_df(E.ph(bt["df"], "No backtest yet"))
    st.markdown("**Decision log**")
    show_df(E.proposal_log_df())
    st.markdown("**Rule versions**")
    show_df(E.versions_df())
    a, b = st.columns([2, 1])
    vers = [v["version"] for v in APP.PM.versions]
    rb = a.selectbox("Roll back to version", vers)

    def _rollback(version, who, why):
        if not who.strip() or len(why.strip()) < 10:
            raise ValueError("Policy owner name and a reason (≥ 10 characters) are required for rollback.")
        v = APP.PM.rollback(version, who, why)
        APP.log(None, who, "policy owner", "Rollback", f"Rules of {version} re-released as v{v}", why)
        return f"Rolled back: rules of v{version} are now active as v{v}. Open cases keep the version they were scored with."

    b.write("")
    if b.button("Roll back"):
        act(_rollback, rb, owner, owner_reason)
    with st.expander("Active policy (thresholds, bounds, weights, lookups)"):
        st.json(APP.PM.current, expanded=False)

# 7 · Ethics & independence ------------------------------------------------------------------------------------------
with tabs[6]:
    st.markdown("**Section C — Ethics training** and **Section D — Conflict of interest** protect the review steps. "
                "Their rule engines are the gate (NON_COMPLIANT people cannot approve or revise; a name match between a "
                "reviewer's disclosure and a party to the option requires recusal). The LLM agents only explain.")
    x, y = st.columns([3, 2])
    with x:
        st.markdown(E.eth_summary_md(), unsafe_allow_html=True)
        show_df(E.eth_status_df())
    with y:
        sample = st.selectbox("Roster", list(E.ETH_SAMPLES),
                              index=list(E.ETH_SAMPLES).index("Transit review team (this system)")
                              if "Transit review team (this system)" in E.ETH_SAMPLES else 0)
        roster_txt = st.text_area("Roster JSON (editable)", json.dumps(E.ETH_SAMPLES[sample], indent=1), height=220,
                                  key=f"roster_{sample}")

        def _load_roster(text):
            n = E.load_eth_roster(json.loads(text))
            APP.log(None, "system", "system", "Ethics roster", "Roster loaded", f"{n} people")
            return f"Roster loaded ({n} people)."

        if st.button("Load roster"):
            act(_load_roster, roster_txt)

    st.markdown("#### Run the governance gate for a person")
    opts = [o for o in (c["analysis"].get("options", []) if c else []) if o["feasible"]]
    a, b, d = st.columns(3)
    g_name = a.text_input("Person", placeholder="e.g. Gabe Ortiz")
    g_dec = b.radio("Decision", ["Approve", "Revise", "Reject"], horizontal=True, key="g_dec")
    g_opt = d.selectbox("Option on the active case", [o["option_id"] for o in opts] or [None],
                        format_func=lambda oid: next((o["label"] for o in opts if o["option_id"] == oid), "—"))
    if st.button("Run governance gate"):
        if not g_name.strip():
            st.warning("Enter a name.")
        else:
            opt = next((o for o in opts if o["option_id"] == g_opt), None)
            probs, notes = E.governance_check(g_name, g_dec, option=opt)
            (st.error if probs else st.success)("Blocked" if probs else "Allowed")
            for p in probs:
                st.markdown(f"- **{p}**")
            for n in notes:
                st.markdown(f"- {n}")
    st.markdown("**Conflict-of-interest disclosures on file (Section D facts)**")
    show_df(E.disclosures_df())
    st.markdown("**Section D rule screen: each person × each option on the active case**")
    show_df(E.coi_screen_df(c))

    with st.expander("Section C agent: ask about the roster (LLM + guardrail)"):
        q = st.text_input("Question", E.ETH_DEFAULT_QUESTION)
        if st.button("Run compliance agent"):
            try:
                rep_ = E.eth_agent.run(q, verbose=False)
                issues = E.verify_report(rep_)
                fin = E.reconcile_report(rep_)
                st.markdown(f"**Answer:** {fin.answer}")
                st.markdown("**Guardrail:** " + ("; ".join(issues) + " (corrected below)" if issues else "✅ matched the rule engine"))
                show_df(pd.DataFrame([{"person": a_.official, "status": a_.status, "missing": ", ".join(a_.missing_courses) or "—",
                                       "priority": a_.priority, "action": a_.action} for a_ in fin.action_items]))
                st.markdown(f"_Ethics notes:_ {fin.ethics_notes}")
            except Exception as ex:
                st.warning(here(f"Agent unavailable: {ex}"))
    with st.expander("Section D multi-agent review of a disclosure (LLM + guardrail)"):
        choices = [f"Staff · {n}" for n in E.STAFF_DISCLOSURES] + [f"Sample · {n}" for n in E.COI_SAMPLES]
        ch = st.selectbox("Case file", choices)
        kind, nm = ch.split(" · ", 1)
        txt = st.text_area("Case text (editable)",
                           E.STAFF_DISCLOSURES[nm]["text"] if kind == "Staff" else E.COI_SAMPLES[nm].strip(),
                           height=120, key=f"coi_{ch}")
        if st.button("Run 5-step conflict-of-interest pipeline"):
            try:
                r = E.coi_run_pipeline(txt, verbose=False)
                fr = r["final"]
                issues = E.coi_guardrail(r)
                st.markdown(f"### {fr.recommendation} · risk {fr.risk_score:.2f} · {fr.official}\n{fr.summary}")
                st.markdown(f"**Rule screener:** score {r['screen']['rule_score']} (threshold {r['screen']['threshold']})")
                st.markdown("**Required actions:** " + "; ".join(fr.required_actions))
                st.markdown(f"**Due process:** {fr.due_process_note}")
                st.markdown("**Guardrail:** " + ("; ".join(issues) if issues else "✅ consistent with flags, mitigations and the rule screen"))
            except Exception as ex:
                st.warning(here(f"Pipeline unavailable: {ex}"))

# 8 · Gallery & evaluation -------------------------------------------------------------------------------------------
with tabs[7]:
    st.markdown("### Scenario gallery — one scripted case for each path through the workflow diagram")
    gdf = getattr(E, "GALLERY_DF", None)
    if gdf is not None and len(gdf):
        show_df(gdf)
    else:
        st.info("The scenario gallery did not run.")
    st.markdown("### Evaluation scorecard (V2–V10)")
    ev = pd.DataFrame(E.EV_RESULTS)
    if len(ev):
        score = ev.pivot_table(index="part", columns="result", values="check", aggfunc="count", fill_value=0)
        n_pass = int((ev["result"] == "pass").sum())
        n_scored = int(ev["result"].isin(["pass", "FAIL"]).sum())
        st.markdown(f"**{n_pass} of {n_scored} scored checks passed** · "
                    f"{int((ev['result'] == 'skipped').sum())} skipped (no data for them in this run) · "
                    f"{int((ev['result'] == 'reported').sum())} reported without a pass mark")
        show_df(score.reset_index())
        fails = ev[ev["result"] == "FAIL"]
        if len(fails):
            st.markdown("**Checks that failed**")
            show_df(fails)
        with st.expander("Every check"):
            show_df(ev)
    st.markdown("### Computer vision (anonymized counts only)")
    cams = E.CV_RESULTS.get("cameras", [])
    if cams:
        show_df(pd.DataFrame([{"camera": x["camera_id"], "role": x["role"], "frames": x["frames_analyzed"],
                               "people (median)": x["person_count"], "peak": x["peak_count"],
                               "load factor": x["load_factor"], "detector score": x["mean_detector_score"]} for x in cams]))
    for err in E.CV_RESULTS.get("errors", []):
        st.caption(f"⚠️ {err}")

# 9 · Audit & export -------------------------------------------------------------------------------------------------
with tabs[8]:
    st.markdown("**All cases (shared case state)**")
    show_df(E.cases_df())
    st.markdown("**Rubber-stamp protection: approval patterns and response times**")
    rs = APP.rubber_stamp_report()
    show_df(rs if len(rs) else E.ph(None, "No decisions yet"))
    st.markdown("**Audit record (newest first)**")
    show_df(E.audit_df(), height=400)
    if st.button("Prepare export (zip: JSON + CSV + map)", type="primary"):
        with APP.lock:
            path = E.export_outputs()
        with open(path, "rb") as fh:
            st.session_state["export"] = (path.split("/")[-1], fh.read())
    if "export" in st.session_state:
        name, data = st.session_state["export"]
        st.download_button("⬇ Download outputs", data=io.BytesIO(data), file_name=name, mime="application/zip")
    st.caption("Governance boundary: human authority is final; thresholds, bounds and weights are versioned policy "
               "assumptions; the recommender cannot change scoring and the evaluator cannot create options; exception "
               "paths never trigger resource action; social reports require corroboration; vision data anonymized; "
               "community data area-level only.")

# 0 · Data inputs (Section A) ----------------------------------------------------------------------------------------
with tabs[9]:
    st.markdown("Edit the seven input sources, then collate → the small language model reads the event report "
                "(when available) → numbers are extracted deterministically → the record is handed to the Signal Scout.")
    if "a_json" not in st.session_state:
        st.session_state["a_json"] = json.dumps(E.section_a_sample_inputs(), indent=1)
    if st.button("Reset to Section A sample"):
        st.session_state["a_json"] = json.dumps(E.section_a_sample_inputs(), indent=1)
        st.rerun()
    a_txt = st.text_area("Seven input sources (JSON)", key="a_json", height=360)
    if st.button("▶ Collate, interpret and hand off to Signal Scout", type="primary"):
        with APP.lock:
            try:
                inputs = json.loads(a_txt)
                ssi = E.build_signal_scout_input(inputs)
                APP.new_case(E.signal_scout_input_to_raw(ssi), use_llm=use_llm)
                st.session_state["a_result"] = ssi
                flash(APP.last_message, "success")
            except (ValueError, KeyError, TypeError) as ex:
                flash(f"⚠️ Section A input error: {ex}", "warning")
        st.rerun()
    ssi = st.session_state.get("a_result")
    if ssi:
        ev_ = ssi["event"]
        st.markdown(f"**SLM reading (supporting only):** {ev_.get('slm_interpretation')}  \n"
                    f"**Deterministic fields:** delay = {ev_.get('delay_min')} min · cause = {ev_.get('cause')}  \n"
                    f"**LLM organizer:** {ssi.get('external_feeds', {}).get('llm')}")
        for fl in ev_.get("slm_cross_check", []):
            st.caption(f"⚠️ {fl}")
        st.json(json.loads(json.dumps(ssi, default=str)), expanded=False)

# Live Newark (real NJ Transit routes and buses) -------------------------------------------------------------------
with tabs[10]:
    render_live_newark()
