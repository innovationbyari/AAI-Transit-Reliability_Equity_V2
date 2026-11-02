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
import types

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
# Sidebar: system status, case selection, run a case
# --------------------------------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚌 Transit Reliability & Equity")
    st.caption("Prototype · deterministic scoring · human-supervised decisions · **not connected to live operations**")

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

    st.divider()
    st.markdown("### Run a case")
    preset = st.selectbox("Scenario", list(PRESETS), index=3 if len(PRESETS) > 3 else 0)
    use_llm = st.checkbox("Live LLM narrative", value=E.LLM_READY, disabled=not E.LLM_READY,
                          help="Add an API key in the app's secrets to enable. LLM text never enters a score or decision.")
    if st.button("▶ Run pipeline", type="primary", width="stretch"):
        act(lambda: APP.new_case(PRESETS[preset](), use_llm=use_llm))

    st.divider()
    st.markdown("### Active case")
    ids = list(APP.cases)[::-1]
    if ids:
        cur = APP.active_id if APP.active_id in ids else ids[0]
        pick = st.selectbox("Case", ids, index=ids.index(cur),
                            format_func=lambda i: f"{i} · {APP.cases[i]['status']} · {APP.cases[i]['scenario_name'][:40]}")
        if pick != APP.active_id:
            with APP.lock:
                APP.active_id = pick
            st.rerun()
    else:
        st.info("No cases yet. Run a scenario above.")

    st.divider()
    if st.button("↻ Refresh", width="stretch"):
        st.rerun()
    with st.expander("Reset demo"):
        st.caption("Everyone using this app shares one demo state. Reset re-runs the setup (about 30 s) and clears all "
                   "cases and decisions.")
        if st.button("Reset all cases and rules", type="secondary"):
            GEN["n"] += 1
            st.session_state.clear()
            st.rerun()


# --------------------------------------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------------------------------------
st.title("Agentic AI for Public Transit Reliability and Equity")
st.caption("AI agents sense, organize and explain · fixed, published rules score · people decide. "
           "A language model's text never enters a score, a routing decision or an approval.")

if "flash" in st.session_state:
    msg, level = st.session_state.pop("flash")
    getattr(st, level if level in ("success", "warning", "info", "error") else "info")(here(msg))

c = APP.active


@st.fragment(run_every="5s")
def deadline_banner():
    """Review timers keep running: check the deadline every few seconds (the dashboard's 2 s timer)."""
    with APP.lock:
        case = APP.active
        expired = bool(case and APP.enforce_deadline(case))
    if case:
        st.markdown(f"{status_pill(case['status'])} **{case['case_id']}** · {case['scenario_name']}",
                    unsafe_allow_html=True)
        st.markdown(E.deadline_md(case), unsafe_allow_html=True)
        if expired:
            st.warning("The review deadline expired and the safe default was applied. Press ↻ Refresh.")


deadline_banner()

tabs = st.tabs(["1 · Intake & verification", "2 · Component scores", "3 · Options & ranking", "4 · Human review",
                "5 · Implementation & outcome", "6 · Policy governance", "7 · Ethics & independence",
                "8 · Gallery & evaluation", "9 · Audit & export", "0 · Data inputs (Section A)"])

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
