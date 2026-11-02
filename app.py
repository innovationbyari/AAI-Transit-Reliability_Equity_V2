"""Gradio web interface: Agentic AI for Public Transit Reliability and Equity.

Two tabs:
  Dashboard              the notebook's own Gradio dashboard (human-in-the-loop review, scores, rules, audit)
  🔴 Live Newark buses   real NJ Transit routes and buses in Newark; start an incident from a real late bus

Runs on Hugging Face Spaces (Gradio SDK) or locally:  python app.py
The engine (engine.py) and the dashboard (gradio_dashboard.py) are generated from the notebook by build_engine.py.
Keys come from the Space's secrets (Settings → Variables and secrets), which arrive as environment variables.
"""
import html as _html
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gradio as gr  # noqa: E402

import engine as E  # noqa: E402  (importing runs the notebook's setup once: live data, vision, gallery, evaluation)
import newark_live as NL  # noqa: E402

# The notebook's dashboard cell, run inside the engine's namespace so it sees every notebook name
with open(os.path.join(HERE, "gradio_dashboard.py"), encoding="utf-8") as _f:
    exec(compile(_f.read(), "gradio_dashboard.py", "exec"), E.__dict__)
dashboard = E.demo

LIVE = NL.LiveCache(E, os.path.join(getattr(E, "OUTPUT_DIR", "transit_outputs"), "njt_cache"))


def _here(text):
    return str(text).replace("Colab Secrets", "the Space's secrets")


def _iframe(folium_map, height=520):
    doc = folium_map.get_root().render()
    return (f'<iframe srcdoc="{_html.escape(doc, quote=True)}" '
            f'style="width:100%;height:{height}px;border:0;border-radius:8px"></iframe>')


NOT_CONNECTED = (
    "### Not connected to NJ Transit yet\n{status}\n\n"
    "To switch this on, open your Space → **Settings → Variables and secrets → New secret** and add:\n"
    "- `NJ_TRANSIT_USERNAME` = your NJ Transit developer username\n"
    "- `NJ_TRANSIT_PASSWORD` = your NJ Transit developer password\n\n"
    "Then open **🔧 Connection check** → **Apply new keys**, or restart the Space (**⋮ → Restart this Space**).")


def _bus_by_id(vid):
    buses = (LIVE.live or {}).get("buses", [])
    return next((b for b in buses if str(b["vehicle_id"]) == str(vid)), None)


def _alerts_for(bus):
    if not bus or not LIVE.live:
        return []
    return [a for a in LIVE.live["alerts"] if bus["route_id"] in a["route_ids"]]


def _bus_info(bus):
    if not bus:
        return ""
    stale = " ⚠️ last report over 5 minutes ago, so the system will treat it as stale." if (bus["age_s"] or 0) > 300 else ""
    return (f"**Route {bus['route']}** {bus['route_name']} · bus {bus['vehicle_id']} · **{NL.delay_text(bus)}** · "
            f"{bus['km_from_penn']} km from Newark Penn.{stale}")


def live_refresh(force=False):
    """Status, map, bus table, bus picker, alert picker, bus details, alerts list."""
    empty_dd = gr.update(choices=[], value=None)
    if not E._NJT.get("token"):
        status = NOT_CONNECTED.format(status=_here(E.LIVE_CONTEXT["nj_transit"].get("status", "")))
        return status, _iframe(NL.build_map(None, None)), None, empty_dd, empty_dd, "", ""
    static = LIVE.get_static()
    try:
        live = LIVE.get_live(force=force)
    except NL.LiveUnavailable as e:
        return f"### Could not get live buses from NJ Transit\n{e}", _iframe(NL.build_map(static, None)), None, \
            empty_dd, empty_dd, "", ""
    buses = live["buses"]
    n_late = sum((b["delay_min"] or 0) >= NL.LATE_MIN or b["cancelled"] for b in buses)
    status = (f"**{len(buses)} buses** in Newark now · **{n_late}** at least {NL.LATE_MIN} min late or cancelled · "
              f"**{len(live['alerts'])}** service alerts")
    status += (f"  \n{len(static['routes'])} real bus routes enter Newark ({static['source']})." if static
               else f"  \nRoute lines are not available right now{': ' + LIVE.static_error if LIVE.static_error else ''}. "
                    "Live buses are still shown.")
    for err in live.get("errors", []):
        status += f"  \n⚠️ {err}"
    if not buses:
        status += "  \nNo NJ Transit buses are reporting inside Newark right now (normal late at night)."
    table = [[b["route"], b["route_name"], b["vehicle_id"], NL.delay_text(b), b["km_from_penn"], b["age_s"]] for b in buses]
    choices = [(NL.bus_label(b), str(b["vehicle_id"])) for b in buses]
    first = buses[0] if buses else None
    alerts = _alerts_for(first)
    alert_dd = gr.update(choices=[("No alert", "")] + [(a["header"][:90] or a["description"][:90], a["id"]) for a in alerts],
                         value=alerts[0]["id"] if alerts else "")
    alerts_md = "\n\n".join(f"**Routes {', '.join(a['routes']) or '—'}:** {a['header']}  \n{a['description']}"
                            for a in live["alerts"]) or "_No NJ Transit service alerts for Newark routes right now._"
    return (status, _iframe(NL.build_map(static, live, highlight=first["route_id"] if first else None)), table,
            gr.update(choices=choices, value=choices[0][1] if choices else None), alert_dd, _bus_info(first), alerts_md)


def live_pick(vid):
    bus = _bus_by_id(vid)
    alerts = _alerts_for(bus)
    alert_dd = gr.update(choices=[("No alert", "")] + [(a["header"][:90] or a["description"][:90], a["id"]) for a in alerts],
                         value=alerts[0]["id"] if alerts else "")
    return _iframe(NL.build_map(LIVE.static, LIVE.live, highlight=bus["route_id"] if bus else None)), alert_dd, _bus_info(bus)


def live_start(vid, alert_id):
    bus = _bus_by_id(vid)
    if not bus:
        return "⚠️ Choose a bus first (press **↻ Update buses** if the list is empty)."
    alert = next((a for a in _alerts_for(bus) if a["id"] == alert_id), None) if alert_id else None
    try:
        with E.APP.lock:
            inputs, name = NL.inputs_for_bus(E, bus, alert)
            ssi = E.build_signal_scout_input(inputs)
            E.APP.new_case(E.signal_scout_input_to_raw(ssi, name=name), use_llm=E.LLM_READY)
            c = E.APP.active
        return (f"✅ **{c['case_id']}** created from bus {bus['vehicle_id']} on route {bus['route']}: "
                f"**{E.STATUS_HELP.get(c['status'], c['status'])}**.  \nOpen the **Dashboard** tab and press "
                "**↻ Refresh** to review it.")
    except Exception as e:  # show the problem instead of a blank error box
        traceback.print_exc()
        return f"⚠️ Could not start the incident: {e}"


with gr.Blocks(title="Live Newark buses") as live_tab:
    gr.Markdown("## 🔴 Real NJ Transit buses in Newark, right now\n"
                "Real bus routes that enter the City of Newark, every bus inside the city, and NJ Transit service alerts. "
                "Pick a late bus to start an incident; it then goes through the same checks, ranking and human review "
                "as every other case. *Rider and neighbourhood figures used in the scoring come from the Census connector "
                "or the documented sample, and responses are planned on the practice network.*")
    with gr.Row():
        status_md = gr.Markdown()
        refresh_btn = gr.Button("↻ Update buses", scale=0)
    map_html = gr.HTML()
    with gr.Row():
        with gr.Column(scale=3):
            bus_table = gr.Dataframe(headers=["Route", "Route name", "Bus", "Status", "km from Newark Penn",
                                              "Seconds since update"], interactive=False, wrap=True,
                                     label="Buses in Newark (cancelled and most-late first)")
        with gr.Column(scale=2):
            bus_dd = gr.Dropdown(label="Choose a bus", choices=[])
            bus_info = gr.Markdown()
            alert_dd = gr.Dropdown(label="NJ Transit alert to include", choices=[])
            start_btn = gr.Button("▶ Start incident from this real bus", variant="primary")
            start_msg = gr.Markdown()
    with gr.Accordion("NJ Transit service alerts for Newark routes", open=False):
        alerts_md = gr.Markdown()

    outs = [status_md, map_html, bus_table, bus_dd, alert_dd, bus_info, alerts_md]
    live_tab.load(lambda: live_refresh(False), outputs=outs)
    refresh_btn.click(lambda: live_refresh(True), outputs=outs)
    bus_dd.input(live_pick, inputs=[bus_dd], outputs=[map_html, alert_dd, bus_info])
    start_btn.click(live_start, inputs=[bus_dd, alert_dd], outputs=[start_msg])


import diagnostics as DG  # noqa: E402


def _md_table(rows):
    if not rows:
        return ""
    cols = list(rows[0])
    esc = lambda v: str(v).replace("|", "/").replace("\n", " ")
    return ("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
            + "".join("| " + " | ".join(esc(r[c]) for c in cols) + " |\n" for r in rows))


def dg_secrets():
    return _md_table(DG.secrets_table(E))


def dg_reload():
    with E.APP.lock:
        r = DG.reload_keys(E)
    return (f"✅ Keys reloaded. AI providers in use: **{', '.join(r['ai_providers']) or 'none'}** · NJ Transit: "
            f"**{_here(r['nj_transit'])}**", dg_secrets())


def dg_run():
    return _md_table(DG.run_all(E, LIVE))


with gr.Blocks(title="Connection check") as check_tab:
    gr.Markdown("## 🔧 Are the keys working?\nThis page tests every key and login with one small real request each. "
                "Keys are never shown.\n\n**1. Which secrets the app can see**")
    secrets_md = gr.Markdown()
    gr.Markdown("If a secret shows **no**: open your Space → **Settings → Variables and secrets**, check the exact name "
                "(capital letters and underscores, e.g. `NJ_TRANSIT_USERNAME`), save it as a **secret**, then press "
                "**Apply new keys**.")
    with gr.Row():
        reload_btn = gr.Button("🔄 Apply new keys")
        run_btn = gr.Button("▶ Run connection check", variant="primary")
    reload_md = gr.Markdown()
    gr.Markdown("**2. Results**")
    results_md = gr.Markdown("_Press **Run connection check**._")
    gr.Markdown("**3. Where each AI agent shows its work**\n\n"
                + _md_table([{"Agent": a, "What the AI writes": w, "Where to see it": where} for a, w, where in DG.AGENT_TABLE])
                + "\nAn AI agent is working when its text names a provider (for example “OpenAI · gpt-4.1-mini”) "
                  "instead of “template”. The AI only writes text: scores, rankings and decisions never depend on it.")
    check_tab.load(dg_secrets, outputs=[secrets_md])
    reload_btn.click(dg_reload, outputs=[reload_md, secrets_md])
    run_btn.click(dg_run, outputs=[results_md])

app = gr.TabbedInterface([dashboard, live_tab, check_tab], ["Dashboard", "🔴 Live Newark buses", "🔧 Connection check"],
                         title="Transit Reliability & Equity — Human-in-the-loop dashboard")

if __name__ == "__main__":
    pw = os.environ.get("DASHBOARD_PASSWORD", "").strip()
    app.queue().launch(auth=("reviewer", pw) if pw else None)
