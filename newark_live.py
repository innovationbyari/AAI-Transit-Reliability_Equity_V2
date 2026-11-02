"""Live NJ Transit bus data for Newark only: real routes, live buses, delays and alerts.

Uses the NJ Transit developer API (bus system GTFSG2) with the same login the notebook uses
(NJ_TRANSIT_USERNAME / NJ_TRANSIT_PASSWORD in the app's secrets):
  - getGTFS              static schedule zip: routes, trips, stops and route shapes (downloaded at most once a day)
  - getVehiclePositions  where each bus is now
  - getTripUpdates       how late each trip is
  - getAlerts            service alerts

"Newark only" means: routes whose path enters the City of Newark, and buses currently inside the city.
Nothing here changes a score; buses picked here are handed to the engine like the notebook's live hand-off.
"""
import csv
import io
import os
import time
import zipfile
from datetime import datetime, timezone

import requests

# City of Newark, NJ (bounding box of the city limits)
NEWARK_BBOX = {"lat_min": 40.6735, "lat_max": 40.7889, "lon_min": -74.2513, "lon_max": -74.1086}
NEWARK_CENTER = (40.7357, -74.1724)
NEWARK_PENN = (40.7347, -74.1642)

GTFS_MAX_AGE_S = 24 * 3600
SHAPE_EVERY_NTH_POINT = 3          # thin route shapes so the map stays light
LATE_MIN = 5                       # minutes late shown amber
VERY_LATE_MIN = 15                 # minutes late shown red (the notebook's hard trigger)


class LiveUnavailable(RuntimeError):
    pass


def in_newark(lat, lon):
    b = NEWARK_BBOX
    return lat is not None and lon is not None and b["lat_min"] <= lat <= b["lat_max"] and b["lon_min"] <= lon <= b["lon_max"]


# ---------------------------------------------------------------------------------------------------------- API calls
def _token(E, refresh=False):
    """Reuse the engine's NJ Transit token; log in again only when it is missing or rejected."""
    if refresh or not E._NJT.get("token"):
        status = E.njt_authenticate()
        if not E._NJT.get("token"):
            raise LiveUnavailable(f"NJ Transit login: {status}")
    return E._NJT["token"], E._NJT.get("system") or "GTFSG2"


def _post(E, endpoint, timeout=60):
    """POST one API call; retry once with a fresh token if NJ Transit rejects the token."""
    for attempt in (0, 1):
        token, system = _token(E, refresh=attempt == 1)
        url = E.NJT_BASE.format(system=system) + endpoint
        try:
            r = requests.post(url, files={"token": (None, token)}, timeout=timeout)
        except requests.RequestException as e:
            raise LiveUnavailable(f"{endpoint}: could not reach NJ Transit ({type(e).__name__})")
        body = r.content or b""
        head = body[:200].lstrip().lower()
        # NJ Transit sometimes answers HTTP 200 with a JSON error or the literal "Null" instead of data
        if b"invalid token" in head or r.status_code in (401, 403):
            if attempt == 0:
                continue
            raise LiveUnavailable(f"{endpoint}: NJ Transit rejected the login token")
        if r.status_code >= 400:
            raise LiveUnavailable(f"{endpoint}: HTTP {r.status_code}")
        if not body or head.startswith((b"null", b"{", b"[", b'"')):
            raise LiveUnavailable(f"{endpoint}: NJ Transit returned no data ({body[:80].decode('utf-8', 'replace')!r})")
        return body
    raise LiveUnavailable(f"{endpoint}: no response")


# ------------------------------------------------------------------------------------------- static routes (getGTFS)
def _read_csv(z, name):
    with z.open(name) as f:
        yield from csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"))


def parse_gtfs(zip_bytes):
    """Routes that enter Newark, one map line per route and direction, and the stops inside Newark."""
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = set(z.namelist())
    routes = {r["route_id"]: {"route_id": r["route_id"], "short": (r.get("route_short_name") or "").strip(),
                              "long": (r.get("route_long_name") or "").strip(),
                              "color": "#" + r["route_color"].strip() if (r.get("route_color") or "").strip() else None}
              for r in _read_csv(z, "routes.txt")}
    shape_route, shape_dir = {}, {}
    for t in _read_csv(z, "trips.txt"):
        if t.get("shape_id"):
            shape_route.setdefault(t["shape_id"], t["route_id"])
            shape_dir.setdefault(t["shape_id"], t.get("direction_id") or "0")

    lines, newark_routes = {}, set()
    if "shapes.txt" in names:
        # pass 1: which shapes enter Newark (streamed, so the file is never held in memory)
        enters = {s["shape_id"] for s in _read_csv(z, "shapes.txt")
                  if in_newark(float(s["shape_pt_lat"]), float(s["shape_pt_lon"]))}
        # pass 2: points of those shapes only
        pts = {}
        for s in _read_csv(z, "shapes.txt"):
            if s["shape_id"] in enters:
                pts.setdefault(s["shape_id"], []).append((int(s["shape_pt_sequence"]), float(s["shape_pt_lat"]),
                                                          float(s["shape_pt_lon"])))
        for sid, p in pts.items():
            rid = shape_route.get(sid)
            if not rid:
                continue
            p.sort()
            path = [(la, lo) for _, la, lo in p[::SHAPE_EVERY_NTH_POINT]] + [(p[-1][1], p[-1][2])]
            key = (rid, shape_dir.get(sid, "0"))
            if key not in lines or len(path) > len(lines[key]):      # keep the longest variant per route/direction
                lines[key] = path
            newark_routes.add(rid)

    stops = {}
    for s in _read_csv(z, "stops.txt"):
        try:
            la, lo = float(s["stop_lat"]), float(s["stop_lon"])
        except (TypeError, ValueError):
            continue
        if in_newark(la, lo):
            stops[s["stop_id"]] = {"stop_id": s["stop_id"], "name": s.get("stop_name", ""), "lat": la, "lon": lo}

    if not newark_routes and "stop_times.txt" in names and stops:
        # no shapes in the feed: fall back to routes that stop in Newark
        trip_route = {t["trip_id"]: t["route_id"] for t in _read_csv(z, "trips.txt")}
        for st_ in _read_csv(z, "stop_times.txt"):
            if st_["stop_id"] in stops and st_["trip_id"] in trip_route:
                newark_routes.add(trip_route[st_["trip_id"]])

    return {"routes": {rid: routes[rid] for rid in newark_routes if rid in routes},
            "lines": [{"route_id": rid, "direction": d, "path": path} for (rid, d), path in lines.items()],
            "stops": list(stops.values()), "all_route_names": {rid: r["short"] for rid, r in routes.items()}}


def load_static(E, cache_dir):
    """Download the bus schedule zip at most once a day; keep a copy on disk for restarts."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "njt_bus_gtfs.zip")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < GTFS_MAX_AGE_S:
        with open(path, "rb") as f:
            data = f.read()
        source = "saved copy from " + datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).astimezone(E.LOCAL_TZ).strftime("%b %d %H:%M %Z") \
            if hasattr(E, "LOCAL_TZ") else "saved copy"
    else:
        data = _post(E, "getGTFS", timeout=180)
        if not data.startswith(b"PK"):
            raise LiveUnavailable("getGTFS did not return a zip file")
        with open(path, "wb") as f:
            f.write(data)
        source = "downloaded from NJ Transit just now"
    out = parse_gtfs(data)
    out["source"] = source
    return out


# --------------------------------------------------------------------------------------------- realtime (live buses)
def _feed(E, endpoint):
    from google.transit import gtfs_realtime_pb2
    msg = gtfs_realtime_pb2.FeedMessage()
    try:
        msg.ParseFromString(_post(E, endpoint))
    except LiveUnavailable:
        raise
    except Exception as e:
        raise LiveUnavailable(f"{endpoint}: unreadable feed ({type(e).__name__})")
    return msg


def _text(ts):
    return ts.translation[0].text.strip() if ts and len(ts.translation) else ""


def load_live(E, static=None):
    """Buses inside Newark right now, with delay, plus alerts for Newark routes."""
    static = static or {"routes": {}, "all_route_names": {}}
    newark_routes = set(static["routes"])
    vp = _feed(E, "getVehiclePositions")
    errors = []
    try:
        tu = _feed(E, "getTripUpdates")
    except LiveUnavailable as e:
        tu, errors = None, errors + [str(e)]
    try:
        al = _feed(E, "getAlerts")
    except LiveUnavailable as e:
        al, errors = None, errors + [str(e)]

    updates = {}
    if tu is not None:
        for ent in tu.entity:
            if not ent.HasField("trip_update"):
                continue
            t = ent.trip_update
            delays = [s.arrival.delay if s.HasField("arrival") else s.departure.delay
                      for s in t.stop_time_update if s.HasField("arrival") or s.HasField("departure")]
            updates[t.trip.trip_id] = {"delay_s": delays[0] if delays else (t.delay if t.HasField("delay") else None),
                                       "cancelled": t.trip.schedule_relationship == t.trip.CANCELED,
                                       "route_id": t.trip.route_id}

    buses = []
    for ent in vp.entity:
        if not ent.HasField("vehicle"):
            continue
        v = ent.vehicle
        la, lo = v.position.latitude, v.position.longitude
        if not in_newark(la, lo):
            continue
        u = updates.get(v.trip.trip_id, {})
        rid = v.trip.route_id or u.get("route_id") or ""      # a bus inside the city is, by definition, serving Newark
        info = static["routes"].get(rid, {})
        ts = v.timestamp or vp.header.timestamp
        delay_s = u.get("delay_s")
        buses.append({
            "route_id": rid, "route": info.get("short") or static.get("all_route_names", {}).get(rid) or rid or "?",
            "route_name": info.get("long", ""), "vehicle_id": v.vehicle.id or v.vehicle.label or ent.id,
            "trip_id": v.trip.trip_id, "lat": la, "lon": lo,
            "delay_s": delay_s, "delay_min": None if delay_s is None else round(delay_s / 60, 1),
            "cancelled": bool(u.get("cancelled", False)), "timestamp": int(ts) if ts else None,
            "age_s": int(time.time() - ts) if ts else None,
            "km_from_penn": round(E.km_between((la, lo), NEWARK_PENN), 2),
        })
    buses.sort(key=lambda b: (not b["cancelled"], -(b["delay_s"] or 0)))

    alerts = []
    if al is not None:
        for ent in al.entity:
            if not ent.HasField("alert"):
                continue
            a = ent.alert
            rids = sorted({ie.route_id for ie in a.informed_entity if ie.route_id})
            stop_ids = {ie.stop_id for ie in a.informed_entity if ie.stop_id}
            newark_stop = bool(stop_ids & {s["stop_id"] for s in static.get("stops", [])})
            if (newark_routes and set(rids) & newark_routes) or newark_stop or (not newark_routes and rids):
                alerts.append({"id": ent.id, "routes": [static["routes"].get(r, {}).get("short") or r for r in rids],
                               "route_ids": rids, "header": _text(a.header_text),
                               "description": _text(a.description_text)[:600],
                               "effect": a.Effect.Name(a.effect) if a.effect else ""})
    return {"buses": buses, "alerts": alerts, "fetched_at": time.time(), "errors": errors,
            "feed_time": int(vp.header.timestamp) if vp.header.timestamp else None}


# ------------------------------------------------------------------------------------- hand a real bus to the engine
def inputs_for_bus(E, bus, alert=None):
    """Section A inputs for one real bus: live service and alert, other sources as the notebook supplies them."""
    inputs = E.section_a_sample_inputs()
    label = f"{bus['route']}" + (f" ({bus['route_name']})" if bus.get("route_name") else "")
    observed = E.iso(datetime.fromtimestamp(bus["timestamp"], timezone.utc)) if bus.get("timestamp") else E.iso(E.utcnow())
    inputs["service_feeds"] = {
        "route": label, "scheduled_headway_min": 15, "headway_source": "assumed (not computed from the schedule)",
        "delay_min": None if bus["delay_s"] is None else max(0.0, round(bus["delay_s"] / 60, 1)),
        "cancelled": bool(bus["cancelled"]), "trip_id": bus["trip_id"],
        "vehicle_position": f"{bus['lat']:.5f},{bus['lon']:.5f} ({bus['km_from_penn']} km from Newark Penn)",
        "source": "NJ Transit GTFSG2 GTFS-realtime (live, Newark)", "observed_at": observed}
    ev = inputs["event_signals"]
    keep = {k: ev[k] for k in ("weather", "weather_alerts", "weather_source") if k in ev}
    if alert:
        text = f"{alert['header']}. {alert['description']}".strip(". ")
        ev = {"source_type": "official_alert", "report": text, "report_source": "NJ Transit GTFS-realtime alert (live)",
              "source": "NJ Transit GTFS-realtime alert (live)",
              "road_closure": any(k in text.lower() for k in ("detour", "road closure", "closed")), "observed_at": E.iso(E.utcnow())}
    else:
        late = inputs["service_feeds"]["delay_min"] or 0
        ev = {"source_type": "none", "report_source": "generated from the live vehicle feed (no active alert)",
              "source": "NJ Transit GTFS-realtime vehicle feed (live; no alert)", "road_closure": False,
              "report": f"Route {label} is running {late:g} minutes late in Newark. No active NJ Transit service alert "
                        f"for this route.", "observed_at": observed}
    ev.update(keep)
    inputs["event_signals"] = ev
    return inputs, f"Live Newark · Route {label} · bus {bus['vehicle_id']}"
