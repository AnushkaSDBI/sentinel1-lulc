"""Prototype: 12 monthly land-cover maps from Sentinel-1 for any place.

Run from the repository root:   streamlit run prototype/app.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
try:                                   # basemap and charts; the app still runs without them (plain maps)
    import folium
    import plotly.express as px
    import plotly.graph_objects as go
    from streamlit_folium import st_folium
    HAVE_MAP = True
except ImportError:
    HAVE_MAP = False
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from lulc import proto
from lulc.classes import CLASS_COLORS, CLASS_NAMES
from lulc.ood import area_signature, ood_check
from lulc.quant import dequantize_db

st.set_page_config(page_title="Sentinel-1 land cover", layout="wide")
CMAP = ListedColormap(["#000000", *CLASS_COLORS])           # 0 = no data, 1..6 = classes
LEGEND = [Patch(color="#000000", label="no data"), *[Patch(color=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)]]
SIZES = {"2.56 km": 2.56, "5.12 km (default)": 5.12, "10.24 km (slow)": 10.24}
CACHE = ROOT / ".cache_lulc"
SAT_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
SAT_ATTR = "Esri, Maxar, Earthstar Geographics"



def show(pred):
    """class id 0..5 -> 1..6, ignore -> 0, for the colour map"""
    return np.where(pred == 255, 0, pred.astype(int) + 1)


@st.cache_resource(show_spinner="Loading models ...")
def get_models(path: str, device: str):
    return proto.load_ensemble(path, torch.device(device))


LEGEND_HINT = {"Tree cover": "forest, plantation", "Grass/shrub": "grass, scrub, pasture", "Cropland": "fields, farmland",
               "Built-up": "towns, roads, buildings", "Bare": "sand, rock, soil", "Water": "rivers, lakes, sea"}


def legend_html():
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;margin:2px 14px 2px 0;font-size:0.9rem">'
        f'<span style="width:16px;height:16px;border-radius:3px;background:{c};border:1px solid #888;margin-right:6px"></span>'
        f'<b>{n}</b>&nbsp;<span style="color:#999;font-size:0.8rem">{LEGEND_HINT.get(n, "")}</span></span>'
        for c, n in zip(CLASS_COLORS, CLASS_NAMES))
    return f'<div style="display:flex;flex-wrap:wrap;line-height:1.6">{chips}</div>'


def metric_card(label, value):
    return (f'<div style="padding:4px 0"><div style="font-size:0.78rem;color:#8a929c;text-transform:uppercase;letter-spacing:0.03em">{label}</div>'
            f'<div style="font-size:1.05rem;font-weight:600;word-break:break-word;line-height:1.3">{value}</div></div>')


def fmt_period(labels):
    a, b = labels[0], labels[-1]
    (ma, ya), (mb, yb) = a.split(), b.split()
    return f"{ma}–{mb} {ya}" if ya == yb else f"{a} – {b}"


def class_rgb(pred, ignore: int = 255):
    pal = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in CLASS_COLORS] + [[18, 20, 24]], np.uint8)
    return pal[np.where(pred == ignore, len(CLASS_COLORS), pred)]


@st.cache_data(show_spinner=False)
def place_name(lat: float, lon: float):
    """Reverse-geocode a point to a readable name. Returns None if the network is blocked (common on a compute server)."""
    try:
        import requests
        r = requests.get("https://nominatim.openstreetmap.org/reverse",
                         params={"lat": lat, "lon": lon, "format": "json", "zoom": 8},
                         headers={"User-Agent": "sar-lulc-prototype"}, timeout=4)
        a = r.json().get("address", {})
        for k in ("city", "town", "village", "county", "state", "region"):
            if a.get(k):
                return a[k] + (f", {a['country']}" if a.get("country") else "")
        return a.get("country")
    except Exception:
        return None


def base_map(bounds, satellite: bool):
    m = folium.Map(location=[(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2], zoom_start=13,
                   tiles=None, control_scale=True)
    if satellite:
        folium.TileLayer(tiles=SAT_TILES, attr=SAT_ATTR, name="Satellite photo").add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)
    m.fit_bounds(bounds)
    return m


def false_colour(store, i):
    vv, vh = dequantize_db(store.s1[i, 0]), dequantize_db(store.s1[i, 1])
    rgb = np.stack([vv, vh, vv - vh], -1)
    for b, (lo, hi) in enumerate([(-20, 0), (-27, -7), (2, 15)]):
        rgb[..., b] = np.clip((rgb[..., b] - lo) / (hi - lo), 0, 1)
    rgb = np.nan_to_num(rgb)
    try:                                   # light smoothing for display only (radar is speckly); the model still sees the raw data
        from scipy.ndimage import uniform_filter
        for b in range(3):
            rgb[..., b] = uniform_filter(rgb[..., b], size=2)
    except Exception:
        pass
    return rgb


# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("1. Model")
weights = sb.text_input("Weights folder", os.environ.get("LULC_WEIGHTS", str(ROOT / "weights")),
                        help="Folder with final_model.pt (from scripts/package_model.py), or one sub-folder per seed with best.pt and model_card.json")
device = "cuda" if torch.cuda.is_available() else "cpu"
models = card = norm = names = None
try:
    models, card, norm, names = get_models(weights, device)
    sb.success(f"{len(models)} model(s): {', '.join(names)} on {device.upper()}")
    if card is not None and proto.uses_coh(card):
        sb.caption("This is the coherence research model. It needs an interferometric coherence file for the area, which is not "
                   "fetched live, so it runs only on a local export that already has one, not on an arbitrary Earth Engine place. "
                   "The main prototype uses the Sentinel-1-only model.")
except Exception as e:
    sb.error(str(e))

pend = st.session_state.pop("pick_pending", None)      # a click on the picker map, applied before the widgets are built
if pend:
    st.session_state["lat"], st.session_state["lon"] = pend
st.session_state.setdefault("lat", 49.4000)
st.session_state.setdefault("lon", 8.6700)
sb.header("2. Place and time")
source = sb.radio("Data source", ["Earth Engine", "Local export (offline test)"])
if source == "Earth Engine":
    project = sb.text_input("Earth Engine project", os.environ.get("EE_PROJECT", ""), type="password")
    c1, c2 = sb.columns(2)
    lat = c1.number_input("Latitude", -80.0, 84.0, key="lat", step=0.01, format="%.4f")
    lon = c2.number_input("Longitude", -180.0, 180.0, key="lon", step=0.01, format="%.4f")
    c1, c2 = sb.columns(2)
    year = int(c1.number_input("Start year", proto.FIRST_YEAR, 2026, 2024, step=1))
    month = int(c2.selectbox("Start month", range(1, 13), index=0, format_func=lambda m: proto.MONTH_NAMES[m - 1]))
    size_km = SIZES[sb.selectbox("Area", list(SIZES), index=1)]
    if HAVE_MAP:
        with sb.expander("Pick the place on a map"):
            pm = folium.Map(location=[lat, lon], zoom_start=10, tiles=None)
            folium.TileLayer(tiles=SAT_TILES, attr=SAT_ATTR, name="Satellite photo").add_to(pm)
            folium.Marker([lat, lon]).add_to(pm)
            hit = st_folium(pm, height=260, width=290, returned_objects=["last_clicked"], key="picker")
            lc = (hit or {}).get("last_clicked")
            if lc and (lc["lat"], lc["lng"]) != st.session_state.get("last_pick"):
                st.session_state["last_pick"] = (lc["lat"], lc["lng"])
                st.session_state["pick_pending"] = (round(lc["lat"], 4), round(lc["lng"], 4))
                st.rerun()
            nm = place_name(round(lat, 4), round(lon, 4))
            st.caption((f"**{nm}** — " if nm else "") + f"{lat:.4f}, {lon:.4f}. Click to move the marker, then press Run.")
else:
    data_dir = sb.text_input("Export folder", "data/raw")
    region_key = sb.text_input("Region", "heidelberg")

sb.header("3. Options")
smooth = sb.checkbox("3-month majority smoothing", True,
                     help="Cuts false change by about a third in our test. Turn off to see raw predictions.")
run = sb.button("Run", type="primary", disabled=models is None)

# ------------------------------------------------------------------ run
st.title("Sentinel-1 land cover: 12 monthly maps")
st.caption("Sentinel-1 radar only. Six classes: " + ", ".join(CLASS_NAMES) + ".")

if run:
    bar = st.progress(0.0, "starting ...")
    try:
        reference = None
        bounds = None
        if source == "Earth Engine":
            if not project:
                st.error("Enter your Earth Engine project ID.")
                st.stop()
            import ee

            ee.Initialize(project=project)
            region = proto.make_region(lat, lon, size_km)
            ym = proto.month_sequence(year, month)
            for w in proto.check_period(ym):
                st.warning(w)
            proto.fetch_months(ee, region, ym, CACHE, lambda f, t: bar.progress(f * 0.5, t))
            store = proto.load_store_ym(CACHE, region.key, ym)
            if sum(y == 2021 for y, _ in ym) >= 6:
                try:
                    reference = proto.fetch_reference(ee, region, CACHE)
                except Exception as e:
                    st.warning(f"Reference maps not available: {e}")
            place = f"{lat:.4f}, {lon:.4f}"
            bounds = proto.bounds_latlon(region)
        else:
            store, ym = proto.load_local(data_dir, region_key)
            reference = proto.load_reference(Path(data_dir) / region_key / "reference.npz")
            place = region_key
            try:
                from lulc.config import load_config
                bounds = proto.bounds_latlon(next(r for r in load_config(ROOT / "configs" / "regions.yaml")[1] if r.key == region_key))
            except Exception:
                bounds = None                                    # region not in the config: plain maps
        use_coh = proto.uses_coh(card)
        if use_coh:                                              # coherence research model: load the area's coherence file
            src_dir = CACHE if source == "Earth Engine" else data_dir
            key = region.key if source == "Earth Engine" else region_key
            if not proto.attach_coherence(store, src_dir, key):
                st.error("The coherence model needs a coherence file for this area, and none is on disk. It is not fetched live. "
                         "Run it on a local export that has coherence_hyp3.npz, or switch to the Sentinel-1-only model.")
                st.stop()
            st.caption("Coherence research model: using the coherence file already on disk for this area.")
        preds, conf, members, months = proto.run_models(
            models, store, norm, card["window_months"], torch.device(device), use_stats=proto.uses_stats(card), use_coh=use_coh,
            progress=lambda f, t: bar.progress(0.5 + f * 0.5 if source == "Earth Engine" else f, t))
        labels = proto.month_labels([ym[i] for i in months])
        ood = ood_check(area_signature(conf, members), proto.load_ood_reference(weights))
        st.session_state["res"] = dict(ood=ood, store=store, ym=[ym[i] for i in months], months=months, labels=labels, preds=preds,
                                       members=members, reference=reference, bounds=bounds, place=place, smooth=smooth, k=len(models))
        bar.empty()
    except Exception as e:
        bar.empty()
        st.error(f"Run failed: {e}")
        if source == "Earth Engine":
            st.info("Earth Engine needs a one-time login: run `earthengine authenticate` in a terminal. "
                    "The project must be registered for Earth Engine.")
        st.stop()

res = st.session_state.get("res")
if res is None:
    st.info("Choose the weights folder, a place and a start month, then press **Run**.")
    st.stop()

out = proto.analyse(res["preds"], res["members"], res["labels"], res["smooth"])
preds, stats, labels = out["preds"], out["stats"], res["labels"]
store, months = res["store"], res["months"]

place_disp = res["place"]
if isinstance(place_disp, str) and "," in place_disp and all(c in "0123456789.,- " for c in place_disp):
    la, lo = (float(x) for x in place_disp.split(","))
    nm = place_name(round(la, 4), round(lo, 4))
    place_disp = f"{nm}<br><span style='color:#8a929c;font-size:0.85rem'>{la:.3f}, {lo:.3f}</span>" if nm else f"{la:.3f}, {lo:.3f}"
m1, m2, m3, m4 = st.columns(4)
m1.markdown(metric_card("Place", place_disp), unsafe_allow_html=True)
m2.markdown(metric_card("Period", fmt_period(labels)), unsafe_allow_html=True)
m3.markdown(metric_card("Models in ensemble", res["k"]), unsafe_allow_html=True)
m4.markdown(metric_card("Pixels used for change", f"{stats.valid_fraction.iloc[0] * 100:.0f}%"), unsafe_allow_html=True)
missing = [proto.month_labels([res['ym'][i]])[0] for i in range(len(months)) if not store.s1[months[i], 0].any()]
if missing:
    st.warning(f"No Sentinel-1 data for: {', '.join(missing)}.")

ood = res.get("ood")
if ood and ood["level"] == "low":
    st.warning(ood["message"] + " Treat the maps as unreliable here (see Method and limits).")
elif ood and ood["level"] == "ok":
    st.caption("Area check: " + ood["message"])

gap = [(labels[k], proto.uncovered_share(store.s1[months[k]])) for k in range(len(months))]
gap = [(a, b) for a, b in gap if b > 0.05]
if gap:
    st.warning("More than 5% of the pixels have no radar value in: " + ", ".join(f"{a} ({100 * b:.0f}%)" for a, b in gap)
               + ". Those pixels are filled with the training mean and carry no information; the change between such months is not real.")

scenes = [store.meta.get("s1_scenes", [float("nan")] * len(store.present))[i] for i in months]
low = proto.scene_flags(scenes)
if any(low):
    st.warning("Few Sentinel-1 scenes in: " + ", ".join(f"{labels[k]} ({scenes[k]:.0f})" for k, f in enumerate(low) if f)
               + ". The composite is noisier in these months, and the maps can change for that reason alone.")

tab_map, tab_share, tab_change, tab_years, tab_check, tab_info = st.tabs(
    ["Maps", "Class shares", "Change", "Change over years", "Check against independent maps", "Method and limits"])

with tab_map:
    i = st.select_slider("Month", options=list(range(len(labels))), format_func=lambda k: labels[k], key="month")
    bounds = res.get("bounds")
    left, right = st.columns(2)
    if HAVE_MAP and bounds is not None:
        c1, c2 = left.columns(2)
        alpha = c1.slider("Class layer opacity", 0.0, 1.0, 0.6, 0.05)
        satellite = c2.radio("Basemap", ["Satellite photo", "Street map"], horizontal=True) == "Satellite photo"
        hl = left.checkbox("Highlight pixels that changed since the previous month", False, disabled=i == 0,
                           help="Unchanged pixels are faded. Most highlighted pixels are noise, see the Change tab.")
        m = base_map(bounds, satellite)
        img = proto.highlight_rgba(preds[i], preds[i - 1], CLASS_COLORS) if hl and i > 0 else proto.overlay_rgba(preds[i], CLASS_COLORS)
        folium.raster_layers.ImageOverlay(image=img, bounds=bounds, opacity=alpha,
                                          name=labels[i]).add_to(m)
        hit = st_folium(m, height=520, use_container_width=True, returned_objects=["last_clicked"], key="resmap")
        left.markdown(legend_html(), unsafe_allow_html=True)
        lc = (hit or {}).get("last_clicked")
        px = proto.click_to_pixel(lc["lat"], lc["lng"], bounds, preds.shape) if lc else None
    else:
        px = None
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(show(preds[i]), cmap=CMAP, vmin=0, vmax=6, interpolation="nearest"); ax.axis("off"); ax.set_title(labels[i])
        fig.legend(handles=LEGEND, loc="lower center", ncol=4, fontsize=8); plt.tight_layout(rect=(0, 0.06, 1, 1))
        left.pyplot(fig); plt.close(fig)
        if bounds is None:
            left.caption("No map position for this region, so no basemap. Earth Engine runs always have one.")
    right.image(class_rgb(preds[i]), caption=f"Predicted land cover, {labels[i]}", clamp=True, use_container_width=True)
    right.image(false_colour(store, months[i]), caption=f"Sentinel-1 radar, {labels[i]}: R = VV, G = VH, B = VV-VH", clamp=True, use_container_width=True)
    share = stats.percent.iloc[i]
    right.bar_chart(share.rename("% of area"), height=220, color="#555555")
    if px is not None:
        series = proto.pixel_series(preds, px[0], px[1], labels)
        color_of = dict(zip(CLASS_NAMES, CLASS_COLORS))
        cells = "".join(f'<td style="background:{color_of.get(c, "#000")};color:{"#000" if c in ("Grass/shrub", "Bare") else "#fff"};'
                        f'padding:2px 6px;font-size:0.75rem;text-align:center">{c}</td>' for c in series["class"])
        heads = "".join(f'<th style="font-size:0.7rem;padding:2px 6px">{mth}</th>' for mth in series["month"])
        st.markdown(f"**Clicked pixel** (row {px[0]}, column {px[1]}): class in each month<br>"
                    f'<table style="width:100%;border-collapse:collapse"><tr>{heads}</tr><tr>{cells}</tr></table>', unsafe_allow_html=True)
        st.caption("A pixel that keeps switching class is the model's noise, not land-use change.")
    elif HAVE_MAP and bounds is not None:
        st.caption("Click a pixel on the map to see its class in every month.")
    with st.expander("All months"):
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for ax, p, lab in zip(axes.ravel(), preds, labels):
            ax.imshow(show(p), cmap=CMAP, vmin=0, vmax=6, interpolation="nearest"); ax.set_title(lab); ax.axis("off")
        fig.legend(handles=LEGEND, loc="lower center", ncol=7); plt.tight_layout(rect=(0, 0.04, 1, 1))
        st.pyplot(fig)
        buf = io.BytesIO(); fig.savefig(buf, dpi=110, bbox_inches="tight")
        st.download_button("Download all maps (PNG)", buf.getvalue(), "lulc_maps.png", "image/png")
        plt.close(fig)

with tab_share:
    st.subheader("Share of the area per class (%)")
    st.dataframe(stats.percent.round(2))
    if HAVE_MAP:
        area = go.Figure()
        for name, color in zip(CLASS_NAMES, CLASS_COLORS):
            area.add_trace(go.Scatter(x=labels, y=stats.percent[name], name=name, mode="lines", stackgroup="one",
                                      line=dict(width=0.5, color="#444"), fillcolor=color))
        area.update_layout(height=340, yaxis=dict(title="% of area", range=[0, 100]), margin=dict(l=10, r=10, t=10, b=10),
                           legend=dict(orientation="h"))
        st.plotly_chart(area, width="stretch")
    st.markdown("**Each class with the spread between ensemble members**")
    if HAVE_MAP:
        line = go.Figure()
        for name, color in zip(CLASS_NAMES, CLASS_COLORS):
            y = stats.percent[name].values
            if out["spread"] is not None:                       # one-std band as a filled envelope
                s = out["spread"][name].values
                line.add_trace(go.Scatter(x=list(labels) + list(labels)[::-1], y=list(y + s) + list((y - s))[::-1],
                                          fill="toself", fillcolor=color, opacity=0.18, line=dict(width=0),
                                          hoverinfo="skip", showlegend=False))
            line.add_trace(go.Scatter(x=labels, y=y, mode="lines+markers", name=name, line=dict(color=color),
                                      hovertemplate=name + ": %{y:.1f}%<extra></extra>"))
        line.update_layout(height=360, yaxis=dict(title="% of area"), margin=dict(l=10, r=10, t=10, b=10),
                           legend=dict(orientation="h"))
        st.plotly_chart(line, width="stretch")
    else:
        fig, ax = plt.subplots(figsize=(11, 4))
        for name, color in zip(CLASS_NAMES, CLASS_COLORS):
            y = stats.percent[name].values
            ax.plot(labels, y, marker="o", color=color, label=name)
            if out["spread"] is not None:
                s = out["spread"][name].values
                ax.fill_between(labels, y - s, y + s, color=color, alpha=0.2)
        ax.set_ylabel("% of area"); ax.tick_params(axis="x", rotation=45); ax.grid(alpha=0.3); ax.legend(ncol=3, fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    if out["spread"] is not None:
        st.caption("Bands show the spread (one standard deviation) between the ensemble members, a measure of model uncertainty.")
    st.download_button("Download shares (CSV)", stats.percent.round(4).to_csv(), "class_shares.csv", "text/csv")

with tab_change:
    floor = out["floor"]
    st.subheader("Month-to-month change (percentage points)")
    steps = stats.delta.iloc[1:].round(2).copy()
    steps["total churn"] = out["churn"].round(2)
    steps["scenes (to month)"] = [("" if not np.isfinite(scenes[k + 1]) else f"{scenes[k + 1]:.0f}") for k in range(len(labels) - 1)]
    steps["vs noise floor"] = np.where(out["within_noise"], "within noise", "ABOVE noise")
    steps.index = [f"{a} to {b}" for a, b in zip(labels[:-1], labels[1:])]
    st.dataframe(steps)
    above = [steps.index[k] for k, w in enumerate(out["within_noise"]) if not w]
    if above:
        st.warning(f"Total churn is above the noise floor ({floor} pp) in: {', '.join(above)}. "
                   "These months are the candidates for real change. Check them against the maps and the radar image.")
    else:
        st.info(f"No month-to-month change is above the noise floor ({floor} pp total churn per month). "
                "Nothing here can be told apart from classification noise.")
    st.subheader(f"Total change {labels[0]} to {labels[-1]} (percentage points)")
    st.dataframe(stats.total_change.round(2).to_frame("pp"))
    st.write(f"Net change across all classes: **{out['net']:.1f} pp**. "
             f"On land that did not change, our test showed about {proto.NOISE_FLOOR['net_first_to_last_pp']} pp between the first and "
             f"last month, so a net change below that is {'' if out['net_within_noise'] else 'not '}within noise here.")
    st.caption(f"Noise floor from: {proto.NOISE_FLOOR['source']}. The labels are annual, so the monthly maps have no ground truth; "
               "seasonal radar effects (snow, wet soil, harvest) look like change.")
    st.download_button("Download change (CSV)", steps.to_csv(), "class_change.csv", "text/csv")
    st.subheader("Where the class differs between the first and the last month")
    ch = proto.change_map(preds[0], preds[-1])
    c1, c2, c3 = st.columns(3)
    for col, arr, title in ((c1, preds[0], labels[0]), (c2, preds[-1], labels[-1]), (c3, ch, "Changed pixels (class of last month)")):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(np.full(arr.shape, 0.92), cmap="gray", vmin=0, vmax=1) if arr is ch else None
        ax.imshow(np.ma.masked_where(show(arr) == 0, show(arr)), cmap=CMAP, vmin=0, vmax=6, interpolation="nearest")
        ax.axis("off"); ax.set_title(title, fontsize=9)
        col.pyplot(fig); plt.close(fig)
    valid = (preds[0] != 255) & (preds[-1] != 255)
    st.caption(f"{100 * (ch != 255).sum() / max(valid.sum(), 1):.1f}% of the pixels have a different class in the last month. "
               f"On land that did not change, our test found about {proto.NOISE_FLOOR['flip_rate_raw_pct']}% flips between months, "
               "so most of this is noise unless it forms large, compact patches.")

with tab_years:
    st.subheader("Same place, several years")
    st.write("Each year is the 12 months from the chosen start month, smoothed, then reduced to one map per year "
             "(most frequent class per pixel). Every year costs 12 Earth Engine downloads plus inference.")
    if source != "Earth Engine":
        st.info("This tab needs the Earth Engine data source.")
    else:
        this_year = pd.Timestamp.today().year
        picked = st.multiselect("Years (start month as in the sidebar)", list(range(proto.FIRST_YEAR, this_year + 1)),
                                default=[2021, 2023], key="years_pick")
        if st.button("Run year comparison", disabled=models is None or not picked or not project):
            pb = st.progress(0.0, "starting ...")
            try:
                import ee

                ee.Initialize(project=project)
                reg = proto.make_region(lat, lon, size_km)
                per_year, scene_med, notes = {}, {}, []
                for n, y in enumerate(sorted(picked)):
                    ym_y = proto.month_sequence(y, month)
                    notes += [f"{y}: {w}" for w in proto.check_period(ym_y)[:2] if "stopped" in w or "Before" in w]
                    proto.fetch_months(ee, reg, ym_y, CACHE, lambda f, t, n=n, y=y: pb.progress((n + f * 0.5) / len(picked), f"{y}: {t}"))
                    st_y = proto.load_store_ym(CACHE, reg.key, ym_y)
                    p_y, _, mem_y, mo_y = proto.run_models(models, st_y, norm, card["window_months"], torch.device(device),
                                                           use_stats=proto.uses_stats(card))
                    p_y = proto.analyse(p_y, mem_y, proto.month_labels([ym_y[i] for i in mo_y]), True)["preds"]
                    per_year[y] = p_y
                    sc = np.asarray(st_y.meta.get("s1_scenes", []), float)
                    scene_med[y] = float(np.nanmedian(sc)) if sc.size and np.isfinite(sc).any() else float("nan")
                    pb.progress((n + 1) / len(picked), f"{y} done")
                summ = proto.yearly_summary(per_year)
                st.session_state["years_res"] = dict(summ=summ, scenes=scene_med, bounds=proto.bounds_latlon(reg),
                                                     notes=sorted(set(notes)), place=f"{lat:.4f}, {lon:.4f}", month=month)
                pb.empty()
            except Exception as e:
                pb.empty()
                st.error(f"Year comparison failed: {e}")
        yr = st.session_state.get("years_res")
        if yr:
            summ, ys = yr["summ"], sorted(yr["summ"]["maps"])
            st.caption(f"{yr['place']}, 12 months from {proto.MONTH_NAMES[yr['month'] - 1]} of each year.")
            for n in yr["notes"]:
                st.warning(n)
            low_y = [y for y in ys if np.isfinite(yr["scenes"][y]) and yr["scenes"][y] < 0.6 * np.nanmax(list(yr["scenes"].values()))]
            if low_y:
                st.warning("Years with clearly fewer scenes per month: " + ", ".join(f"{y} ({yr['scenes'][y]:.0f})" for y in low_y)
                           + ". Differences to other years can come from that alone.")
            shares = summ["shares"]
            st.markdown("**Share of the area per class (%)**")
            st.dataframe(shares.round(2))
            if HAVE_MAP:
                fig_y = go.Figure()
                for name, color in zip(CLASS_NAMES, CLASS_COLORS):
                    fig_y.add_trace(go.Bar(x=[str(y) for y in shares.index], y=shares[name], name=name, marker_color=color))
                fig_y.update_layout(barmode="stack", height=320, yaxis=dict(title="% of area", range=[0, 100]),
                                    margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig_y, width="stretch")
                if len(ys) >= 2:                                   # animated view: press play to watch the shares change year to year
                    long = shares.reset_index().melt(id_vars=shares.index.name or "year", var_name="class", value_name="share")
                    long = long.rename(columns={shares.index.name or "year": "year"})
                    long["year"] = long["year"].astype(str)
                    anim = px.bar(long, x="class", y="share", color="class", animation_frame="year",
                                  color_discrete_map=dict(zip(CLASS_NAMES, CLASS_COLORS)),
                                  category_orders={"class": list(CLASS_NAMES)},
                                  range_y=[0, min(100, float(long["share"].max()) * 1.15 + 1)])
                    anim.update_layout(height=400, showlegend=False, xaxis_title="", yaxis_title="% of area",
                                       title="Class share by year (press play)", margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(anim, width="stretch")
                    st.caption("Press play to step through the years. There is no ground truth for any year; differences can be noise or "
                               "fewer scenes, not real change.")
            if len(ys) >= 2:
                st.markdown("**Change from year to year (percentage points)**")
                d = shares.diff().iloc[1:].round(2)
                d["total change"] = (shares.diff().iloc[1:].abs().sum(axis=1) / 2).round(2)
                d["vs noise"] = np.where(d["total change"] < proto.NOISE_FLOOR["net_first_to_last_pp"], "within noise", "ABOVE noise")
                d.index = [f"{a} to {b}" for a, b in zip(ys[:-1], ys[1:])]
                st.dataframe(d)
                st.caption(f"Noise reference: {proto.NOISE_FLOOR['net_first_to_last_pp']} pp net change between two months on land that "
                           "did not change. Year maps use 12 months, so real noise should be lower, but we did not measure it. "
                           "Years after 2021 are also noisier (fewer scenes) and unchecked.")
                c1, c2 = st.columns(2)
                ya = c1.selectbox("From year", ys, index=0)
                yb = c2.selectbox("To year", ys, index=len(ys) - 1)
                if ya != yb:
                    ma, mb = summ["maps"][ya], summ["maps"][yb]
                    cols = st.columns(3)
                    for col, arr, title in ((cols[0], ma, str(ya)), (cols[1], mb, str(yb)), (cols[2], proto.change_map(ma, mb), f"Changed pixels (class in {yb})")):
                        fig, ax = plt.subplots(figsize=(4, 4))
                        if title.startswith("Changed"):
                            ax.imshow(np.full(arr.shape, 0.92), cmap="gray", vmin=0, vmax=1)
                        ax.imshow(np.ma.masked_where(show(arr) == 0, show(arr)), cmap=CMAP, vmin=0, vmax=6, interpolation="nearest")
                        ax.axis("off"); ax.set_title(title, fontsize=9)
                        col.pyplot(fig); plt.close(fig)
                    st.markdown(f"**Where the area went, {ya} to {yb} (% of the area; rows = {ya}, columns = {yb})**")
                    st.dataframe(proto.transition_matrix(ma, mb).round(2))
                    st.caption("The diagonal is land that kept its class. Off-diagonal cells are change or noise. Large, compact patches "
                               "are more likely real than scattered pixels. There is no ground truth for these years.")
            st.download_button("Download yearly shares (CSV)", shares.round(4).to_csv(), "yearly_shares.csv", "text/csv")

with tab_check:
    ref = res["reference"]
    if ref is None:
        st.info("Reference maps (WorldCover and Esri 2021) are only used when at least 6 of the 12 months are in 2021, "
                "or when a local export includes reference.npz.")
    else:
        labels_dw = store.label[months] if store.label is not None else None
        cmp = proto.comparison_table(preds, ref, labels_dw)
        st.dataframe(cmp.round(3))
        st.caption("**Dynamic World (ceiling)** is the label source the model was trained to imitate: the model cannot be expected to "
                   "match the reference maps better than it does. **Baseline** paints the whole area with the reference's most "
                   "frequent class. **share_of_area_%** is the part of the area where the two references agree: the 'agree' scores "
                   "cover only that easy part.")
        with st.expander("Confusion matrix against Esri (% of each Esri class predicted as each class)"):
            st.dataframe(proto.confusion_vs(preds, ref["esri"]).round(1))
        cols = st.columns(4)
        annual = np.stack([(preds == k).sum(0) for k in range(6)]).argmax(0)
        for c, (name, arr) in zip(cols, [("Model, most frequent class", annual), ("WorldCover 2021", ref["wc"]),
                                         ("Esri 2021", ref["esri"]), ("Both agree", ref["agree"])]):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(show(arr), cmap=CMAP, vmin=0, vmax=6, interpolation="nearest"); ax.axis("off"); ax.set_title(name, fontsize=9)
            c.pyplot(fig); plt.close(fig)
        st.caption("Both reference maps are annual and use their own class definitions, so part of every disagreement is definitional.")

with tab_info:
    st.markdown(f"""
**Method.** Sentinel-1 IW, VV and VH, from Earth Engine (calibrated, terrain-corrected, in dB). For every month: temporal median of the
dB scenes of the region's dominant orbit direction. A U-Net (ResNet-34) reads a 3-month window and predicts one map per month. The
{res['k']} ensemble members are trained with different seeds; the maps are the mean of their probabilities.

**Change.** Class shares use only pixels valid in all months. Change is the difference of shares in percentage points.
Smoothing replaces each month's class with the majority over the month and its two neighbours.

**What to trust.**
- The monthly maps have **no ground truth**: labels are annual, and Dynamic World, our training label, is itself a model.
- On land that did not change, the model still reports about {proto.NOISE_FLOOR['monthly_churn_raw_pp']} pp total churn per month
  ({proto.NOISE_FLOOR['monthly_churn_smoothed_pp']} pp with smoothing). Smaller changes are noise.
- Grass/shrub is weak (IoU about 0.15 against Dynamic World). Radar cannot reliably separate grass, shrub and crops.
- Weak in arid shrubland and in landscapes unlike the training regions (mostly European cities plus a few rural tiles).
- The area check compares mean confidence and seed disagreement with a few held-out test regions. It is a coarse flag, not a
  probability of being wrong, and it only exists when `ood_reference.json` is in the weights folder.
- No radiometric terrain flattening: hilly areas carry slope-dependent errors.
- Sentinel-1B stopped in December 2021, so later years have about half as many scenes per month. The model was trained on 2021.
- Any place and any month from 2015 works if Earth Engine has radar there, but only 2021 can be checked against independent maps.
- Rural areas are covered by 15 test and train tiles; the rural score is lower than the city score (see report). Dry shrubland, bare mountains and snow are the weakest.
""")
