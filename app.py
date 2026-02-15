import time
from pathlib import Path
from datetime import datetime, timedelta
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Page setup + CSS tightening
# -----------------------------
st.set_page_config(page_title="Inventory Replenishment Engine", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.0rem;}
      .stMetric {padding: 6px 10px;}
      div[data-testid="stVerticalBlockBorderWrapper"] {padding: 10px;}
      .tight-card {padding: 12px 14px; border-radius: 14px; border: 1px solid rgba(49,51,63,0.15);}
      .muted {opacity: 0.75;}
      .sku-pill {display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid rgba(49,51,63,0.18);}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📦 Inventory Replenishment Engine")
st.caption("Left: MP4 feed + SKU selector | Right: Orders + Inventory + Replenishment telemetry + next replenishment prediction + GenBI query (demo).")


# -----------------------------
# Video loading
# -----------------------------
VIDEO_DIR = Path("videos")
fallback_uploaded = Path("/mnt/data/3817769649-preview.mp4")  # user-uploaded file

video_files = []
if VIDEO_DIR.exists():
    video_files = sorted([p for p in VIDEO_DIR.glob("*.mp4")])

if not video_files and fallback_uploaded.exists():
    video_files = [fallback_uploaded]

if not video_files:
    st.error("❌ No MP4 found. Create /videos and add MP4 files, or provide a valid MP4 path.")
    st.stop()


# -----------------------------
# SKU catalog (demo)
# -----------------------------
# You can tune these to match any client narrative (fast-movers vs slow-movers, bulky vs small, etc.)
SKU_CATALOG = {
    "SKU-A (Fast mover)": {"units_per_order": 10.0, "start_inventory": 5200, "base_orders": 44, "batch_base": 1100},
    "SKU-B (Medium mover)": {"units_per_order": 8.0, "start_inventory": 4200, "base_orders": 36, "batch_base": 950},
    "SKU-C (Slow mover)": {"units_per_order": 6.0, "start_inventory": 3200, "base_orders": 28, "batch_base": 800},
    "SKU-D (Bulky / constrained)": {"units_per_order": 14.0, "start_inventory": 2400, "base_orders": 22, "batch_base": 650},
}


# -----------------------------
# Sidebar controls (global)
# -----------------------------
with st.sidebar:
    st.header("Controls")
    autoplay = st.toggle("Autoplay telemetry", value=True)
    tick_ms = st.slider("Refresh speed (ms)", 150, 1500, 350, 10)

    st.divider()
    st.subheader("Video selection")

    if "video_idx" not in st.session_state:
        st.session_state.video_idx = 0

    chosen = st.selectbox(
        "Pick a video",
        options=list(range(len(video_files))),
        format_func=lambda i: f"{i+1}. {video_files[i].name}",
        index=st.session_state.video_idx
    )
    st.session_state.video_idx = chosen

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("⏮ Prev"):
            st.session_state.video_idx = (st.session_state.video_idx - 1) % len(video_files)
    with colB:
        if st.button("▶ Next"):
            st.session_state.video_idx = (st.session_state.video_idx + 1) % len(video_files)
    with colC:
        if st.button("🔁 Reset"):
            st.session_state.video_idx = 0

    st.divider()
    st.subheader("Warehouse realism (demo)")
    demand_spike = st.slider("Demand spike", 0.0, 2.5, 0.8, 0.1)
    noise = st.slider("Signal noise", 0.0, 3.0, 0.7, 0.1)
    lead_time = st.slider("Supplier lead time (hrs)", 4, 72, 24, 1)
    service_level = st.slider("Service level (safety)", 0.0, 3.0, 1.2, 0.1)
    reorder_policy = st.selectbox("Policy", ["(s,S) reorder-up-to", "Reorder point"], index=0)


# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.25, 1.0], gap="large")
current_video = video_files[st.session_state.video_idx]


# -----------------------------
# Telemetry generation (SKU-aware demo)
# -----------------------------
def make_warehouse_series(
    seed: int,
    sku_cfg: dict,
    n: int = 320,
    noise: float = 0.7,
    demand_spike: float = 0.8
):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    base_orders = float(sku_cfg["base_orders"])
    units_per_order = float(sku_cfg["units_per_order"])
    start_inventory = float(sku_cfg["start_inventory"])
    batch_base = float(sku_cfg["batch_base"])

    # Orders processed / tick
    base = base_orders + 8 * np.sin(2 * np.pi * t / 95) + 5 * np.sin(2 * np.pi * t / 29)
    spike = demand_spike * 10 * np.maximum(0, np.sin(2 * np.pi * (t - 25) / 120))
    orders_rate = base + spike + rng.normal(0, noise * 2.2, size=n)
    orders_rate = np.clip(orders_rate, max(6, base_orders * 0.35), base_orders * 2.2)

    # Replenishment inflow: periodic receipts
    repl_inflow = np.zeros(n)
    for k in range(50, n, 60):
        batch = batch_base + (0.35 * batch_base * rng.random())
        width = rng.integers(2, 6)
        repl_inflow[k:k + width] += batch / width

    repl_inflow += rng.normal(0, noise * 3.5, size=n)
    repl_inflow = np.clip(repl_inflow, 0, None)

    # Inventory dynamics
    inventory = np.zeros(n)
    backlog = np.zeros(n)
    inventory[0] = start_inventory + 0.15 * start_inventory * rng.random()

    for i in range(1, n):
        consume = orders_rate[i] * units_per_order
        inv = inventory[i - 1] - consume + repl_inflow[i]

        if inv < 0:
            backlog[i] = backlog[i - 1] + abs(inv)
            inv = 0
        else:
            backlog[i] = max(0, backlog[i - 1] - 0.25 * repl_inflow[i])

        inventory[i] = inv

    return t, orders_rate, inventory, repl_inflow, backlog


def compute_replenishment_prediction(orders_rate, inventory, lead_time_hrs: int, service_level: float, units_per_order: float):
    # Demand estimation window
    w = min(70, len(orders_rate))
    mu_orders = float(np.mean(orders_rate[-w:]))
    sigma_orders = float(np.std(orders_rate[-w:]))

    # Convert orders/tick to units/hr
    tick_minutes = 6.0
    orders_per_hr = mu_orders * (60.0 / tick_minutes)
    demand_units_per_hr = orders_per_hr * float(units_per_order)

    ltd = demand_units_per_hr * lead_time_hrs
    safety = service_level * sigma_orders * (60.0 / tick_minutes) * float(units_per_order) * np.sqrt(max(lead_time_hrs, 1))
    rop = ltd + safety

    inv_now = float(inventory[-1])
    burn = max(demand_units_per_hr, 1e-6)

    hrs_to_rop = (inv_now - rop) / burn
    hrs_to_rop = float(np.clip(hrs_to_rop, -72, 240))

    review_horizon = 24.0
    up_to = rop + demand_units_per_hr * review_horizon
    rec_qty = max(0.0, up_to - inv_now)

    confidence = float(np.clip(70 + (service_level * 6) - (abs(sigma_orders) * 0.6), 45, 92))

    return {
        "demand_units_per_hr": demand_units_per_hr,
        "rop": float(rop),
        "hrs_to_rop": hrs_to_rop,
        "up_to": float(up_to),
        "rec_qty": float(rec_qty),
        "confidence": confidence,
        "tick_minutes": tick_minutes
    }


def status_from_inventory(inv_now: float, rop: float):
    if inv_now <= rop:
        return "ALERT"
    if inv_now <= rop * 1.25:
        return "WATCH"
    return "NORMAL"


# -----------------------------
# SKU selection (LEFT pane, left-most position)
# -----------------------------
with left:
    st.subheader("🧾 SKU Selector")
    sku_names = list(SKU_CATALOG.keys())

    if "sku" not in st.session_state:
        st.session_state.sku = sku_names[0]

    sku_choice = st.selectbox("Select SKU", sku_names, index=sku_names.index(st.session_state.sku))
    st.session_state.sku = sku_choice
    sku_cfg = SKU_CATALOG[sku_choice]

    st.markdown(
        f"<span class='sku-pill'><b>Active:</b> {sku_choice}</span>",
        unsafe_allow_html=True
    )

    # Small SKU card (keeps it “client ready”)
    st.markdown('<div class="tight-card">', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Units / Order", f"{sku_cfg['units_per_order']:.0f}")
    s2.metric("Start Inv (demo)", f"{sku_cfg['start_inventory']:,.0f}")
    s3.metric("Base Orders/tick", f"{sku_cfg['base_orders']:.0f}")
    s4.metric("Receipt Batch (demo)", f"{sku_cfg['batch_base']:,.0f}")
    st.markdown(f"<span class='muted'>Tip: change SKU to show fast-mover vs slow-mover behavior instantly.</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Telemetry generation (seeded by video + SKU)
# -----------------------------
seed = abs(hash((current_video.name, st.session_state.sku))) % (10**6)
t, orders_rate, inventory, repl_inflow, backlog = make_warehouse_series(
    seed,
    sku_cfg=sku_cfg,
    noise=noise,
    demand_spike=demand_spike
)


# -----------------------------
# Cursor + autoplay
# -----------------------------
if "cursor" not in st.session_state:
    st.session_state.cursor = 0

if st.session_state.get("last_key") != (current_video.name, st.session_state.sku):
    st.session_state.last_key = (current_video.name, st.session_state.sku)
    st.session_state.cursor = 0

cursor = int(np.clip(st.session_state.cursor, 0, len(t) - 1))
st.session_state.cursor = cursor

if autoplay:
    st.session_state.cursor = min(st.session_state.cursor + 2, len(t) - 1)
    time.sleep(tick_ms / 1000.0)
    st.rerun()


# Slice up to cursor
tt = t[:cursor + 1]
oo = orders_rate[:cursor + 1]
inv = inventory[:cursor + 1]
rep = repl_inflow[:cursor + 1]
bl = backlog[:cursor + 1]

pred = compute_replenishment_prediction(
    oo, inv,
    lead_time_hrs=lead_time,
    service_level=service_level,
    units_per_order=sku_cfg["units_per_order"]
)

inv_now = float(inv[-1])
rop = float(pred["rop"])
hrs_to_rop = float(pred["hrs_to_rop"])
status = status_from_inventory(inv_now, rop)

eta_dt = datetime.now() + timedelta(hours=max(0.0, hrs_to_rop))
eta_str = eta_dt.strftime("%d %b %Y, %I:%M %p")

policy_label = "Reorder-up-to" if reorder_policy.startswith("(s,S)") else "Reorder point"
rec_qty = pred["rec_qty"] if policy_label == "Reorder-up-to" else max(0.0, rop - inv_now)


# -----------------------------
# GENBI: rule-based query engine (offline, SKU-aware)
# -----------------------------
def genbi_answer(q: str, cursor_now: int):
    ql = q.strip().lower()
    if not ql:
        return None, None

    sku_label = st.session_state.sku

    if ("inventory" in ql or "stock" in ql) and ("current" in ql or "now" in ql):
        return f"[{sku_label}] Current on-hand inventory is **{inv_now:,.0f} units**. Status is **{status}**.", None

    if "reorder" in ql and ("point" in ql or "rop" in ql):
        return f"[{sku_label}] Reorder point (ROP) is **{rop:,.0f} units** (lead time **{lead_time}h**, safety **{service_level}x**).", None

    if ("when" in ql or "eta" in ql or "next" in ql) and ("replen" in ql or "reorder" in ql):
        if hrs_to_rop <= 0:
            return f"[{sku_label}] Inventory is at/under ROP **now** → replenish immediately (confidence **{pred['confidence']:.0f}%**, demo).", None
        return f"[{sku_label}] Time to ROP is **{hrs_to_rop:.1f} hrs** → trigger around **{eta_str}**.", None

    if ("how much" in ql or "quantity" in ql or "order" in ql) and ("replen" in ql or "po" in ql):
        return f"[{sku_label}] Recommended replenishment quantity is **{rec_qty:,.0f} units** using **{policy_label}** policy (confidence **{pred['confidence']:.0f}%**, demo).", None

    if "backlog" in ql:
        return f"[{sku_label}] Current backlog proxy is **{float(bl[-1]):,.0f} units** (demo).", None

    # trend plots
    m = re.search(r"last\s+(\d+)\s+ticks", ql)
    n = int(m.group(1)) if m else 90
    n = int(np.clip(n, 20, 240))
    s = max(0, cursor_now - n)

    def line_fig(x, y, name, ytitle):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Tick", yaxis_title=ytitle)
        return fig

    if "orders" in ql:
        fig = line_fig(t[s:cursor_now + 1], orders_rate[s:cursor_now + 1], "Orders processed / tick", "Orders")
        return f"[{sku_label}] Showing last **{cursor_now - s}** ticks of **Orders Processed**.", fig

    if "inventory" in ql or "stock" in ql:
        fig = line_fig(t[s:cursor_now + 1], inventory[s:cursor_now + 1], "Inventory on-hand", "Units")
        fig.add_hline(y=rop, line_width=1)
        return f"[{sku_label}] Showing last **{cursor_now - s}** ticks of **Inventory** (includes ROP line).", fig

    if "replen" in ql or "inflow" in ql:
        fig = line_fig(t[s:cursor_now + 1], repl_inflow[s:cursor_now + 1], "Replenishment inflow", "Units")
        return f"[{sku_label}] Showing last **{cursor_now - s}** ticks of **Replenishment Inflow**.", fig

    return f"[{sku_label}] I can answer: current inventory, ROP, next replenishment ETA, recommended replenishment quantity, backlog, and show last N ticks for orders/inventory/replenishment.", None


# -----------------------------
# LEFT: Video + summary + quick query
# -----------------------------
with left:
    st.subheader("🎥 Live Warehouse Feed")
    st.write(f"**Now playing:** {current_video.name}")
    st.video(str(current_video))

    st.markdown('<div class="tight-card">', unsafe_allow_html=True)
    st.markdown(f"### 📌 Replenishment Executive Summary — {st.session_state.sku}")
    a, b, c = st.columns(3)
    a.metric("State", status)
    b.metric("On-hand Inventory", f"{inv_now:,.0f} units")
    c.metric("Reorder Point (ROP)", f"{rop:,.0f} units")

    d, e, f = st.columns(3)
    d.metric("Time to ROP", "NOW" if hrs_to_rop <= 0 else f"{hrs_to_rop:.1f} hrs")
    e.metric("Next Replenishment ETA", eta_str if hrs_to_rop > 0 else "Immediate trigger")
    f.metric("Confidence (demo)", f"{pred['confidence']:.0f}%")

    st.markdown(f"<span class='muted'>Policy: {policy_label} | Lead time: {lead_time}h | Safety: {service_level}x</span>", unsafe_allow_html=True)

    st.markdown("#### 🔎 GenBI Quick Query")
    quick_q = st.text_input(
        "Ask about inventory, ROP, ETA, trends…",
        placeholder="e.g., 'when is next replenishment' or 'show last 120 ticks inventory trend'"
    )
    st.markdown("</div>", unsafe_allow_html=True)

quick_answer, quick_fig = genbi_answer(quick_q, cursor) if quick_q else (None, None)
if quick_q and quick_answer:
    with left:
        st.info(quick_answer)
        if quick_fig is not None:
            st.plotly_chart(quick_fig, use_container_width=True)


# -----------------------------
# RIGHT: KPIs + Tabs
# -----------------------------
with right:
    st.subheader(f"📟 Warehouse Telemetry Dashboard — {st.session_state.sku}")

    r1, r2, r3 = st.columns(3)
    r1.metric("Orders / tick", f"{oo[-1]:.0f}")
    r2.metric("Inventory (units)", f"{inv_now:,.0f}")
    r3.metric("Replen Inflow (units)", f"{rep[-1]:.0f}")

    r4, r5, r6 = st.columns(3)
    r4.metric("Backlog (proxy)", f"{float(bl[-1]):,.0f}")
    r5.metric("Demand (units/hr)", f"{pred['demand_units_per_hr']:,.0f}")
    r6.metric("State", status)

    tabs = st.tabs(["📈 Live Timeline", "🔮 Replenishment Prediction", "💬 GenBI Query"])

    with tabs[0]:
        window = 150
        start = max(0, cursor - window)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t[start:cursor+1], y=orders_rate[start:cursor+1], mode="lines", name="Orders / tick"))
        fig.add_trace(go.Scatter(x=t[start:cursor+1], y=inventory[start:cursor+1], mode="lines", name="Inventory (units)", yaxis="y2"))
        fig.add_trace(go.Scatter(x=t[start:cursor+1], y=repl_inflow[start:cursor+1], mode="lines", name="Replen Inflow (units)", yaxis="y3"))

        fig.add_hline(y=rop, line_width=1)
        fig.add_vline(x=t[cursor], line_width=2)

        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Telemetry Tick",
            yaxis=dict(title="Orders"),
            yaxis2=dict(title="Inventory", overlaying="y", side="right"),
            yaxis3=dict(title="Replen Inflow", overlaying="y", side="right", position=0.97, showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        colX, colY = st.columns([1, 2])
        with colX:
            if st.button("⏩ Advance telemetry"):
                st.session_state.cursor = min(st.session_state.cursor + 10, len(t) - 1)
                st.rerun()
        with colY:
            st.progress(int((cursor / (len(t) - 1)) * 100))

    with tabs[1]:
        c1, c2 = st.columns(2)

        with c1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=inv_now,
                number={"suffix": " units"},
                gauge={
                    "axis": {"range": [0, max(rop * 2.0, inv_now * 1.2, 1)]},
                    "bar": {"thickness": 0.35},
                    "threshold": {"line": {"width": 3}, "value": rop},
                },
                title={"text": "Inventory vs Reorder Point (ROP)"}
            ))
            gauge.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(gauge, use_container_width=True)

        with c2:
            w = 150
            s = max(0, cursor - w)
            dist = inventory[s:cursor+1] - rop
            risk_like = 100 * np.clip(1 - (dist / (rop + 1e-6)), 0, 1)

            risk_fig = go.Figure()
            risk_fig.add_trace(go.Scatter(x=t[s:cursor+1], y=risk_like, mode="lines"))
            risk_fig.add_hline(y=40, line_width=1)
            risk_fig.add_hline(y=70, line_width=1)
            risk_fig.add_vline(x=t[cursor], line_width=2)
            risk_fig.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Tick",
                yaxis_title="Risk (0-100)",
                showlegend=False
            )
            st.plotly_chart(risk_fig, use_container_width=True)

        st.markdown("### 📅 Predicted replenishment trigger")
        m1, m2, m3 = st.columns(3)
        m1.metric("Trigger (ROP breach)", "Immediate" if hrs_to_rop <= 0 else eta_str)
        m2.metric("Recommended Quantity", f"{rec_qty:,.0f} units")
        m3.metric("Confidence (demo)", f"{pred['confidence']:.0f}%")

        if status == "ALERT":
            st.error("🚨 Recommendation: Replenish immediately. Risk of stockout / backlog escalation is high.")
        elif status == "WATCH":
            st.warning("⚠️ Recommendation: Prepare replenishment. Inventory is approaching ROP.")
        else:
            st.success("✅ Recommendation: Normal operations. No near-term replenishment trigger predicted.")

    with tabs[2]:
        st.markdown("### 💬 GenBI Query")
        st.caption("Plain English (rule-based/offline). Upgrade to LLM later.")

        q = st.text_input("Your question", placeholder="e.g., What is current inventory and when is next replenishment due?")
        ans, fig = genbi_answer(q, cursor) if q else (None, None)
        if ans:
            st.info(ans)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
