"""
generate_dashboard.py  —  FinContagion-GNN  (Research Dashboard v2)
=====================================================================
4-tab interactive dashboard designed to tell the story of financial
contagion to a professor / conference audience.

Tab 1 — THE NETWORK          How are firms connected? Who infects whom?
Tab 2 — CONTAGION SPREAD     How does a crash in one sector ripple to others?
Tab 3 — PORTFOLIO DEFENCE    How does the GNN protect the portfolio?
Tab 4 — MODEL PERFORMANCE    How well does the GNN actually predict crashes?
"""

import os, json, textwrap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE  (dark, publication-quality)
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#060B18"          # page background
CARD    = "#0E1626"          # card / panel
CARD2   = "#131D30"          # slightly lighter card
BORDER  = "#1E2D45"
MUTED   = "#8899AA"
TEXT    = "#DDE6F0"
GREEN   = "#00D4A0"
RED     = "#FF4D6A"
AMBER   = "#FFB347"
BLUE    = "#4EA8DE"
PURPLE  = "#A78BFA"
TEAL    = "#2DD4BF"

SECTOR_COLORS = {
    "Tech":       "#4EA8DE",
    "Finance":    "#00D4A0",
    "Energy":     "#FFB347",
    "Healthcare": "#A78BFA",
    "Consumer":   "#FF6B9D",
}

CRISES = {
    "COVID-19 Crash  (Feb – Apr 2020)":   ("2019-10-01", "2020-06-30", "2020-02-19", "2020-03-23"),
    "Fed Rate Shock  (Jan – Dec 2022)":   ("2021-10-01", "2022-12-31", "2022-01-03", "2022-10-12"),
    "SVB Bank Crisis  (Mar 2023)":        ("2022-12-01", "2023-06-30", "2023-03-08", "2023-03-24"),
    "Tech Selloff    (Q1 2022)":          ("2021-10-01", "2022-05-31", "2022-01-03", "2022-05-20"),
}

STOCK_UNIVERSE = {
    "Tech":       ["AAPL","MSFT","NVDA","AVGO","ORCL","ADBE","CRM","AMD","CSCO","INTC"],
    "Finance":    ["JPM","BAC","WFC","MS","GS","C","BLK","SPGI","AXP","V"],
    "Energy":     ["XOM","CVX","COP","EOG","SLB","MPC","HAL","VLO","OXY","PSX"],
    "Healthcare": ["UNH","JNJ","LLY","MRK","ABBV","PFE","TMO","DHR","ABT","BMY"],
    "Consumer":   ["AMZN","WMT","PG","HD","COST","KO","PEP","MCD","NKE","SBUX"],
}
SECTOR_MAP  = {t: s for s, ts in STOCK_UNIVERSE.items() for t in ts}
ALL_TICKERS = [t for v in STOCK_UNIVERSE.values() for t in v]
SECTORS     = list(STOCK_UNIVERSE.keys())

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _load(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None

gnn  = _load("gnn_predictions.csv", index_col=0)
bt   = _load("backtest_results.csv", index_col=0, parse_dates=True)
bm   = _load("backtest_metrics.csv", index_col=0)
mlog = _load("monthly_rebalance_log.csv", parse_dates=["date"])
dr   = _load("daily_returns.csv", index_col=0, parse_dates=True)
A_df = _load("edge_matrix_A.csv", index_col=0)

# add sector col to gnn
if gnn is not None:
    gnn["sector"] = [SECTOR_MAP.get(t, "Unknown") for t in gnn.index]
    risk_col = "mean_contagion" if "mean_contagion" in gnn.columns else "Contagion_Risk"
else:
    risk_col = "mean_contagion"

# ─────────────────────────────────────────────────────────────────────────────
# SHARED LAYOUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def dark(fig, height=None, margin=None, legend=True):
    m = margin or dict(l=50, r=30, t=50, b=40)
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=m,
        legend=dict(bgcolor=CARD2, bordercolor=BORDER,
                    font=dict(color=TEXT)) if legend else dict(visible=False),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED))
    if height:
        fig.update_layout(height=height)
    return fig


def card(children, padding="24px 28px", style=None):
    base = {"backgroundColor": CARD, "border": f"1px solid {BORDER}",
            "borderRadius": "12px", "padding": padding}
    if style:
        base.update(style)
    return html.Div(children, style=base)


def section_title(text, sub=None):
    return html.Div([
        html.H5(text, style={"color": TEXT, "fontWeight": "700",
                              "marginBottom": "4px", "letterSpacing": "-0.01em"}),
        html.P(sub, style={"color": MUTED, "fontSize": "13px",
                            "marginBottom": "0"}) if sub else None,
    ], style={"marginBottom": "16px"})


def kpi(label, value, color=GREEN, sub=""):
    return html.Div([
        html.P(label, style={"color": MUTED, "fontSize": "11px",
                              "textTransform": "uppercase", "letterSpacing": "0.06em",
                              "marginBottom": "4px"}),
        html.H4(value, style={"color": color, "fontFamily": "monospace",
                               "fontWeight": "700", "marginBottom": "2px",
                               "fontSize": "22px"}),
        html.P(sub, style={"color": MUTED, "fontSize": "11px", "marginBottom": "0"}),
    ], style={"backgroundColor": CARD2, "border": f"1px solid {BORDER}",
               "borderRadius": "10px", "padding": "16px 20px"})


def explainer(text, icon="ℹ️"):
    return html.Div([
        html.Span(icon + " ", style={"fontSize": "14px"}),
        html.Span(text, style={"color": MUTED, "fontSize": "12px",
                                "lineHeight": "1.6"}),
    ], style={"backgroundColor": "#0A1525", "border": f"1px solid {BORDER}",
               "borderRadius": "8px", "padding": "12px 16px", "marginBottom": "16px"})


def tab_style(color):
    return {"color": MUTED, "borderBottom": f"2px solid transparent",
            "padding": "12px 20px", "fontSize": "13px", "fontWeight": "600",
            "cursor": "pointer"}


def active_tab_style(color):
    return {"color": color, "borderBottom": f"2px solid {color}",
            "padding": "12px 20px", "fontSize": "13px", "fontWeight": "600",
            "backgroundColor": "transparent"}


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Force-directed network ────────────────────────────────────────────────
def build_network(color_mode="risk", selected_sector=None):
    if A_df is None:
        return go.Figure()

    G = nx.from_pandas_adjacency(A_df.fillna(0))
    pos = nx.spring_layout(G, k=2.2, seed=42, iterations=120)

    # filter by sector if requested
    active_tickers = ALL_TICKERS if not selected_sector else STOCK_UNIVERSE.get(selected_sector, ALL_TICKERS)

    edge_x, edge_y, edge_w = [], [], []
    for u, v, d in G.edges(data=True):
        if u not in active_tickers and v not in active_tickers:
            continue
        xu, yu = pos[u]; xv, yv = pos[v]
        edge_x += [xu, xv, None]; edge_y += [yu, yv, None]
        edge_w.append(d.get("weight", 0.3))

    avg_w = np.mean(edge_w) if edge_w else 0.5
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.8, color=f"rgba(78,168,222,0.25)"),
        hoverinfo="none", showlegend=False,
    )

    traces = [edge_trace]
    for sector, tickers in STOCK_UNIVERSE.items():
        xs, ys, texts, sizes, colors, hovers = [], [], [], [], [], []
        for t in tickers:
            if t not in pos:
                continue
            x, y = pos[t]
            risk = float(gnn.loc[t, risk_col]) if (gnn is not None and t in gnn.index) else 0.5
            xs.append(x); ys.append(y); texts.append(t)
            sizes.append(18 + risk * 28)

            if color_mode == "risk":
                r = int(255 * risk)
                g2 = int(212 * (1 - risk))
                colors.append(f"rgb({r},{g2},80)")
            else:
                alpha = "ff" if (not selected_sector or sector == selected_sector) else "44"
                colors.append(SECTOR_COLORS.get(sector, "#888"))

            degree = G.degree(t)
            hovers.append(
                f"<b style='font-size:14px'>{t}</b><br>"
                f"<span style='color:{SECTOR_COLORS.get(sector,'#aaa')}'>{sector}</span><br><br>"
                f"🔴 Contagion Risk: <b>{risk:.1%}</b><br>"
                f"🔗 Network Connections: <b>{degree}</b><br>"
                f"📊 Crash Rate (historical): <b>{float(gnn.loc[t,'crash_rate']):.1%}</b>"
                if gnn is not None and t in gnn.index else f"<b>{t}</b>"
            )

        opacity = 1.0 if (not selected_sector or sector == selected_sector) else 0.2
        traces.append(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=sizes, color=colors,
                         line=dict(color=BORDER, width=1.5), opacity=opacity),
            text=texts,
            textposition="middle center",
            textfont=dict(size=7.5, color="white", family="monospace"),
            hovertext=hovers, hoverinfo="text",
            name=sector,
            showlegend=(color_mode == "sector"),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0), height=500,
        legend=dict(bgcolor=CARD2, bordercolor=BORDER, font=dict(color=TEXT),
                    orientation="v", x=1.01, y=0.5),
        hoverlabel=dict(bgcolor=CARD2, font_size=12,
                         font_family="Inter", font_color=TEXT, bordercolor=BORDER),
    )
    return fig


# ── 2. Sector-to-sector contagion chord / heatmap ────────────────────────────
def build_sector_flow(crisis_key=None):
    """
    Shows how strongly each sector pair co-moved during a given crisis.
    The diagonal is intra-sector correlation, off-diagonal is cross-sector.
    Darker = more contagion transmission.
    """
    if dr is None:
        return go.Figure()

    if crisis_key and crisis_key in CRISES:
        s, e, _, _ = CRISES[crisis_key]
        period = dr.loc[s:e]
        title_suffix = f"during {crisis_key.split('(')[0].strip()}"
    else:
        period = dr
        title_suffix = "full period (2015–2024)"

    corr = period.corr()
    sec_corr = pd.DataFrame(index=SECTORS, columns=SECTORS, dtype=float)
    for s1 in SECTORS:
        for s2 in SECTORS:
            t1s = [t for t in STOCK_UNIVERSE[s1] if t in corr.columns]
            t2s = [t for t in STOCK_UNIVERSE[s2] if t in corr.columns]
            vals = [corr.loc[a, b] for a in t1s for b in t2s if a != b]
            sec_corr.loc[s1, s2] = np.mean(vals) if vals else 0.0

    z = sec_corr.values.astype(float)
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=SECTORS, y=SECTORS,
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#0E1626"], [0.4, "#1a3a6b"],
                    [0.7, "#2563eb"], [1.0, "#ef4444"]],
        zmin=0, zmax=1,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>Avg Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color=TEXT)),
            tickfont=dict(color=MUTED), bgcolor=CARD, bordercolor=BORDER,
        ),
    ))
    dark(fig, height=380, margin=dict(l=80, r=20, t=40, b=60))
    fig.update_layout(
        title=dict(text=f"Cross-Sector Contagion Matrix — {title_suffix}",
                    font=dict(color=TEXT, size=13)),
        xaxis=dict(side="bottom", tickfont=dict(color=TEXT, size=11),
                    gridcolor=BORDER),
        yaxis=dict(tickfont=dict(color=TEXT, size=11), gridcolor=BORDER),
    )
    return fig


# ── 3. Contagion wave animation — price waterfall ────────────────────────────
def build_contagion_wave(crisis_key):
    if dr is None or crisis_key not in CRISES:
        return go.Figure()

    full_s, full_e, crash_s, crash_e = CRISES[crisis_key]
    period = dr.loc[full_s:full_e]
    cum = (1 + period).cumprod()

    # colour every line by sector
    fig = go.Figure()

    # shade crash window
    fig.add_vrect(
        x0=crash_s, x1=crash_e,
        fillcolor="rgba(255,77,106,0.10)", line_width=0,
        annotation_text="⚡ Crash Window",
        annotation_font=dict(color=RED, size=11),
        annotation_position="top left",
    )

    for sector, tickers in STOCK_UNIVERSE.items():
        clr = SECTOR_COLORS[sector]
        ts_in = [t for t in tickers if t in cum.columns]
        # sector average line — thick
        sec_avg = cum[ts_in].mean(axis=1)
        fig.add_trace(go.Scatter(
            x=sec_avg.index, y=(sec_avg - 1) * 100,
            name=sector, line=dict(color=clr, width=2.5),
            hovertemplate=f"<b>{sector}</b><br>%{{x|%b %d %Y}}<br>Cum return: %{{y:.1f}}%<extra></extra>",
            legendgroup=sector,
        ))
        # individual firm lines — thin, faded
        for t in ts_in:
            fig.add_trace(go.Scatter(
                x=cum.index, y=(cum[t] - 1) * 100,
                name=t, line=dict(color=clr, width=0.6),
                opacity=0.25,
                hovertemplate=f"<b>{t}</b>  ({sector})<br>%{{x|%b %d %Y}}<br>%{{y:.1f}}%<extra></extra>",
                legendgroup=sector, showlegend=False,
            ))

    dark(fig, height=400, margin=dict(l=60, r=20, t=20, b=50))
    fig.update_layout(
        yaxis_title="Cumulative Return (%)", yaxis_ticksuffix="%",
        legend=dict(bgcolor=CARD2, bordercolor=BORDER, font=dict(color=TEXT),
                    orientation="h", y=-0.18, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig


# ── 4. Sector drawdown bar race  (peak-to-trough per sector) ─────────────────
def build_sector_drawdown(crisis_key):
    if dr is None or crisis_key not in CRISES:
        return go.Figure()

    _, _, crash_s, crash_e = CRISES[crisis_key]
    window = dr.loc[crash_s:crash_e]
    cum    = (1 + window).cumprod()
    dd     = (cum / cum.cummax() - 1).min() * 100   # worst DD per ticker

    rows = []
    for sector, tickers in STOCK_UNIVERSE.items():
        for t in tickers:
            if t in dd.index:
                rows.append({"Sector": sector, "Ticker": t, "MaxDD": float(dd[t]),
                              "Risk": float(gnn.loc[t, risk_col]) if gnn is not None and t in gnn.index else 0.5})
    df = pd.DataFrame(rows).sort_values("MaxDD")

    fig = go.Figure()
    for sector in SECTORS:
        sub = df[df["Sector"] == sector]
        fig.add_trace(go.Bar(
            y=sub["Ticker"], x=sub["MaxDD"],
            name=sector, orientation="h",
            marker=dict(color=SECTOR_COLORS[sector], opacity=0.85,
                         line=dict(color=CARD, width=0.5)),
            hovertemplate="<b>%{y}</b><br>Max Drawdown: %{x:.1f}%<extra></extra>",
        ))

    dark(fig, height=420, margin=dict(l=70, r=30, t=20, b=40))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Max Drawdown (%)", xaxis_ticksuffix="%",
        yaxis=dict(tickfont=dict(size=9, color=TEXT)),
        legend=dict(bgcolor=CARD2, bordercolor=BORDER, font=dict(color=TEXT),
                    orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    return fig


# ── 5. GNN contagion score — ranked bar ──────────────────────────────────────
def build_risk_bar():
    if gnn is None:
        return go.Figure()
    df = gnn[[risk_col, "sector"]].copy().rename(columns={risk_col: "risk"})
    df = df.sort_values("risk", ascending=True)

    colors = [SECTOR_COLORS.get(df.loc[t, "sector"], BLUE) for t in df.index]
    thresh_line = 0.5

    fig = go.Figure(go.Bar(
        y=df.index, x=df["risk"],
        orientation="h",
        marker=dict(color=colors, line=dict(color=CARD, width=0.4)),
        text=[f"{v:.0%}" for v in df["risk"]],
        textposition="outside",
        textfont=dict(color=MUTED, size=9),
        hovertemplate="<b>%{y}</b><br>Contagion Risk: %{x:.1%}<extra></extra>",
    ))
    fig.add_vline(x=thresh_line, line_dash="dash", line_color=RED,
                   line_width=1, annotation_text="Risk Threshold",
                   annotation_font_color=RED, annotation_font_size=10)
    dark(fig, height=500, margin=dict(l=70, r=60, t=20, b=40), legend=False)
    fig.update_layout(
        xaxis=dict(tickformat=".0%", range=[0, 0.75], title="GNN Contagion Probability"),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


# ── 6. Sector risk donut ─────────────────────────────────────────────────────
def build_sector_donut():
    labels, vals, colors_list = [], [], []
    for s in SECTORS:
        ts = [t for t in STOCK_UNIVERSE[s] if gnn is not None and t in gnn.index]
        if not ts:
            continue
        avg = np.mean([float(gnn.loc[t, risk_col]) for t in ts])
        labels.append(s); vals.append(avg); colors_list.append(SECTOR_COLORS[s])

    fig = go.Figure(go.Pie(
        labels=labels, values=vals,
        hole=0.55,
        marker=dict(colors=colors_list, line=dict(color=CARD, width=3)),
        textfont=dict(color=TEXT, size=12),
        hovertemplate="<b>%{label}</b><br>Avg Contagion: %{value:.1%}<extra></extra>",
        textinfo="label+percent",
    ))
    fig.add_annotation(text="Avg Risk<br>by Sector", x=0.5, y=0.5,
                        font=dict(color=MUTED, size=11), showarrow=False)
    dark(fig, height=280, margin=dict(l=10, r=10, t=10, b=10), legend=False)
    return fig


# ── 7. Equity curve ──────────────────────────────────────────────────────────
def build_equity():
    if bt is None:
        return go.Figure()

    fig = make_subplots(rows=2, cols=1, row_heights=[0.68, 0.32],
                         shared_xaxes=True, vertical_spacing=0.06)
    C = {"GNN_Agent": GREEN, "EqualWeight": AMBER, "SPY": BLUE}
    LABELS = {"GNN_Agent": "🤖 GNN Agent", "EqualWeight": "⚖️ Equal Weight", "SPY": "📈 SPY Index"}

    for col, clr in C.items():
        if col not in bt.columns:
            continue
        fig.add_trace(go.Scatter(
            x=bt.index, y=bt[col],
            name=LABELS[col], line=dict(color=clr, width=2.2),
            hovertemplate=f"<b>{LABELS[col]}</b><br>%{{x|%b %Y}}<br>${{y:,.0f}}<extra></extra>",
        ), row=1, col=1)

        eq  = bt[col].values
        cum = np.maximum.accumulate(eq)
        dd  = (eq - cum) / cum * 100
        fig.add_trace(go.Scatter(
            x=bt.index, y=dd, name=col,
            fill="tozeroy", line=dict(color=clr, width=0.8),
            fillcolor=_hex_rgba(clr, 0.12),
            showlegend=False,
            hovertemplate=f"<b>DD</b><br>%{{x|%b %Y}}<br>%{{y:.1f}}%<extra></extra>",
        ), row=2, col=1)

    fig.add_vrect(x0="2020-02-19", x1="2020-03-23", row=1, col=1,
                   fillcolor="rgba(255,77,106,0.07)", line_width=0,
                   annotation_text="COVID", annotation_font_color=RED,
                   annotation_font_size=10, annotation_position="top left")
    fig.add_vrect(x0="2022-01-03", x1="2022-10-12", row=1, col=1,
                   fillcolor="rgba(255,179,71,0.06)", line_width=0,
                   annotation_text="Rate Shock", annotation_font_color=AMBER,
                   annotation_font_size=10, annotation_position="top left")

    dark(fig, height=480, margin=dict(l=70, r=20, t=20, b=50))
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1,
                      tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, ticksuffix="%")
    fig.update_xaxes(row=2, col=1, tickfont=dict(color=MUTED))
    fig.update_layout(hovermode="x unified",
                       legend=dict(bgcolor=CARD2, bordercolor=BORDER,
                                   font=dict(color=TEXT), orientation="h",
                                   y=1.04, x=0.5, xanchor="center"))
    return fig


# ── 8. Sector weight area chart ──────────────────────────────────────────────
def build_sector_area():
    if mlog is None:
        return go.Figure()
    fig = go.Figure()
    for s in SECTORS:
        col = f"w_{s}"
        if col not in mlog.columns:
            continue
        fig.add_trace(go.Scatter(
            x=mlog["date"], y=mlog[col] * 100,
            name=s, stackgroup="one",
            line=dict(color=SECTOR_COLORS[s], width=0),
            fillcolor=_hex_rgba(SECTOR_COLORS[s], 0.55),
            hovertemplate=f"<b>{s}</b><br>%{{x|%b %Y}}<br>Weight: %{{y:.1f}}%<extra></extra>",
        ))
    # shade crisis periods
    for cname, (s, e, cs, ce) in CRISES.items():
        fig.add_vrect(x0=cs, x1=ce,
                       fillcolor="rgba(255,77,106,0.07)", line_width=0)

    dark(fig, height=260, margin=dict(l=60, r=20, t=20, b=50))
    fig.update_layout(
        yaxis=dict(range=[0, 100], ticksuffix="%", title="Allocation (%)"),
        legend=dict(bgcolor=CARD2, bordercolor=BORDER, font=dict(color=TEXT),
                    orientation="h", y=-0.25, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig


# ── 9. Contagion timeline ─────────────────────────────────────────────────────
def build_contagion_timeline():
    if mlog is None:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mlog["date"], y=mlog["avg_contagion"] * 100,
        name="Avg Contagion", fill="tozeroy",
        fillcolor="rgba(255,77,106,0.12)", line=dict(color=RED, width=1.8),
        hovertemplate="%{x|%b %Y}<br>Avg: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=mlog["date"], y=mlog["max_contagion"] * 100,
        name="Peak Contagion", line=dict(color=AMBER, width=1.2, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>Peak: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color=BORDER,
                   annotation_text="50% threshold", annotation_font_color=MUTED,
                   annotation_font_size=10)
    for cname, (s, e, cs, ce) in CRISES.items():
        fig.add_vrect(x0=cs, x1=ce, fillcolor="rgba(255,77,106,0.07)", line_width=0)
    dark(fig, height=200, margin=dict(l=60, r=20, t=20, b=40))
    fig.update_layout(
        yaxis=dict(range=[0, 100], ticksuffix="%", title="Score (%)"),
        legend=dict(bgcolor=CARD2, bordercolor=BORDER, font=dict(color=TEXT),
                    orientation="h", y=-0.3, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig


# ── 10. Metrics table rows ────────────────────────────────────────────────────
def build_metrics_rows():
    if bm is None:
        return []
    C = {"GNN_Agent": GREEN, "EqualWeight": AMBER, "SPY": BLUE}
    LABELS = {"GNN_Agent": "🤖 GNN Agent", "EqualWeight": "⚖️ Equal Weight", "SPY": "📈 SPY Index"}
    rows = []
    for idx, row in bm.iterrows():
        clr = C.get(str(idx), TEXT)
        cells = [html.Td(LABELS.get(str(idx), idx),
                          style={"color": clr, "fontWeight": "700",
                                  "fontSize": "13px", "padding": "10px 14px"})]
        for col, val in row.items():
            v_style = {"color": TEXT, "fontSize": "13px", "padding": "10px 14px",
                        "textAlign": "right", "fontFamily": "monospace"}
            try:
                fval = float(val)
                if "Return" in col or "Sharpe" in col or "Calmar" in col or "Sortino" in col:
                    v_style["color"] = GREEN if fval > 0 else RED
                if "Drawdown" in col:
                    v_style["color"] = RED if fval < -20 else AMBER
            except Exception:
                pass
            cells.append(html.Td(str(val), style=v_style))
        rows.append(html.Tr(cells, style={"borderBottom": f"1px solid {BORDER}"}))
    return rows


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _hex_rgba(h, alpha=0.5):
    r, g, b = _hex_rgb(h)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.SLATE,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    title="FinContagion-GNN",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI VALUES
# ─────────────────────────────────────────────────────────────────────────────
def _kv(idx, col, fmt):
    try:
        return fmt.format(float(bm.loc[idx, col]))
    except Exception:
        return "—"

gnn_ret  = _kv("GNN_Agent",  "Annual Return (%)",    "{:.1f}%")
gnn_sr   = _kv("GNN_Agent",  "Sharpe Ratio",          "{:.3f}")
gnn_mdd  = _kv("GNN_Agent",  "Max Drawdown (%)",     "{:.1f}%")
gnn_vol  = _kv("GNN_Agent",  "Annual Volatility (%)", "{:.1f}%")
spy_ret  = _kv("SPY",        "Annual Return (%)",    "{:.1f}%")
gnn_fv   = _kv("GNN_Agent",  "Final Value ($)",      "${:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
PAGE_STYLE = {
    "backgroundColor": BG,
    "minHeight": "100vh",
    "fontFamily": "Inter, sans-serif",
    "color": TEXT,
}

HEADER = html.Div([
    dbc.Container(fluid=True, children=[
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("⬡ ", style={"color": GREEN, "fontSize": "24px"}),
                        html.Span("FinContagion", style={"color": GREEN, "fontWeight": "700",
                                                          "fontSize": "20px"}),
                        html.Span("-GNN", style={"color": TEXT, "fontWeight": "300",
                                                  "fontSize": "20px"}),
                    ]),
                    html.P("Modeling Financial Contagion & Adaptive Portfolio Optimization using Graph Neural Networks",
                           style={"color": MUTED, "fontSize": "12px", "marginTop": "2px",
                                   "marginBottom": "0"}),
                ])
            ], width=4),
            dbc.Col([
                dbc.Row([
                    dbc.Col(kpi("GNN Ann. Return",  gnn_ret, GREEN,  f"SPY: {spy_ret}"), width=3),
                    dbc.Col(kpi("Sharpe Ratio",     gnn_sr,  BLUE), width=2),
                    dbc.Col(kpi("Max Drawdown",     gnn_mdd, AMBER), width=3),
                    dbc.Col(kpi("Portfolio (Final)", gnn_fv, PURPLE, "from $1,000,000"), width=4),
                ], className="g-2"),
            ], width=8),
        ], align="center", className="py-3"),
    ], style={"maxWidth": "100%"}),
], style={"backgroundColor": CARD, "borderBottom": f"1px solid {BORDER}",
           "position": "sticky", "top": "0", "zIndex": "1000"})

# ── TAB NAV ──────────────────────────────────────────────────────────────────
TAB_BAR = html.Div([
    dbc.Container(fluid=True, children=[
        dbc.Row([
            dbc.Col([
                dcc.Tabs(id="tabs", value="network", children=[
                    dcc.Tab(label="🕸  Network & Risk Map",      value="network",
                             style=tab_style(GREEN),
                             selected_style=active_tab_style(GREEN)),
                    dcc.Tab(label="⚡  Contagion Spread",        value="contagion",
                             style=tab_style(RED),
                             selected_style=active_tab_style(RED)),
                    dcc.Tab(label="🛡  Portfolio Defence",       value="portfolio",
                             style=tab_style(BLUE),
                             selected_style=active_tab_style(BLUE)),
                    dcc.Tab(label="🧠  GNN Model Performance",  value="gnn",
                             style=tab_style(PURPLE),
                             selected_style=active_tab_style(PURPLE)),
                ], style={"borderBottom": "none"},
                   colors={"border": "transparent", "primary": GREEN,
                            "background": "transparent"}),
            ]),
        ]),
    ], style={"maxWidth": "100%"}),
], style={"backgroundColor": CARD2, "borderBottom": f"1px solid {BORDER}"})

# ─────────────────────────────────────────────────────────────────────────────
# TAB CONTENTS
# ─────────────────────────────────────────────────────────────────────────────

# ── TAB 1: NETWORK ───────────────────────────────────────────────────────────
TAB_NETWORK = dbc.Container(fluid=True, className="py-4", children=[

    # Explainer banner
    html.Div([
        html.H6("What am I looking at?", style={"color": TEXT, "marginBottom": "6px",
                                                  "fontWeight": "600"}),
        html.P([
            "Each ", html.B("circle (node)"), " is a real firm. Lines between firms show ",
            html.B("financial connections"), " — if two firms move together strongly (correlation ≥ 0.45), ",
            "they are linked. The ", html.Span("redder", style={"color": RED, "fontWeight": "600"}),
            " a node, the higher the GNN predicts its risk of causing or suffering a crash. ",
            html.Span("Greener", style={"color": GREEN, "fontWeight": "600"}),
            " means safer. Node size grows with risk.",
        ], style={"color": MUTED, "fontSize": "13px", "lineHeight": "1.7",
                   "marginBottom": "0"}),
    ], style={"backgroundColor": "#0A1525", "border": f"1px solid {BORDER}",
               "borderRadius": "10px", "padding": "16px 20px", "marginBottom": "20px"}),

    dbc.Row([
        # Controls
        dbc.Col([
            card([
                section_title("Controls"),
                html.P("Colour mode", style={"color": MUTED, "fontSize": "12px",
                                              "marginBottom": "6px"}),
                dcc.RadioItems(
                    id="net-color",
                    options=[{"label": "🔴  Contagion Risk", "value": "risk"},
                              {"label": "🎨  Sector Colour",  "value": "sector"}],
                    value="risk", inline=False,
                    inputStyle={"marginRight": "8px", "accentColor": GREEN},
                    labelStyle={"color": TEXT, "fontSize": "13px",
                                 "display": "block", "marginBottom": "10px",
                                 "cursor": "pointer"},
                ),
                html.Hr(style={"borderColor": BORDER, "margin": "16px 0"}),
                html.P("Highlight sector", style={"color": MUTED, "fontSize": "12px",
                                                    "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="net-sector-filter",
                    options=[{"label": "All sectors", "value": ""}] +
                             [{"label": s, "value": s} for s in SECTORS],
                    value="",
                    clearable=False,
                    style={"backgroundColor": CARD2, "color": TEXT,
                            "border": f"1px solid {BORDER}", "borderRadius": "6px"},
                ),
                html.Hr(style={"borderColor": BORDER, "margin": "16px 0"}),
                html.Div(id="node-info", children=[
                    html.P("👆 Click any firm node to see its risk details here.",
                           style={"color": MUTED, "fontSize": "12px",
                                   "fontStyle": "italic", "lineHeight": "1.6"}),
                ]),
            ]),
        ], width=2),

        # Main graph
        dbc.Col([
            card([
                html.Div([
                    html.Span("Financial Dependency Network — 50 Firms",
                              style={"color": TEXT, "fontWeight": "600", "fontSize": "14px"}),
                    html.Span(" (2015–2024)", style={"color": MUTED, "fontSize": "12px"}),
                ], style={"marginBottom": "12px"}),
                dcc.Graph(id="net-graph",
                           figure=build_network(),
                           config={"displayModeBar": False},
                           style={"borderRadius": "8px"}),
            ], padding="20px"),
        ], width=7),

        # Right column: sector summary
        dbc.Col([
            card([
                section_title("GNN Risk by Sector",
                               "Average contagion probability across all firms in each sector"),
                dcc.Graph(figure=build_sector_donut(),
                           config={"displayModeBar": False}),
            ], padding="20px"),
            html.Div(style={"height": "16px"}),
            card([
                section_title("Top Risk Firms",
                               "Click a firm in the network for full detail"),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Span(f"{t}  ", style={"color": TEXT, "fontWeight": "600",
                                                         "fontSize": "13px",
                                                         "fontFamily": "monospace"}),
                            html.Span(SECTOR_MAP.get(t, ""),
                                       style={"color": SECTOR_COLORS.get(SECTOR_MAP.get(t,""), MUTED),
                                               "fontSize": "11px"}),
                        ]),
                        html.Div([
                            html.Div(style={
                                "width": f"{float(gnn.loc[t, risk_col])*100:.0f}%",
                                "height": "4px",
                                "backgroundColor": f"rgb({int(255*float(gnn.loc[t,risk_col]))},{int(212*(1-float(gnn.loc[t,risk_col])))},80)",
                                "borderRadius": "2px", "marginTop": "4px",
                            }),
                            html.Span(f"{float(gnn.loc[t, risk_col]):.0%}",
                                       style={"color": MUTED, "fontSize": "11px",
                                               "fontFamily": "monospace"}),
                        ]),
                    ], style={"marginBottom": "10px"})
                    for t in (gnn.nlargest(8, risk_col).index.tolist() if gnn is not None else [])
                ]),
            ], padding="20px"),
        ], width=3),
    ], className="g-3"),
])


# ── TAB 2: CONTAGION SPREAD ───────────────────────────────────────────────────
TAB_CONTAGION = dbc.Container(fluid=True, className="py-4", children=[

    html.Div([
        html.H6("How does financial contagion spread?", style={"color": TEXT,
                                                                "marginBottom": "6px",
                                                                "fontWeight": "600"}),
        html.P([
            "When one sector crashes, it ", html.B("does not stay isolated"),
            ". Because firms borrow from each other, hold each other's assets, "
            "and share the same investors, distress ", html.B("propagates through the network"),
            ". The charts below show exactly how each historical crisis rippled "
            "sector-by-sector. The ", html.B("heatmap"),
            " shows how correlated sectors became during the crash — ",
            "the darker the cell, the more the two sectors moved together (contagion). "
            "The ", html.B("waterfall chart"), " shows cumulative returns so you can see which sector fell first and which pulled others down.",
        ], style={"color": MUTED, "fontSize": "13px", "lineHeight": "1.7",
                   "marginBottom": "0"}),
    ], style={"backgroundColor": "#0A1525", "border": f"1px solid {BORDER}",
               "borderRadius": "10px", "padding": "16px 20px", "marginBottom": "20px"}),

    # Crisis selector
    dbc.Row([
        dbc.Col([
            card([
                html.P("Select a crisis event", style={"color": MUTED, "fontSize": "12px",
                                                         "marginBottom": "8px"}),
                dcc.RadioItems(
                    id="crisis-radio",
                    options=[{"label": k, "value": k} for k in CRISES],
                    value=list(CRISES.keys())[0], inline=False,
                    inputStyle={"marginRight": "8px", "accentColor": RED},
                    labelStyle={"color": TEXT, "fontSize": "13px",
                                 "display": "block", "marginBottom": "12px",
                                 "cursor": "pointer", "lineHeight": "1.5"},
                ),
                html.Hr(style={"borderColor": BORDER, "margin": "12px 0"}),
                html.Div(id="crisis-stats"),
            ], padding="20px"),
        ], width=2),

        dbc.Col([
            dbc.Row([
                dbc.Col([
                    card([
                        html.Div([
                            html.Span("Cross-Sector Contagion Matrix",
                                      style={"color": TEXT, "fontWeight": "600",
                                              "fontSize": "14px"}),
                            html.Br(),
                            html.Span("How strongly each sector infected the others during the crisis — darker = more contagion",
                                      style={"color": MUTED, "fontSize": "12px"}),
                        ], style={"marginBottom": "12px"}),
                        dcc.Graph(id="sector-flow",
                                   figure=build_sector_flow(list(CRISES.keys())[0]),
                                   config={"displayModeBar": False}),
                    ], padding="20px"),
                ], width=6),
                dbc.Col([
                    card([
                        html.Div([
                            html.Span("Max Drawdown Per Firm",
                                      style={"color": TEXT, "fontWeight": "600",
                                              "fontSize": "14px"}),
                            html.Br(),
                            html.Span("How far each stock fell from its peak during the crash window",
                                      style={"color": MUTED, "fontSize": "12px"}),
                        ], style={"marginBottom": "12px"}),
                        dcc.Graph(id="sector-dd",
                                   figure=build_sector_drawdown(list(CRISES.keys())[0]),
                                   config={"displayModeBar": False}),
                    ], padding="20px"),
                ], width=6),
            ], className="g-3"),
            html.Div(style={"height": "16px"}),
            card([
                html.Div([
                    html.Span("Cumulative Return by Sector",
                              style={"color": TEXT, "fontWeight": "600", "fontSize": "14px"}),
                    html.Br(),
                    html.Span("Thick lines = sector average. Thin lines = individual firms. Red shaded area = peak crash window.",
                              style={"color": MUTED, "fontSize": "12px"}),
                ], style={"marginBottom": "12px"}),
                dcc.Graph(id="wave-chart",
                           figure=build_contagion_wave(list(CRISES.keys())[0]),
                           config={"displayModeBar": False}),
            ], padding="20px"),
        ], width=10),
    ], className="g-3"),
])


# ── TAB 3: PORTFOLIO DEFENCE ─────────────────────────────────────────────────
TAB_PORTFOLIO = dbc.Container(fluid=True, className="py-4", children=[

    html.Div([
        html.H6("How does the GNN protect the portfolio?", style={"color": TEXT,
                                                                    "marginBottom": "6px",
                                                                    "fontWeight": "600"}),
        html.P([
            "Every month, the GNN scores each firm's contagion risk. These scores are fed into a ",
            html.B("Black-Litterman optimizer"), " which tilts the portfolio away from high-risk firms "
            "and toward safer sectors. The agent also penalises firms that are ",
            html.B("highly connected in the network"), " (shock amplifiers). "
            "Below you can see the equity curves vs benchmarks, how sector allocation shifted month-by-month, "
            "and the full performance metrics.",
        ], style={"color": MUTED, "fontSize": "13px", "lineHeight": "1.7",
                   "marginBottom": "0"}),
    ], style={"backgroundColor": "#0A1525", "border": f"1px solid {BORDER}",
               "borderRadius": "10px", "padding": "16px 20px", "marginBottom": "20px"}),

    dbc.Row([
        dbc.Col([
            card([
                html.Div([
                    html.Span("Portfolio Equity Curves  (2020–2024)",
                              style={"color": TEXT, "fontWeight": "600", "fontSize": "14px"}),
                    html.Br(),
                    html.Span("Top: portfolio value from $1M. Bottom: drawdown from peak. Shaded = crisis periods.",
                              style={"color": MUTED, "fontSize": "12px"}),
                ], style={"marginBottom": "12px"}),
                dcc.Graph(figure=build_equity(), config={"displayModeBar": False}),
            ], padding="20px"),
        ], width=8),
        dbc.Col([
            card([
                section_title("Performance Summary",
                               "Walk-forward backtest · Monthly rebalancing · $1M initial"),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Strategy", style={"color": MUTED, "fontSize": "11px",
                                                    "padding": "6px 14px", "textTransform": "uppercase",
                                                    "letterSpacing": "0.06em",
                                                    "borderBottom": f"1px solid {BORDER}"}),
                        *[html.Th(c, style={"color": MUTED, "fontSize": "11px",
                                             "padding": "6px 14px", "textAlign": "right",
                                             "textTransform": "uppercase",
                                             "letterSpacing": "0.06em",
                                             "borderBottom": f"1px solid {BORDER}"})
                          for c in (bm.columns.tolist() if bm is not None else [])],
                    ])),
                    html.Tbody(build_metrics_rows()),
                ], style={"width": "100%", "borderCollapse": "collapse",
                            "fontSize": "13px"}),
            ], padding="20px"),
            html.Div(style={"height": "16px"}),
            card([
                section_title("GNN Contagion Index",
                               "Monthly average & peak contagion score — spikes = market stress periods"),
                dcc.Graph(figure=build_contagion_timeline(),
                           config={"displayModeBar": False}),
            ], padding="20px"),
        ], width=4),
    ], className="g-3"),

    html.Div(style={"height": "16px"}),

    dbc.Row([
        dbc.Col([
            card([
                html.Div([
                    html.Span("Sector Allocation Over Time",
                              style={"color": TEXT, "fontWeight": "600", "fontSize": "14px"}),
                    html.Br(),
                    html.Span("How the GNN agent shifted money between sectors as contagion risk evolved. "
                              "Notice how it rotates out of Energy/Finance during crises toward Healthcare/Consumer.",
                              style={"color": MUTED, "fontSize": "12px"}),
                ], style={"marginBottom": "12px"}),
                dcc.Graph(figure=build_sector_area(), config={"displayModeBar": False}),
            ], padding="20px"),
        ], width=12),
    ]),
])


# ── TAB 4: GNN ANALYTICS ─────────────────────────────────────────────────────
TAB_GNN = dbc.Container(fluid=True, className="py-4", children=[

    html.Div([
        html.H6("How does the GNN model work?", style={"color": TEXT,
                                                        "marginBottom": "6px",
                                                        "fontWeight": "600"}),
        html.P([
            "The model is a ", html.B("Hierarchical Temporal Contagion GNN (HTC-GNN)"),
            ". It has two levels: (1) ", html.B("Micro"), " — Graph Attention layers let each firm "
            "aggregate information from its neighbours, learning which connections matter most. "
            "(2) ", html.B("Macro"), " — each sector is pooled into a super-node that feeds back into firm embeddings. "
            "A ", html.B("GRU cell"), " tracks how risk evolves month-to-month. "
            "The model is trained on data from 2015–2018, validated on 2019, and tested on 2020–2024 — "
            "completely out-of-sample, including the COVID crash.",
        ], style={"color": MUTED, "fontSize": "13px", "lineHeight": "1.7",
                   "marginBottom": "0"}),
    ], style={"backgroundColor": "#0A1525", "border": f"1px solid {BORDER}",
               "borderRadius": "10px", "padding": "16px 20px", "marginBottom": "20px"}),

    dbc.Row([
        # Architecture diagram (text-based)
        dbc.Col([
            card([
                section_title("Model Architecture", "HTC-GNN — 5 processing stages"),
                *[
                    html.Div([
                        html.Div([
                            html.Span(icon, style={"fontSize": "20px", "marginRight": "12px"}),
                            html.Div([
                                html.P(title, style={"color": TEXT, "fontWeight": "600",
                                                      "marginBottom": "2px", "fontSize": "13px"}),
                                html.P(desc, style={"color": MUTED, "fontSize": "12px",
                                                     "marginBottom": "0", "lineHeight": "1.5"}),
                            ]),
                        ], style={"display": "flex", "alignItems": "flex-start"}),
                        html.Div(style={"height": "1px", "backgroundColor": BORDER,
                                         "margin": "12px 0"}) if i < 4 else None,
                    ])
                    for i, (icon, title, desc) in enumerate([
                        ("📥", "Input Projection",
                         "13 node features per firm: returns, volatility, RSI, beta, sector one-hot → projected to 64-dim space"),
                        ("🔗", "Graph Attention (×2)",
                         "Multi-head GAT: each firm attends to its correlated neighbours. Attention weights reveal the contagion channels."),
                        ("🏛", "Sector Pooling",
                         "Firms are grouped into 5 sector super-nodes via mean pooling — captures macro-level stress."),
                        ("🔀", "Cross-Level Fusion",
                         "Each firm's micro-embedding is fused with its sector's macro-embedding via cross-attention."),
                        ("⏱", "Temporal GRU",
                         "A GRU cell carries the hidden state across monthly snapshots — the model remembers how risk has been evolving."),
                    ])
                ],
                html.Hr(style={"borderColor": BORDER, "margin": "16px 0"}),
                html.Div([
                    html.P("Training Setup", style={"color": TEXT, "fontWeight": "600",
                                                     "fontSize": "13px", "marginBottom": "8px"}),
                    *[html.Div([
                        html.Span(k + ":  ", style={"color": MUTED, "fontSize": "12px"}),
                        html.Span(v, style={"color": TEXT, "fontSize": "12px",
                                             "fontFamily": "monospace"}),
                    ], style={"marginBottom": "4px"})
                      for k, v in [
                          ("Loss function",   "Focal Loss  (α=0.75, γ=2)"),
                          ("Optimizer",       "AdamW  (lr=3e-4, wd=1e-4)"),
                          ("LR schedule",     "Cosine Annealing Warm Restarts"),
                          ("Train split",     "2015–2018  (42 monthly snapshots)"),
                          ("Val split",       "2019  (12 snapshots)"),
                          ("Test split",      "2020–2024  (60 snapshots)"),
                          ("Val AUROC",       "0.7575"),
                          ("Test AUROC",      "0.6600"),
                          ("Test Avg Prec",   "0.3152"),
                      ]],
                ]),
            ], padding="24px"),
        ], width=4),

        # Risk ranking
        dbc.Col([
            card([
                section_title("GNN Contagion Score — All 50 Firms",
                               "Red dashed line = 50% risk threshold. Firms above it are flagged as high-risk."),
                dcc.Graph(figure=build_risk_bar(), config={"displayModeBar": False}),
            ], padding="20px"),
        ], width=5),

        # Test metrics
        dbc.Col([
            card([
                section_title("Test Set Metrics",
                               "Out-of-sample 2020–2024 incl. COVID crash"),
                *[
                    html.Div([
                        html.Div([
                            html.P(label, style={"color": MUTED, "fontSize": "11px",
                                                  "textTransform": "uppercase",
                                                  "letterSpacing": "0.06em",
                                                  "marginBottom": "2px"}),
                            html.H5(value, style={"color": color, "fontFamily": "monospace",
                                                   "fontWeight": "700", "marginBottom": "0"}),
                            html.P(note, style={"color": MUTED, "fontSize": "11px",
                                                 "marginTop": "2px"}),
                        ], style={"backgroundColor": CARD2, "borderRadius": "8px",
                                   "padding": "14px 16px", "marginBottom": "10px",
                                   "border": f"1px solid {BORDER}"}),
                    ])
                    for label, value, color, note in [
                        ("AUROC", "0.6600", GREEN,
                         "Area under ROC curve — 0.5 = random, 1.0 = perfect"),
                        ("Average Precision", "0.3152", BLUE,
                         "Precision-recall AUC — reflects rare crash class"),
                        ("Safe Firm F1", "0.8740", GREEN,
                         "Correctly identifies safe-to-hold firms"),
                        ("Crash Firm F1", "0.3319", AMBER,
                         "Correctly flags crash-prone firms (hard due to imbalance)"),
                        ("Overall Accuracy", "78.8%", PURPLE,
                         "2,364 / 3,000 firms correctly classified"),
                    ]
                ],
                html.Hr(style={"borderColor": BORDER, "margin": "12px 0"}),
                html.P([
                    html.B("Why is crash F1 lower? "),
                    html.Span("Only 13% of (firm, month) pairs experienced a crash — "
                              "class imbalance is the main challenge. Focal Loss helps, but more "
                              "data from rare crashes (2008, 2018 corrections) would improve recall further.",
                              style={"color": MUTED}),
                ], style={"fontSize": "12px", "lineHeight": "1.6", "color": MUTED}),
            ], padding="24px"),
        ], width=3),
    ], className="g-3"),
])


# ─────────────────────────────────────────────────────────────────────────────
# ROOT LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    HEADER,
    TAB_BAR,
    html.Div(id="tab-content", style={"minHeight": "calc(100vh - 130px)"}),
    # footer
    html.Div([
        html.P("FinContagion-GNN  ·  Sharanya Rao · Prajakta Patil · Niranjan S Kaithota  ·  1RV23CD049 · 1RV23BT043 · 1RV23AI067",
               style={"color": MUTED, "fontSize": "11px", "textAlign": "center",
                       "margin": "0", "padding": "16px"}),
    ], style={"borderTop": f"1px solid {BORDER}", "backgroundColor": CARD}),
], style=PAGE_STYLE)


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "network":   return TAB_NETWORK
    if tab == "contagion": return TAB_CONTAGION
    if tab == "portfolio": return TAB_PORTFOLIO
    if tab == "gnn":       return TAB_GNN
    return TAB_NETWORK


@app.callback(Output("net-graph", "figure"),
              [Input("net-color", "value"), Input("net-sector-filter", "value")])
def update_network(color_mode, sector):
    return build_network(color_mode=color_mode,
                          selected_sector=sector if sector else None)


@app.callback(Output("node-info", "children"), Input("net-graph", "clickData"))
def show_node_info(click_data):
    if not click_data:
        return html.P("👆 Click any firm to see details.",
                       style={"color": MUTED, "fontSize": "12px", "fontStyle": "italic"})
    pt = click_data["points"][0]
    ticker = pt.get("text", "")
    if not ticker or gnn is None or ticker not in gnn.index:
        return html.P("Click a firm node.", style={"color": MUTED, "fontSize": "12px"})

    risk    = float(gnn.loc[ticker, risk_col])
    cr      = float(gnn.loc[ticker, "crash_rate"])
    sector  = SECTOR_MAP.get(ticker, "?")
    sec_clr = SECTOR_COLORS.get(sector, MUTED)
    risk_clr = RED if risk > 0.5 else (AMBER if risk > 0.35 else GREEN)
    G = nx.from_pandas_adjacency(A_df.fillna(0)) if A_df is not None else None
    degree  = G.degree(ticker) if G and ticker in G else "—"
    neighbours = sorted(G.neighbors(ticker), key=lambda n: -A_df.loc[ticker, n]
                         if A_df is not None and n in A_df.columns else 0)[:4] if G and ticker in G else []

    return html.Div([
        html.Div([
            html.Span(ticker, style={"color": GREEN, "fontWeight": "700",
                                      "fontSize": "18px", "fontFamily": "monospace"}),
            html.Span(f"  {sector}", style={"color": sec_clr, "fontSize": "12px"}),
        ], style={"marginBottom": "12px"}),
        *[html.Div([
            html.Span(label + "  ", style={"color": MUTED, "fontSize": "11px"}),
            html.Span(value, style={"color": vc, "fontWeight": "600",
                                     "fontFamily": "monospace", "fontSize": "13px"}),
        ], style={"marginBottom": "6px"})
          for label, value, vc in [
              ("Contagion Risk",    f"{risk:.1%}", risk_clr),
              ("Historical Crash Rate", f"{cr:.1%}", AMBER),
              ("Network Connections", str(degree), BLUE),
              ("Verdict", "⚠️ HIGH RISK" if risk > 0.5 else
               "⚡ MODERATE" if risk > 0.35 else "✅ SAFE", risk_clr),
          ]],
        html.Hr(style={"borderColor": BORDER, "margin": "10px 0"}),
        html.P("Top correlated firms:", style={"color": MUTED, "fontSize": "11px",
                                                "marginBottom": "4px"}),
        html.Div([
            html.Span(n + "  ", style={"color": SECTOR_COLORS.get(SECTOR_MAP.get(n,""), TEXT),
                                        "fontSize": "11px", "fontFamily": "monospace",
                                        "fontWeight": "600"})
            for n in neighbours
        ]),
    ])


@app.callback(
    [Output("sector-flow", "figure"),
     Output("wave-chart",  "figure"),
     Output("sector-dd",   "figure"),
     Output("crisis-stats", "children")],
    Input("crisis-radio", "value"),
)
def update_crisis(crisis_key):
    flow   = build_sector_flow(crisis_key)
    wave   = build_contagion_wave(crisis_key)
    dd_fig = build_sector_drawdown(crisis_key)

    # crisis stats panel
    stats = html.Div([html.P("—", style={"color": MUTED})])
    if dr is not None and crisis_key in CRISES:
        _, _, cs, ce = CRISES[crisis_key]
        window = dr.loc[cs:ce]
        cum = (1 + window).cumprod()
        worst_sector = min(SECTORS,
                            key=lambda s: cum[[t for t in STOCK_UNIVERSE[s]
                                               if t in cum.columns]].mean(axis=1).iloc[-1])
        best_sector  = max(SECTORS,
                            key=lambda s: cum[[t for t in STOCK_UNIVERSE[s]
                                               if t in cum.columns]].mean(axis=1).iloc[-1])
        mdd = (cum / cum.cummax() - 1).min().min() * 100
        stats = html.Div([
            html.Hr(style={"borderColor": BORDER, "margin": "0 0 12px 0"}),
            html.P("Crisis Stats", style={"color": MUTED, "fontSize": "11px",
                                           "textTransform": "uppercase",
                                           "letterSpacing": "0.06em",
                                           "marginBottom": "8px"}),
            *[html.Div([
                html.Span(label + " ", style={"color": MUTED, "fontSize": "11px"}),
                html.Div(val, style={"color": vc, "fontWeight": "600",
                                      "fontSize": "12px", "fontFamily": "monospace",
                                      "marginBottom": "6px"}),
            ]) for label, val, vc in [
                ("Hardest hit:", worst_sector, RED),
                ("Most resilient:", best_sector, GREEN),
                ("Worst firm DD:", f"{mdd:.1f}%", RED),
            ]],
        ])

    return flow, wave, dd_fig, stats


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  FinContagion-GNN Dashboard  v2")
    print("  → http://127.0.0.1:8050")
    print("═" * 55 + "\n")
    app.run(debug=False, port=8050, host="0.0.0.0")
