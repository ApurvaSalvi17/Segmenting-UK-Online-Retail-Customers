#!/usr/bin/env python3
"""
UK Online Retail – Customer Explorer Dashboard
(cluster averages + scatter with up to 4 highlighted customers)
"""
from __future__ import annotations
from pathlib import Path
import base64, pickle

import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# ═══════════ DATA (load once) ═══════════ #
DATA_DIR, MODEL_DIR = Path("data"), Path("models")
rfm = pd.read_csv(DATA_DIR / "rfm_clusters.csv.gz", compression="infer")
tx  = pd.read_csv(DATA_DIR / "online_retail_clean.csv.gz",
                  compression="infer", parse_dates=["InvoiceDate"])

desc_map = (tx.drop_duplicates("StockCode")
              .set_index("StockCode")["Description"]
              .str.title().to_dict())

with open(MODEL_DIR / "stock_to_idx.pkl", "rb") as f:
    stock_to_idx = pickle.load(f)
idx_to_stock = {v: k for k, v in stock_to_idx.items()}
item_vecs    = sparse.load_npz(MODEL_DIR / "item_vectors.npz")

PALETTE = px.colors.qualitative.Set2
cluster_color_map = {str(cl): PALETTE[i % len(PALETTE)]
                     for i, cl in enumerate(sorted(rfm["Cluster"].unique()))}

TEMPLATE = "plotly_white"

def recommend(code: str, k: int = 5) -> list[str]:
    if code not in stock_to_idx:
        return []
    sims = cosine_similarity(item_vecs[stock_to_idx[code]], item_vecs).ravel()
    return [idx_to_stock[i] for i in sims.argsort()[-k-1:-1][::-1]]

# ═══════════ DASH APP ═══════════ #
ext_css = ["https://cdnjs.cloudflare.com/ajax/libs/bootswatch/5.3.3/lux/bootstrap.min.css"]
app = Dash(__name__, external_stylesheets=ext_css, title="Online Retail Explorer")

app.layout = html.Div(className="container my-4", children=[
    html.H2("UK Online Retail – Customer Explorer"),

    # Multi-select dropdown (max 4 IDs for clarity)
    dcc.Dropdown(
        id="cust-dd",
        options=[{"label": int(cid), "value": int(cid)} for cid in sorted(rfm["CustomerID"])],
        placeholder="Select up to 4 Customer IDs",
        multi=True,                       # ← allows multiple
        className="mb-4",
    ),

    # Summary + recommendations
    html.Div(className="row", children=[
        html.Div(className="col-md-8", children=[
            dcc.Markdown(id="summary-md"),
            dcc.Markdown(id="explain-md", style={"whiteSpace": "pre-line"}),
        ]),
        html.Div(className="col-md-4", children=[
            html.H4("Recommended Products"),
            html.Ul(id="rec-list", className="mb-3"),
        ]),
    ]),

    # Tabs & figure
    dcc.Tabs(id="tabs", value="bar", children=[
        dcc.Tab(label="📊 Cluster Averages", value="bar"),
        dcc.Tab(label="🗺️ Customer Scatter", value="scatter"),
        dcc.Tab(label="ℹ️ About",             value="about"),
    ]),
    html.Div(id="fig-area", className="my-3"),

    # Download
    html.Button("Download RFM + Cluster CSV", id="dl-btn", className="btn btn-primary"),
    dcc.Download(id="dl-out"),
])

# ═══════════ CALLBACKS ═══════════ #
@app.callback(
    Output("summary-md", "children"),
    Output("explain-md", "children"),
    Output("fig-area",   "children"),
    Output("rec-list",   "children"),
    Input("cust-dd", "value"),
    Input("tabs",   "value"),
)
def render(cust_ids, tab):
    # handle None / empty
    if not cust_ids:
        if tab == "about":
            about = dcc.Markdown(
                "- **Data**: UCI *Online Retail* (UK invoices 2010-2011)\n"
                "- **RFM**: Recency, Frequency, Monetary\n"
                "- **Clusters**: K-Means  (k = 2–4+)\n"
                "- **Recommender**: TF-IDF cosine similarity",
                style={"whiteSpace": "pre-line"})
            return "", "", about, []
        return "⬆️ Pick up to four customers.", "", html.Div(), []

    # cap to four selections
    cust_ids = cust_ids[:4]

    # use the first selected ID for the textual summary
    first_id = cust_ids[0]
    row = rfm.set_index("CustomerID").loc[first_id]
    clan = str(int(row["Cluster"]))

    summary = f"**Customer {first_id}** → Cluster **{clan}**" + \
              (f"  (plus {len(cust_ids)-1} more)" if len(cust_ids) > 1 else "")
    explain = (f"- Recency **{row['Recency']:.0f} days**\n"
               f"- Frequency **{row['Frequency']:.0f} orders**\n"
               f"- Monetary **£{row['Monetary']:.2f}**")

    # ── Tab 1: cluster averages ──────────────────────────────────────────
    if tab == "bar":
        metrics = ["Recency", "Frequency", "Monetary"]
        units   = {"Recency":"Days", "Frequency":"Orders", "Monetary":"£"}
        avg = rfm.groupby("Cluster")[metrics].mean().astype(float)
        avg.index = avg.index.astype(str)

        fig = make_subplots(rows=1, cols=3, subplot_titles=metrics)
        for col_i, m in enumerate(metrics, start=1):
            for cl in avg.index:
                fig.add_bar(
                    x=[cl], y=[avg.loc[cl, m]], name=f"Cluster {cl}",
                    marker_color=cluster_color_map[cl],
                    marker_line_width=2 if cl == clan else 0,
                    marker_line_color="black",
                    showlegend=(col_i == 1), row=1, col=col_i)
            fig.update_yaxes(title_text=units[m], row=1, col=col_i)

        fig.update_layout(template=TEMPLATE, font_size=12,
                          title_text="Average metrics per cluster",
                          legend_title_text="Cluster", bargap=0.25)
        graph_block = dcc.Graph(figure=fig)

    # ── Tab 2: scatter with up to four highlighted customers ─────────────
    elif tab == "scatter":
        sc = rfm.copy()
        sc["Cluster"] = sc["Cluster"].astype(str)

        fig = px.scatter(
            sc, x="Recency", y="Monetary", size="Frequency",
            color="Cluster", color_discrete_map=cluster_color_map,
            size_max=50, opacity=0.70, template=TEMPLATE,
            labels={"Recency": "Recency (days)", "Monetary": "Monetary (£)"},
            title="Recency × Monetary – bubble size = Frequency",
        )
        fig.update_traces(marker=dict(sizemode="area"))
        fig.update_layout(font_size=12, legend_title_text="Cluster")

        # up to four marker shapes
        shapes = ["pentagon", "triangle-up", "diamond", "star"]
        for i, cid in enumerate(cust_ids):
            crow = rfm.set_index("CustomerID").loc[cid]
            fig.add_trace(go.Scatter(
                x=[crow["Recency"]], y=[crow["Monetary"]],
                mode="markers",
                name=f"Cust {cid}",
                marker=dict(symbol=shapes[i], size=40,
                            color="red", line=dict(width=2, color="white")),
                legendgroup=f"Selected {cid}", showlegend=True
            ))

        graph_block = dcc.Graph(figure=fig)

    else:  # about tab already handled above
        graph_block = html.Div()

    # ── Recommendations use first selected customer ──────────────────────
    last_code = (tx[tx["CustomerID"] == first_id]
                 .sort_values("InvoiceDate").iloc[-1]["StockCode"])
    recs = recommend(last_code)
    rec_items = ([html.Li(f"{c} — {desc_map.get(c, 'No description')}") for c in recs]
                 or [html.I("No recommendations")])

    return summary, explain, graph_block, rec_items

# ═══════════ CSV download ═══════════ #
@app.callback(
    Output("dl-out", "data"),
    Input("dl-btn", "n_clicks"), prevent_initial_call=True,
)
def download(_):
    b64 = base64.b64encode(rfm.to_csv(index=False).encode()).decode()
    return dict(content=b64, filename="rfm_clusters.csv", type="text/csv;base64")

# ═════════════════════════════════════ #
if __name__ == "__main__":
    app.run(debug=True, port=8050)
