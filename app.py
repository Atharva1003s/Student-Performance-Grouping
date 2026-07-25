import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import os

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Clustering",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — minimal overrides for a polished look
# ──────────────────────────────────────────────
_CSS = """
<style>
/* Metric cards */
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="metric-container"] label {
    font-weight: 500;
    color: #64748b !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

/* Accent divider */
.section-divider {
    height: 3px;
    background: linear-gradient(90deg, #6d28d9, #a78bfa, transparent);
    border: none;
    margin: 0.25rem 0 1.5rem;
    border-radius: 2px;
}

/* Cluster result cards */
.cluster-card {
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
}
.cluster-0 {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border-left: 5px solid #059669;
}
.cluster-1 {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border-left: 5px solid #2563eb;
}
.cluster-2 {
    background: linear-gradient(135deg, #fef2f2, #fecaca);
    border-left: 5px solid #dc2626;
}

/* Sidebar footer */
.sidebar-footer {
    position: fixed;
    bottom: 1rem;
    font-size: 0.72rem;
    opacity: 0.45;
}

/* Subtle table improvements */
div[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

# ──────────────────────────────────────────────
# Cluster label descriptions
# ──────────────────────────────────────────────
CLUSTER_LABELS = {
    0: ("🟢", "High Performers", "#059669"),
    1: ("🔵", "Average Performers", "#2563eb"),
    2: ("🔴", "Low Performers", "#dc2626"),
}

CLUSTER_CARD_CLASSES = {0: "cluster-0", 1: "cluster-1", 2: "cluster-2"}

# ──────────────────────────────────────────────
# Data Loading & Pipeline (cached)
# ──────────────────────────────────────────────
def _csv_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "StudentsPerformance.csv")


@st.cache_data
def load_raw_data():
    """Load the unprocessed student performance dataset."""
    return pd.read_csv(_csv_path())


@st.cache_resource
def run_clustering_pipeline():
    """
    Reproduce the notebook preprocessing & clustering pipeline.
    Returns everything needed for the UI.
    """
    df = pd.read_csv(_csv_path())

    # Keep a copy of original categorical values for display
    original_df = df.copy()

    # Label-encode categorical columns (same as notebook)
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include="object").columns
    label_mappings = {}
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
        label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Standard scaling (same as notebook)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

    # Elbow / WCSS (same as notebook: range 2-10)
    wcss = []
    k_range = range(2, 11)
    for i in k_range:
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(scaled_df)
        wcss.append(km.inertia_)

    # Silhouette scores
    sil_scores = []
    for i in k_range:
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_df)
        sil_scores.append(silhouette_score(scaled_df, labels))

    # K-Means with k=3 (same as notebook)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    original_df["KMeans_Cluster"] = kmeans.fit_predict(scaled_df)

    # Agglomerative Clustering with k=3 (same as notebook)
    agg = AgglomerativeClustering(n_clusters=3)
    original_df["Agg_Cluster"] = agg.fit_predict(scaled_df)

    # PCA for visualization (same as notebook)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled_df)
    original_df["PC1"] = X_pca[:, 0]
    original_df["PC2"] = X_pca[:, 1]

    # Linkage matrix for dendrogram
    linkage_matrix = linkage(scaled_df, method="ward")

    # Silhouette comparison
    k_sil = silhouette_score(scaled_df, original_df["KMeans_Cluster"])
    a_sil = silhouette_score(scaled_df, original_df["Agg_Cluster"])

    # Correlation matrix (on encoded df)
    corr_matrix = df.corr()

    return {
        "df": original_df,
        "encoded_df": df,
        "scaled_df": scaled_df,
        "scaler": scaler,
        "kmeans_model": kmeans,
        "label_mappings": label_mappings,
        "wcss": list(wcss),
        "k_range": list(k_range),
        "sil_scores": list(sil_scores),
        "linkage_matrix": linkage_matrix,
        "k_sil": k_sil,
        "a_sil": a_sil,
        "pca": pca,
        "corr_matrix": corr_matrix,
    }


# ──────────────────────────────────────────────
#  Helper — assign meaningful cluster labels
# ──────────────────────────────────────────────
def assign_cluster_labels(df, cluster_col):
    """
    Rank clusters by average total score and map to
    High / Average / Low labels.
    """
    df = df.copy()
    df["_total"] = df["math score"] + df["reading score"] + df["writing score"]
    means = df.groupby(cluster_col)["_total"].mean().sort_values(ascending=False)
    rank_map = {c: i for i, c in enumerate(means.index)}  # 0=high, 1=avg, 2=low
    df.drop(columns="_total", inplace=True)
    return rank_map


# ──────────────────────────────────────────────
#  PAGES
# ──────────────────────────────────────────────

# ---------- Dashboard ----------
def page_dashboard(pipeline):
    df = pipeline["df"]

    st.markdown("## 📊 Student Performance Overview")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    total = len(df)
    avg_math = df["math score"].mean()
    avg_read = df["reading score"].mean()
    avg_write = df["writing score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", f"{total:,}")
    c2.metric("Avg Math Score", f"{avg_math:.1f}")
    c3.metric("Avg Reading Score", f"{avg_read:.1f}")
    c4.metric("Avg Writing Score", f"{avg_write:.1f}")

    st.markdown("")

    # Row 1 — Gender split + Ethnicity breakdown
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### Gender Distribution")
        counts = df["gender"].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map={"female": "#8b5cf6", "male": "#3b82f6"},
            hole=0.45,
        )
        fig.update_traces(textinfo="value+percent", textfont_size=13)
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=360,
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.markdown("#### Scores by Race/Ethnicity")
        grouped = df.groupby("race/ethnicity")[["math score", "reading score", "writing score"]].mean().round(1)
        fig = px.bar(
            grouped, barmode="group",
            color_discrete_sequence=["#6d28d9", "#3b82f6", "#10b981"],
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=360,
            xaxis_title="", yaxis_title="Average Score",
            legend_title="Subject",
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 — Lunch impact + Test prep impact
    left_col2, right_col2 = st.columns(2)

    with left_col2:
        st.markdown("#### Score Distribution by Lunch Type")
        fig = px.box(
            df, x="lunch", y="math score",
            color="lunch",
            color_discrete_map={"standard": "#3b82f6", "free/reduced": "#ef4444"},
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=360,
            xaxis_title="Lunch Type", yaxis_title="Math Score",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right_col2:
        st.markdown("#### Test Preparation Course Impact")
        prep_means = df.groupby("test preparation course")[["math score", "reading score", "writing score"]].mean().round(1)
        fig = px.bar(
            prep_means, barmode="group",
            color_discrete_sequence=["#6d28d9", "#3b82f6", "#10b981"],
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=360,
            xaxis_title="", yaxis_title="Average Score",
            legend_title="Subject",
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------- Explore Data ----------
def page_explore(pipeline):
    df_raw = pipeline["df"]
    corr = pipeline["corr_matrix"]

    st.markdown("## 🔍 Explore the Dataset")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab_preview, tab_corr, tab_dist = st.tabs(
        ["📋 Data Preview", "🔗 Correlations", "📊 Distributions"]
    )

    # --- Preview ---
    with tab_preview:
        st.markdown(
            f"Showing **{len(df_raw):,}** records across "
            f"**{len(df_raw.columns) - 4}** original features."
        )
        display_cols = [c for c in df_raw.columns if c not in ["KMeans_Cluster", "Agg_Cluster", "PC1", "PC2"]]

        filter_col, _ = st.columns([1, 2])
        with filter_col:
            genders = st.multiselect(
                "Filter by Gender",
                options=df_raw["gender"].unique().tolist(),
                default=df_raw["gender"].unique().tolist(),
            )
        filtered = df_raw[df_raw["gender"].isin(genders)]
        st.dataframe(filtered[display_cols], height=420, use_container_width=True)

        st.markdown("#### Quick Statistics")
        st.dataframe(
            filtered[["math score", "reading score", "writing score"]].describe().round(2),
            use_container_width=True,
        )

    # --- Correlations ---
    with tab_corr:
        st.markdown("#### Feature Correlation Heatmap")
        st.caption("Numeric encoding applied to categorical columns for correlation analysis.")

        fig = px.imshow(
            corr, color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
            text_auto=".2f",
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=600, font=dict(size=9),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top correlated features with math score
        st.markdown("#### Top Features Correlated with Math Score")
        top = (
            corr["math score"].drop("math score")
            .abs().sort_values(ascending=True).tail(7)
        )
        fig = px.bar(
            x=top.values, y=top.index, orientation="h",
            color=top.values,
            color_continuous_scale=["#c4b5fd", "#6d28d9"],
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10), height=340,
            xaxis_title="Absolute Correlation", yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Distributions ---
    with tab_dist:
        st.markdown("#### Score Distributions")

        d1, d2, d3 = st.columns(3)
        for col_widget, score_col, color in [
            (d1, "math score", "#6d28d9"),
            (d2, "reading score", "#3b82f6"),
            (d3, "writing score", "#10b981"),
        ]:
            with col_widget:
                fig = px.histogram(
                    df_raw, x=score_col, nbins=25, opacity=0.8,
                    color_discrete_sequence=[color],
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20), height=350,
                    title=score_col.title(),
                    xaxis_title="Score", yaxis_title="Count",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Pair Plot — Score Relationships")
        fig = px.scatter_matrix(
            df_raw,
            dimensions=["math score", "reading score", "writing score"],
            color="gender",
            color_discrete_map={"female": "#8b5cf6", "male": "#3b82f6"},
            opacity=0.5,
        )
        fig.update_layout(
            margin=dict(l=30, r=30, t=30, b=30), height=550,
            legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
        )
        fig.update_traces(diagonal_visible=False, marker=dict(size=3))
        st.plotly_chart(fig, use_container_width=True)


# ---------- Optimal Clusters ----------
def page_optimal_clusters(pipeline):
    st.markdown("## 🎯 Finding Optimal Number of Clusters")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        "Two standard methods — the **Elbow Method** (WCSS) and the "
        "**Silhouette Score** — are used to determine the best value of **k**."
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### Elbow Method (WCSS)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pipeline["k_range"], y=pipeline["wcss"],
            mode="lines+markers",
            marker=dict(size=10, color="#6d28d9"),
            line=dict(color="#6d28d9", width=2),
        ))
        # Highlight k=3
        idx_3 = pipeline["k_range"].index(3)
        fig.add_trace(go.Scatter(
            x=[3], y=[pipeline["wcss"][idx_3]],
            mode="markers",
            marker=dict(size=16, color="#ef4444", symbol="star"),
            name="k = 3 (selected)",
        ))
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=420,
            xaxis_title="Number of Clusters (k)",
            yaxis_title="WCSS (Within-Cluster Sum of Squares)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.06)")
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.markdown("#### Silhouette Score")
        colors = ["#6d28d9" if k != 3 else "#ef4444" for k in pipeline["k_range"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pipeline["k_range"], y=pipeline["sil_scores"],
            mode="lines+markers",
            marker=dict(size=10, color=colors),
            line=dict(color="#a78bfa", width=2),
        ))
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=420,
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Silhouette Score",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.06)")
        st.plotly_chart(fig, use_container_width=True)

    # Scores table
    st.markdown("#### Silhouette Scores Table")
    scores_df = pd.DataFrame({
        "k": pipeline["k_range"],
        "WCSS": [f"{w:,.1f}" for w in pipeline["wcss"]],
        "Silhouette Score": [f"{s:.4f}" for s in pipeline["sil_scores"]],
    })
    st.dataframe(scores_df, use_container_width=True, hide_index=True)

    st.info(
        "💡 Based on both the Elbow Method and Silhouette analysis, "
        "**k = 3** is selected as the optimal number of clusters."
    )


# ---------- Clustering Results ----------
def page_clustering_results(pipeline):
    df = pipeline["df"]

    st.markdown("## 🔬 Clustering Results")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab_kmeans, tab_agg, tab_compare = st.tabs(
        ["🎯 K-Means Clustering", "🌳 Hierarchical Clustering", "⚖️ Comparison"]
    )

    # ── K-Means ──
    with tab_kmeans:
        st.markdown("#### K-Means Clusters (k = 3)")

        # Assign meaningful labels
        km_rank = assign_cluster_labels(df, "KMeans_Cluster")
        df["KMeans_Label"] = df["KMeans_Cluster"].map(
            lambda x: CLUSTER_LABELS[km_rank[x]][1]
        )

        # PCA scatter
        fig = px.scatter(
            df, x="PC1", y="PC2",
            color="KMeans_Label",
            color_discrete_map={
                "High Performers": "#059669",
                "Average Performers": "#2563eb",
                "Low Performers": "#dc2626",
            },
            opacity=0.6,
            title="K-Means Clusters (PCA Projection)",
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20), height=500,
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            legend_title="Cluster",
            legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
        )
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(fig, use_container_width=True)

        # Cluster counts
        st.markdown("#### Cluster Distribution")
        km_counts = df["KMeans_Label"].value_counts()
        for label in ["High Performers", "Average Performers", "Low Performers"]:
            if label in km_counts.index:
                info = CLUSTER_LABELS[[k for k, v in CLUSTER_LABELS.items() if v[1] == label][0]]
                count = km_counts[label]
                pct = count / len(df) * 100
                st.markdown(
                    f"**{info[0]} {label}**: {count} students ({pct:.1f}%)"
                )

        # Cluster means
        st.markdown("#### Average Scores by Cluster")
        means = df.groupby("KMeans_Label")[["math score", "reading score", "writing score"]].mean().round(1)
        fig = px.bar(
            means, barmode="group",
            color_discrete_sequence=["#6d28d9", "#3b82f6", "#10b981"],
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=380,
            xaxis_title="Cluster", yaxis_title="Average Score",
            legend_title="Subject",
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Hierarchical ──
    with tab_agg:
        st.markdown("#### Agglomerative Clustering (k = 3)")

        agg_rank = assign_cluster_labels(df, "Agg_Cluster")
        df["Agg_Label"] = df["Agg_Cluster"].map(
            lambda x: CLUSTER_LABELS[agg_rank[x]][1]
        )

        # PCA scatter
        fig = px.scatter(
            df, x="PC1", y="PC2",
            color="Agg_Label",
            color_discrete_map={
                "High Performers": "#059669",
                "Average Performers": "#2563eb",
                "Low Performers": "#dc2626",
            },
            opacity=0.6,
            title="Agglomerative Clusters (PCA Projection)",
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20), height=500,
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            legend_title="Cluster",
            legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
        )
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(fig, use_container_width=True)

        # Cluster counts
        st.markdown("#### Cluster Distribution")
        agg_counts = df["Agg_Label"].value_counts()
        for label in ["High Performers", "Average Performers", "Low Performers"]:
            if label in agg_counts.index:
                info = CLUSTER_LABELS[[k for k, v in CLUSTER_LABELS.items() if v[1] == label][0]]
                count = agg_counts[label]
                pct = count / len(df) * 100
                st.markdown(
                    f"**{info[0]} {label}**: {count} students ({pct:.1f}%)"
                )

        # Dendrogram
        st.markdown("#### Dendrogram")
        fig_dend, ax = plt.subplots(figsize=(14, 5))
        dendrogram(
            pipeline["linkage_matrix"],
            truncate_mode="lastp", p=30,
            leaf_rotation=90, leaf_font_size=8,
            ax=ax,
        )
        ax.set_title("Hierarchical Clustering Dendrogram (Ward's Method)", fontsize=13)
        ax.set_xlabel("Sample index or (cluster size)", fontsize=10)
        ax.set_ylabel("Distance", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig_dend)

        # Cluster means
        st.markdown("#### Average Scores by Cluster")
        means_agg = df.groupby("Agg_Label")[["math score", "reading score", "writing score"]].mean().round(1)
        fig = px.bar(
            means_agg, barmode="group",
            color_discrete_sequence=["#6d28d9", "#3b82f6", "#10b981"],
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=380,
            xaxis_title="Cluster", yaxis_title="Average Score",
            legend_title="Subject",
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Comparison ──
    with tab_compare:
        st.markdown("#### K-Means vs Agglomerative Clustering")

        comp_left, comp_right = st.columns(2)

        with comp_left:
            st.markdown("##### Silhouette Score Comparison")
            comp_df = pd.DataFrame({
                "Method": ["K-Means", "Agglomerative"],
                "Silhouette Score": [pipeline["k_sil"], pipeline["a_sil"]],
            })

            best_score = max(pipeline["k_sil"], pipeline["a_sil"])
            colors = [
                "#6d28d9" if s == best_score else "#c4b5fd"
                for s in [pipeline["k_sil"], pipeline["a_sil"]]
            ]

            fig = go.Figure(go.Bar(
                x=comp_df["Method"], y=comp_df["Silhouette Score"],
                marker_color=colors,
                text=[f"{s:.4f}" for s in comp_df["Silhouette Score"]],
                textposition="outside",
            ))
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20), height=400,
                yaxis=dict(title="Silhouette Score", range=[0, max(pipeline["k_sil"], pipeline["a_sil"]) * 1.3]),
                xaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.06)")
            st.plotly_chart(fig, use_container_width=True)

        with comp_right:
            st.markdown("##### Detailed Metrics")
            st.markdown("")

            best_method = "K-Means" if pipeline["k_sil"] >= pipeline["a_sil"] else "Agglomerative"

            metrics_table = pd.DataFrame({
                "Metric": ["Silhouette Score", "Number of Clusters", "Best Method"],
                "K-Means": [f"{pipeline['k_sil']:.4f}", "3", "✓" if best_method == "K-Means" else ""],
                "Agglomerative": [f"{pipeline['a_sil']:.4f}", "3", "✓" if best_method == "Agglomerative" else ""],
            })
            st.dataframe(metrics_table, use_container_width=True, hide_index=True)

            if best_method == "K-Means":
                st.success(
                    f"🏆 **K-Means** achieves a higher silhouette score "
                    f"({pipeline['k_sil']:.4f} vs {pipeline['a_sil']:.4f}), "
                    f"indicating slightly better-defined clusters."
                )
            else:
                st.success(
                    f"🏆 **Agglomerative Clustering** achieves a higher silhouette score "
                    f"({pipeline['a_sil']:.4f} vs {pipeline['k_sil']:.4f}), "
                    f"indicating slightly better-defined clusters."
                )

        # Side-by-side PCA comparison
        st.markdown("---")
        st.markdown("#### Side-by-Side PCA Visualization")

        pca_left, pca_right = st.columns(2)

        with pca_left:
            fig = px.scatter(
                df, x="PC1", y="PC2",
                color="KMeans_Label",
                color_discrete_map={
                    "High Performers": "#059669",
                    "Average Performers": "#2563eb",
                    "Low Performers": "#dc2626",
                },
                opacity=0.5, title="K-Means",
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=50, b=20), height=420,
                xaxis_title="PC1", yaxis_title="PC2",
                legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            )
            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(fig, use_container_width=True)

        with pca_right:
            fig = px.scatter(
                df, x="PC1", y="PC2",
                color="Agg_Label",
                color_discrete_map={
                    "High Performers": "#059669",
                    "Average Performers": "#2563eb",
                    "Low Performers": "#dc2626",
                },
                opacity=0.5, title="Agglomerative",
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=50, b=20), height=420,
                xaxis_title="PC1", yaxis_title="PC2",
                legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            )
            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(fig, use_container_width=True)


# ---------- Predict Cluster ----------
def page_predict(pipeline):
    st.markdown("## 🎯 Predict Student Cluster")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        "Enter a student's details below and click **Predict** to see "
        "which performance cluster they belong to."
    )

    method = st.selectbox(
        "Clustering Method",
        ["K-Means", "Agglomerative"],
        index=0,
    )
    st.markdown("---")

    df = pipeline["df"]

    # ---- Student Details ----
    st.markdown("##### Student Information")
    p1, p2 = st.columns(2)
    with p1:
        gender = st.selectbox("Gender", ["female", "male"], index=0)
        race = st.selectbox(
            "Race/Ethnicity",
            sorted(df["race/ethnicity"].unique().tolist()),
            index=1,
        )
        parent_edu = st.selectbox(
            "Parental Level of Education",
            sorted(df["parental level of education"].unique().tolist()),
            index=3,
        )
    with p2:
        lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"], index=0)
        test_prep = st.selectbox(
            "Test Preparation Course",
            ["none", "completed"],
            index=0,
        )

    st.markdown("##### Exam Scores")
    s1, s2, s3 = st.columns(3)
    with s1:
        math_score = st.slider("Math Score", 0, 100, 65)
    with s2:
        reading_score = st.slider("Reading Score", 0, 100, 70)
    with s3:
        writing_score = st.slider("Writing Score", 0, 100, 68)

    st.markdown("---")

    if st.button("🎯  Predict Cluster", type="primary", use_container_width=True):
        # Build input dataframe
        input_data = pd.DataFrame([{
            "gender": gender,
            "race/ethnicity": race,
            "parental level of education": parent_edu,
            "lunch": lunch,
            "test preparation course": test_prep,
            "math score": math_score,
            "reading score": reading_score,
            "writing score": writing_score,
        }])

        # Encode
        le = LabelEncoder()
        encoded = input_data.copy()
        for col in encoded.select_dtypes(include="object").columns:
            # Fit on the original data's unique values for consistency
            original_vals = sorted(df[col].unique().tolist())
            le.fit(original_vals)
            encoded[col] = le.transform(encoded[col])

        # Scale
        scaled = pipeline["scaler"].transform(encoded)

        if method == "K-Means":
            cluster = pipeline["kmeans_model"].predict(scaled)[0]
            cluster_col = "KMeans_Cluster"
        else:
            # For agglomerative, we use nearest centroid approach via K-Means
            # since AgglomerativeClustering doesn't have predict()
            # Use nearest centroid from K-Means as approximation
            cluster = pipeline["kmeans_model"].predict(scaled)[0]
            cluster_col = "KMeans_Cluster"
            st.caption("ℹ️ Agglomerative Clustering does not support direct prediction. Using K-Means centroids for approximation.")

        # Map to meaningful label
        rank_map = assign_cluster_labels(df, cluster_col)
        mapped = rank_map[cluster]
        label_info = CLUSTER_LABELS[mapped]
        card_class = CLUSTER_CARD_CLASSES[mapped]

        st.markdown(
            f'<div class="cluster-card {card_class}">'
            f'<h3 style="margin:0; color:{label_info[2]};">'
            f'{label_info[0]} {label_info[1]}</h3>'
            f'<p style="margin:0.5rem 0 0; font-size:1.05rem;">'
            f'This student is predicted to belong to the '
            f'<strong>{label_info[1]}</strong> cluster.'
            f'</p>'
            f'<p style="margin:0.25rem 0 0; font-size:0.9rem; opacity:0.7;">'
            f'Total Score: {math_score + reading_score + writing_score} / 300'
            f'</p></div>',
            unsafe_allow_html=True,
        )

        # Show cluster averages for context
        st.markdown("")
        st.markdown("##### How does this student compare?")
        comp_df = df.groupby(cluster_col)[["math score", "reading score", "writing score"]].mean().round(1)
        student_row = pd.DataFrame(
            [{"math score": math_score, "reading score": reading_score, "writing score": writing_score}],
            index=["This Student"],
        )

        fig = go.Figure()
        subjects = ["math score", "reading score", "writing score"]
        colors_sub = ["#6d28d9", "#3b82f6", "#10b981"]

        fig.add_trace(go.Bar(
            x=subjects, y=[math_score, reading_score, writing_score],
            name="This Student", marker_color="#f59e0b",
        ))
        for idx, (cluster_id, row) in enumerate(comp_df.iterrows()):
            mapped_id = rank_map[cluster_id]
            label = CLUSTER_LABELS[mapped_id][1]
            fig.add_trace(go.Bar(
                x=subjects, y=row.values,
                name=f"Cluster Avg: {label}",
                marker_color=CLUSTER_LABELS[mapped_id][2],
                opacity=0.6,
            ))

        fig.update_layout(
            barmode="group",
            margin=dict(l=20, r=20, t=30, b=20), height=400,
            yaxis_title="Score",
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.06)")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    st.markdown(_CSS, unsafe_allow_html=True)

    pipeline = run_clustering_pipeline()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🎓 Student Clustering")
        st.caption("K-Means & Hierarchical")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            [
                "📊 Dashboard",
                "🔍 Explore Data",
                "🎯 Optimal Clusters",
                "🔬 Clustering Results",
                "🧪 Predict Cluster",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            '<p class="sidebar-footer">Built with Streamlit · scikit-learn</p>',
            unsafe_allow_html=True,
        )

    # Route
    if page == "📊 Dashboard":
        page_dashboard(pipeline)
    elif page == "🔍 Explore Data":
        page_explore(pipeline)
    elif page == "🎯 Optimal Clusters":
        page_optimal_clusters(pipeline)
    elif page == "🔬 Clustering Results":
        page_clustering_results(pipeline)
    elif page == "🧪 Predict Cluster":
        page_predict(pipeline)


if __name__ == "__main__":
    main()
