"""Dashboard de métricas y resultados experimentales.

Exporta render_dashboard() para ser llamado desde app_streamlit.py.
También puede ejecutarse como página independiente:
  streamlit run dashboard_resultados.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

_TRAINING_REPORT = Path("results/training_report.json")
_QUALITY_REPORT = Path("results/quality_report.json")


# ---------------------------------------------------------------------------
# Punto de entrada público
# ---------------------------------------------------------------------------

def render_dashboard() -> None:
    """Renderiza el dashboard completo dentro de una página Streamlit."""
    st.header("Dashboard de Resultados")

    if not _TRAINING_REPORT.exists():
        st.warning(
            "No se encontró results/training_report.json. "
            "Ejecutar primero: python -m src.training_pipeline"
        )
        return

    with open(_TRAINING_REPORT, encoding="utf-8") as f:
        report = json.load(f)

    _mostrar_metricas_principales(report)
    st.divider()
    _mostrar_curvas_entrenamiento(report)
    st.divider()
    _mostrar_distribucion_clases(report)
    st.divider()
    _mostrar_tabla_metricas_completa(report)
    st.divider()
    _mostrar_info_arquitectura(report)


# ---------------------------------------------------------------------------
# Secciones del dashboard
# ---------------------------------------------------------------------------

def _mostrar_metricas_principales(report: dict) -> None:
    st.subheader("Metricas en Conjunto de Test")
    m = report.get("metrics", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{m.get('accuracy', 0):.2%}")
    col2.metric("F1 Macro", f"{m.get('f1_macro', 0):.2%}")
    col3.metric("Precision Macro", f"{m.get('precision_macro', 0):.2%}")
    col4.metric("Recall Macro", f"{m.get('recall_macro', 0):.2%}")

    st.caption(
        f"Early stopping en época {report.get('stopped_epoch', '?')}  |  "
        f"Features: {report.get('architecture', {}).get('input_dim', '?')}  |  "
        f"Clases: {report.get('architecture', {}).get('output_dim', '?')}"
    )


def _mostrar_curvas_entrenamiento(report: dict) -> None:
    st.subheader("Curvas de Entrenamiento")
    history = report.get("history", {})
    if not history:
        st.info("Historial de entrenamiento no disponible.")
        return

    try:
        import plotly.graph_objects as go
        _curvas_plotly(history)
    except ImportError:
        _curvas_nativa(history)


def _curvas_plotly(history: dict) -> None:
    import plotly.graph_objects as go

    epochs = list(range(1, len(history.get("loss", [])) + 1))
    tab_loss, tab_acc = st.tabs(["Perdida (Loss)", "Accuracy"])

    with tab_loss:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=history.get("loss", []),
            name="Train", line=dict(color="#1f77b4"),
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=history.get("val_loss", []),
            name="Validacion", line=dict(color="#ff7f0e", dash="dash"),
        ))
        fig.update_layout(
            xaxis_title="Epoca",
            yaxis_title="Loss",
            height=350,
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_acc:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=history.get("accuracy", []),
            name="Train", line=dict(color="#2ca02c"),
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=history.get("val_accuracy", []),
            name="Validacion", line=dict(color="#d62728", dash="dash"),
        ))
        fig.update_layout(
            xaxis_title="Epoca",
            yaxis_title="Accuracy",
            yaxis=dict(tickformat=".0%"),
            height=350,
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)


def _curvas_nativa(history: dict) -> None:
    """Fallback con line_chart nativo de Streamlit cuando plotly no está."""
    epochs = list(range(1, len(history.get("loss", [])) + 1))
    tab_loss, tab_acc = st.tabs(["Perdida (Loss)", "Accuracy"])
    with tab_loss:
        df = pd.DataFrame({
            "Epoca": epochs,
            "Train Loss": history.get("loss", []),
            "Val Loss": history.get("val_loss", []),
        }).set_index("Epoca")
        st.line_chart(df)
    with tab_acc:
        df = pd.DataFrame({
            "Epoca": epochs,
            "Train Acc": history.get("accuracy", []),
            "Val Acc": history.get("val_accuracy", []),
        }).set_index("Epoca")
        st.line_chart(df)


def _mostrar_distribucion_clases(report: dict) -> None:
    st.subheader("Distribucion de Clases en el Dataset")
    class_dist = report.get("class_distribution", {})
    if not class_dist:
        st.info("Distribucion de clases no disponible.")
        return

    clases = list(class_dist.keys())
    totales = [
        sum(v.values()) if isinstance(v, dict) else int(v)
        for v in class_dist.values()
    ]
    df = pd.DataFrame({"Categoria": clases, "Total": totales}).sort_values(
        "Total", ascending=False
    )

    try:
        import plotly.express as px
        fig = px.bar(
            df,
            x="Total",
            y="Categoria",
            orientation="h",
            height=max(420, len(clases) * 22),
            color="Total",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(df.set_index("Categoria")["Total"])


def _mostrar_tabla_metricas_completa(report: dict) -> None:
    st.subheader("Tabla de Metricas")
    m = report.get("metrics", {})
    if not m:
        return
    filas = [
        {"Metrica": k.replace("_", " ").title(), "Valor": round(v, 4)}
        for k, v in m.items()
    ]
    st.dataframe(
        pd.DataFrame(filas),
        use_container_width=True,
        hide_index=True,
    )


def _mostrar_info_arquitectura(report: dict) -> None:
    st.subheader("Arquitectura del Modelo")
    arch = report.get("architecture", {})
    hp = report.get("hyperparameters", {})
    if not arch:
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Capas**")
        for capa in arch.get("layers", []):
            st.markdown(f"- {capa}")
    with col2:
        st.markdown("**Hiperparametros**")
        for k, v in hp.items():
            st.markdown(f"- {k.replace('_', ' ').title()}: `{v}`")
        shapes = report.get("shapes", {})
        if shapes:
            st.markdown("**Shapes**")
            for split, shape in shapes.items():
                st.markdown(f"- {split}: `{shape}`")


# ---------------------------------------------------------------------------
# Ejecución independiente
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(page_title="Dashboard — Facturas Inteligentes", layout="wide")
    render_dashboard()
