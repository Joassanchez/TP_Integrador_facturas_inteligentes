"""Interfaz Streamlit para clasificacion y deteccion de anomalias en facturas.

Ejecutar:
  streamlit run app_streamlit.py

Estructura:
  Tab "Clasificador"  — Bloques 1, 2 y 3 del flujo principal
  Tab "Dashboard"     — Metricas y graficos de resultados experimentales
"""

import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuracion de pagina (debe ser el primer comando Streamlit)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Clasificador de Facturas",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_TIPOS_COMPROBANTE = [
    "Factura A",
    "Factura B",
    "Factura C",
    "Nota de Debito A",
    "Nota de Debito B",
    "Nota de Credito A",
    "Nota de Credito B",
    "Recibo X",
]

_UMBRAL_CONFIANZA_BAJA = 0.70


# ---------------------------------------------------------------------------
# Carga de modelos (cached — solo una vez por sesion)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelos...")
def _cargar_modelos():
    from src.modelo_inferencia import (
        predecir_factura,
        validar_consistencia,
        predecir_con_encoder,
        artefactos_disponibles,
    )
    return predecir_factura, validar_consistencia, predecir_con_encoder, artefactos_disponibles


# ---------------------------------------------------------------------------
# Bloque 1 — Modos de carga de factura
# ---------------------------------------------------------------------------

def _bloque_manual():
    with st.form("form_manual"):
        col1, col2 = st.columns(2)
        proveedor = col1.text_input("Proveedor", placeholder="Ej: SOLVER SYSTEM INFORMATICA")
        tipo = col2.selectbox("Tipo de comprobante", _TIPOS_COMPROBANTE)
        descripcion = st.text_input(
            "Descripcion",
            placeholder="Ej: Servicio de soporte tecnico mensual",
        )
        col3, col4 = st.columns(2)
        monto = col3.number_input("Monto ($)", min_value=0.0, step=100.0, format="%.2f")
        fecha = col4.date_input("Fecha", value=datetime.date.today())
        submitted = st.form_submit_button("Clasificar", type="primary", use_container_width=True)

    if submitted:
        if not proveedor.strip() or not descripcion.strip():
            st.warning("Completar al menos Proveedor y Descripcion.")
            return None
        return {
            "proveedor": proveedor.strip(),
            "descripcion": descripcion.strip(),
            "monto": monto,
            "tipo_comprobante": tipo,
            "fecha": fecha.isoformat(),
        }
    return None


def _bloque_archivo():
    archivo = st.file_uploader("Cargar archivo CSV o XLSX", type=["csv", "xlsx"])
    if archivo is None:
        return None, None

    try:
        from src.lector_csv_xlsx import cargar_desde_bytes
        ext = Path(archivo.name).suffix
        df = cargar_desde_bytes(archivo.read(), ext)
    except Exception as exc:
        st.error(f"Error al leer el archivo: {exc}")
        return None, None

    st.write(f"Archivo cargado: **{archivo.name}** — {len(df)} filas")
    st.dataframe(df.head(5), use_container_width=True)

    columnas_req = {"proveedor", "descripcion", "monto", "tipo_comprobante", "fecha"}
    faltantes = columnas_req - set(df.columns)
    if faltantes:
        st.warning(f"Columnas faltantes: {', '.join(sorted(faltantes))}")
        return None, None

    if len(df) == 1:
        return df.iloc[0].to_dict(), None

    opciones = ["Clasificar todas las filas"] + [f"Fila {i+1}" for i in range(len(df))]
    seleccion = st.selectbox("Que filas clasificar", opciones)

    if st.button("Clasificar", type="primary"):
        if seleccion == "Clasificar todas las filas":
            return None, df
        idx = int(seleccion.split()[-1]) - 1
        return df.iloc[idx].to_dict(), None

    return None, None


def _bloque_pdf():
    from src.extractor_pdfs import PDFPLUMBER_DISPONIBLE, extraer_desde_bytes

    if not PDFPLUMBER_DISPONIBLE:
        st.warning("pdfplumber no esta instalado. Instalar con: pip install pdfplumber")
        return None

    archivo = st.file_uploader("Cargar factura PDF", type=["pdf"])
    if archivo is None:
        return None

    with st.spinner("Extrayendo campos del PDF..."):
        campos = extraer_desde_bytes(archivo.read())

    if campos["errores"]:
        for err in campos["errores"]:
            st.error(err)
        return None

    st.success("Campos extraidos del PDF — verificar antes de clasificar:")

    with st.form("form_pdf"):
        col1, col2 = st.columns(2)
        proveedor = col1.text_input("Proveedor", value=campos["proveedor"] or "")
        tipo_default = campos["tipo_comprobante"] or _TIPOS_COMPROBANTE[0]
        tipo_idx = (
            _TIPOS_COMPROBANTE.index(tipo_default)
            if tipo_default in _TIPOS_COMPROBANTE
            else 0
        )
        tipo = col2.selectbox("Tipo de comprobante", _TIPOS_COMPROBANTE, index=tipo_idx)
        descripcion = st.text_input("Descripcion", value=campos["descripcion"] or "")
        col3, col4 = st.columns(2)
        monto = col3.number_input(
            "Monto ($)",
            value=float(campos["monto"] or 0.0),
            min_value=0.0,
            step=100.0,
            format="%.2f",
        )
        fecha_str = campos["fecha"] or datetime.date.today().isoformat()
        try:
            fecha_parsed = datetime.date.fromisoformat(fecha_str.replace("/", "-"))
        except Exception:
            fecha_parsed = datetime.date.today()
        fecha = col4.date_input("Fecha", value=fecha_parsed)
        submitted = st.form_submit_button("Clasificar", type="primary", use_container_width=True)

    if submitted:
        return {
            "proveedor": proveedor.strip(),
            "descripcion": descripcion.strip(),
            "monto": monto,
            "tipo_comprobante": tipo,
            "fecha": fecha.isoformat(),
        }
    return None


# ---------------------------------------------------------------------------
# Bloque 3 — Visualizacion de resultados
# ---------------------------------------------------------------------------

def _mostrar_resultado_individual(pred: dict, anomalia: dict, experimento: str) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Rubro", pred["rubro"])
    col2.metric("Subrubro", pred["subrubro"])
    col3.metric("Confianza", f"{pred['confianza']:.1%}")

    requiere_revision = pred["confianza"] < _UMBRAL_CONFIANZA_BAJA

    if requiere_revision:
        st.warning(
            f"Confianza baja ({pred['confianza']:.1%}). Se recomienda revision manual."
        )

    if "1" in experimento:
        if anomalia.get("disponible"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Error de Reconstruccion", f"{anomalia['error_reconstruccion']:.5f}")
            c2.metric("Umbral", f"{anomalia['umbral']:.5f}")
            c3.metric("Estado", anomalia["estado"])

            if anomalia["es_atipico"]:
                st.error(
                    f"Factura ATIPICA — el error de reconstruccion supera el umbral definido."
                )
                requiere_revision = True
            else:
                st.success("Factura dentro del patron normal.")
        else:
            st.info(
                "Deteccion de anomalias no disponible — "
                "autoencoder.keras no encontrado en models/."
            )
    else:
        encoder_msg = (
            "encoder.keras (Experimento 2)"
            if pred.get("encoder_usado")
            else "clasificador directo (encoder.keras no encontrado — fallback a Exp. 1)"
        )
        st.caption(f"Pipeline: {encoder_msg}")

    st.metric("Requiere Revision", "Si" if requiere_revision else "No")

    with st.expander("Ver top-10 probabilidades por categoria"):
        probs = pred.get("probabilidades", {})
        if probs:
            df_p = (
                pd.DataFrame(list(probs.items()), columns=["Categoria", "Probabilidad"])
                .sort_values("Probabilidad", ascending=False)
                .head(10)
            )
            df_p["Probabilidad"] = df_p["Probabilidad"].apply(lambda x: f"{x:.2%}")
            st.dataframe(df_p, use_container_width=True, hide_index=True)


def _predecir_individual(row: dict, experimento: str) -> None:
    try:
        predecir_factura, validar_consistencia, predecir_con_encoder, _ = _cargar_modelos()
    except Exception as exc:
        st.error(f"Error cargando modelos: {exc}")
        return

    with st.spinner("Clasificando..."):
        try:
            if "1" in experimento:
                pred = predecir_factura(row)
                anomalia = validar_consistencia(row)
            else:
                pred = predecir_con_encoder(row)
                anomalia = {"disponible": False}
        except Exception as exc:
            st.error(f"Error durante la prediccion: {exc}")
            return

    _mostrar_resultado_individual(pred, anomalia, experimento)


def _predecir_batch(df: pd.DataFrame, experimento: str) -> None:
    try:
        predecir_factura, validar_consistencia, predecir_con_encoder, _ = _cargar_modelos()
    except Exception as exc:
        st.error(f"Error cargando modelos: {exc}")
        return

    resultados = []
    barra = st.progress(0, text="Clasificando facturas...")
    n = len(df)

    for i, (_, fila) in enumerate(df.iterrows()):
        row = fila.to_dict()
        try:
            if "1" in experimento:
                pred = predecir_factura(row)
                anomalia = validar_consistencia(row)
            else:
                pred = predecir_con_encoder(row)
                anomalia = {"disponible": False}

            resultados.append({
                "Proveedor": str(row.get("proveedor", ""))[:40],
                "Descripcion": str(row.get("descripcion", ""))[:55],
                "Rubro": pred["rubro"],
                "Subrubro": pred["subrubro"],
                "Confianza": f"{pred['confianza']:.1%}",
                "Estado": anomalia.get("estado", "—") if anomalia.get("disponible") else "—",
                "Requiere Revision": (
                    "Si" if anomalia.get("requiere_revision") or pred["confianza"] < _UMBRAL_CONFIANZA_BAJA
                    else "No"
                ),
            })
        except Exception as exc:
            resultados.append({
                "Proveedor": str(row.get("proveedor", ""))[:40],
                "Descripcion": str(row.get("descripcion", ""))[:55],
                "Rubro": "ERROR",
                "Subrubro": str(exc)[:50],
                "Confianza": "—",
                "Estado": "—",
                "Requiere Revision": "—",
            })

        barra.progress((i + 1) / n, text=f"Clasificando... {i+1}/{n}")

    barra.empty()
    df_res = pd.DataFrame(resultados)
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    csv_bytes = df_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar resultados (CSV)",
        data=csv_bytes,
        file_name="resultados_clasificacion.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Tab: Clasificador
# ---------------------------------------------------------------------------

def _render_clasificador() -> None:
    # Bloque 1 — Carga
    st.subheader("1. Cargar Factura")
    modo = st.radio(
        "modo_carga",
        ["Manual", "CSV / XLSX", "PDF"],
        horizontal=True,
        label_visibility="collapsed",
    )

    factura_row = None
    filas_batch = None

    if modo == "Manual":
        factura_row = _bloque_manual()
    elif modo == "CSV / XLSX":
        factura_row, filas_batch = _bloque_archivo()
    else:
        factura_row = _bloque_pdf()

    st.divider()

    # Bloque 2 — Experimento
    st.subheader("2. Seleccion de Experimento")
    experimento = st.radio(
        "experimento",
        [
            "Experimento 1 — Clasificador + Autoencoder",
            "Experimento 2 — Encoder + Clasificador",
        ],
        label_visibility="collapsed",
    )

    if "1" in experimento:
        st.caption(
            "Pipeline: clasificador supervisado (rubro/subrubro) → "
            "autoencoder (error de reconstruccion y estado Normal/Atipico)."
        )
    else:
        st.caption(
            "Pipeline: encoder (representacion latente) → "
            "clasificador supervisado (rubro/subrubro y confianza)."
        )

    st.divider()

    # Bloque 3 — Resultados
    st.subheader("3. Resultado")

    if filas_batch is not None:
        _predecir_batch(filas_batch, experimento)
    elif factura_row is not None:
        _predecir_individual(factura_row, experimento)
    else:
        st.info("Ingresa o carga una factura para obtener la prediccion.")


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

st.title("Clasificador Inteligente de Facturas")
st.caption("TP Integrador — Sistemas Inteligentes  |  Sanchez, Nunez, Geneyro")

tab_pred, tab_dash = st.tabs(["Clasificador", "Dashboard de Resultados"])

with tab_pred:
    _render_clasificador()

with tab_dash:
    from dashboard_resultados import render_dashboard
    render_dashboard()
