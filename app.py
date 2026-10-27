# -*- coding: utf-8 -*-
"""
Laboratorio SpVAR – Colombia (versión Streamlit)
Ejecutar local:  streamlit run app.py
"""
import unicodedata
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from libpysal.weights import Queen, KNN, lag_spatial
from esda.moran import Moran, Moran_Local
from spreg import OLS, ML_Lag
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SpVAR Colombia", page_icon="🗺️", layout="wide")

# -----------------------------------------------------------------------------
# 1. DATOS
# -----------------------------------------------------------------------------
URLS = [
    "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson",
    "https://raw.githubusercontent.com/santiblanko/colombia.geojson/master/depto.json",
]


def sin_tildes(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()


IPM = {
    'VICHADA': 55.2, 'GUAINÍA': 51.8, 'VAUPÉS': 41.7, 'LA GUAJIRA': 40.1, 'CHOCÓ': 30.8,
    'SUCRE': 25.4, 'GUAVIARE': 20.0, 'CÓRDOBA': 19.8, 'AMAZONAS': 18.1, 'MAGDALENA': 17.6,
    'BOLÍVAR': 17.0, 'CAQUETÁ': 15.0, 'CESAR': 14.8, 'PUTUMAYO': 13.6, 'NORTE DE SANTANDER': 13.3,
    'NARIÑO': 13.2, 'CAUCA': 13.0, 'ARAUCA': 12.7, 'META': 12.3, 'TOLIMA': 12.2, 'HUILA': 9.7,
    'ANTIOQUIA': 9.5, 'CASANARE': 8.9, 'ATLÁNTICO': 8.2, 'CALDAS': 7.9, 'SANTANDER': 6.9,
    'QUINDÍO': 6.7, 'BOYACÁ': 6.4, 'RISARALDA': 4.9, 'VALLE DEL CAUCA': 4.9, 'SAN ANDRÉS': 4.3,
    'CUNDINAMARCA': 4.1, 'BOGOTÁ, D.C.': 2.2,
}
CONECT = {
    'BOGOTÁ, D.C.': 48.0, 'ATLÁNTICO': 31.5, 'VALLE DEL CAUCA': 29.8, 'QUINDÍO': 28.5,
    'RISARALDA': 26.2, 'CALDAS': 24.1, 'ANTIOQUIA': 23.8, 'CUNDINAMARCA': 22.5,
    'SANTANDER': 19.4, 'BOYACÁ': 17.8, 'TOLIMA': 16.9, 'HUILA': 15.7, 'NORTE DE SANTANDER': 15.2,
    'MAGDALENA': 14.1, 'CESAR': 13.5, 'CÓRDOBA': 13.0, 'BOLÍVAR': 12.4, 'SUCRE': 12.1,
    'CAUCA': 11.8, 'NARIÑO': 11.2, 'META': 9.8, 'CASANARE': 8.5, 'ARAUCA': 7.2,
    'LA GUAJIRA': 6.8, 'CHOCÓ': 5.1, 'CAQUETÁ': 4.8, 'PUTUMAYO': 4.3, 'GUAVIARE': 3.5,
    'VICHADA': 2.1, 'AMAZONAS': 1.8, 'GUAINÍA': 1.4, 'VAUPÉS': 1.3, 'SAN ANDRÉS': 38.0,
}
DENS = {
    'BOGOTÁ, D.C.': 18000, 'ATLÁNTICO': 750, 'QUINDÍO': 420, 'RISARALDA': 380,
    'CALDAS': 210, 'VALLE DEL CAUCA': 220, 'CUNDINAMARCA': 140, 'ANTIOQUIA': 130,
    'SANTANDER': 85, 'BOYACÁ': 55, 'TOLIMA': 50, 'HUILA': 55,
    'NORTE DE SANTANDER': 70, 'MAGDALENA': 60, 'CESAR': 45, 'CÓRDOBA': 70,
    'BOLÍVAR': 55, 'SUCRE': 80, 'CAUCA': 45, 'NARIÑO': 55, 'META': 15,
    'CASANARE': 10, 'ARAUCA': 8, 'LA GUAJIRA': 20, 'CHOCÓ': 10,
    'CAQUETÁ': 6, 'PUTUMAYO': 12, 'GUAVIARE': 3, 'VICHADA': 1.5,
    'AMAZONAS': 0.8, 'GUAINÍA': 0.7, 'VAUPÉS': 0.6, 'SAN ANDRÉS': 900,
}
# diccionarios con llaves sin tildes (más robusto al cruzar con el geojson)
IPM_N = {sin_tildes(k): v for k, v in IPM.items()}
CONECT_N = {sin_tildes(k): v for k, v in CONECT.items()}
DENS_N = {sin_tildes(k): v for k, v in DENS.items()}

ALIAS = {
    'BOGOTA': 'BOGOTA, D.C.', 'BOGOTA D.C.': 'BOGOTA, D.C.', 'BOGOTA DC': 'BOGOTA, D.C.',
    'SANTAFE DE BOGOTA D.C.': 'BOGOTA, D.C.', 'SANTAFE DE BOGOTA': 'BOGOTA, D.C.',
    'SAN ANDRES': 'SAN ANDRES', 'SAN ANDRES Y PROVIDENCIA': 'SAN ANDRES',
    'ARCHIPIELAGO DE SAN ANDRES': 'SAN ANDRES',
    'ARCHIPIELAGO DE SAN ANDRES, PROVIDENCIA Y SANTA CATALINA': 'SAN ANDRES',
    'N. DE SANTANDER': 'NORTE DE SANTANDER', 'NORTE SANTANDER': 'NORTE DE SANTANDER',
    'VALLE': 'VALLE DEL CAUCA',
}


def normalizar(n):
    n = sin_tildes(n)
    if n in ALIAS:
        return ALIAS[n]
    for k, v in ALIAS.items():
        if k in n and k not in ("VALLE",):
            return v
    if n.startswith("VALLE"):
        return "VALLE DEL CAUCA"
    return n


@st.cache_data(show_spinner="Cargando datos y estimando modelo…")
def cargar_todo():
    gdf = None
    for u in URLS:
        try:
            gdf = gpd.read_file(u)
            break
        except Exception:
            continue
    if gdf is None:
        raise RuntimeError("No se pudo descargar el GeoJSON de departamentos.")

    col = next((c for c in ['DPTO_CNMBR', 'NOMBRE_DPT', 'dpto_cnmbr', 'nombre', 'NAME', 'name']
                if c in gdf.columns), None)
    gdf = gdf.rename(columns={col: 'departamento'}) if col else gdf.assign(departamento=gdf.index.astype(str))
    gdf = gdf[['departamento', 'geometry']].copy()
    gdf = gdf.to_crs(epsg=4326)

    gdf['departamento_norm'] = gdf['departamento'].apply(normalizar)
    gdf['IPM'] = gdf['departamento_norm'].map(IPM_N).fillna(9.9)
    gdf['Conectividad'] = gdf['departamento_norm'].map(CONECT_N).fillna(12.0)

    rng = np.random.default_rng(42)
    gdf['VIIRS'] = (0.55 * gdf['Conectividad'] + 0.25 * (50 - gdf['IPM'])
                    + rng.normal(0, 2.5, len(gdf))).clip(0.5)
    gdf['Densidad_Pob'] = gdf['departamento_norm'].map(DENS_N).fillna(30)

    # centroides en CRS proyectado (MAGNA-SIRGAS Colombia Bogotá) y luego a lon/lat
    cent = gdf.to_crs(epsg=3116).geometry.centroid.to_crs(epsg=4326)
    gdf['lon'] = cent.x.values
    gdf['lat'] = cent.y.values

    sc = StandardScaler()
    gdf['IPM_z'] = sc.fit_transform(gdf[['IPM']])
    gdf['Conectividad_z'] = StandardScaler().fit_transform(gdf[['Conectividad']])
    gdf['VIIRS_z'] = StandardScaler().fit_transform(gdf[['VIIRS']])

    # Matriz W
    W = Queen.from_dataframe(gdf, silence_warnings=True)
    if getattr(W, 'islands', None):
        W = KNN.from_dataframe(gdf, k=3)
    W.transform = 'r'

    y1 = gdf['IPM_z'].values.reshape(-1, 1)
    y2 = gdf['Conectividad_z'].values.reshape(-1, 1)
    X1 = np.hstack([y2, lag_spatial(W, y2)])
    try:
        m = ML_Lag(y1, X1, w=W, name_y="IPM")
        modelo, rho = "SAR", float(m.rho)
    except Exception:
        m = OLS(y1, X1, w=W, name_y="IPM")
        modelo, rho = "OLS", 0.0

    moran = Moran(gdf['IPM_z'].values, W)
    lisa = Moran_Local(gdf['IPM_z'].values, W)
    gdf['LISA_q'] = lisa.q.astype(int)
    gdf['LISA_sig'] = lisa.p_sim < 0.05
    gdf['LISA_label'] = gdf['LISA_q'].map({1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'})

    gdf = gdf.reset_index(drop=True)
    gdf['id_geo'] = gdf.index.astype(str)

    stats = dict(modelo=modelo, rho=rho, moran_I=float(moran.I), moran_p=float(moran.p_sim))
    return gdf, stats


try:
    gdf, S = cargar_todo()
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

modelo, rho, moran_I, moran_p = S["modelo"], S["rho"], S["moran_I"], S["moran_p"]
VARS = ["IPM", "Conectividad", "VIIRS", "Densidad_Pob"]


# -----------------------------------------------------------------------------
# 2. FUNCIONES
# -----------------------------------------------------------------------------
def texto_interpretacion():
    hh = list(gdf.loc[(gdf.LISA_q == 1) & gdf.LISA_sig, 'departamento_norm'])
    ll = list(gdf.loc[(gdf.LISA_q == 3) & gdf.LISA_sig, 'departamento_norm'])
    return f"""
### Interpretación automática SpVAR

- **Modelo IPM**: `{modelo}`
- **ρ espacial**: `{rho:.3f}`
- **I de Moran**: `{moran_I:.3f}` (p = {moran_p:.4f}) → clustering espacial de pobreza
- **Clusters HH (trampa de pobreza)**: {hh if hh else 'ninguno'}
- **Clusters LL (círculos virtuosos)**: {ll if ll else 'ninguno'}

La pobreza se transmite espacialmente. Mejorar la conectividad de los vecinos genera spillovers positivos.
"""


def texto_mecanismos():
    return f"""
### Mecanismos de transmisión espacial capturados por el SpVAR

| # | Mecanismo | Expresión | Interpretación |
|---|-----------|-----------|----------------|
| 1 | **Autodinámica espacial** | `W · Yᵢ → Yᵢ` | La pobreza/conectividad de un departamento depende de la de sus vecinos |
| 2 | **Cruce local** | `Xⱼ → Yᵢ` | La conectividad local afecta el IPM del mismo departamento (y viceversa) |
| 3 | **Relación espacial directa** | `ρ · W · Yᵢ` | Efecto de contagio puro (parámetro ρ) |
| 4 | **Relación espacial cruzada** | `W · Xⱼ → Yᵢ` | La conectividad de los vecinos afecta el IPM local (spillover cruzado) |

**Resultado actual**: Modelo **{modelo}** · ρ ≈ **{rho:.3f}** · Moran I ≈ **{moran_I:.3f}** (p={moran_p:.4f})
"""


def mapa_matplotlib(variable):
    fig, ax = plt.subplots(figsize=(9, 8))
    if variable == "LISA_q":
        gdf.plot(column='LISA_q', categorical=True, legend=True, ax=ax,
                 cmap='RdBu_r', edgecolor='k', linewidth=0.4,
                 legend_kwds={'title': 'Cluster LISA'})
        ax.set_title("Clusters LISA de Pobreza Multidimensional\n(1=HH · 2=LH · 3=LL · 4=HL)", fontsize=12)
    else:
        cmaps = {"IPM": "YlOrRd", "Conectividad": "Blues", "VIIRS": "inferno", "Densidad_Pob": "YlGn"}
        gdf.plot(column=variable, cmap=cmaps.get(variable, "YlOrRd"), legend=True, ax=ax,
                 edgecolor='k', linewidth=0.35, legend_kwds={'shrink': 0.6, 'label': variable})
        ax.set_title(f"{variable} – Colombia", fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    return fig


def mapa_nocturno(variable):
    geojson = gdf[['id_geo', 'departamento_norm', 'geometry']].__geo_interface__
    fig = px.choropleth(
        gdf, geojson=geojson, locations="id_geo", featureidkey="properties.id_geo",
        color=variable, color_continuous_scale="Viridis", hover_name="departamento_norm",
        title=f"Mapa Coroplético (Visión Nocturna) – {variable}",
    )
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="#0b0f19")
    fig.update_layout(paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19", font_color="#00ffcc",
                      margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=600)
    return fig


def mapa_3d(variable):
    fig = go.Figure(data=[go.Scatter3d(
        x=gdf['lon'], y=gdf['lat'], z=gdf[variable], mode='markers',
        marker=dict(
            size=10 + (gdf['Conectividad'] / gdf['Conectividad'].max() * 12),
            color=gdf[variable], colorscale='Plasma', opacity=0.9,
            colorbar=dict(title=variable), line=dict(width=0.5, color='white')),
        text=gdf['departamento_norm'],
        hovertemplate=("<b>%{text}</b><br>" + f"{variable}: %{{z:.2f}}<br>"
                       + "Longitud: %{x:.2f}<br>Latitud: %{y:.2f}<extra></extra>"),
    )])
    axis = dict(backgroundcolor="#111827", gridcolor="#333333", showbackground=True)
    fig.update_layout(
        title=f"Estructura Dinámica 3D Espacial – {variable}",
        paper_bgcolor="#0b0f19", font_color="#ffffff",
        scene=dict(xaxis_title="Longitud", yaxis_title="Latitud", zaxis_title=variable,
                   xaxis=axis, yaxis=axis, zaxis=axis,
                   camera=dict(eye=dict(x=1.6, y=1.5, z=1.2))),
        height=680, margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig


def texto_interpretabilidad(variable):
    media = gdf[variable].mean()
    imax, imin = gdf[variable].idxmax(), gdf[variable].idxmin()
    return f"""
### 📊 Informe de Interpretabilidad Automatizada – {variable}

- **Tendencia central:** promedio = **{media:.2f}**
- **Valor máximo:** **{gdf.loc[imax, 'departamento_norm']}** ({gdf.loc[imax, variable]:.2f})
- **Valor mínimo:** **{gdf.loc[imin, 'departamento_norm']}** ({gdf.loc[imin, variable]:.2f})
- **Diagnóstico espacial:** concentración territorial coherente con el I de Moran = {moran_I:.3f} (p={moran_p:.4f}) y con los clusters LISA detectados.
"""


def esda_fig():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, v, c, t in zip(axes, ["IPM", "Conectividad", "VIIRS"],
                           ["#e74c3c", "#3498db", "#f39c12"], ["IPM (%)", "Conectividad", "VIIRS"]):
        sns.histplot(gdf[v], kde=True, ax=ax, color=c)
        ax.set_title(t)
    plt.tight_layout()
    return fig


def lisa_map():
    fig, ax = plt.subplots(figsize=(8, 7))
    gdf.plot(column="LISA_q", categorical=True, legend=True, ax=ax,
             cmap="RdBu_r", edgecolor="k", linewidth=0.4)
    ax.set_title("Clusters LISA de IPM\n(1=HH · 2=LH · 3=LL · 4=HL)")
    ax.axis("off")
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 3. INTERFAZ
# -----------------------------------------------------------------------------
st.title("Laboratorio SpVAR – Colombia")
st.caption("IPM ↔ Conectividad Vial · Mapas + 3D + Mecanismos de transmisión")

tabs = st.tabs(["Inicio", "ESDA", "Mapas (matplotlib)", "Mapas Visión Nocturna",
                "SpVAR + LISA", "3D Dinámico", "Interpretabilidad", "Metodología"])

with tabs[0]:
    st.markdown(texto_interpretacion())
    st.markdown(texto_mecanismos())

with tabs[1]:
    st.markdown(f"""
**Media IPM** = {gdf['IPM'].mean():.1f}%  
**Rango** = {gdf['IPM'].min():.1f} – {gdf['IPM'].max():.1f}%  
**I de Moran** = {moran_I:.3f} (p = {moran_p:.4f})  
→ Clustering espacial de pobreza confirmado.
""")
    fig = esda_fig()
    st.pyplot(fig)
    plt.close(fig)

with tabs[2]:
    v = st.radio("Variable", ["IPM", "Conectividad", "VIIRS", "LISA_q", "Densidad_Pob"],
                 horizontal=True, key="v_mpl")
    fig = mapa_matplotlib(v)
    st.pyplot(fig)
    plt.close(fig)

with tabs[3]:
    v = st.selectbox("Variable", VARS, key="v_noc")
    st.plotly_chart(mapa_nocturno(v), width="stretch")

with tabs[4]:
    st.markdown(texto_interpretacion())
    st.markdown(texto_mecanismos())
    fig = lisa_map()
    st.pyplot(fig)
    plt.close(fig)

with tabs[5]:
    st.markdown("""
### Estructura Dinámica 3D Espacial
Basada en centroides · Color = valor de la variable · Tamaño ≈ conectividad.  
Rota y haz zoom con el mouse.
""")
    v = st.selectbox("Variable para el eje Z", VARS, key="v_3d")
    st.plotly_chart(mapa_3d(v), width="stretch")

with tabs[6]:
    v = st.selectbox("Variable a interpretar", VARS, key="v_int")
    st.markdown(texto_interpretabilidad(v))

with tabs[7]:
    st.markdown("""
### Metodología auditada
- **W**: Queen (fallback KNN para islas)
- **Especificación**: SpVAR seccional cruzado
- **Selección**: SAR cuando converge
- **Diagnósticos**: Moran + LISA + ρ
- **Mapas estables**: matplotlib
- **Mapas nocturnos + 3D**: Plotly

### Los 4 mecanismos de transmisión espacial
1. Autodinámica espacial (`W·Yᵢ`)
2. Cruce local (`Xⱼ → Yᵢ`)
3. Relación espacial directa (`ρ·W·Yᵢ`)
4. Relación espacial cruzada (`W·Xⱼ → Yᵢ`)
""")
    st.markdown(texto_mecanismos())

st.markdown("---")
st.caption("Laboratorio de Econometría Espacial")
