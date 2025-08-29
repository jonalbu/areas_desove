import os
import io
import subprocess
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point
from geopandas.tools import sjoin_nearest
import zipfile
import tempfile

# Configuración de la app
st.set_page_config(page_title="Rutas Acuáticas", layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("<h1>Estimación Áreas de desove de Peces dulceacuícolas para la cuenca Magdalena-Cauca</h1>", unsafe_allow_html=True)

st.sidebar.image("Logo/002.jpg", use_container_width=True)
st.sidebar.image("Logo/004.jpg", use_container_width=True)
st.sidebar.image("Logo/007.jpg", use_container_width=True)
st.sidebar.markdown("""<p id="autor"> Foto por Jose L. Londoño Lopez</p>""", unsafe_allow_html=True)

# Instrucciones de formato de datos
st.markdown("""
<p id="in"> Se requiere que el archivo de Excel de entrada incluya las siguientes columnas, utilizando los mismos nombres de encabezado:</p>

<ul>  <strong id="in"> Sitios de colecta (Excel, .xlsx):</strong>
  <li> <code> sample_id:</code> identificador único de cada muestra.</li>
  <li> <code> place_name:</code> nombre del sitio de colecta.</li>
  <li> <code> latitude:</code> coordenada de latitud del sitio de colecta en WGS84 (p.e. 7,2345678).</li>
  <li> <code> longitude:</code> coordenada de longitud del sitio de colecta en WGS84. (p.e. -73,4567890).</li>
  <li> <code> species_name:</code> nombre de la especie.</li>
  <li> <code> min_time:</code> tiempo mínimo en horas de desarrollo de ictioplancton colectado.</li>
  <li> <code> max_time:</code> tiempo máximo en horas de desarrollo de ictioplancton colectado.</li>
</ul>

<p id="nota"> <strong> NOTA:</strong> Asegúrese de que los nombres de columnas coincidan exactamente.</p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------
# Parámetros de filtrado
# ---------------------------------------------------------------------------------
st.sidebar.header("Parámetros de Filtrado")

# Cargar shapefile de centrales para obtener posibles estados
def_shp_hydro = "DBase_Proyectos_Hidroelectricos_Magdalena/Proyectos_Hidroelectricos.shp"
try:
    hydro_tmp = gpd.read_file(def_shp_hydro)
    hydro_tmp.columns = hydro_tmp.columns.str.strip().str.lower().str.replace(' ', '_')
    status_vals = hydro_tmp['status'].unique().tolist()
except Exception:
    status_vals = ["operativa", "planeado", "en construcción"]  # valores por defecto

status_sel = st.sidebar.multiselect(
    "Estatus de centrales",
    options=status_vals,
    default=status_vals
)

elev_max = st.sidebar.slider("Elevación máxima (m)", 0, 5000, 1200)
grid_min = st.sidebar.slider("Strahler mínimo", 0, 10, 2)

# Subida de Excel de sitios
xls_file = st.sidebar.file_uploader("Subir Excel de sitios de Colecta (.xlsx)", type=["xlsx"])

# Rutas fijas de shapefiles auxiliares
def_shp_arcs = "SHP_Magdalena/Con_altitud/Red_Magdalena_n.shp"

# ---------------------------------------------------------------------------------
# “Ejecutar Análisis”
# ---------------------------------------------------------------------------------
if st.sidebar.button("Ejecutar Análisis"):
    # Validar que se subió Excel
    if xls_file is None:
        st.error("Por favor suba el archivo Excel de sitios.")
        st.stop()

    # ---------------------------------------------------------------------------------
    # 1. Carga de todos los datos
    # ---------------------------------------------------------------------------------
    arcs  = gpd.read_file(def_shp_arcs)        # Red hidro
    sites = pd.read_excel(xls_file, engine='openpyxl')  # Excel de sitios
    hydro = gpd.read_file(def_shp_hydro)       # Centrales hidroeléctricas

    # ---------------------------------------------------------------------------------
    # 2. Normalizar nombre de columnas
    # ---------------------------------------------------------------------------------
    arcs.columns  = arcs.columns.str.strip().str.lower().str.replace(' ', '_')
    sites.columns = sites.columns.str.strip().str.lower().str.replace(' ', '_')
    hydro.columns = hydro.columns.str.strip().str.lower().str.replace(' ', '_')

    # ---------------------------------------------------------------------------------
    # 3. Filtrar centrales según estados seleccionados
    # ---------------------------------------------------------------------------------
    if status_sel:
        hydro = hydro[hydro['status'].isin(status_sel)]

    # ---------------------------------------------------------------------------------
    # 4. Filtrar arcos por elevación y Strahler
    # ---------------------------------------------------------------------------------
    darcs = arcs[(arcs['elevmed'] <= elev_max) & (arcs['grid_code'] >= grid_min)].copy()
    # El peso “time” ya está en horas:
    darcs['weight_time'] = darcs['time']

    # Longitud de cada arco en el CRS actual
    darcs['length_m'] = darcs.geometry.length
    darcs['weight_len'] = darcs['length_m']

    # ---------------------------------------------------------------------------------
    # 5. Construir dos grafos dirigidos: tiempo y longitud
    # ---------------------------------------------------------------------------------
    G_time = nx.DiGraph()
    G_len  = nx.DiGraph()
    for _, r in darcs.iterrows():
        u = int(r['from_node'])
        v = int(r['to_node'])
        G_time.add_edge(u, v, weight=float(r['weight_time']))
        G_len.add_edge(u, v, weight=float(r['weight_len']))
    G_time_rev = G_time.reverse(copy=True)
    G_len_rev  = G_len.reverse(copy=True)

    # -----------------------
    # 6. Crear GeoDataFrame de nodos de arcos (manejar MultiLineString)
    # -----------------------
    def get_start_point(geom):
        if geom.geom_type == 'MultiLineString':
            return Point(geom.geoms[0].coords[0])
        return Point(geom.coords[0])

    def get_end_point(geom):
        if geom.geom_type == 'MultiLineString':
            return Point(geom.geoms[-1].coords[-1])
        return Point(geom.coords[-1])

    df_start = darcs.assign(
        geometry=darcs.geometry.apply(get_start_point)
    )[['from_node','geometry']].rename(columns={'from_node':'node'})

    df_end = darcs.assign(
        geometry=darcs.geometry.apply(get_end_point)
    )[['to_node','geometry']].rename(columns={'to_node':'node'})

    nodes_all = pd.concat([df_start, df_end], ignore_index=True).drop_duplicates('node')
    nodes_all = gpd.GeoDataFrame(nodes_all, geometry='geometry', crs=darcs.crs)

    # ---------------------------------------------------------------------------------
    # 7. Poda en centrales: cortar el ARCO más cercano a cada central
    #    y bloquear cualquier avance río arriba desde ese punto
    # ---------------------------------------------------------------------------------
    HYDRO_SNAP_MAX_DIST = 800  # ajusta según tu red (unidades del CRS de darcs)

    # Emparejar cada central con el TRAMO más cercano
    hydro2arc = sjoin_nearest(
        hydro.to_crs(darcs.crs),
        darcs[['from_node', 'to_node', 'geometry']],
        how='left',
        distance_col='hydro_dist'
    )

    for _, h in hydro2arc.iterrows():
        # si está muy lejos de la red, no cortamos
        if pd.isna(h.get('hydro_dist')) or (h['hydro_dist'] > HYDRO_SNAP_MAX_DIST):
            continue

        u = int(h['from_node'])   # nodo aguas arriba del tramo
        v = int(h['to_node'])     # nodo aguas abajo del tramo

        # En el grafo REVERSO, la subida es v -> u. Cortamos esa arista.
        if G_time_rev.has_edge(v, u):
            G_time_rev.remove_edge(v, u)
        if G_len_rev.has_edge(v, u):
            G_len_rev.remove_edge(v, u)

        # Además bloqueamos TODA conexión río arriba desde 'u' (refuerzo)
        if G_time_rev.has_node(u):
            for succ in list(G_time_rev.successors(u)):
                G_time_rev.remove_edge(u, succ)
        if G_len_rev.has_node(u):
            for succ in list(G_len_rev.successors(u)):
                G_len_rev.remove_edge(u, succ)

    # ---------------------------------------------------------------------------------
    # 8. Snap (vincular) cada sitio al nodo más cercano
    # ---------------------------------------------------------------------------------
    gdf_sites = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites['longitude'], sites['latitude']),
        crs='EPSG:4326'
    ).to_crs(darcs.crs)

    gdf_sites = sjoin_nearest(
        gdf_sites,
        nodes_all[['node','geometry']],
        how='left', distance_col='dist'
    ).rename(columns={'node':'node_init'})

    # ---------------------------------------------------------------------------------
    # 9. Calcular rutas y acumular horas y longitud
    # ---------------------------------------------------------------------------------
    records = []
    for _, site in gdf_sites.iterrows():
        nid = int(site['node_init'])
        # Horas acumuladas aguas arriba
        lengths_time = nx.single_source_dijkstra_path_length(G_time_rev, nid, weight='weight')
        # Longitudes acumuladas aguas arriba
        lengths_len  = nx.single_source_dijkstra_path_length(G_len_rev,  nid, weight='weight')

        valid = {n for n, d in lengths_time.items() if site['min_time'] <= d <= site['max_time']}
        sub = darcs[
            darcs['from_node'].isin(valid) &
            darcs['to_node'].isin(valid)
        ].copy()

        sub['sample_id']  = site['sample_id']
        sub['place_name'] = site['place_name']
        sub['species_nm'] = site['species_name']
        # Horas acumuladas hasta el nodo destino de cada arco:
        sub['cum_hrs'] = sub['to_node'].map(lengths_time)
        # Longitud acumulada hasta el nodo destino de cada arco:
        sub['cum_len_meters'] = sub['to_node'].map(lengths_len)

        # Heredar datos
        sub['departamen'] = sub['departamen']
        sub['nombre_ent'] = sub['nombre_ent']
        sub['cod_munici'] = sub['cod_munici']
        sub['arcid_1']    = sub['arcid_1']
        sub['waterbody']  = sub['nombre_geo']

        records.append(sub)

    # ---------------------------------------------------------------------------------
    # 10. Consolidar resultados en un GeoDataFrame
    # ---------------------------------------------------------------------------------
    if records:
        result_gdf = gpd.GeoDataFrame(
            pd.concat(records, ignore_index=True),
            geometry='geometry',
            crs=darcs.crs
        )
    else:
        cols = list(darcs.columns) + [
            'sample_id','place_name','species_nm','cum_hrs','cum_len_meters',
            'departamen','nombre_ent','cod_munici','arcid_1','waterbody'
        ]
        result_gdf = gpd.GeoDataFrame(columns=cols, crs=darcs.crs)

    # ---------------------------------------------------------------------------------
    # 11. Agregar totales por muestra
    # ---------------------------------------------------------------------------------
    totals = (
        result_gdf
        .groupby('sample_id')[['cum_hrs','cum_len_meters']]
        .sum()
        .reset_index()
        .rename(columns={'cum_hrs':'total_hrs','cum_len_meters':'total_len'})
    )
    totals['total_min'] = totals['total_hrs'] * 60
    result_gdf = result_gdf.merge(totals, on='sample_id', how='left')

    # ---------------------------------------------------------------------------------
    # 12. Longitud de cada arco
    # ---------------------------------------------------------------------------------
    result_gdf['length'] = result_gdf.geometry.length

    # ---------------------------------------------------------------------------------
    # 13. Mostrar mapa de rutas
    # ---------------------------------------------------------------------------------
    df_map = result_gdf.to_crs('EPSG:4326').copy()
    df_map['lat'] = df_map.geometry.centroid.y
    df_map['lon'] = df_map.geometry.centroid.x
    st.subheader("Rutas extraídas")
    st.map(df_map[['lat','lon']])

    # -----------------------
    # 14. Exportar CSV
    # -----------------------
    csv_df = df_map[[
        'from_node','to_node','elevmed','arcid_1','time',
        'length','cum_len_meters','departamen','nombre_ent','cod_munici',
        'place_name','species_nm','cum_hrs','waterbody'
    ]].copy()
    csv_df = csv_df.rename(columns={'length':'longitud'})
    csv_bytes = csv_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')

    st.markdown("""
**Los siguientes archivos de salida contienen la información relevante que se detalla a continuación:**
- `elevmed`: Elevación media del tramo.
- `arcid_1`: Identificador del tramo.
- `time`: Tiempo (horas) que demora el agua a lo largo de ese tramo.
- `longitud`: Longitud de ese tramo (metros).
- `cum_len_meters`: Longitud acumulada desde el sitio de colecta hasta el final de ese tramo (metros).
- `departamen`, `nombre_ent`, `cod_munici`: atributos administrativos y de cuenca.
- `place_name`: Nombre del punto de colecta.
- `species_nm`: Especie muestreada.
- `cum_hrs`: Tiempo acumulado (horas) desde el sitio de colecta hasta el final de ese tramo.
- `waterbody`: Nombre del cuerpo de agua.
""")
    st.download_button("Descargar CSV", data=csv_bytes, file_name="tramos_de_desove.csv", mime="text/csv")

    # -----------------------
    # 15. Exportar GeoJSON
    # -----------------------
    arcos_wgs = result_gdf.to_crs("EPSG:4326")
    arcos_wgs.to_file("tramos_desove.geojson", driver="GeoJSON")
    with open("tramos_desove.geojson", "rb") as f:
        geojson_bytes = f.read()
    st.download_button(
        "Descargar GeoJSON",
        data=geojson_bytes,
        file_name="tramos_desove.geojson",
        mime="application/json"
    )

    # -----------------------
    # 16. Exportar Shapefile (ZIP)
    # -----------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_pref = os.path.join(tmpdir, "tramos_desove")
        result_gdf.to_file(shp_pref + ".shp")
        files = [f for f in os.listdir(tmpdir) if f.startswith("tramos_desove.")]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(os.path.join(tmpdir, f), arcname=f)
        buf.seek(0)
        st.download_button(
            "Descargar Shapefile (.zip)",
            data=buf.getvalue(),
            file_name="tramos_desove_shp.zip",
            mime="application/zip"
        )
    st.markdown("""<div id="proceso">Procesamiento completado! </div>""", unsafe_allow_html=True)

    # ----------------------------
    # 17. Código R sugerido
    # ----------------------------
    r_snippet = """
        # -------------------- Paquetes necesarios --------------------
        install.packages(c("sf", "ggplot2", "dplyr", "viridis"), dependencies = TRUE)

        library(sf)
        library(ggplot2)
        library(dplyr)
        library(viridis)

        # -------------------- 1. Cargar capas --------------------
        dep <- st_read("Departamento/Dpto_84.shp") %>% 
        st_transform(4326)

        rios <- st_read("/Rios/Red_Magdalena.shp") %>%
        st_transform(4326)

        # Leer GeoJSON generado por Streamlit
        arcos_geo <- st_read("tramos_desove.geojson") %>%
        st_transform(4326)

        # -------------------- 2. Mapa general --------------------
        mapa_general <- ggplot() +
        geom_sf(data = dep, fill = "gray95", color = "gray70", size = 0.3) +
        geom_sf(data = rios, color = "lightblue", size = 0.15, alpha = 0.3) +
        geom_sf(data = arcos_geo, aes(color = as.factor(sample_id)), size = 1.2, alpha = 0.9) +
        scale_color_viridis_d(name = "Sample ID", option = "C") +
        labs(title = "Contexto nacional: rutas de desove") +
        theme_minimal(base_size = 11) +
        theme(
            panel.background = element_rect(fill = "aliceblue", color = NA),
            plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
        )

        # -------------------- 3. Mapa con zoom --------------------
        bbox <- st_bbox(arcos_geo)
        xrange <- bbox["xmax"] - bbox["xmin"]
        yrange <- bbox["ymax"] - bbox["ymin"]
        margen <- 0.1

        mapa_zoom <- ggplot() +
        geom_sf(data = dep, fill = "gray95", color = "gray70", size = 0.3) +
        geom_sf(data = rios, color = "lightblue", size = 0.15, alpha = 0.3) +
        geom_sf(data = arcos_geo, aes(color = as.factor(sample_id)), size = 1.2, alpha = 0.9) +
        scale_color_viridis_d(name = "Sample ID", option = "C") +
        coord_sf(
            xlim = c(bbox["xmin"] - margen * xrange, bbox["xmax"] + margen * xrange),
            ylim = c(bbox["ymin"] - margen * yrange, bbox["ymax"] + margen * yrange)
        ) +
        labs(title = "Zoom automático sobre rutas de desove") +
        theme_minimal(base_size = 11) +
        theme(
            panel.background = element_rect(fill = "aliceblue", color = NA),
            plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
        )

        # -------------------- 4. Exportar Mapas --------------------
        ggsave("mapa_general_colombia.png", mapa_general, width = 10, height = 8, dpi = 300)
        ggsave("mapa_zoom_desove.png", mapa_zoom, width = 10, height = 8, dpi = 300)
    """

    st.subheader("Código R para reproducir el mapa fuera de esta aplicación")
    st.code(r_snippet, language="r")
