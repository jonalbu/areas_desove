import os
import io
import streamlit as st
import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import Point
from geopandas.tools import sjoin_nearest
import zipfile
import tempfile

st.set_page_config(page_title="Rutas Acuáticas", layout="wide")
st.title("Cálculo de Áreas de desove en Red Hídrica")

# Instrucciones de formato de datos
st.markdown("""
**Los archivos de entrada deben contener las siguientes columnas con la información correspondiente**

Asegurese que los nombres de columnas coincidan exactamente y que las geometrías estén válidas.

- **Red hidrográfica (.shp)**

El archivo debe estar en formato .shp y contener las siguientes columnas:

  - `from_node` (int): Identificador único del nodo origen de cada arco.
  - `to_node` (int): Identificador único del nodo destino.
  - `time` (float): Duración en **minutos** de recorrer cada arco.
  - `elevmed` (float): Elevación media del arco en metros.
  - `grid_code` (int): Orden de Strahler del arco (>= 0).
  - Geometría: líneas (`LineString`) representando el tramo.

- **Sitios de colecta (Excel)**

El archivo debe estar en formato .xlsx y contener las siguientes columnas:

  - `sample_id` (string o int): Identificador de la muestra.
  - `place_name` (string): Nombre de la ubicación de la muestra.
  - `latitude` (float): Latitud en grados decimales (WGS84).
  - `longitude` (float): Longitud en grados decimales (WGS84).
  - `species_name` (string): Nombre de la especie muestreada.
  - `min_time` (float): Tiempo mínimo en **horas** para filtrar recorrido.
  - `max_time` (float): Tiempo máximo en **horas** para filtrar recorrido.

- **Centrales hidroeléctricas (.shp)**

el archivo debe estar en formato .shp y contener las siguientes columnas:

  - `status` (string): Estado de la central (p.ej. "Operativa", "En construcción").
  - Geometría: puntos (`Point`) ubicados en la red.

Asegúrate de que los nombres de columnas coincidan exactamente y que las geometrías estén válidas.
"""
)


# Sidebar inputs
st.sidebar.header("Parámetros de Entrada")
shp_arcs = st.sidebar.text_input(
    "Red hidrográfica (.shp)",
    "SHP_Magdalena/Con_altitud/streamsline_Magdalena_Times_altitud.shp"
)
xls_sites = st.sidebar.text_input(
    "Sitios de colecta (Excel)",
    "Sitios_Colecta/Colecta.xlsx"
)
shp_hydro = st.sidebar.text_input(
    "Centrales hidroeléctricas (.shp)",
    "DBase_Proyectos_Hidroelectricos_Magdalena/Proyectos_Hidroelectricos.shp"
)

# Load hydro for status options
try:
    hydro_tmp = gpd.read_file(shp_hydro)
    hydro_tmp.columns = (
        hydro_tmp.columns.str.strip()
                            .str.lower()
                            .str.replace(' ', '_')
    )
    status_vals = hydro_tmp['status'].unique().tolist()
except Exception:
    status_vals = []
status_sel = st.sidebar.multiselect(
    "Estatus centrales",
    options=status_vals,
    default=status_vals
)

elev_max = st.sidebar.slider(
    "Elevación máxima (m)",
    min_value=0,
    max_value=5000,
    value=1200
)
grid_min = st.sidebar.slider(
    "Strahler mínimo",
    min_value=0,
    max_value=10,
    value=2
)

if st.sidebar.button("Ejecutar Análisis"):
    # Carga de datos
    try:
        arcs = gpd.read_file(shp_arcs)
        sites = pd.read_excel(xls_sites, engine='openpyxl')
        hydro = gpd.read_file(shp_hydro)
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.stop()

    # Min/max de tiempo en minutos
    sites['min_time_h'] = sites['min_time']
    sites['max_time_h'] = sites['max_time']
    sites['min_time'] = sites['min_time'] * 60
    sites['max_time'] = sites['max_time'] * 60

    # Normalizar nombres
    arcs.columns = arcs.columns.str.strip().str.lower().str.replace(' ', '_')
    sites.columns = sites.columns.str.strip().str.lower().str.replace(' ', '_')
    hydro.columns = hydro.columns.str.strip().str.lower().str.replace(' ', '_')

    # Filtrar centrales
    if status_sel:
        hydro = hydro[hydro['status'].isin(status_sel)]

    # Filtrar arcos por elevación y Strahler
    arcs = arcs[(arcs['elevmed'] <= elev_max) & (arcs['grid_code'] >= grid_min)].copy()
    arcs['weight'] = arcs['time']

    # Construir grafo
    G = nx.DiGraph()
    for _, r in arcs.iterrows():
        G.add_edge(int(r['from_node']), int(r['to_node']), weight=float(r['weight']))
    G_rev = G.reverse(copy=True)

    # Crear nodos de arcos
    df_start = arcs.assign(
        geometry=arcs.geometry.apply(lambda L: Point(L.coords[0]))
    )[['from_node','geometry']].rename(columns={'from_node':'node'})
    df_end = arcs.assign(
        geometry=arcs.geometry.apply(lambda L: Point(L.coords[-1]))
    )[['to_node','geometry']].rename(columns={'to_node':'node'})
    nodes_all = pd.concat([df_start, df_end], ignore_index=True).drop_duplicates('node')
    nodes_all = gpd.GeoDataFrame(nodes_all, geometry='geometry', crs=arcs.crs)

        # Crear GeoDataFrame de nodos de arcos
    df_start = arcs.assign(
        geometry=arcs.geometry.apply(lambda L: Point(L.coords[0]))
    )[['from_node','geometry']].rename(columns={'from_node':'node'})
    df_end = arcs.assign(
        geometry=arcs.geometry.apply(lambda L: Point(L.coords[-1]))
    )[['to_node','geometry']].rename(columns={'to_node':'node'})
    nodes_all = pd.concat([df_start, df_end], ignore_index=True).drop_duplicates('node')
    nodes_all = gpd.GeoDataFrame(nodes_all, geometry='geometry', crs=arcs.crs)

    # Poda de grafo en centrales (detener ruteo aguas arriba)
    hydro_nodes = sjoin_nearest(
        hydro.to_crs(arcs.crs),
        nodes_all[['node','geometry']],
        how='left',
        distance_col='hydro_dist'
    )['node'].unique()
    for hn in hydro_nodes:
        if G_rev.has_node(hn):
            for succ in list(G_rev.successors(hn)):
                G_rev.remove_edge(hn, succ)

    # Snap sitios a nodos
    gdf_sites = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites['longitude'], sites['latitude']),
        crs='EPSG:4326'
    ).to_crs(arcs.crs)
    gdf_sites = sjoin_nearest(
        gdf_sites,
        nodes_all[['node','geometry']],
        how='left', distance_col='dist'
    ).rename(columns={'node':'node_init'})

    # Calcular rutas por sitio
    records = []
    for _, site in gdf_sites.iterrows():
        nid = int(site['node_init'])
        sid = site['sample_id']
        pl = site['place_name']
        sp = site['species_name']
        min_t = float(site['min_time'])
        max_t = float(site['max_time'])
        lengths = nx.single_source_dijkstra_path_length(G_rev, nid, weight='weight')
        valid = {n for n, d in lengths.items() if min_t <= d <= max_t}
        sub = arcs[arcs['from_node'].isin(valid) & arcs['to_node'].isin(valid)].copy()
        sub['sample_id'] = sid
        sub['place_name'] = pl
        sub['species_nm'] = sp
        sub['cum_min'] = sub['to_node'].map(lambda x: lengths.get(x))
        sub['cum_hrs'] = sub['cum_min'] / 60.0
        records.append(sub)

    # Consolidar resultados
    if records:
        result_gdf = gpd.GeoDataFrame(
            pd.concat(records, ignore_index=True),
            geometry='geometry',
            crs=arcs.crs
        )
    else:
        cols = list(arcs.columns) + ['sample_id','place_name','species_nm','cum_min','cum_hrs']
        result_gdf = gpd.GeoDataFrame(columns=cols, crs=arcs.crs)

    # Mostrar mapa
    df_map = result_gdf.to_crs('EPSG:4326').copy()
    df_map['lat'] = df_map.geometry.centroid.y
    df_map['lon'] = df_map.geometry.centroid.x
    st.subheader("Rutas extraídas")
    st.map(df_map[['lat','lon']])
    
    
    st.subheader("A continuación se presentan los resultados en CSV, GeoJSON y Shapefile de las áreas posibles de desove")
    # Descargas
    csv_data = df_map.drop(columns=['geometry','lat','lon']).to_csv(index=False).encode('utf-8')
    st.download_button("Descargar CSV", csv_data, file_name="arcos_desove.csv")
    st.download_button("Descargar GeoJSON", df_map.to_json(), file_name="arcos_desove.geojson")
    st.markdown("El archivo GeoJSON se puede visualizar en la siguiente aplicación web: https://geojson.io/")
    
    # Exportar SHP zip
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_pref = os.path.join(tmpdir, "arcos_desove")
        result_gdf.to_file(shp_pref + ".shp")
        files = [f for f in os.listdir(tmpdir) if f.startswith("arcos_desove.")]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(os.path.join(tmpdir, f), arcname=f)
        buf.seek(0)
        st.download_button("Descargar Shapefile (.zip)", buf.getvalue(), file_name="arcos_desove_shp.zip")
    st.markdown("El archivo Shapefile se puede visualizar en QGIS o ArcGIS.")
    
   
    
    st.success("Procesamiento completo!")
