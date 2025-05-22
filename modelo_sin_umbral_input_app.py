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

Asegúrese que los nombres de columnas coincidan exactamente y que las geometrías estén válidas.

- **Red hidrográfica (.shp)**
  - `from_node` (int): Identificador único del nodo origen de cada arco.
  - `to_node` (int): Identificador único del nodo destino.
  - `time` (float): Duración en **minutos** de recorrer cada arco.
  - `elevmed` (float): Elevación media del arco en metros.
  - `grid_code` (int): Orden de Strahler del arco (>= 0).
  - Geometría: líneas (`LineString`).

- **Sitios de colecta (Excel)**
  - `sample_id` (string o int): Identificador de la muestra.
  - `place_name` (string): Nombre de la ubicación de la muestra.
  - `latitude` (float): Latitud en grados decimales (WGS84).
  - `longitude` (float): Longitud en grados decimales (WGS84).
  - `species_name` (string): Nombre de la especie muestreada.
  - `min_time` (float): Tiempo mínimo en **horas** para filtrar recorrido.
  - `max_time` (float): Tiempo máximo en **horas** para filtrar recorrido.

- **Centrales hidroeléctricas (.shp)**
  - `status` (string): Estado de la central (p.ej. "Operativa").
  - Geometría: puntos (`Point`).
"""
)

# Parámetros de entrada
st.sidebar.header("Entradas de Usuario")
shp_zip = st.sidebar.file_uploader(
    "Subir ZIP con los archivos Shapefile de la red hidrográfica", type="zip"
)
xls_file = st.sidebar.file_uploader(
    "Subir Excel de sitios de Colecta (.xlsx)", type=["xlsx"]
)
hydro_zip = st.sidebar.file_uploader(
    "Subir ZIP con los archivos Shapefile de las Centrales Hidroeléctricas", type="zip"
)

# Función para cargar shapefile desde ZIP
def load_shapefile_from_zip(uploaded_zip):
    import zipfile, tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(uploaded_zip) as zf:
            zf.extractall(tmpdir)
        shp_files = [f for f in os.listdir(tmpdir) if f.lower().endswith('.shp')]
        if not shp_files:
            st.error("No se encontró ningún .shp en el ZIP cargado.")
            return None
        return gpd.read_file(os.path.join(tmpdir, shp_files[0]))

# Cargar datos subidos
arcs = load_shapefile_from_zip(shp_zip) if shp_zip else None
sites = pd.read_excel(xls_file, engine='openpyxl') if xls_file else None
hydro = load_shapefile_from_zip(hydro_zip) if hydro_zip else None

# Estatus centrales
status_vals = []
if hydro is not None:
    hydro.columns = hydro.columns.str.strip().str.lower().str.replace(' ', '_')
    if 'status' in hydro.columns:
        status_vals = hydro['status'].unique().tolist()
status_sel = st.sidebar.multiselect(
    "Estatus centrales", options=status_vals, default=status_vals
)

# Filtros de arcos
elev_max = st.sidebar.slider("Elevación máxima (m)", 0, 5000, 1200)
grid_min = st.sidebar.slider("Strahler mínimo", 0, 10, 2)

# Ejecutar análisis
if st.sidebar.button("Ejecutar Análisis"):
    if arcs is None or sites is None or hydro is None:
        st.error("Por favor suba todos los archivos requeridos.")
        st.stop()

    # Convertir horas a minutos
    sites['min_time_h'] = sites['min_time']
    sites['max_time_h'] = sites['max_time']
    sites['min_time'] = sites['min_time'] * 60
    sites['max_time'] = sites['max_time'] * 60

    # Normalizar nombres de columnas
    arcs.columns = arcs.columns.str.strip().str.lower().str.replace(' ', '_')
    sites.columns = sites.columns.str.strip().str.lower().str.replace(' ', '_')
    hydro.columns = hydro.columns.str.strip().str.lower().str.replace(' ', '_')

    # Filtrar centrales por estado
    if status_sel:
        hydro = hydro[hydro['status'].isin(status_sel)]

    # Filtrar arcos de red
    arcs = arcs[(arcs['elevmed'] <= elev_max) & (arcs['grid_code'] >= grid_min)].copy()
    arcs['weight'] = arcs['time']

    # Construir grafo dirigido
    G = nx.DiGraph()
    for _, r in arcs.iterrows():
        G.add_edge(int(r['from_node']), int(r['to_node']), weight=float(r['weight']))
    G_rev = G.reverse(copy=True)

    # Crear GeoDataFrame de nodos de arcos
    df_start = arcs.assign(geometry=arcs.geometry.apply(lambda L: Point(L.coords[0])))[['from_node','geometry']].rename(columns={'from_node':'node'})
    df_end = arcs.assign(geometry=arcs.geometry.apply(lambda L: Point(L.coords[-1])))[['to_node','geometry']].rename(columns={'to_node':'node'})
    nodes_all = pd.concat([df_start, df_end], ignore_index=True).drop_duplicates('node')
    nodes_all = gpd.GeoDataFrame(nodes_all, geometry='geometry', crs=arcs.crs)

    # Poda de grafo en centrales (detener rutas aguas arriba)
    hydro_nodes = sjoin_nearest(
        hydro.to_crs(arcs.crs), nodes_all[['node','geometry']], how='left', distance_col='hydro_dist'
    )['node'].unique()
    for hn in hydro_nodes:
        if G_rev.has_node(hn):
            for succ in list(G_rev.successors(hn)):
                G_rev.remove_edge(hn, succ)

    # Asignar sitio a nodo más cercano
    gdf_sites = gpd.GeoDataFrame(
        sites, geometry=gpd.points_from_xy(sites['longitude'], sites['latitude']), crs='EPSG:4326'
    ).to_crs(arcs.crs)
    gdf_sites = sjoin_nearest(
        gdf_sites, nodes_all[['node','geometry']], how='left', distance_col='dist'
    ).rename(columns={'node':'node_init'})

    # Calcular rutas y acumular tiempos
    records = []
    for _, site in gdf_sites.iterrows():
        nid = int(site['node_init'])
        lengths = nx.single_source_dijkstra_path_length(G_rev, nid, weight='weight')
        valid = {n for n, d in lengths.items() if site['min_time'] <= d <= site['max_time']}
        sub = arcs[arcs['from_node'].isin(valid) & arcs['to_node'].isin(valid)].copy()
        sub['sample_id'] = site['sample_id']
        sub['place_name'] = site['place_name']
        sub['species_nm'] = site['species_name']
        sub['cum_min'] = sub['to_node'].map(lambda x: lengths.get(x))
        sub['cum_hrs'] = sub['cum_min'] / 60.0
        records.append(sub)

    # Consolidar resultados
    if records:
        result_gdf = gpd.GeoDataFrame(pd.concat(records, ignore_index=True), geometry='geometry', crs=arcs.crs)
    else:
        cols = list(arcs.columns) + ['sample_id','place_name','species_nm','cum_min','cum_hrs']
        result_gdf = gpd.GeoDataFrame(columns=cols, crs=arcs.crs)

    # Mostrar mapa de rutas
    df_map = result_gdf.to_crs('EPSG:4326').copy()
    df_map['lat'] = df_map.geometry.centroid.y
    df_map['lon'] = df_map.geometry.centroid.x
    st.subheader("Rutas extraídas")
    st.map(df_map[['lat','lon']])
    
    st.subheader("A continuación se presentan los resultados en CSV, GeoJSON y Shapefile de las áreas posibles de desove")
    # Descargas CSV y GeoJSON
    csv_data = df_map.drop(columns=['geometry','lat','lon']).to_csv(index=False).encode('utf-8')
    st.download_button("Descargar CSV", csv_data, file_name="arcos_desove.csv")
    st.download_button("Descargar GeoJSON", df_map.to_json(), file_name="arcos_desove.geojson")
    st.markdown("El archivo GeoJSON se puede visualizar en la siguiente aplicación web: https://geojson.io/")
    
    # Descargar SHP comprimido
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
