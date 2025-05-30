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
st.title("Cálculo de Áreas de desove en las Cuencas Magdalena y Cauca")

# Instrucciones de formato de datos
st.markdown("""
**Los archivos de entrada deben contener las siguientes columnas con la información correspondiente**

Asegúrese de que los nombres de columnas coincidan exactamente y que las geometrías estén válidas.

- **Red hidrográfica (.shp)**
  - `from_node` (int): Identificador único del nodo origen de cada arco.
  - `to_node` (int): Identificador único del nodo destino.
  - `time` (float): Duración en **horas** de recorrer cada arco.
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
st.sidebar.header("Parámetros de Filtrado")
# Slider para elevación máxima
elev_max = st.sidebar.slider("Elevación máxima (m)", 0, 5000, 1200)
# Slider para Strahler mínimo
grid_min = st.sidebar.slider("Strahler mínimo", 0, 10, 2)
# File uploader solo para sitios
xls_file = st.sidebar.file_uploader("Subir Excel de sitios de Colecta (.xlsx)", type=["xlsx"])
# Rutas fijas a shapefiles
shp_arcs = "SHP_Magdalena/Con_altitud/Red_Magdalena.shp"
shp_hydro = "DBase_Proyectos_Hidroelectricos_Magdalena/Proyectos_Hidroelectricos.shp"

# Ejecutar análisis
if st.sidebar.button("Ejecutar Análisis"):
    if xls_file is None:
        st.error("Por favor suba el archivo Excel de sitios.")
        st.stop()

    # Carga de datos
    arcs = gpd.read_file(shp_arcs)
    sites = pd.read_excel(xls_file, engine='openpyxl')
    hydro = gpd.read_file(shp_hydro)

    # No convertir unidades: tiempo ya en horas
    sites['min_time_h'] = sites['min_time']
    sites['max_time_h'] = sites['max_time']

    # Normalizar nombres de columnas
    arcs.columns = arcs.columns.str.strip().str.lower().str.replace(' ', '_')
    sites.columns = sites.columns.str.strip().str.lower().str.replace(' ', '_')
    hydro.columns = hydro.columns.str.strip().str.lower().str.replace(' ', '_')

    # Filtrar centrales operativas
    hydro = hydro[hydro['status'].isin(["Operativa", "Operation"])]

    # Filtrar arcos por elevación y Strahler
    darcs = arcs[(arcs['elevmed'] <= elev_max) & (arcs['grid_code'] >= grid_min)].copy()
    darcs['weight'] = darcs['time']  # time en horas

    # Construir grafo dirigido
    G = nx.DiGraph()
    for _, r in darcs.iterrows():
        G.add_edge(int(r['from_node']), int(r['to_node']), weight=float(r['weight']))
    G_rev = G.reverse(copy=True)

    # Crear GeoDataFrame de nodos de arcos
    df_start = darcs.assign(geometry=darcs.geometry.apply(lambda L: Point(L.coords[0])))[['from_node','geometry']].rename(columns={'from_node':'node'})
    df_end = darcs.assign(geometry=darcs.geometry.apply(lambda L: Point(L.coords[-1])))[['to_node','geometry']].rename(columns={'to_node':'node'})
    nodes_all = pd.concat([df_start, df_end], ignore_index=True).drop_duplicates('node')
    nodes_all = gpd.GeoDataFrame(nodes_all, geometry='geometry', crs=darcs.crs)

    # Poda de grafo en centrales hidroeléctricas
    hydro_nodes = sjoin_nearest(hydro.to_crs(darcs.crs), nodes_all[['node','geometry']], how='left', distance_col='hydro_dist')['node'].unique()
    for hn in hydro_nodes:
        if G_rev.has_node(hn):
            for succ in list(G_rev.successors(hn)):
                G_rev.remove_edge(hn, succ)

    # Asignar nodo más cercano a cada sitio
    gdf_sites = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites['longitude'], sites['latitude']), crs='EPSG:4326').to_crs(darcs.crs)
    gdf_sites = sjoin_nearest(gdf_sites, nodes_all[['node','geometry']], how='left', distance_col='dist').rename(columns={'node':'node_init'})

    # Calcular rutas por sitio y acumular
    records = []
    for _, site in gdf_sites.iterrows():
        nid = int(site['node_init'])
        lengths = nx.single_source_dijkstra_path_length(G_rev, nid, weight='weight')
        valid = {n for n,d in lengths.items() if site['min_time'] <= d <= site['max_time']}
        sub = darcs[darcs['from_node'].isin(valid) & darcs['to_node'].isin(valid)].copy()
        sub['sample_id'] = site['sample_id']
        sub['place_name'] = site['place_name']
        sub['species_nm'] = site['species_name']
        sub['cum_min'] = sub['to_node'].map(lambda x: lengths.get(x))
        sub['cum_hrs'] = sub['cum_min']
        records.append(sub)

    # Consolidar resultados
    if records:
        result_gdf = gpd.GeoDataFrame(pd.concat(records, ignore_index=True), geometry='geometry', crs=darcs.crs)
    else:
        cols = list(darcs.columns) + ['sample_id','place_name','species_nm','cum_min','cum_hrs']
        result_gdf = gpd.GeoDataFrame(columns=cols, crs=darcs.crs)

    # Agregar totales por muestra
    totals = result_gdf.groupby('sample_id')['cum_hrs'].sum().reset_index().rename(columns={'cum_hrs':'total_hrs'})
    totals['total_min'] = totals['total_hrs'] * 60
    result_gdf = result_gdf.merge(totals, on='sample_id', how='left')

    # Agregar longitud de cada arco
    result_gdf['length'] = result_gdf.geometry.length

    # Mostrar mapa de rutas
    df_map = result_gdf.to_crs('EPSG:4326').copy()
    df_map['lat'] = df_map.geometry.centroid.y
    df_map['lon'] = df_map.geometry.centroid.x
    st.subheader("Rutas extraídas")
    st.map(df_map[['lat','lon']])

    # Descargas CSV y GeoJSON
    csv_data = df_map.drop(columns=['geometry','lat','lon']).to_csv(index=False).encode('utf-8')
    st.download_button("Descargar CSV", csv_data, file_name="arcos_desove.csv")
    st.download_button("Descargar GeoJSON", df_map.to_json(), file_name="arcos_desove.geojson")

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

    st.success("Procesamiento completo!")

