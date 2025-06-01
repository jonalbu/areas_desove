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

# Configuración de la app
st.set_page_config(page_title="Rutas Acuáticas", layout="wide")
st.title("Cálculo de Áreas de desove en las Cuencas Magdalena y Cauca")

# Instrucciones de formato de datos
st.markdown("""
**Los archivos de entrada deben contener las siguientes columnas:**

- **Sitios de colecta (Excel, .xlsx):**
  - `sample_id`, `place_name`, `latitude`, `longitude`, `species_name`, `min_time`, `max_time`.

Asegúrate de que los nombres de columnas coincidan exactamente.
"""
)

# ----------------------
# Parámetros de filtrado
# ----------------------
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

# Rutas fijas de shapefiles auxiliares en el repositorio
# Asegúrate que estos archivos existan en tu estructura de carpetas
def_shp_arcs   = "SHP_Magdalena/Con_altitud/Red_Magdalena.shp"
# ---------------------------------------
# Cuando el usuario hace clic en “Ejecutar Análisis”
# ---------------------------------------
if st.sidebar.button("Ejecutar Análisis"):
    # Validar que se subió Excel
    if xls_file is None:
        st.error("Por favor suba el archivo Excel de sitios.")
        st.stop()

    # -----------------------
    # 1. Carga de todos los datos
    # -----------------------
    arcs   = gpd.read_file(def_shp_arcs)        # Red hidro
    sites  = pd.read_excel(xls_file, engine='openpyxl')       # Excel de sitios
    hydro  = gpd.read_file(def_shp_hydro)       # Centrales hidroeléctricas

    # -----------------------
    # 2. Normalizar nombre de columnas
    # -----------------------
    arcs.columns  = arcs.columns.str.strip().str.lower().str.replace(' ', '_')
    sites.columns = sites.columns.str.strip().str.lower().str.replace(' ', '_')
    hydro.columns = hydro.columns.str.strip().str.lower().str.replace(' ', '_')

    # -----------------------
    # 3. Filtrar centrales según estados seleccionados
    # -----------------------
    if status_sel:
        hydro = hydro[hydro['status'].isin(status_sel)]
    # Sino, se considera todas

    # -----------------------
    # 4. Filtrar arcos por elevación y Strahler
    # -----------------------
    darcs = arcs[(arcs['elevmed'] <= elev_max) & (arcs['grid_code'] >= grid_min)].copy()
    darcs['weight'] = darcs['time']  # 'time' ya está en horas

    # -----------------------
    # 5. Construir grafo dirigido
    # -----------------------
    G = nx.DiGraph()
    for _, r in darcs.iterrows():
        G.add_edge(int(r['from_node']), int(r['to_node']), weight=float(r['weight']))
    G_rev = G.reverse(copy=True)

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

    # -----------------------
    # 7. Poda en centrales: cortar aristas aguas arriba de nodos de centrales
    # -----------------------
    hydro_nodes = sjoin_nearest(
        hydro.to_crs(darcs.crs),
        nodes_all[['node','geometry']],
        how='left', distance_col='hydro_dist'
    )['node'].unique()
    for hn in hydro_nodes:
        if G_rev.has_node(hn):
            for succ in list(G_rev.successors(hn)):
                G_rev.remove_edge(hn, succ)

    # -----------------------
    # 8. Snap (vincular) cada sitio al nodo más cercano
    # -----------------------
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

    # -----------------------
    # 9. Calcular rutas y acumular horas
    # -----------------------
    records = []
    for _, site in gdf_sites.iterrows():
        nid     = int(site['node_init'])
        lengths = nx.single_source_dijkstra_path_length(G_rev, nid, weight='weight')
        valid   = {n for n, d in lengths.items() if site['min_time'] <= d <= site['max_time']}
        sub = darcs[darcs['from_node'].isin(valid) & darcs['to_node'].isin(valid)].copy()
        sub['sample_id']  = site['sample_id']
        sub['place_name'] = site['place_name']
        sub['species_nm'] = site['species_name']
        sub['cum_hrs']    = sub['to_node'].map(lengths)
        # Heredar datos de departamento, municipio, río, cod_munici, arcid_1
        sub['departamen'] = sub['departamen']
        sub['nombre_ent'] = sub['nombre_ent']
        sub['nombre_geo'] = sub['nombre_geo']
        sub['cod_munici'] = sub['cod_munici']
        sub['arcid_1']    = sub['arcid_1']
        records.append(sub)

    # -----------------------
    # 10. Consolidar resultados en un GeoDataFrame
    # -----------------------
    if records:
        result_gdf = gpd.GeoDataFrame(pd.concat(records, ignore_index=True), geometry='geometry', crs=darcs.crs)
    else:
        cols = list(darcs.columns) + [
            'sample_id','place_name','species_nm','cum_hrs',
            'departamen','nombre_ent','nombre_geo','cod_munici','arcid_1'
        ]
        result_gdf = gpd.GeoDataFrame(columns=cols, crs=darcs.crs)

    # -----------------------
    # 11. Agregar totales por muestra (horas y minutos)
    # -----------------------
    totals = (result_gdf.groupby('sample_id')['cum_hrs']
                       .sum().reset_index().rename(columns={'cum_hrs':'total_hrs'}))
    totals['total_min'] = totals['total_hrs'] * 60
    result_gdf = result_gdf.merge(totals, on='sample_id', how='left')

    # -----------------------
    # 12. Agregar longitud de cada arco (en unidades de CRS)
    # -----------------------
    result_gdf['length'] = result_gdf.geometry.length

    # -----------------------
    # 13. Mostrar mapa de rutas
    # -----------------------
    df_map = result_gdf.to_crs('EPSG:4326').copy()
    df_map['lat'] = df_map.geometry.centroid.y
    df_map['lon'] = df_map.geometry.centroid.x
    st.subheader("Rutas extraídas")
    st.map(df_map[['lat','lon']])

    # -----------------------
    # 14. Exportar CSV (columnas específicas)
    # -----------------------
    csv_df = df_map[[
        'from_node','to_node','elevmed','arcid_1','time',
        'lon','departamen','nombre_ent','nombre_geo','cod_munici',
        'place_name','species_nm','cum_hrs','total_hrs','total_min'
    ]].copy()
    # Renombrar lon a longitud
    csv_df = csv_df.rename(columns={'lon':'longitud'})
    csv_bytes = csv_df.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar CSV", data=csv_bytes, file_name="arcos_desove.csv", mime="text/csv")

    # -----------------------
    # 15. Exportar GeoJSON (con todos los atributos)
    # -----------------------
    st.download_button("Descargar GeoJSON", data=df_map.to_json(), file_name="arcos_desove.geojson", mime="application/json")

    # -----------------------
    # 16. Exportar Shapefile completo (ZIP)
    # -----------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_pref = os.path.join(tmpdir, "arcos_desove")
        result_gdf.to_file(shp_pref + ".shp")
        files = [f for f in os.listdir(tmpdir) if f.startswith("arcos_desove.")]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(os.path.join(tmpdir, f), arcname=f)
        buf.seek(0)
        st.download_button("Descargar Shapefile (.zip)", data=buf.getvalue(), file_name="arcos_desove_shp.zip", mime="application/zip")

    st.success("Procesamiento completo!")
