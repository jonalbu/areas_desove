import os
import io
import streamlit as st
import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import Point
from geopandas.tools import sjoin_nearest
import zipfile

st.set_page_config(page_title="Rutas Acuáticas", layout="wide")
st.title("Cálculo de Áreas de Desove en Red Hídrica")

# Sidebar inputs
st.sidebar.header("Parámetros de Entrada")
shp_arcs = st.sidebar.text_input(
    "Shapefile de arcos",
    "SHP_Magdalena/Con_altitud/streamsline_Magdalena_Times_altitud.shp"
)
xls_sites = st.sidebar.text_input(
    "Excel de sitios",
    "Sitios_Colecta/Colecta.xlsx"
)
shp_hydro = st.sidebar.text_input(
    "Shapefile de centrales",
    "DBase_Proyectos_Hidroelectricos_Magdalena/Proyectos_Hidroelectricos.shp"
)
# Populate status options
try:
    hydro_tmp = gpd.read_file(shp_hydro)
    hydro_tmp.columns = (
        hydro_tmp.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
    )
    status_vals = list(hydro_tmp['status'].unique())
except Exception:
    status_vals = []
status_sel = st.sidebar.multiselect(
    "Estatus centrales",
    options=status_vals,
    default=status_vals
)
# Additional filters
elev_max = st.sidebar.slider(
    "Elevación máxima (m)", 0, 5000, 1200
)
grid_min = st.sidebar.slider(
    "Strahler mínimo",
    0, 10, 2
)
dist_tol = st.sidebar.slider(
    "Umbral central (m)",
    0, 5000, 1000
)

# Execute analysis
if st.sidebar.button("Ejecutar Análisis"):
    # Load data
    try:
        arcs = gpd.read_file(shp_arcs)
        sites = pd.read_excel(xls_sites, engine='openpyxl')
        hydro = gpd.read_file(shp_hydro)
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.stop()

    # Convert hours to minutes
    sites['min_time_h'] = sites['min_time']
    sites['max_time_h'] = sites['max_time']
    sites['min_time'] *= 60
    sites['max_time'] *= 60

    # Normalize column names
    arcs.columns = arcs.columns.str.strip().str.lower().str.replace(' ', '_')
    sites.columns = sites.columns.str.strip().str.lower().str.replace(' ', '_')
    hydro.columns = hydro.columns.str.strip().str.lower().str.replace(' ', '_')

    # Filter hydro
    if status_sel:
        hydro = hydro[hydro['status'].isin(status_sel)]

    # Filter arcs by elevation and strahler
    arcs = arcs[(arcs['elevmed'] <= elev_max) & (arcs['grid_code'] >= grid_min)].copy()
    arcs['weight'] = arcs['time']

    # Build graph
    G = nx.DiGraph()
    for _, r in arcs.iterrows():
        G.add_edge(int(r['from_node']), int(r['to_node']), weight=float(r['weight']))
    G_rev = G.reverse(copy=True)

    # Create nodes GeoDataFrame
    df_start = arcs.assign(geometry=arcs.geometry.apply(lambda L: Point(L.coords[0])))[['from_node','geometry']].rename(columns={'from_node':'node'})
    df_end   = arcs.assign(geometry=arcs.geometry.apply(lambda L: Point(L.coords[-1])))[['to_node','geometry']].rename(columns={'to_node':'node'})
    nodes_all = pd.concat([df_start, df_end], ignore_index=True).drop_duplicates('node')
    nodes_all = gpd.GeoDataFrame(nodes_all, geometry='geometry', crs=arcs.crs)

    # Prune at hydro nodes
    hydro_near = sjoin_nearest(
        hydro.to_crs(arcs.crs),
        nodes_all[['node','geometry']],
        how='left', distance_col='hydro_dist', max_distance=dist_tol
    )
    hydro_nodes = hydro_near.loc[~hydro_near['hydro_dist'].isna(), 'node'].unique()
    for hn in hydro_nodes:
        if G_rev.has_node(hn):
            for succ in list(G_rev.successors(hn)):
                G_rev.remove_edge(hn, succ)

    # Snap sites to nodes
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

    # Compute routes per site
    records=[]
    for _, site in gdf_sites.iterrows():
        nid, sid = int(site['node_init']), site['sample_id']
        pl, sp = site['place_name'], site['species_name']
        min_t, max_t = float(site['min_time']), float(site['max_time'])
        lengths = nx.single_source_dijkstra_path_length(G_rev, nid, weight='weight')
        valid = {n for n,d in lengths.items() if min_t <= d <= max_t}
        sub = arcs[arcs['from_node'].isin(valid) & arcs['to_node'].isin(valid)].copy()
        sub['sample_id'], sub['place_name'], sub['species_nm'] = sid, pl, sp
        sub['cum_min'] = sub['to_node'].map(lambda x: lengths.get(x))
        sub['cum_hrs'] = sub['cum_min'] / 60.0
        records.append(sub)

    # Consolidate
    if records:
        result_gdf = gpd.GeoDataFrame(pd.concat(records, ignore_index=True), geometry='geometry', crs=arcs.crs)
    else:
        cols = list(arcs.columns)+['sample_id','place_name','species_nm','cum_min','cum_hrs']
        result_gdf = gpd.GeoDataFrame(columns=cols, crs=arcs.crs)

        # Prepare map
    df_map = result_gdf.to_crs('EPSG:4326').copy()
    df_map['latitude'] = df_map.geometry.centroid.y
    df_map['longitude'] = df_map.geometry.centroid.x
    st.subheader("Rutas extraídas")
    st.map(df_map[['latitude','longitude']])

    # Offer downloads
    csv_data = df_map.drop(columns=['geometry','latitude','longitude']).to_csv(index=False).encode('utf-8')
    st.download_button(
        "Descargar CSV arcos",
        csv_data,
        file_name="arcos_desove.csv",
        mime="text/csv"
    )
    st.download_button(
        "Descargar GeoJSON arcos",
        df_map.to_json(),
        file_name="arcos_desove.geojson",
        mime="application/json"
    )

        # Export Shapefile zipped using a temporary directory for speed
    import tempfile, io, zipfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write all shapefile components
        shp_prefix = os.path.join(tmpdir, "arcos_desove")
        result_gdf.to_file(shp_prefix + ".shp")
        # Collect component files
        files = [f for f in os.listdir(tmpdir) if f.startswith("arcos_desove.")]
        # Zip in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in files:
                zf.write(os.path.join(tmpdir, fname), arcname=fname)
        zip_buffer.seek(0)
        st.download_button(
            "Descargar Shapefile (.zip)",
            zip_buffer.getvalue(),
            file_name="arcos_desove_shp.zip",
            mime="application/zip"
        )

    # Final message
    st.success("Procesamiento completo")
   
