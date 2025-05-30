# -*- coding: utf-8 -*-
"""
Script: calcular_tiempo_acumulado_y_exportar.py
Descripción:
  - Carga un shapefile de arcos con GeoPandas
  - Lee un Excel con sitios de colecta (incluye place_name, min_time y max_time en horas)
  - Convierte min_time y max_time de horas a minutos para procesamiento
  - Carga shapefile de centrales hidroeléctricas y filtra por status
  - Verifica y renombra columnas si es necesario
  - Filtra arcos por elevación (<1200) y Strahler (>2)
  - Usa el campo 'time' (minutos) como peso de arista
  - Construye un grafo dirigido con NetworkX (weight en minutos)
  - Identifica nodos de centrales hidroeléctricas y pruna el grafo aguas arriba
  - Asigna a cada sitio el nodo más cercano
  - Para cada muestra, recorre aguas arriba interrumpiendo en centrales y extrae subgrafo de arcos
    cuya suma de minutos acumulados esté entre min_time y max_time
  - Exporta resultados a CSV con place_name y min/max en horas
  - Calcula cum_hrs (horas) para cada arco y exporta GeoJSON/Shapefile
"""
import os
import sys
import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import Point
from geopandas.tools import sjoin_nearest

# 1. Paths relativos
def get_paths():
    base = os.getcwd()
    return (
        os.path.join(base, "SHP_Magdalena", "Con_altitud", "Red_Magdalena.shp"),
        os.path.join(base, "Sitios_Colecta", "Colecta.xlsx"),
        os.path.join(base, "DBase_Proyectos_Hidroelectricos_Magdalena", "Proyectos_Hidroelectricos.shp")
    )
shp_arcs, xls_sites, shp_hydro = get_paths()

# 2. Verificar existencia de archivos
#for p in (shp_arcs, xls_sites, shp_hydro):
   # if not os.path.exists(p):
    #    sys.exit(f"ERROR: Archivo no encontrado: {p}")

# 3. Carga de datos
darcs = gpd.read_file(shp_arcs)
try:
    sites = pd.read_excel(xls_sites, engine='openpyxl')
except Exception as e:
    sys.exit(f"ERROR al leer Excel: {e}")
hydro = gpd.read_file(shp_hydro)

# 4. Convertir min/max originales (horas) a minutos
sites['min_time_h'] = sites['min_time']
sites['max_time_h'] = sites['max_time']
sites['min_time']   = sites['min_time'] 
sites['max_time']   = sites['max_time']

# 5. Normalizar nombres de columnas
darcs.columns = darcs.columns.str.strip().str.lower().str.replace(' ', '_')
sites.columns = sites.columns.str.strip().str.lower().str.replace(' ', '_')
hydro.columns = hydro.columns.str.strip().str.lower().str.replace(' ', '_')

# 6. Filtrar centrales por status (ajustar lista)
status_sel = ['Operation']
hydro = hydro[hydro['status'].isin(status_sel)].copy()

# 7. Validar campos necesarios
req_arcs  = {'from_node','to_node','time','elevmed','grid_code'}
req_sites = {'sample_id','place_name','latitude','longitude','species_name','min_time','max_time','min_time_h','max_time_h'}
req_hydro = {'status'}  # plus geometry
missing = req_arcs - set(darcs.columns)
if missing: sys.exit(f"Faltan en arcos: {missing}")
missing = req_sites - set(sites.columns)
if missing: sys.exit(f"Faltan en sitios: {missing}")
missing = req_hydro - set(hydro.columns)
if missing: sys.exit(f"Faltan en hidro: {missing}")

# 8. Filtrar arcos
darcs = darcs[(darcs['elevmed'] < 1200) & (darcs['grid_code'] > 2)].copy()

# 9. Peso de arista en minutos
darcs['weight'] = darcs['time']

# 10. Construir grafo dirigido
G = nx.DiGraph()
for _, r in darcs.iterrows():
    G.add_edge(int(r['from_node']), int(r['to_node']), weight=float(r['weight']))
# Invertido para aguas arriba
G_rev = G.reverse(copy=True)

# 11. Mapear centrales al grafo: nodos hidros
df_start = darcs.assign(geometry=darcs.geometry.apply(lambda L: Point(L.coords[0])))[['from_node','geometry']].rename(columns={'from_node':'node'})
df_end   = darcs.assign(geometry=darcs.geometry.apply(lambda L: Point(L.coords[-1])))[['to_node','geometry']].rename(columns={'to_node':'node'})
nodes_all = pd.concat([df_start, df_end],ignore_index=True).drop_duplicates('node')
nodes_all = gpd.GeoDataFrame(nodes_all, geometry='geometry', crs=darcs.crs)
# Unir centrales a nodos
hydro_nodes = sjoin_nearest(hydro.to_crs(darcs.crs), nodes_all[['node','geometry']], how='left').node.unique()
# Prunar grafo aguas arriba en centrales
for hn in hydro_nodes:
    if G_rev.has_node(hn):
        for succ in list(G_rev.successors(hn)):
            G_rev.remove_edge(hn, succ)

# 12. GeoDataFrame de sitios y reproyección
gdf_sites = gpd.GeoDataFrame(
    sites,
    geometry=gpd.points_from_xy(sites['longitude'], sites['latitude']),
    crs='EPSG:4326'
).to_crs(darcs.crs)

# 13. Asignar nodo más cercano a cada sitio
gdf_sites = sjoin_nearest(gdf_sites, nodes_all[['node','geometry']], how='left', distance_col='dist').rename(columns={'node':'node_init'})

# 14. Extraer subgrafo de arcos por muestra
records = []
for _, site in gdf_sites.iterrows():
    nid   = int(site['node_init'])
    sid   = site['sample_id']
    pl    = site['place_name']
    sp    = site['species_name']
    min_t = float(site['min_time'])
    max_t = float(site['max_time'])
    lengths = nx.single_source_dijkstra_path_length(G_rev, nid, weight='weight')
    valid   = {n for n,d in lengths.items() if min_t <= d <= max_t}
    sub = darcs[darcs['from_node'].isin(valid) & darcs['to_node'].isin(valid)].copy()
    sub['sample_id']  = sid
    sub['place_name'] = pl
    sub['species_nm'] = sp
    sub['cum_min']    = sub['to_node'].map(lambda x: lengths.get(x))
    sub['cum_hrs']    = sub['cum_min'] / 60.0
    records.append(sub)

# 15. Consolidar resultados
def_cols = list(darcs.columns) + ['sample_id','place_name','species_nm','cum_min','cum_hrs']
if records:
    result_gdf = gpd.GeoDataFrame(pd.concat(records,ignore_index=True), geometry='geometry', crs=darcs.crs)
else:
    result_gdf = gpd.GeoDataFrame(columns=def_cols, crs=darcs.crs)

# 16. Exportar CSV de sitios con tiempos en horas
out_csv = 'sitios_rango_tiempo_horas.csv'
gdf_sites[['sample_id','place_name','species_name','node_init','min_time_h','max_time_h']].to_csv(out_csv,index=False)

# 17. Exportar GeoJSON y Shapefile con atributos finales
out_dir = 'Resultados'; os.makedirs(out_dir, exist_ok=True)
# GeoJSON en lat/lon
result_gdf.to_crs('EPSG:4326').to_file(f"{out_dir}/arcos_desove.geojson", driver='GeoJSON')
# Shapefile con nombres <=10 chars
export_gdf = result_gdf.rename(columns={'place_name':'place_nam','species_nm':'species_nm','node_init':'node_init','cum_hrs':'cum_hrs'})
export_gdf.to_file(f"{out_dir}/arcos_desove.shp")

print('Procesamiento completo:')