#!/usr/bin/env python3
"""
modelo.py

Esta versión no usa argparse; en cambio, toma rutas/valores por defecto o 
te los solicita interactivamente si dejas las variables vacías.

En consola debes ejecutar la siguiente linea del nombre del archivo, este nombre debe ser igual al creado en el IDE, sino, aparece error:
    python modelo.py

Si quieres cambiar rutas/valores, edita directamente las variables en la sección
“PARÁMETROS A EDITAR” más abajo, o déjalas vacías ("") para que las pida en tiempo de ejecución.

Requisitos:
    pip install geopandas pandas networkx shapely matplotlib
"""

import os
import sys
import zipfile
import tempfile
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point
from geopandas.tools import sjoin_nearest
import matplotlib.pyplot as plt

# =========================
# PARÁMETROS A EDITAR
# =========================
# Si alguna ruta queda vacía (""), el script te preguntará por consola dónde encontrar el archivo.
# De lo contrario, usará lo que especifiques aquí directamente.

# 1. Shapefile red hidrográfica (arcos)
RED_SHP = ""  
# 2. Excel de sitios de colecta
SITES_XLSX = ""  
# 3. Shapefile centrales hidroeléctricas
HYDRO_SHP = ""  
# 4. Elevación máxima para filtrar (en metros)
ELEV_MAX = 1200  
# 5. Strahler mínimo para filtrar (entero)
GRID_MIN = 2  
# 6. Carpeta de salida (se creará si no existe)
OUT_DIR = "resultados"  
# =========================


def pedir_si_vacio(variable, mensaje):
    """
    Si 'variable' está vacía (""), pregunta al usuario por consola con 'mensaje',
    devolviendo la cadena que ingrese. En caso contrario, retorna 'variable' tal cual.
    """
    if (variable is None) or (str(variable).strip() == ""):
        valor = input(f"{mensaje}: ").strip()
        if valor == "":
            print("Debes ingresar un valor válido. Saliendo.")
            sys.exit(1)
        return valor
    return variable


def main():
    # 1) Validar/solicitar rutas de entrada
    red_shp = pedir_si_vacio(
        RED_SHP,
        "Ruta al shapefile de la red hidrográfica (arcos)"
    )
    sites_xlsx = pedir_si_vacio(
        SITES_XLSX,
        "Ruta al Excel (.xlsx) con sitios de colecta"
    )
    hydro_shp = pedir_si_vacio(
        HYDRO_SHP,
        "Ruta al shapefile de centrales hidroeléctricas"
    )

    # 2) Crear carpeta de salida si no existe
    os.makedirs(OUT_DIR, exist_ok=True)

    # 3) Cargar archivos
    print("Cargando red hidrográfica …")
    arcs = gpd.read_file(red_shp)

    print("Cargando Excel de sitios de colecta …")
    sites = pd.read_excel(sites_xlsx, engine="openpyxl")

    print("Cargando centrales hidroeléctricas …")
    hydro = gpd.read_file(hydro_shp)

    # 4) Normalizar nombres de columna (quitar espacios, pasar a minúsculas, etc.)
    arcs.columns = arcs.columns.str.strip().str.lower().str.replace(" ", "_")
    sites.columns = sites.columns.str.strip().str.lower().str.replace(" ", "_")
    hydro.columns = hydro.columns.str.strip().str.lower().str.replace(" ", "_")

    # 5) Filtrar arcos por elevación y Strahler
    print(f"Filtrando tramos con elevmed ≤ {ELEV_MAX} y grid_code ≥ {GRID_MIN} …")
    darcs = arcs[
        (arcs["elevmed"] <= float(ELEV_MAX)) &
        (arcs["grid_code"] >= int(GRID_MIN))
    ].copy()

    # Añadir pesos: horas y longitud
    darcs["weight_time"] = darcs["time"].astype(float)
    darcs["length_m"] = darcs.geometry.length
    darcs["weight_len"] = darcs["length_m"]

    # 6) Construir grafos dirigidos inversos (para enrutar hacia arriba)
    print("Construyendo grafos de enrutamiento …")
    G_time = nx.DiGraph()
    G_len = nx.DiGraph()
    for _, r in darcs.iterrows():
        u = int(r["from_node"])
        v = int(r["to_node"])
        G_time.add_edge(u, v, weight=float(r["weight_time"]))
        G_len.add_edge(u, v, weight=float(r["weight_len"]))

    G_time_rev = G_time.reverse(copy=True)
    G_len_rev = G_len.reverse(copy=True)

    # 7) Crear GeoDataFrame de nodos (punto inicio/fin de cada arco)
    def get_start_point(geom):
        if geom.geom_type == "MultiLineString":
            return Point(geom.geoms[0].coords[0])
        return Point(geom.coords[0])

    def get_end_point(geom):
        if geom.geom_type == "MultiLineString":
            return Point(geom.geoms[-1].coords[-1])
        return Point(geom.coords[-1])

    df_start = darcs.assign(
        geometry=darcs.geometry.apply(get_start_point)
    )[[ "from_node", "geometry"]].rename(columns={"from_node": "node"})
    df_end = darcs.assign(
        geometry=darcs.geometry.apply(get_end_point)
    )[[ "to_node", "geometry"]].rename(columns={"to_node": "node"})

    nodes_all = pd.concat([df_start, df_end], ignore_index=True).drop_duplicates("node")
    nodes_all = gpd.GeoDataFrame(nodes_all, geometry="geometry", crs=darcs.crs)

    # 8) Poda de aristas aguas arriba de nodos de centrales
    print("Recortando flechas aguas arriba de cada central …")
    hydro_nodes = sjoin_nearest(
        hydro.to_crs(darcs.crs),
        nodes_all[["node", "geometry"]],
        how="left",
        distance_col="hydro_dist"
    )["node"].unique()

    for hn in hydro_nodes:
        if G_time_rev.has_node(hn):
            for succ in list(G_time_rev.successors(hn)):
                G_time_rev.remove_edge(hn, succ)
        if G_len_rev.has_node(hn):
            for succ in list(G_len_rev.successors(hn)):
                G_len_rev.remove_edge(hn, succ)

    # 9) Asignar cada sitio de colecta a su nodo más cercano
    print("Asociando sitios de colecta a la red …")
    gdf_sites = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites["longitude"], sites["latitude"]),
        crs="EPSG:4326"
    ).to_crs(darcs.crs)

    gdf_sites = sjoin_nearest(
        gdf_sites,
        nodes_all[["node", "geometry"]],
        how="left",
        distance_col="dist"
    ).rename(columns={"node": "node_init"})

    # 10) Calcular rutas (“áreas posibles de desove”) para cada sitio
    print("Calculando rutas hacia posibles áreas de desove …")
    records = []
    for _, site in gdf_sites.iterrows():
        nid = int(site["node_init"])
        lengths_time = nx.single_source_dijkstra_path_length(G_time_rev, nid, weight="weight")
        lengths_len = nx.single_source_dijkstra_path_length(G_len_rev, nid, weight="weight")

        valid = {n for n, d in lengths_time.items()
                 if (site["min_time"] <= d <= site["max_time"])}
        sub = darcs[
            darcs["from_node"].isin(valid) &
            darcs["to_node"].isin(valid)
        ].copy()

        sub["sample_id"] = site["sample_id"]
        sub["place_name"] = site["place_name"]
        sub["species_nm"] = site["species_name"]
        sub["cum_hrs"] = sub["to_node"].map(lengths_time)
        sub["cum_len"] = sub["to_node"].map(lengths_len)

        # Heredar atributos administrativos si existen
        for attr in ["departamen", "nombre_ent", "nombre_geo", "cod_munici", "arcid_1"]:
            if attr in sub.columns:
                sub[attr] = sub[attr]
            else:
                sub[attr] = None

        records.append(sub)

    # 11) Consolidar resultados en un solo GeoDataFrame
    if records:
        result_gdf = gpd.GeoDataFrame(
            pd.concat(records, ignore_index=True),
            geometry="geometry",
            crs=darcs.crs
        )
    else:
        cols = list(darcs.columns) + [
            "sample_id", "place_name", "species_nm", "cum_hrs", "cum_len",
            "departamen", "nombre_ent", "nombre_geo", "cod_munici", "arcid_1"
        ]
        result_gdf = gpd.GeoDataFrame(columns=cols, crs=darcs.crs)

    # 12) Calcular totales por muestra (horas/minutos/longitud)
    totals = (
        result_gdf
        .groupby("sample_id")[["cum_hrs", "cum_len"]]
        .sum()
        .reset_index()
        .rename(columns={"cum_hrs": "total_hrs", "cum_len": "total_len"})
    )
    totals["total_min"] = totals["total_hrs"] * 60
    result_gdf = result_gdf.merge(totals, on="sample_id", how="left")

    # 13) Añadir columna “length” (longitud de cada arco, unidad CRS)
    result_gdf["length"] = result_gdf.geometry.length

    # 14) Exportar CSV
    print("Guardando CSV de resultados …")
    df_map = result_gdf.to_crs("EPSG:4326").copy()
    df_map["lat"] = df_map.geometry.centroid.y
    df_map["lon"] = df_map.geometry.centroid.x

    csv_df = df_map[[
        "from_node", "to_node", "elevmed", "arcid_1", "time",
        "length", "cum_len", "departamen", "nombre_ent", "nombre_geo", "cod_munici",
        "place_name", "species_nm", "cum_hrs"
    ]].copy()
    csv_df = csv_df.rename(columns={"length": "longitud"})
    csv_path = os.path.join(OUT_DIR, "arcos_desove.csv")
    csv_df.to_csv(csv_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"CSV guardado en: {csv_path}")

    # 15) Exportar GeoJSON
    print("Guardando GeoJSON completo …")
    geojson_path = os.path.join(OUT_DIR, "arcos_desove.geojson")
    arcos_wgs = result_gdf.to_crs("EPSG:4326")
    arcos_wgs.to_file(geojson_path, driver="GeoJSON")
    print(f"GeoJSON guardado en: {geojson_path}")

    # 16) Exportar Shapefile completo en ZIP
    print("Guardando Shapefile (ZIP) …")
    shp_dir = tempfile.mkdtemp()
    shp_pref = os.path.join(shp_dir, "arcos_desove")
    result_gdf.to_file(shp_pref + ".shp")
    zip_path = os.path.join(OUT_DIR, "arcos_desove_shp.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(shp_dir):
            if fname.startswith("arcos_desove."):
                zf.write(os.path.join(shp_dir, fname), arcname=fname)
    print(f"Shapefile ZIP guardado en: {zip_path}")

    # 17) Dibujar y guardar mapa de fondo (Departamentos + Ríos + rutas coloreadas)
    try:
        print("Generando mapa PNG …")
        arcos = gpd.read_file(geojson_path).to_crs(4326)
        # Se asume que tienes “Departamentos/Dpto_84.shp” junto al resto de archivos
        depts = gpd.read_file("Departamentos/Dpto_84.shp").to_crs(4326)
        rios = gpd.read_file(red_shp).to_crs(4326)

        fig, ax = plt.subplots(figsize=(10, 8))
        depts.plot(ax=ax, facecolor="lightgray", edgecolor="gray", linewidth=0.3)
        rios.plot(ax=ax, color="steelblue", linewidth=0.4)

        unique_ids = arcos["sample_id"].unique()
        cmap = plt.get_cmap("viridis", len(unique_ids))
        color_map = {sid: cmap(i) for i, sid in enumerate(unique_ids)}

        # Crear “proxy artists” para la leyenda
        for sid in unique_ids:
            ax.plot([], [], color=color_map[sid], linewidth=2, label=str(sid))

        # Dibujar cada ruta coloreada
        for _, row in arcos.iterrows():
            sid = row["sample_id"]
            geom = row.geometry
            color = color_map[sid]
            if geom.geom_type == "MultiLineString":
                for part in geom.geoms:
                    xs, ys = part.xy
                    ax.plot(xs, ys, color=color, linewidth=0.7, alpha=0.8)
            else:
                xs, ys = geom.xy
                ax.plot(xs, ys, color=color, linewidth=0.7, alpha=0.8)

        ax.legend(
            title="ID Muestra",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8,
            title_fontsize=9
        )

        xs = np.arange(-82, -66, 1)
        ys = np.arange(-4, 15, 1)
        ax.set_xticks(xs)
        ax.set_yticks(ys)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.7)

        ax.set_title(
            "Rutas de posibles áreas de desove sobre mapa de Colombia",
            fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")

        plt.tight_layout()
        png_path = os.path.join(OUT_DIR, "mapa_arcos_desove.png")
        fig.savefig(png_path, dpi=600)
        plt.close(fig)
        print(f"Mapa PNG guardado en: {png_path}")

    except Exception as e:
        print(f"[ERROR] No se pudo generar el mapa: {e}")

    print("Proceso completado.")


if __name__ == "__main__":
    main()
