import os
import io
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point
from geopandas.tools import sjoin_nearest
import zipfile
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Configuración de la app
st.set_page_config(page_title="Rutas Acuáticas", layout="wide")
st.title("Estimación Áreas de desove de Peces dulceacuícolas para la cuenca Magdalena-Cauca")

# Instrucciones de formato de datos
st.markdown("""
**Los archivos de entrada deben contener las siguientes columnas con la información correspondiente, utilizando los mismos nombres de encabezado:**

- **Red hidrográfica (.shp)**
  - `from_node` (int): Identificador único del nodo origen de cada arco.
  - `to_node` (int): Identificador único del nodo destino.
  - `time` (float): Duración en **horas** de recorrer cada tramo.
  - `elevmed` (float): Elevación media del tramo en metros.
  - `grid_code` (int): Orden de Strahler del tramo>= 0).
  - `longitud`: Longitud de ese tramo (metros).
  - Geometría: líneas (`LineString`).
  - Los siguientes atributos administrativos son opcionales, esto es útil si requiere que los datos de salida incluya estos datos.
    - `departamen`: Nombre del Departamento
    - `nombre_ent`: Nombre del Municipio o Distrito
    - `nombre_geo`: Nombre del cuerpo de agua

- **Sitios de colecta (Excel, .xlsx):**
  - `sample_id`, `place_name`, `latitude`, `longitude`, `species_name`, `min_time`, `max_time`.
  - `sample_id:` identificador único de cada muestra.
  - `place_name:` nombre del sitio de colecta.
  - `latitude:` coordenada de latitud del sitio de colecta en WGS84 (p.e. 7,2345678).
  - `longitude:` coordenada de longitud del sitio de colecta en WGS84 (p.e. -73,4567890).
  - `species_name:` nombre de la especie.
  - `min_time:` tiempo mínimo en horas de desarrollo de ictioplancton colectado.
  - `max_time:` tiempo máximo en horas de desarrollo de ictioplancton colectado.
  
- **Centrales hidroeléctricas (.shp)**
  - `status` (string): Estado de la central (p.ej. "Operativa").
  - Geometría: puntos (`Point`).

Asegúrate de que los nombres de columnas coincidan exactamente.
""")

# ----------------------
# CARGA ANTICIPADA: Centrales Hidroeléctricas para el multiselect
# ----------------------
def_shp_hydro = "DBase_Proyectos_Hidroelectricos_Magdalena/Proyectos_Hidroelectricos.shp"
try:
    hydro_full = gpd.read_file(def_shp_hydro)
    hydro_full.columns = hydro_full.columns.str.strip().str.lower().str.replace(' ', '_')
    status_vals = hydro_full["status"].unique().tolist()
except Exception:
    hydro_full = None
    status_vals = []

# ----------------------
# Parámetros de entrada
# ----------------------
st.sidebar.header("Entradas de Usuario")
shp_zip = st.sidebar.file_uploader("Subir ZIP de Shapefile de red hidrográfica", type="zip")
xls_file = st.sidebar.file_uploader("Subir Excel de sitios de Colecta (.xlsx)", type=["xlsx"])
hydro_zip = st.sidebar.file_uploader("Subir ZIP de Shapefile de Centrales Hidroeléctricas", type="zip")

# ----------------------
# Parámetros de filtrado
# ----------------------
st.sidebar.header("Parámetros de Filtrado")
status_sel = st.sidebar.multiselect(
    "Estatus de centrales",
    options=status_vals,
    default=status_vals
)
elev_max = st.sidebar.slider("Elevación máxima (m)", 0, 5000, 1200)
grid_min = st.sidebar.slider("Strahler mínimo", 0, 10, 2)

# ---------------------------------------
# “Ejecutar Análisis”
# ---------------------------------------
if st.sidebar.button("Ejecutar Análisis"):
    # Validar entradas
    if shp_zip is None:
        st.error("Por favor suba el ZIP del shapefile de red hidrográfica.")
        st.stop()
    if xls_file is None:
        st.error("Por favor suba el archivo Excel de sitios.")
        st.stop()
    if hydro_zip is None and hydro_full is None:
        st.error("Por favor suba el ZIP del shapefile de centrales hidroeléctricas.")
        st.stop()

    # -----------------------
    # 1. Descomprimir y cargar shapefiles desde ZIPs
    # -----------------------
    # Red hidrográfica
    with tempfile.TemporaryDirectory() as tmp_arcs:
        with open(os.path.join(tmp_arcs, "red.zip"), "wb") as f:
            f.write(shp_zip.read())
        with zipfile.ZipFile(os.path.join(tmp_arcs, "red.zip"), "r") as zf:
            zf.extractall(tmp_arcs)
        # Buscar .shp dentro de tmp_arcs
        for fname in os.listdir(tmp_arcs):
            if fname.lower().endswith(".shp"):
                path_arcs = os.path.join(tmp_arcs, fname)
                break
        arcs = gpd.read_file(path_arcs)

    # Centrales hidroeléctricas
    if hydro_zip is not None:
        with tempfile.TemporaryDirectory() as tmp_hydro:
            with open(os.path.join(tmp_hydro, "hydro.zip"), "wb") as f:
                f.write(hydro_zip.read())
            with zipfile.ZipFile(os.path.join(tmp_hydro, "hydro.zip"), "r") as zf:
                zf.extractall(tmp_hydro)
            for fname in os.listdir(tmp_hydro):
                if fname.lower().endswith(".shp"):
                    path_hydro = os.path.join(tmp_hydro, fname)
                    break
            hydro = gpd.read_file(path_hydro)
    else:
        hydro = hydro_full.copy()

    # Excel de sitios
    sites = pd.read_excel(xls_file, engine='openpyxl')

    # -----------------------
    # 2. Normalizar nombres de columnas
    # -----------------------
    arcs.columns = arcs.columns.str.strip().str.lower().str.replace(' ', '_')
    sites.columns = sites.columns.str.strip().str.lower().str.replace(' ', '_')
    hydro.columns = hydro.columns.str.strip().str.lower().str.replace(' ', '_')

    # -----------------------
    # 3. Filtrar centrales según estados seleccionados
    # -----------------------
    if status_sel:
        hydro = hydro[hydro["status"].isin(status_sel)].copy()

    # -----------------------
    # 4. Filtrar arcos por elevación y Strahler
    # -----------------------
    darcs = arcs[(arcs['elevmed'] <= elev_max) & (arcs['grid_code'] >= grid_min)].copy()
    # El peso “time” ya está en horas:
    darcs['weight_time'] = darcs['time']
    # Longitud de cada arco en unidades de CRS:
    darcs['length_m'] = darcs.geometry.length
    darcs['weight_len'] = darcs['length_m']

    # -----------------------
    # 5. Construir dos grafos dirigidos:
    #    G_time: peso = horas
    #    G_len:  peso = longitud
    # -----------------------
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

    # -----------------------
    # 7. Poda en centrales: cortar aristas aguas arriba de nodos de centrales
    # -----------------------
    hydro_nodes = sjoin_nearest(
        hydro.to_crs(darcs.crs),
        nodes_all[['node','geometry']],
        how='left',
        distance_col='hydro_dist'
    )['node'].unique()
    for hn in hydro_nodes:
        if G_time_rev.has_node(hn):
            for succ in list(G_time_rev.successors(hn)):
                G_time_rev.remove_edge(hn, succ)
        if G_len_rev.has_node(hn):
            for succ in list(G_len_rev.successors(hn)):
                G_len_rev.remove_edge(hn, succ)

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
        how='left',
        distance_col='dist'
    ).rename(columns={'node':'node_init'})

    # -----------------------
    # 9. Calcular rutas y acumular horas y longitud
    # -----------------------
    records = []
    for _, site in gdf_sites.iterrows():
        nid = int(site['node_init'])
        lengths_time = nx.single_source_dijkstra_path_length(G_time_rev, nid, weight='weight')
        lengths_len  = nx.single_source_dijkstra_path_length(G_len_rev,  nid, weight='weight')

        valid = {n for n, d in lengths_time.items() if site['min_time'] <= d <= site['max_time']}
        sub = darcs[
            darcs['from_node'].isin(valid) &
            darcs['to_node'].isin(valid)
        ].copy()

        sub['sample_id']  = site['sample_id']
        sub['place_name'] = site['place_name']
        sub['species_nm'] = site['species_name']
        sub['cum_hrs'] = sub['to_node'].map(lengths_time)
        sub['cum_len'] = sub['to_node'].map(lengths_len)
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
        result_gdf = gpd.GeoDataFrame(
            pd.concat(records, ignore_index=True),
            geometry='geometry',
            crs=darcs.crs
        )
    else:
        cols = list(darcs.columns) + [
            'sample_id','place_name','species_nm','cum_hrs','cum_len',
            'departamen','nombre_ent','nombre_geo','cod_munici','arcid_1'
        ]
        result_gdf = gpd.GeoDataFrame(columns=cols, crs=darcs.crs)

    # -----------------------
    # 11. Agregar totales por muestra (horas, minutos y longitud total)
    # -----------------------
    totals = (
        result_gdf
        .groupby('sample_id')[['cum_hrs','cum_len']]
        .sum()
        .reset_index()
        .rename(columns={'cum_hrs':'total_hrs','cum_len':'total_len'})
    )
    totals['total_min'] = totals['total_hrs'] * 60
    result_gdf = result_gdf.merge(totals, on='sample_id', how='left')

    # -----------------------
    # 12. Longitud de cada arco (en unidades de CRS)
    # -----------------------
    result_gdf['length'] = result_gdf.geometry.length

    # -----------------------
    # 13. Mostrar mapa de rutas (puntos) en Streamlit
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
        'length','cum_len','departamen','nombre_ent','nombre_geo','cod_munici',
        'place_name','species_nm','cum_hrs',
    ]].copy()
    csv_df = csv_df.rename(columns={'length':'longitud'})
    csv_bytes = csv_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')

    st.markdown("""
**Los archivos de salida contienen la información relevante que se detalla a continuación:**
- `elevmed`: Elevación media del tramo.
- `arcid_1`: Identificador del tramo.
- `time`: Tiempo (horas) que demora el agua a lo largo de ese tramo.
- `longitud`: Longitud de ese tramo (metros).
- `cum_len`: Longitud acumulada desde el sitio de colecta hasta el final de ese tramo.
- `departamen`, `nombre_ent`, `nombre_geo`, `cod_munici`: atributos administrativos y de cuenca.
- `place_name`: Nombre del punto de colecta.
- `species_nm`: Especie muestreada.
- `cum_hrs`: Tiempo acumulado (horas) desde el sitio de colecta hasta el final de ese tramo.
""")
    st.download_button("Descargar CSV", data=csv_bytes, file_name="arcos_desove.csv", mime="text/csv")

    # -----------------------
    # 15. Exportar GeoJSON (con todos los atributos) y ofrecer descarga
    # -----------------------
    arcos_wgs = result_gdf.to_crs("EPSG:4326")
    arcos_wgs.to_file("arcos_desove.geojson", driver="GeoJSON")
    with open("arcos_desove.geojson", "rb") as f:
        geojson_bytes = f.read()
    st.download_button(
        "Descargar GeoJSON",
        data=geojson_bytes,
        file_name="arcos_desove.geojson",
        mime="application/json"
    )

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
        st.download_button(
            "Descargar Shapefile (.zip)",
            data=buf.getvalue(),
            file_name="arcos_desove_shp.zip",
            mime="application/zip"
        )

    st.success("Procesamiento completado!")

    # ----------------------------
    # 17. Generar y mostrar mapa en Python
    # ----------------------------
    try:
        arcos = gpd.read_file("arcos_desove.geojson").to_crs(4326)
        depts = gpd.read_file("Departamentos/Dpto_84.shp").to_crs(4326)
        rios  = gpd.read_file("SHP_Magdalena/Con_altitud/Red_Magdalena.shp").to_crs(4326)

        fig, ax = plt.subplots(figsize=(10, 8))
        depts.plot(ax=ax, facecolor="lightgray", edgecolor="gray", linewidth=0.3)
        rios.plot(ax=ax, color="steelblue", linewidth=0.4)

        unique_ids = arcos["sample_id"].unique()
        cmap = plt.get_cmap("viridis", len(unique_ids))
        color_map = {sid: cmap(i) for i, sid in enumerate(unique_ids)}

        # Proxy artists para la leyenda
        for sid in unique_ids:
            ax.plot([], [], color=color_map[sid], linewidth=2, label=str(sid))

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
        fig.savefig("mapa_arcos_desove.png", dpi=600)

        with open("mapa_arcos_desove.png", "rb") as img_file:
            st.download_button(
                "Descargar mapa de desove",
                img_file.read(),
                file_name="mapa_arcos_desove.png",
                mime="image/png"
            )
        st.subheader("Mapa de desove")
        st.image(
            "mapa_arcos_desove.png",
            caption="Rutas de desove sobre mapa de Colombia",
            use_container_width=True
        )
        plt.close(fig)

    except Exception as e:
        st.warning(f"No se pudo generar el mapa en Python: {e}")

    r_snippet = """
    # -------------------- Paquetes necesarios --------------------
    install.packages(c("sf", "ggplot2", "dplyr", "viridis"), dependencies = TRUE)

    library(sf)
    library(ggplot2)
    library(dplyr)
    library(viridis)

    # -------------------- 1. Cargar capas --------------------
    dep <- st_read("Departamentos/Dpto_84.shp") %>%
      st_transform(4326)

    rios <- st_read("SHP_Magdalena/Con_altitud/Red_Magdalena.shp") %>%
      st_transform(4326)

    arcos_geo <- st_read("arcos_desove.geojson") %>%
      st_transform(4326)

    # -------------------- 2. Mapa general --------------------
    mapa_general <- ggplot() +
      geom_sf(data = dep, fill = "gray95", color = "gray70", size = 0.3) +
      geom_sf(data = rios, color = "lightblue", size = 0.15, alpha = 0.3) +
      geom_sf(data = arcos_geo,
              aes(color = as.factor(sample_id)),
              size = 1.2, alpha = 0.9) +
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
      geom_sf(data = arcos_geo,
              aes(color = as.factor(sample_id)),
              size = 1.2, alpha = 0.9) +
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
