import os
import io
import subprocess  # <— Asegúrate de importar subprocess
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
st.title("Estimación Áreas de desove de Peces dulceacuícolas para la cuenca Magdalena-Cauca")

# Instrucciones de formato de datos
st.markdown("""
**Se requiere que el archivo de Excel de entrada incluya las siguientes columnas, utilizando los mismos nombres de encabezado:**

- **Sitios de colecta (Excel, .xlsx):**
  - `sample_id`, `place_name`, `latitude`, `longitude`, `species_name`, `min_time`, `max_time`.
  - `sample_id:` identificador único de cada muestra.
  - `place_name:` nombre del sitio de colecta.
  - `latitude:` coordenada de latitud del sitio de colecta en WGS84 (p.e. 7,2345678).
  - `longitude:` coordenada de longitud del sitio de colecta en WGS84. (p.e. -73,4567890).
  - `species_name:` nombre de la especie.
  - `min_time:` tiempo mínimo en horas de desarrollo de ictioplancton colectado.
  - `max_time:` tiempo máximo en horas de desarrollo de ictioplancton colectado.

Asegúrate de que los nombres de columnas coincidan exactamente.
""")

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
def_shp_arcs = "SHP_Magdalena/Con_altitud/Red_Magdalena.shp"

# ---------------------------------------
# “Ejecutar Análisis”
# ---------------------------------------
if st.sidebar.button("Ejecutar Análisis"):
    # Validar que se subió Excel
    if xls_file is None:
        st.error("Por favor suba el archivo Excel de sitios.")
        st.stop()

    # -----------------------
    # 1. Carga de todos los datos
    # -----------------------
    arcs  = gpd.read_file(def_shp_arcs)        # Red hidro
    sites = pd.read_excel(xls_file, engine='openpyxl')  # Excel de sitios
    hydro = gpd.read_file(def_shp_hydro)       # Centrales hidroeléctricas

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
    # El peso “time” ya está en horas:
    darcs['weight_time'] = darcs['time']

    # Calculamos la longitud de cada arco en el CRS actual (metros o unidades de proyección)
    darcs['length_m'] = darcs.geometry.length
    # Usamos ese valor como peso de longitud:
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
        how='left', distance_col='hydro_dist'
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
        how='left', distance_col='dist'
    ).rename(columns={'node':'node_init'})

    # -----------------------
    # 9. Calcular rutas y acumular horas y longitud
    # -----------------------
    records = []
    for _, site in gdf_sites.iterrows():
        nid = int(site['node_init'])
        # Longitudes acumuladas hacia aguas arriba
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
        # Horas acumuladas hasta el nodo destino de cada arco:
        sub['cum_hrs'] = sub['to_node'].map(lengths_time)
        # Longitud acumulada hasta el nodo destino de cada arco:
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
        'length','cum_len','departamen','nombre_ent','nombre_geo','cod_munici',
        'place_name','species_nm','cum_hrs',
    ]].copy()
    csv_df = csv_df.rename(columns={'length':'longitud'})
    # Usamos punto decimal y “;” como separador, UTF-8 con BOM:
    csv_bytes = csv_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')

    st.markdown("""
** Los archivos de salida contienen la información relevante que se detalla a continuación:**
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
    # 15. Exportar GeoJSON (con todos los atributos) y guardarlo en disco
    # -----------------------
    # Convertir a EPSG:4326 y guardar a archivo físico
    arcos_wgs = result_gdf.to_crs("EPSG:4326")
    arcos_wgs.to_file("arcos_desove.geojson", driver="GeoJSON")

    # Crear el botón de descarga para quienes quieran el GeoJSON
    st.download_button(
        "Descargar GeoJSON",
        data=arcos_wgs.to_json(),
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
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe  # Para contorno blanco en texto

    try:
        # Leer el GeoJSON que acabamos de generar
        arcos = gpd.read_file("arcos_desove.geojson").to_crs(4326)

        # Leer límites de departamentos y ríos (ajusta rutas si hace falta)
        depts = gpd.read_file("Departamentos/Dpto_84.shp").to_crs(4326)
        rios  = gpd.read_file("SHP_Magdalena/Con_altitud/Red_Magdalena.shp").to_crs(4326)

        # Preparar figura
        fig, ax = plt.subplots(figsize=(10, 8))

        # 1) Dibuja departamentos en gris
        depts.plot(ax=ax, facecolor="lightgray", edgecolor="gray", linewidth=0.3)

        # 2) Dibuja ríos en azul
        rios.plot(ax=ax, color="steelblue", linewidth=0.4)

        # 3) Dibuja cada ruta coloreada por sample_id
        #    y prepara proxy artists para la leyenda:

        unique_ids = arcos["sample_id"].unique()
        cmap = plt.get_cmap("viridis", len(unique_ids))
        color_map = {sid: cmap(i) for i, sid in enumerate(unique_ids)}

        # Crear un proxy artist (línea vacía) por cada sample_id
        for sid in unique_ids:
            ax.plot([], [], color=color_map[sid], linewidth=2, label=str(sid))

        # Ahora recorremos cada tramo y dibujamos las líneas
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

        # Finalmente, dibujar la leyenda con un rótulo por cada sample_id
        ax.legend(
            title="ID Muestra",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8,
            title_fontsize=9
        )

        # 4) Agregar cuadrícula y ejes
        import numpy as np
        xs = np.arange(-82, -66, 1)    # longitudes de ejemplo
        ys = np.arange(-4, 15, 1)      # latitudes de ejemplo

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

        # Botón de descarga para el PNG
        with open("mapa_arcos_desove.png", "rb") as img_file:
            st.download_button(
                "Descargar mapa de desove",
                img_file.read(),
                file_name="mapa_arcos_desove.png",
                mime="image/png"
            )
        plt.close(fig)

        # Mostrar la imagen en Streamlit
        st.subheader("Mapa de desove generado en Python")
        st.image(
            "mapa_arcos_desove.png",
            caption="Rutas de desove sobre mapa de Colombia",
            use_column_width=True
        )

    except Exception as e:
        st.warning(f"No se pudo generar el mapa en Python: {e}")
        
    r_snippet = """
        # Código en R para la creación de mapa a partir

        library(sf)
        library(ggplot2)
        library(dplyr)

        # 1. Leer GeoJSON generado por Streamlit (ajusta ruta si está en subcarpeta):
        arcos_geo <- st_read("arcos_desove.geojson") %>% st_transform(4326)

        # 2. Leer límites de departamentos desde ZIP:
        dep  <- st_read("Municipio_84.shp") %>% st_transform(4326) # la ruta debe contener todos los archivos del .shp. 
        Se puede descargar en la siguiente ruta https://github.com/jonalbu/areas_desove/tree/main/Departamentos/Dpto_84.zip

        # 3. Leer ríos principales desde ZIP:
        rios <- st_read("Red_Magdalena.shp") %>% st_transform(4326) # la ruta debe contener todos los archivos del .shp. 
        # Se puede descargar en la siguiente ruta https://github.com/jonalbu/areas_desove/tree/main/SHP_Magdalena/Con_altitud/Red_Magdalena.zip

        # 4. Graficar todo junto
        p <- ggplot() +
        geom_sf(data = dep, fill="gray95", color="gray40", size=0.3) +
        geom_sf(data = rios, color="steelblue", size=0.4) +
        geom_sf(data = arcos_geo,
                aes(color = as.factor(sample_id)),
                size=0.7, alpha=0.8) +
        scale_color_viridis_d(name = "Sample ID") +
        labs(
            title    = "Rutas de posibles áreas de desove sobre mapa de Colombia",
            x = "Longitud", y = "Latitud"
        ) +
        theme_minimal() +
        theme(
            legend.position = "right",
            plot.title      = element_text(face="bold", size=16)
        )

        # 5. Guardar como PNG en la raíz del proyecto
        ggsave("mapa_arcos_desove.png", plot = p, width = 10, height = 8, dpi = 300)
        """

    st.subheader("Código R para reproducir el mapa fuera de Streamlit")
    st.code(r_snippet, language="r")
