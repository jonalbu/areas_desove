

## Áreas de Estimación de desove 

Las estimaciones presentadas se basan en la información sobre las velocidades de las corrientes de agua de la red hídrica de la cuenca de los ríos Magdalena y Cauca, según lo reportado por **López-Casas et al. (2025)**. En dicho artículo, además, se exponen los resultados de la estimación de las áreas de desove de las especies potádromas presentes en la cuenca.

Debido a las restricciones de derechos de uso asociadas al aplicativo **Matlab** empleado para el cálculo de las áreas de desove, disponible en el repositorio: https://github.com/N4W-Facility/Spawning_Ground_Model, se desarrolló un modelo alternativo de estimación utilizando software de código abierto **(Python 3.12.4)**. Este modelo se fundamenta en la información de la red hídrica y permite incorporar datos independientes de la cuenca, facilitando así su adaptación y uso en diferentes contextos.

**Es importante destacar que, los archivos de datos a utilizar en el modelo debe contener los nombres de columnas correspondientes que se presentan a continuación:**



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

El archivo debe estar en formato .shp y contener las siguientes columnas:

  - `status` (string): Estado de la central (p.ej. "Operativa", "En construcción").
  - Geometría: puntos (`Point`) ubicados en la red.

Asegúrate de que los nombres de columnas coincidan exactamente y que las geometrías estén válidas.


### Referencias
López-Casas, S., Rogéliz-Prada, C. A., Atencio-García, V., Moreno-Árias, C., Arenas, D., Rivera-Coley, K., & Jimenez-Segura, L. (2025). Spawning grounds model for neotropical potamodromous fishes: conservation and management implications. Frontiers in Environmental Science, 13, 1425804. URL=https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2025.1425804, DOI=10.3389/fenvs.2025.1425804

