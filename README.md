# TFG_Machine_Learning
Una de las enfermedades del cerebro y de la médula espinal que afecta al sistema nervioso central es la esclerosis múltiple, y en este proyecto encontrarás la metodología utilizada para poder clasificar si un sujeto tiene o no la enfermedad mediante la inteligencia artificial. El trabajo está enfocado de dos maneras distintas: la primera es hacer una clasificación utilizando un conjunto de matrices aplicando un grupo de técnicas de reducción de atributos y finalmente entrenar con varios modelos. La segunda, se basa en aplicar unos procesos basados en características derivadas de los grafos (aplicar métricas) para finalmente entrenar con varios modelos. Al final valoramos todos los resultados obtenidos escogiendo los mejores procesamientos y modelos. Un pequeño adelanto de los resultados, en más de un modelo obtenemos un Accuracy de más del 85%.

# Carpetas  y su contenido.
  - classificationmodels: el contenido de esta carpeta es entrenar todos los modelos y poderlos guardar
  - confidata: el contenido de esta carpeta contiene todo lo necesario para poder hacer la transformación de todas las matrices a solo 3 y pasar de matrices a grafos aplicando métricas
  - data: el contenido de esta carpeta contiene toda la base de datos con otra información que vamos generando a lo largo del proyecto.
  - datapreparation: el contenido de esta carpeta contiene toda lo necesario para realizar el análisis de la bd y crear matrices con PCA y t_strudent
  
  # Ejecutar el contenido:
  Solo es necesario ejecutar el main, ya que está preparado para hacer todos los distintos procedimientos
