""" TODO: continuar este notebook completando:
- Crea un modelo para clasificar si una flor es de tipo virgínica (```iris.target == 2```) o no lo es (**clasificación binaria**) solo en función de la longitud del pétalo (columna 2 de ```iris.data```).
- Representa gráficamente el modelo y los datos de entrenamiento con ```matplotlib```.
- Calcula y representa cuál sería la predicción del modelo de que una flor con longitud de pétalo de 5 cm sea de tipo virgínica.
 """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_iris
# Cargar el conjunto de datos Iris
iris = load_iris()


# print(type(iris)) # Tipo de objeto
# print(iris.DESCR) # Información del dataset
print(type(iris.target)) # Clases de las flores
print(iris.target_names) # Nombre de las clases
print(iris.target) # Clases de las flores

# print(iris.feature_names) # Datos de las flores
# print(iris.data) # Datos de las flores

#creamos el dataframe con los datos del dataset y las columnas
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Seleccionar únicamente la longitud del pétalo (columna 2) como característica
x = iris_df['petal length (cm)'].values.reshape(-1, 1)

# Etiquetas binarias: 1 si la flor es de tipo virgínica (iris.target == 2), 0 en caso contrario
y = (iris_df['target'] == 2).astype(int).values

# Inicializar el clasificador de regresión logística
log_reg = LogisticRegression()

# Entrenar el clasificador con todos los datos
log_reg.fit(x, y)

# Crear una malla para graficar la función de decisión
x_values = pd.DataFrame({'petal length (cm)': pd.Series(np.linspace(x.min(), x.max(), 100))})

# Obtener las probabilidades predichas
y_values = log_reg.predict_proba(x_values)

# Graficar los datos y la función de decisión
plt.figure(figsize=(10, 5))
plt.scatter(iris_df['petal length (cm)'], y, color='blue', label='Datos')
plt.plot(x_values, y_values[:, 1], color='red', label='Probabilidad de ser virgínica')
plt.xlabel('Longitud del pétalo')
plt.ylabel('Probabilidad de ser de tipo virgínica')
plt.title('Clasificación de flores de tipo virgínica')

# Calcular la predicción para una longitud de pétalo de 5 cm
longitud_petalo = 5
probabilidad = log_reg.predict_proba([[longitud_petalo]])[:, 1]
plt.scatter(longitud_petalo, probabilidad, color='green', label=f'Predicción para longitud 5cm: {probabilidad[0]:.2f}', marker='x', s=100)

plt.legend()
plt.show()
