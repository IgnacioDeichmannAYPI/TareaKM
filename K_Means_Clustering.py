from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Trabajo Práctico: Ignacio Deichmann.

filepath = "perros.csv"
K_clusters = 3


def eliminar_campo(campo, arreglo):
    m = []
    for i, x in enumerate(arreglo):
        if i != campo:
            m.append(x)
    return m


def elemento_en_campo(campo, arreglo):
    for i, x in enumerate(arreglo):
        if i == campo:
            return x


def vector_aleatorio(linea):
    m = []
    for txt in linea:
        try:
            m.append(float(txt))
        except:
            m.append(0)
    return m


guardar = []
registros = []
with open(filepath, mode='r', encoding='UTF-8') as archivo:
    leido = archivo.readlines()
    leidos = []
    for linea in leido:
        leidos.append(linea.strip().split(","))

    for i, registro in enumerate(leidos):
        if i != 0:
            linea = registro
            guardar.append(elemento_en_campo(0, linea)+")" +
                           elemento_en_campo(1, linea))
            linea = eliminar_campo(1, linea)
            linea = vector_aleatorio(linea)
            registros.append(linea)

varianzas = np.std(np.array(registros), axis=0)
medias = np.mean(np.array(registros), axis=0)

print("VARIANZA")
print(varianzas)
scaler = StandardScaler()
scaler.fit(registros)
print("MEDIA")
print(scaler.mean_)
print("Z")
X_scaled = scaler.transform(registros)
print(X_scaled)

print("Algoritmo KMeans Clustering + Algoritmo De LLoyd")
kmeans = KMeans(n_clusters=K_clusters, init='k-means++',
                max_iter=300, n_init=67, random_state=0)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
            c=cluster_labels, cmap='viridis', s=40, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='blue', marker="x", s=300, label='Centroides')

plt.xlabel('Agrupamiento 1 (Estandarizado)')
plt.ylabel('Agrupamiento 2 (Estandarizado)')
plt.title('KMeans Clustering ('+str(K_clusters)+')')

plt.legend()
plt.show()

print("AGRUPAMIENTOS DEL CLUSTERING")
resultado_clustering = zip(cluster_labels, guardar)
ordenar = dict()
for key, value in resultado_clustering:
    print(key, value)
