from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import pandas as pd
import matplotlib.pyplot as plt


file_path = "./data/abstracts.csv"
data = pd.read_csv(file_path, sep=';')
#data.head()

# Extraire les abstracts et enlever les valeurs manquantes
abstracts = data['Abstract'].dropna().tolist()
#len(abstracts)

# Vectorisation TF-IDF des abstracts
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(abstracts)

# Déterminer le nombre optimal de clusters avec la méthode du coude

inertia = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Tracer la courbe du coude

plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.title("Méthode du coude pour déterminer k optimal")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.show()


n_clusters = 6
kmeans = KMeans(n_clusters = n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

data['Cluster'] = -1  # Initialisation
data.loc[data['Abstract'].notna(), 'Cluster'] = clusters

print(f"{data[data['Cluster'] == -1].shape[0]} articles sans abstracts dans la base de données")
for c in range(n_clusters):
  print(f"In cluster {c} : ", data[data["Cluster"] == c].shape[0])


top_terms_per_cluster = []
centroids = kmeans.cluster_centers_
terms = vectorizer.get_feature_names_out()

for i in range(n_clusters):

    top_terms = [terms[ind] for ind in centroids[i].argsort()[-15:]]

    top_terms_per_cluster.append(top_terms)


for i, top_term_in_cluster in enumerate(top_terms_per_cluster):
  print(f"Top terms in cluster {i} : ",top_term_in_cluster)