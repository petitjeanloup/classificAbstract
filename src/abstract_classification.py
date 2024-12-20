import openai
import pandas as pd

# Clé d'API OpenAI (remplace par ta propre clé d'API)

openai.api_key = "your-api-key-here"

# Liste d'abstracts d'exemple
abstracts = [

    "L'agriculture de précision utilise des technologies pour optimiser l'utilisation des ressources. Cela inclut l'utilisation de drones pour surveiller les cultures.",
    "Le changement climatique a un impact direct sur la productivité des cultures. Des stratégies d'adaptation sont nécessaires pour maintenir la sécurité alimentaire.",
    "Les pratiques de culture durable sont essentielles pour réduire l'empreinte carbone de l'agriculture tout en maintenant la rentabilité des fermes."

]

# Mettre les abstracts dans un DataFrame pandas pour plus de flexibilité
df = pd.DataFrame(abstracts, columns=["Abstract"])

# Liste d'étiquettes existantes pour la classification
# Exemple d'étiquettes : "Précision", "Changement Climatique", "Culture Durable"

existing_labels = ["Précision", "Changement Climatique", "Culture Durable"]

# Fonction pour interroger l'API OpenAI avec un prompt
def get_label_from_openai(abstract, existing_labels=None):

    # Préparation du prompt

    prompt = f"Voici un abstract sur l'agronomie :\n\n{abstract}\n\nQuel est le sujet principal de cet abstract ?"

    # Si tu as une liste d'étiquettes existantes, tu peux l'ajouter au prompt pour aider à la classification
    if existing_labels:

        prompt += "\n\nLes sujets possibles sont les suivants : " + ", ".join(existing_labels)
        prompt += "\n\nAttribue l'étiquette la plus appropriée parmi ces options."
    
    # Appel à l'API OpenAI pour obtenir la réponse
    response = openai.Completion.create(

        engine="text-davinci-003",  # Ou un modèle similaire si nécessaire
        prompt=prompt,
        max_tokens=50,  # Limite la longueur de la réponse
        n=1,  # Nombre de résultats
        stop=None,  # Pas de caractère de fin spécifique
        temperature=0.7  # Le niveau de créativité, 0.7 est un bon compromis pour des réponses cohérentes

    )

    # Retourner la réponse générée (c'est l'étiquette suggérée)
    return response.choices[0].text.strip()

# Appliquer la fonction pour chaque abstract et ajouter les étiquettes au DataFrame
df['Label'] = df['Abstract'].apply(lambda x: get_label_from_openai(x, existing_labels))

# Afficher les résultats
print(df)
