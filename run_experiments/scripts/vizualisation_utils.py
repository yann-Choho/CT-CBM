# visualization_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import plotly.graph_objects as go


### ------------- COMMON STRATEGY VISUALIZATION --------------------------------


### ------------- PROJECTION MODEL VISUALIZATION --------------------------------

def visualize_concepts_weights(projection_model):
    # Extraire les noms des concepts
    concepts_name = projection_model.concepts_name
    concepts_name = list(set(concepts_name))
    
    # Extraire les poids de la couche linéaire pour toutes les classes
    concepts_weights = projection_model.classifier.linear.weight.detach().cpu().numpy()  # Poids pour toutes les classes

    num_classes = concepts_weights.shape[0]  # Nombre de classes

    # Parcourir chaque classe et créer un graphique pour chaque classe
    for i in range(num_classes):
        # Créer un DataFrame pour la classe actuelle
        concepts_weights_df = pd.DataFrame({
            'Concepts': concepts_name,
            'Weights': concepts_weights[i]  # Poids pour la classe i
        })

        # Créer un bar plot pour visualiser les poids de cette classe
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Concepts', y='Weights', data=concepts_weights_df)
        plt.xticks(rotation=45, ha='right')  # Rotation pour mieux voir les concepts
        plt.title(f"Concept Weights for Class {i} in the Projection Model")
        plt.tight_layout()  # Assure que tout rentre dans le cadre
        plt.show()

    return concepts_weights_df

### ------------- JOINT MODEL VISUALIZATION --------------------------------

import matplotlib.pyplot as plt
import pandas as pd

def visualize_concepts_weights(joint_model, class_to_name_dict=None):
    """
    Visualiser les poids des concepts pour chaque classe dans le modèle de projection.
    
    Paramètres :
    - joint_model : Le modèle de projection déjà chargé contenant les concepts et les poids.
    - class_to_name_dict : (Optionnel) Dictionnaire mappant les indices de classes aux noms réels (ex: {0: "documentary"}).
    """
    # Extraire les noms des concepts depuis le modèle
    concepts_name = list(set(joint_model.concepts_name))  # Assurer que les noms des concepts sont uniques
    
    # Extraire les poids de la couche linéaire du classifieur pour toutes les classes
    classifier_weights = joint_model.ModelXtoCtoY_layer.sec_model.linear.weight.detach().cpu().numpy()  # Poids pour toutes les classes

    num_classes = classifier_weights.shape[0]  # Nombre de classes

    print(concepts_name)
    print(classifier_weights.shape)
    # Parcourir chaque classe et créer un graphique pour chaque classe
    for i in range(num_classes):
        # Créer un DataFrame pour la classe actuelle avec tri des poids par valeur absolue
        concepts_weights_df = pd.DataFrame({
            'Concepts': concepts_name,
            'Weights': classifier_weights[i]  # Poids pour la classe i
        })

        # Trier les poids en fonction de la valeur absolue mais garder les valeurs réelles
        concepts_weights_df['abs_weights'] = concepts_weights_df['Weights'].abs()
        concepts_weights_df = concepts_weights_df.sort_values(by='abs_weights', ascending=False)

        # Récupérer le nom de la classe s'il est disponible dans le dictionnaire
        class_name = class_to_name_dict[i] if class_to_name_dict and i in class_to_name_dict else f"Class {i}"

        # Créer un bar plot pour visualiser les poids de cette classe
        plt.figure(figsize=(12, 8))
        plt.barh(concepts_weights_df['Concepts'], concepts_weights_df['Weights'], color='skyblue')
        plt.xlabel('Weights', fontsize=12)
        plt.ylabel('Concepts', fontsize=12)
        plt.title(f"Concept Weights for {class_name} in the Projection Model", fontsize=14, fontweight='bold')
        
        # Inverser l'axe Y pour que les concepts avec les plus grands poids soient en haut
        plt.gca().invert_yaxis()
        
        # Affichage propre du graphique
        plt.tight_layout()
        plt.show()


#### ------------------------------- DIAGRAM DE SANKEY ---------------------
# Fonction pour visualiser les poids des concepts pour chaque classe via un Sankey Diagram
def visualize_top_5_concepts_per_class_sankey(joint_model, class_to_name_dict=None):
    """
    Visualiser les 5 concepts les plus influents pour chaque classe dans le modèle de projection sous forme de Sankey Diagram.

    Paramètres :
    - joint_model : Le modèle de projection déjà chargé contenant les concepts et les poids.
    - class_to_name_dict : (Optionnel) Dictionnaire mappant les indices de classes aux noms réels (ex: {0: "documentary"}).
    """
    # Extraire les noms des concepts depuis le modèle
    concepts_name = list(set(joint_model.concepts_name))  # Assurer que les noms des concepts sont uniques

    # Extraire les poids de la couche linéaire du classifieur pour toutes les classes
    classifier_weights = joint_model.ModelXtoCtoY_layer.sec_model.linear.weight.detach().cpu().numpy()  # Poids pour toutes les classes

    num_classes = classifier_weights.shape[0]  # Nombre de classes

    # Palette de couleurs spécifique pour les classes
    palette = ["#AFC2D5", "#CCDDD3", "#DFEFCA", "#FFF9A5", "#B48B7D"]

    # Préparer les données pour le Sankey Diagram
    labels = []
    concept_index_map = {}

    for concept in concepts_name:
        labels.append(concept)
        concept_index_map[concept] = len(labels) - 1

    if class_to_name_dict:
        for i in range(num_classes):
            labels.append(class_to_name_dict[i])
    else:
        for i in range(num_classes):
            labels.append(f"Class {i}")

    sources = []
    targets = []
    values = []
    link_colors = []

    # Construire les liens entre concepts et classes
    for i in range(num_classes):
        class_index = len(concepts_name) + i

        # Obtenir les indices des 5 concepts les plus influents pour la classe actuelle
        top_5_indices = (-classifier_weights[i]).argsort()[:5]

        for j in top_5_indices:
            weight = classifier_weights[i][j]
            if abs(weight) > 0:  # Inclure les poids non nuls
                concept_label = f"NOT {concepts_name[j]}" if weight < 0 else concepts_name[j]
                if concept_label not in labels:
                    labels.append(concept_label)
                    concept_index_map[concept_label] = len(labels) - 1
                sources.append(concept_index_map[concepts_name[j]])
                targets.append(class_index)  # Index de la classe
                values.append(abs(weight))  # Utiliser la valeur absolue des poids

                # Définir la couleur du lien comme la couleur du nœud cible (classe)
                class_color = palette[i % len(palette)]  # Utiliser une couleur de la palette
                link_colors.append(class_color)

    # Créer le Sankey Diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=10,  # Réduire l'espace entre les nœuds
            thickness=15,  # Réduire l'épaisseur des nœuds
            line=dict(color="black", width=0.5),
            label=labels,
            color=["#AFC2D5", "#CCDDD3", "#DFEFCA", "#FFF9A5", "#B48B7D"] * (len(labels) // 5 + 1)  # Appliquer les couleurs aux nœuds
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors  # Appliquer les couleurs aux liens
        )
    ))

    fig.update_layout(
        title_text="Sankey Diagram des 5 concepts principaux par classe",
        font_size=10,
        margin=dict(l=50, r=50, t=50, b=50)  # Réduire les marges pour un rendu plus compact
    )
    fig.show()

#### ------------------------------- OTHERS ---------------------
