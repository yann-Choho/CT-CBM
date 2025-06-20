import torch
import pandas as pd
import numpy as np
import ast
from ast import literal_eval
import re
# from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import joblib
from sklearn.metrics import pairwise
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 


# KEPT TO LOAD Gemma in the notebook (need to be called there)
def load_model(model_name, access_token):
    n_gpus = torch.cuda.device_count()
    # max_memory = "10000MB"
    max_memory = "80GB"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
        token = access_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name , token = access_token)
 
    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token
 
    return model, tokenizer

# MODIFIED
def extract_target_words(attributions, top_n=1, min_score=0):
    """ Extrait les n mots avec les scores d'attributions les plus élevés. """
    # Filtrer les attributions supérieures à min_score 
    filtered_words = [(token, score) for token, score in attributions if score > min_score]
    top_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)[:top_n]
    return top_words  # Retourne le mot et son score

# NEW
def find_all_occurrences(text, target_word):
    """ Trouve toutes les positions (index) du mot cible dans le texte. """
    positions = []
    start = 0
    while start < len(text):
        pos = text.find(target_word, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + len(target_word)
    return positions

# MODIFIED
def create_context_window(text, word_position, window_size=5):
    """ Crée une sous-phrase autour d'une occurrence spécifique du mot cible. """
    words = text.split()
    
    # Calculer l'index du mot cible en comptant le nombre de mots avant la position donnée
    target_word_index = len(text[:word_position].split())  # Nombre de mots avant le mot cible
    
    # Définir les limites de la fenêtre de contexte
    start = max(0, target_word_index - window_size)
    end = min(len(words), target_word_index + window_size + 1)
    
    # Retourner la fenêtre de contexte
    return ' '.join(words[start:end])


# PS: Version utilisant le centroid de chaque cluster
def run_concepts_discovery(df_attribution, model=None, tokenizer=None, top_n=1, save_path=''):
    """ Function to run the concepts discovery pipeline
    Args:
        df_attribution (pd.DataFrame): the dataframe containing the attributions
    Returns:
        unique_concepts (list): the list of unique concepts extracted
    """
    import hdbscan

    sentence_transformer_name='all-mpnet-base-v2'
    mpnet_model = SentenceTransformer(sentence_transformer_name)

    # Charger UMAP et DBSCAN
    reducer = joblib.load(f'{save_path}/concepts_discovery/umap_model.pkl')
    clusterer = joblib.load(f'{save_path}/concepts_discovery/dbscan_model.pkl')

    # Charger le mapping entre clusters et macro-concepts
    with open(f'{save_path}/concepts_discovery/number_to_macro.json', 'r') as f:
        number_to_macro = json.load(f)

    responses = []

    # Parcours des lignes du DataFrame
    new_concepts = []
    for index, row in df_attribution.iterrows():
        text = row['text']
        text = text.replace(" ##", "")  #recoller les mot couper dans le text original
        attributions = row['word_attributions'] #eval(row['word_attributions'])  # Évaluer la chaîne de caractères en liste

        # Extraire les mots avec les top-n attributions les plus élevées
        target_words_with_scores = extract_target_words(attributions, top_n=top_n, min_score=0)
        target_words = [word for word, score in target_words_with_scores]

        for i, (target_word, highest_score) in enumerate(target_words_with_scores):
            # Trouver toutes les positions du mot dans le texte
            target_word_positions = find_all_occurrences(text, target_word)

            # Identifier l'occurrence du mot ayant la plus haute attribution
            occurrence_count = 0
            exact_position = None
            for word, score in attributions:
                if word == target_word:
                    if score == highest_score:  # Si c'est l'occurrence ayant le plus haut score
                        
                        exact_position = target_word_positions[occurrence_count]
                        break
                    occurrence_count += 1

            if exact_position is not None:
                # Créer une fenêtre de contexte autour du mot
                context = create_context_window(text, word_position=exact_position)
                # print(f"Contexte autour de l'occurrence '{target_word}' avec la plus haute attribution : {context}")
                
                # Obtenir le vecteur d'embedding du contexte
                context_vector = mpnet_model.encode(context)

                # Réduire la dimensionnalité du vecteur du contexte
                reduced_vector = reducer.transform(context_vector.reshape(1, -1))

                # Prédire le cluster pour le vecteur réduit
                cluster_label = hdbscan.approximate_predict(clusterer, reduced_vector) #HBDSCAN use approximate_predict and not predict :it output a array pf label

                # Associer le cluster du point à un macro-concept
                if str(cluster_label[0][0]) in number_to_macro:  #applique la condition sur les keys() directly
                    best_macro_concept = number_to_macro[str(cluster_label[0][0])]
                    # print("best_macro_concept", best_macro_concept)
                else:
                    best_macro_concept = None
                    # print("cluster_label[0][0] not in number_to_macro")
                # Ajouter le meilleur concept macro au DataFrame
                df_attribution.loc[index, 'targeted_context'] = context
                df_attribution.loc[index, 'targeted_word'] = target_word
                df_attribution.loc[index, f'best_macro_concept_{i}'] = best_macro_concept
            else:
                # print(f"Aucune occurrence de '{target_word}' trouvée avec le plus haut score.")
                continue

    # Sauvegarder le DataFrame mis à jour avec les nouveaux concepts
    df_attribution.to_csv(f'{save_path}/df_attribution_v2.csv', index=False)

    return df_attribution

# TODO: à coder
# PS: Version utilisant le centroid de chaque cluster
def run_concepts_discovery_v_centroid(df_attribution, model=None, tokenizer=None, top_n=1, save_path=''):
    """ Function to run the concepts discovery pipeline
    Args:
        df_attribution (pd.DataFrame): the dataframe containing the attributions
    Returns:
        unique_concepts (list): the list of unique concepts extracted
    """
    import hdbscan

    sentence_transformer_name='all-mpnet-base-v2'
    mpnet_model = SentenceTransformer(sentence_transformer_name)

    # Charger le dictionnaire des centroïdes des concepts macro
    # with open(f'{save_path}/macro_to_centroid.json', 'r') as f:
    #     macro_to_centroid = json.load(f)  # {'macro_concept1': [embedding of centroid], ...}
    # Convertir les centroïdes de listes Python en tableaux NumPy
    # for cluster, centroid in number_to_centroids_.items():
    #     number_to_centroids_[cluster] = np.array(centroid)

    # Charger UMAP et DBSCAN
    reducer = joblib.load(f'{save_path}/concepts_discovery/umap_model.pkl')
    clusterer = joblib.load(f'{save_path}/concepts_discovery/dbscan_model.pkl')

    # Charger le mapping entre clusters et macro-concepts
    with open(f'{save_path}/concepts_discovery/number_to_macro.json', 'r') as f:
        number_to_macro = json.load(f)

    responses = []

    # Parcours des lignes du DataFrame
    new_concepts = []
    for index, row in df_attribution.iterrows():
        text = row['text']
        text = text.replace(" ##", "")  #recoller les mot couper dans le text original
        attributions = row['word_attributions'] #eval(row['word_attributions'])  # Évaluer la chaîne de caractères en liste

        # Extraire les mots avec les top-n attributions les plus élevées
        target_words_with_scores = extract_target_words(attributions, top_n=top_n, min_score=0)
        target_words = [word for word, score in target_words_with_scores]

        for i, (target_word, highest_score) in enumerate(target_words_with_scores):
            # Trouver toutes les positions du mot dans le texte
            target_word_positions = find_all_occurrences(text, target_word)

            # Identifier l'occurrence du mot ayant la plus haute attribution
            occurrence_count = 0
            exact_position = None
            for word, score in attributions:
                if word == target_word:
                    if score == highest_score:  # Si c'est l'occurrence ayant le plus haut score
                        
                        exact_position = target_word_positions[occurrence_count]
                        break
                    occurrence_count += 1

            if exact_position is not None:
                # Créer une fenêtre de contexte autour du mot
                context = create_context_window(text, word_position=exact_position)
                # print(f"Contexte autour de l'occurrence '{target_word}' avec la plus haute attribution : {context}")
                
                # Obtenir le vecteur d'embedding du contexte
                context_vector = mpnet_model.encode(context)

                # Réduire la dimensionnalité du vecteur du contexte
                reduced_vector = reducer.transform(context_vector.reshape(1, -1))

                # Prédire le cluster pour le vecteur réduit
                cluster_label = hdbscan.approximate_predict(clusterer, reduced_vector) #HBDSCAN use approximate_predict and not predict :it output a array pf label

                # Associer le cluster du point à un macro-concept
                if str(cluster_label[0][0]) in number_to_macro:  #applique la condition sur les keys() directly
                    best_macro_concept = number_to_macro[str(cluster_label[0][0])]
                    # print("best_macro_concept", best_macro_concept)
                else:
                    best_macro_concept = None
                    # print("cluster_label[0][0] not in number_to_macro")
                # Ajouter le meilleur concept macro au DataFrame
                df_attribution.loc[index, 'targeted_context'] = context
                df_attribution.loc[index, 'targeted_word'] = target_word
                df_attribution.loc[index, f'best_macro_concept_{i}'] = best_macro_concept
            else:
                # print(f"Aucune occurrence de '{target_word}' trouvée avec le plus haut score.")
                continue

            # LOGIC COSINE SIMILARITY
            # # Calculer la similarité cosinus entre le vecteur du contexte et les centroïdes
            # best_macro_concept = None
            # max_similarity = -1
            # for macro_concept, centroid_vector in macro_to_centroid.items():
            #     similarity = cosine_similarity([context_vector], [centroid_vector])[0][0]
            #     if similarity > max_similarity:
            #         max_similarity = similarity
            #         best_macro_concept = macro_concept

            # Ajouter le meilleur concept macro au DataFrame
            # df_attribution.loc[index, 'targeted_context'] = context
            # df_attribution.loc[index, 'targeted_word'] = target_word
            # df_attribution.loc[index, f'best_macro_concept_{i}'] = best_macro_concept
            # best_macro_concept_per_target_word.append(best_macro_concept)

        # Sauvegarder les réponses obtenues pour cette ligne
        # responses.append(best_macro_concept_per_target_word)

    # Sauvegarder le DataFrame mis à jour avec les nouveaux concepts
    df_attribution.to_csv(f'{save_path}/df_attribution_v2.csv', index=False)

    return df_attribution

def calculate_macro_concept_frequencies(df, save_path, iter, column='best_macro_concept_0'):
    """ Calcule la fréquence des macro-concepts dans une colonne spécifique du DataFrame, 
    et ajoute les concepts manquants avec une fréquence de 0.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les attributions et les macro-concepts.
        macro_concepts_keys (list): Les clés des macro-concepts provenant de macro_to_centroid.
        column (str): La colonne du DataFrame contenant les macro-concepts à analyser.
        
    Returns:
        concept_frequencies (dic): Un dict contenant les fréquences des macro-concepts, 
                                         incluant ceux absents de la colonne avec une fréquence de 0.
        most_frequent_concept (str): Le macro-concept le plus fréquent.
    """
    # Charger le mapping entre clusters et macro-concepts
    with open(f'{save_path}/concepts_discovery/number_to_macro.json', 'r') as f:
        number_to_macro = json.load(f)
    macro_concepts_keys = list(number_to_macro.values())

    # Vérifier que la colonne existe dans le DataFrame
    if column not in df.columns:
        raise ValueError(f"La colonne {column} n'existe pas dans le DataFrame.")
    
    # Calculer la fréquence des concepts dans la colonne spécifiée
    concept_frequencies = df[column].value_counts()

    # Initialiser les fréquences à 0 pour les concepts absents
    for concept in macro_concepts_keys:
        if concept not in concept_frequencies:
            concept_frequencies[concept] = 0

    # Trier les fréquences des concepts pour garantir un ordre logique
    concept_frequencies = concept_frequencies.reindex(macro_concepts_keys, fill_value=0)

    # Trouver le concept le plus fréquent (en excluant les cas d'égalité)
    most_frequent_concept = concept_frequencies.idxmax()

    # Convert int64 to int
    concept_frequencies_dict = {k: int(v) if isinstance(v, np.int64) else v for k, v in concept_frequencies.items()}

    with open(f'{save_path}/frequencies/current_freq_iter_{iter}.json', 'w') as f:
        json.dump(concept_frequencies_dict, f) 
    
    return concept_frequencies_dict, most_frequent_concept

def update_concept_frequencies(concept_frequencies, concept_frequencies_v2, iter, save_path):
    """ Met à jour les fréquences des concepts en ajoutant les valeurs de concept_frequencies_v2.

    Args:
        concept_frequencies (dict): Les fréquences des concepts actuelles sous forme de dictionnaire.
        concept_frequencies_v2 (dict): Les nouvelles fréquences à ajouter sous forme de dictionnaire.
        iter (int): Le numéro de l'itération pour la sauvegarde.
        save_path (str): Le chemin où enregistrer les fréquences mises à jour.
    
    Returns:
        concept_frequencies_v3 (dict): Une version actualisée des fréquences des concepts.
    """
    # Copier le dictionnaire original pour ne pas modifier directement l'original
    concept_frequencies_v3 = concept_frequencies.copy()

    # Mettre à jour les fréquences en additionnant celles des deux dictionnaires
    for concept, freq in concept_frequencies_v2.items():
        if concept in concept_frequencies_v3:
            concept_frequencies_v3[concept] += freq
        else:
            concept_frequencies_v3[concept] = freq

    # Sauvegarder le dictionnaire mis à jour dans un fichier JSON
    with open(f'{save_path}/frequencies/updated_freq_iter_{iter}.json', 'w') as f:
        json.dump(concept_frequencies_v3, f)

    return concept_frequencies_v3

def find_most_frequent_macro_concept(concept_frequencies_v3, already_taken_macro_concept = []):
    """ Trouve le macro-concept le plus fréquent qui n'est pas déjà dans la liste already_taken_macro_concept.
    
    Args:
        concept_frequencies_v3 (dict): Un dictionnaire contenant les fréquences des concepts.
        already_taken_macro_concept (list): Une liste de concepts déjà sélectionnés.
    
    Returns:
        str: Le macro-concept le plus fréquent non présent dans already_taken_macro_concept.
    """
    print(" concept_frequencies_v3 :", concept_frequencies_v3)
    print(" already_taken_macro_concept :", already_taken_macro_concept)
    # Filtrer les concepts pour ne garder que ceux qui ne sont pas dans already_taken_macro_concept
    filtered_concepts = {concept: freq for concept, freq in concept_frequencies_v3.items() 
                         if concept not in already_taken_macro_concept}
    
    # Si tous les concepts sont déjà dans la liste, retourner None
    if not filtered_concepts:
        return None
    
    # Trouver le concept avec la fréquence la plus élevée
    most_frequent_concept = max(filtered_concepts, key=filtered_concepts.get)
    print(" most_frequent_concept :", most_frequent_concept)
    return most_frequent_concept

