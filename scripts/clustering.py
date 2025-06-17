import os
import json
import pandas as pd
from sklearn.metrics import pairwise_distances
import hdbscan
from umap import UMAP
import matplotlib.pyplot as plt
from load_config import load_config

def cluster_and_visualize_topics(
    model_name: str,
    dataset: str,
    annotation: str,
    min_cluster_size: int = 2,
    random_state: int = 42
):
    """
    1) Charge le bon fichier de configuration en fonction de model_name et dataset.
    2) Charge et pré-traite df_aug_train selon le type d'annotation ('C3M' ou 'our_annotation').
    3) Extrait la matrice binaire de concepts, calcule la distance de Hamming, fait un clustering HDBSCAN,
       réassigne les points 'noise' ou les garde séparés (deux stratégies),
       construit les dictionnaires cluster→[concepts], affiche les embeddings UMAP, et sauvegarde les JSON.

    Arguments :
    - model_name        : 'bert-base-uncased' ou 'deberta-large'
    - dataset           : 'movies' / 'agnews' / 'dbpedia' / 'medical'
    - annotation        : 'C3M' ou 'our_annotation'
    - min_cluster_size  : taille minimale pour HDBSCAN (défaut=2)
    - random_state      : graine aléatoire pour UMAP (défaut=42)
    """


    config = load_config(model_name, dataset)

        
    # --- B) Chargement et prétraitement du DataFrame selon l'annotation ---
    if annotation == 'C3M':
        # 1) Charger le CSV C3M
        path_csv = f"{config.SAVE_PATH_CONCEPTS}/df_with_topics_v4_C3M.csv"
        df_aug_train = pd.read_csv(path_csv)

        # 2) Renommer les colonnes pour retirer d'éventuels préfixes indésirables
        # df_aug_train.rename(columns=lambda x: x.replace("dummy_", ""), inplace=True)

        # 3) S'assurer que 'text' est str et strip()
        df_aug_train['text'] = df_aug_train['text'].astype(str).str.strip()
        # S'assurer que 'label' est int (ou strip de chaines puis converti)
        if df_aug_train['label'].dtype != int:
            df_aug_train['label'] = df_aug_train['label'].astype(str).str.strip()

        # 4) Filtrer les étiquettes inconnues puis mapper en int
        with open(f"{config.SAVE_PATH_CONCEPTS}/dictionary_{config.DATASET}.json", "r", encoding="utf-8") as f:
            caption_to_number = json.load(f)

        if df_aug_train['label'].dtype != int:
            df_aug_train = df_aug_train[df_aug_train["label"].isin(caption_to_number.keys())]
            df_aug_train["label"] = df_aug_train["label"].map(caption_to_number)

        # 5) Identifier les colonnes de la CBM (dummy) vs les colonnes C3M
        all_feature_cols = [
            col for col in df_aug_train.columns
            if col not in ['Unnamed: 0', 'text', 'label']
        ]
        columns_CBM = [col for col in all_feature_cols if 'dummy' in col]
        columns_C3M = [col for col in all_feature_cols if 'dummy' not in col]
        n_concepts = len(columns_C3M)

        # 6) Supprimer les colonnes CBM et 'Unnamed: 0'
        df_aug_train = df_aug_train.drop(columns=columns_CBM + ['Unnamed: 0'])

        # 7) Repréfixer chaque colonne C3M par 'dummy_' et nettoyer noms
        cleaned_cols = []
        for col in df_aug_train.columns:
            if col in columns_C3M:
                new_col = "dummy_" + col.replace("\n", "").strip()
            else:
                new_col = col.replace("\n", "").strip()
            cleaned_cols.append(new_col)
        df_aug_train.columns = cleaned_cols

        # 8) Convertir les colonnes de concepts (type 'O') en int (0 ou 1)
        for col in df_aug_train.columns:
            if col not in ['text', 'label'] and df_aug_train[col].dtype == 'O':
                df_aug_train[col] = df_aug_train[col].apply(lambda x: int(x) if str(x).isdigit() else 0)

    elif annotation == 'our_annotation':
        # 1) Charger utilitaire de nettoyage
        from concepts_bank_utils import clean_concept_name

        # 2) Charger le CSV ‘our_annotation’
        path_csv = f"{config.SAVE_PATH_CONCEPTS}/df_with_topics_v4.csv"
        df_aug_train = pd.read_csv(path_csv)

        # 3) Nettoyer les noms de colonnes (concepts)
        df_aug_train.rename(columns=lambda x: clean_concept_name(x), inplace=True)

    elif annotation == 'combined_annotation':
        # --- i) Prétraitement C3M (mêmes étapes que ci-dessus) ---
        path_csv = f"{config.SAVE_PATH_CONCEPTS}/df_with_topics_v4_C3M.csv"
        df_aug_train = pd.read_csv(path_csv)

        # adapt lines of code above to replace train_data and test_data by df_aug_train & df_aug_test 
        df_aug_train['text'] = df_aug_train['text'].astype(str).str.strip()
        
        if(df_aug_train['label'].dtype != int):
            df_aug_train['label'] = df_aug_train["label"].astype(str).str.strip()
        
        #dataset length
        print("df_aug_train", len(df_aug_train))
        print(df_aug_train['label'].unique())
        
        # import json file
        with open(f"{config.SAVE_PATH_CONCEPTS}/dictionary_{config.DATASET}.json", "r") as f:
            caption_to_number = json.load(f)
        
        print(caption_to_number)
        
        if(df_aug_train['label'].dtype != int):
            df_aug_train = df_aug_train[df_aug_train["label"].isin(caption_to_number.keys())]
            df_aug_train["label"] = df_aug_train["label"].map(caption_to_number)
        
        print("df_aug_train", len(df_aug_train))
        print(df_aug_train['label'].unique())
        
        # determine total number of concepts
        columns_CBM = [col for col in df_aug_train.drop(columns=['Unnamed: 0','text','label']) if 'dummy' in col]
        columns_C3M = [col for col in df_aug_train.drop(columns=['Unnamed: 0','text','label']) if not 'dummy' in col]
        n_concepts = len(columns_C3M)
        print("..n_concepts", n_concepts)
        print(columns_C3M)
        
        # define backbone model & tokenizers
        
        
        df_aug_train = df_aug_train.drop(columns=columns_CBM)
        df_aug_train = df_aug_train.drop(columns=['Unnamed: 0'])
        
        # clean column names
        df_aug_train.columns = ["dummy_"+col.replace("\n", "").strip() if col in columns_C3M else col.replace("\n", "").strip() for col in df_aug_train]
    
    
        # clean types to int for "missing values"
        for col in [col for col in df_aug_train.columns if (df_aug_train[col].dtype == 'O') and (col !='text') and (col!='label')]:
            df_aug_train[col] = df_aug_train[col].apply(lambda x: int(x) if str(x).isdigit() else 0)

        # 1) Charger utilitaire de nettoyage
        from concepts_bank_utils import clean_concept_name

        # 2) Charger le CSV ‘our_annotation’
        path_csv = f"{config.SAVE_PATH_CONCEPTS}/df_with_topics_v4.csv"
        df_aug_train_1 = pd.read_csv(path_csv)

        # 3) Nettoyer les noms de colonnes (concepts)
        df_aug_train_1.rename(columns=lambda x: clean_concept_name(x), inplace=True)
        
        # Merge des DataFrames
        df_aug_train_final = df_aug_train.merge(df_aug_train_1, on=['text', 'label'])
        
        
        # Reconvertir la colonne 'label' en int
        df_aug_train_final['label'] = df_aug_train_final['label'].astype(int)

        # df_aug_train = df_aug_train_final
        # return df_aug_train
    else:
        raise ValueError("Entrez un type d’annotation valide parmi ['C3M','our_annotation']")

    # --- C) Construction de la matrice binaire de concepts ---
    df_bin = df_aug_train.drop(columns=["text", "label"])
    df_transposed = df_bin.T  # chaque ligne = un concept binaire

    # --- D) Calcul de la matrice de distance Hamming ---
    dist_matrix = pairwise_distances(df_transposed, metric="hamming")

    # --- E) Clustering HDBSCAN ---
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
    labels = clusterer.fit_predict(dist_matrix)

    # --- F) Stratégie 1 : réassigner les noise à leur cluster le plus proche ---
    # F1) Centroides (moyennes binaires) pour chaque cluster ≠ -1
    cluster_ids = [c for c in set(labels) if c != -1]
    centroids = {
        c: df_transposed.iloc[labels == c].mean(axis=0).values
        for c in cluster_ids
    }

    # F2) Réassignation des points noise (label == -1)
    labels_strat1 = labels.copy()
    noise_idx = [i for i, l in enumerate(labels) if l == -1]

    for i in noise_idx:
        point = df_transposed.iloc[i].values.reshape(1, -1)
        dists = {
            c: ((point - centroids[c])**2).sum() ** 0.5
            for c in cluster_ids
        }
        nearest = min(dists, key=dists.get)
        labels_strat1[i] = nearest

    # --- G) Stratégie 2 : garder le noise comme cluster séparé (-1) ---
    labels_strat2 = labels.copy()

    # --- H) Construction des dictionnaires cluster→liste de concepts ---
    def build_cluster_dict(labels_arr):
        d = {}
        for lbl in sorted(set(labels_arr)):
            concept_list = df_transposed.index[labels_arr == lbl].tolist()
            d[lbl] = concept_list
        return d

    cluster_dict_strat1 = build_cluster_dict(labels_strat1)
    cluster_dict_strat2 = build_cluster_dict(labels_strat2)

    # --- I) Visualisation UMAP pour les deux stratégies ---
    reducer = UMAP(metric="precomputed", random_state=random_state)
    embed = reducer.fit_transform(dist_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, (lbls, title) in zip(
        axes,
        [
            (labels_strat1, "Stratégie 1 : noise réassigné"),
            (labels_strat2, "Stratégie 2 : noise séparé")
        ]
    ):
        scatter = ax.scatter(embed[:, 0], embed[:, 1], c=lbls, s=80)
        for i, concept in enumerate(df_transposed.index):
            ax.text(embed[i, 0] + 0.002, embed[i, 1] + 0.002, concept, fontsize=7)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True)

        # Légende uniquement sur le deuxième subplot
        if title.startswith("Stratégie 2"):
            unique_labels = sorted(set(lbls))
            handles = [
                plt.Line2D([], [], marker="o", linestyle="", label=f"Cluster {u}")
                for u in unique_labels
            ]
            ax.legend(handles=handles, title="Clusters",
                      bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    # --- J) Affichage des dictionnaires dans la console ---
    print("Dictionnaire Stratégie 1 (noise réassigné) :")
    for cluster_id, concepts in cluster_dict_strat1.items():
        print(f"Cluster {cluster_id} : {concepts}\n")

    print("Dictionnaire Stratégie 2 (noise séparé) :")
    for cluster_id, concepts in cluster_dict_strat2.items():
        print(f"Cluster {cluster_id} : {concepts}\n")

    # --- K) Sauvegarde des dictionnaires au format JSON ---
    os.makedirs(config.SAVE_PATH_CONCEPTS, exist_ok=True)

    filename1 = f"cluster_dict_strat_reassignation_{annotation}.json"
    filename2 = f"cluster_dict_strat_noise_cluster_{annotation}.json"

    path1 = os.path.join(config.SAVE_PATH_CONCEPTS, filename1)
    path2 = os.path.join(config.SAVE_PATH_CONCEPTS, filename2)

    # Convertir clés en string pour la sérialisation JSON
    cluster_dict_strat1_str = {str(k): v for k, v in cluster_dict_strat1.items()}
    cluster_dict_strat2_str = {str(k): v for k, v in cluster_dict_strat2.items()}

    with open(path1, "w", encoding="utf-8") as f:
        json.dump(cluster_dict_strat1_str, f, ensure_ascii=False, indent=4)

    with open(path2, "w", encoding="utf-8") as f:
        json.dump(cluster_dict_strat2_str, f, ensure_ascii=False, indent=4)

    print(f"Les dictionnaires JSON ont été enregistrés :\n- {path1}\n- {path2}")
