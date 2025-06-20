

def compute_cavs_mean_minus(df_aug, embedder_model, embedder_tokenizer, baseline_model, config):
    import os
    import json
    import torch
    from tqdm import tqdm

    device = config.device
    embedder_model.to(device)

    print('Calculating CAVs using embeddings (mean_minus_others) without DataLoader...')
    concept_name_list = []

    df_aug['consolidated_concepts'] = [[] for _ in range(len(df_aug))]
    for column in df_aug.columns:
        if 'dummy' not in column:
            continue
        concept_name = column.replace('dummy_', '')
        concept_name_list.append(concept_name)
        for i in df_aug.loc[df_aug[column] == 1].index:
            df_aug.at[i, 'consolidated_concepts'].append(concept_name)
    
    train_data_saved = df_aug[['text', 'label', 'consolidated_concepts']].copy()
    print('Consolidated concepts in data:', train_data_saved['consolidated_concepts'].head())
    
    train_data_saved['embeddings'] = None
    train_data_saved['embeddings'] = train_data_saved['embeddings'].astype(object)
    
    for idx, row in tqdm(train_data_saved.iterrows(), total=len(train_data_saved), unit="exemple"):
        text = row['text']
        inputs = embedder_tokenizer(
            text,
            max_length=config.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            if config.model_name == 'gemma':
                outputs = baseline_model.get_pooled_output(input_ids)
            else:        
                outputs = baseline_model.get_pooled_output(input_ids, attention_mask)
        
        embedding_cpu = outputs.flatten().detach().cpu()
        train_data_saved.at[idx, 'embeddings'] = embedding_cpu
        
        del input_ids, attention_mask, outputs
        torch.cuda.empty_cache()
    
    # Calcul des CAVs en CPU uniquement
    cavs_concepts = {}
    for concept in concept_name_list:
        concept_data = train_data_saved.loc[train_data_saved['consolidated_concepts'].apply(lambda x: concept in x)]
        if concept_data.empty:
            print(f"Aucun échantillon trouvé pour le concept '{concept}'.")
            continue
        
        concept_embeddings = torch.stack([emb for emb in concept_data['embeddings'].values]).cpu()
        
        other_data = train_data_saved.loc[train_data_saved['consolidated_concepts'].apply(lambda x: concept not in x)]
        if other_data.empty:
            print(f"Aucun échantillon non associé au concept '{concept}'.")
            continue
        
        other_embeddings = torch.stack([torch.tensor(emb) for emb in other_data['embeddings'].values]).cpu()
        
        cav = concept_embeddings.mean(dim=0) - other_embeddings.mean(dim=0)
        cavs_concepts[concept] = cav.cpu()
    
    json_save_path = os.path.join(config.SAVE_PATH, "blue_checkpoints", config.model_name, "cavs", config.cavs_type)
    os.makedirs(json_save_path, exist_ok=True)
    # json_file_path = os.path.join(json_save_path, 'cavs_mean.json')
    json_file_path = os.path.join(json_save_path, f'cavs_mean_{config.annotation}.json')

    with open(json_file_path, 'w') as f:
        json.dump({key: value.tolist() for key, value in cavs_concepts.items()}, f, indent=4)
    print(f"CAVs saved in JSON at {json_file_path}")
    
    torch.cuda.empty_cache()
    
    return cavs_concepts


def calculate_concept_acc(data_df, cavs_concepts, device, baseline_model, tokenizer, max_len, equal_prop=True):
    """
    Calcule, pour chaque concept, des métriques de performance (accuracy et f1-score)
    sur un DataFrame contenant les textes et les concepts (dans la colonne 'consolidated_concepts')
    sans utiliser de DataLoader. Pour chaque exemple, la prédiction pour un concept est 1 si la similarité
    cosinus entre le CAV du concept et l'embedding de l'exemple est > 0, et 0 sinon.
    
    Args:
        data_df (pd.DataFrame): DataFrame contenant au moins les colonnes :
            - 'text' : le texte à évaluer.
            - 'consolidated_concepts' : liste des concepts présents pour cet exemple.
        cavs_concepts (dict): Dictionnaire {concept: torch.Tensor} des CAVs.
        device (torch.device or str): Le device sur lequel effectuer les calculs (ex. "cuda" ou "cpu").
        baseline_model (torch.nn.Module): Modèle fournissant la méthode get_pooled_output.
        tokenizer: Tokenizer associé au modèle.
        max_len (int): Longueur maximale des séquences pour la tokenisation.
        equal_prop (bool): Si True, on limite le nombre d'instances prises en compte pour chaque concept afin de garder un effectif équilibré.
        
    Returns:
        dict: Dictionnaire de confusion pour chaque concept contenant TP, FP, FN, TN, l'accuracy et la f1-score.
    """
    import torch
    import os
    import pickle
    import json
    import pandas as pd
    from tqdm import tqdm
    import numpy as np

    print('Calculating concept accuracy on test data without DataLoader...')
    
    # Initialisation de la matrice de confusion pour chaque concept
    confusion_matrix = {
        concept: {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}
        for concept in cavs_concepts.keys()
    }
    
    # Si equal_prop est True, déterminer pour chaque concept le nombre maximal d'instances à prendre en compte
    if equal_prop:
        # Aplatir la colonne des concepts (chaque ligne est une liste)
        all_concepts = sum(data_df['consolidated_concepts'], [])
        counts = pd.Series(all_concepts).value_counts().to_dict()
        # Pour chaque concept, limiter à min(n_concepts, n_non_concepts)
        min_instances = {
            concept: min(count, len(data_df) - count)
            for concept, count in counts.items()
        }
    else:
        min_instances = None
    
    # Préparation de la colonne 'embeddings'
    data_df['embeddings'] = None
    data_df['embeddings'] = data_df['embeddings'].astype(object)
    
    # Calcul de tous les embeddings en itérant directement sur chaque exemple du DataFrame
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), unit="exemple"):
        text = row['text']
        # Tokenisation du texte
        inputs = tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Calcul de l'embedding sans calcul des gradients
        with torch.no_grad():
            outputs = baseline_model.get_pooled_output(input_ids, attention_mask)
        
        # Stockage de l'embedding sur CPU pour libérer la mémoire GPU
        embedding_cpu = outputs.flatten().detach().cpu()
        data_df.at[idx, 'embeddings'] = embedding_cpu
        
        # Nettoyage et libération de la mémoire GPU
    #     del input_ids, attention_mask, outputs
    #     torch.cuda.empty_cache()
    
    # # Pour chaque exemple du DataFrame (parcours dans un ordre aléatoire)
    # for idx in np.random.permutation(data_df.index):
    #     # Récupération de l'embedding et mise sur le device
    #     embedding = data_df.loc[idx, 'embeddings']
    #     if not torch.is_tensor(embedding):
    #         embedding = torch.tensor(embedding, dtype=torch.float)
    #     output = embedding.to(device)
        
    #     # Construction d'un tenseur contenant tous les CAVs
    #     cavs_tensor = torch.stack(list(cavs_concepts.values())).to(device)
        
    #     # Calcul de la similarité cosinus entre l'embedding et chacun des CAVs
    #     projections = torch.nn.functional.cosine_similarity(cavs_tensor, output, dim=1)
        
    #     # Association de chaque concept à sa similarité
    #     dict_proj = {
    #         concept: proj.item() 
    #         for concept, proj in zip(cavs_concepts.keys(), projections)
    #     }
        
    #     # Récupération des concepts réels associés à cet exemple
    #     instance_concepts = data_df.loc[idx, 'consolidated_concepts']
        
    #     # Mise à jour de la matrice de confusion pour chaque concept
    #     for concept in cavs_concepts.keys():
    #         sim_value = dict_proj[concept]
    #         if sim_value > 0:  # Prédiction = 1
    #             if concept in instance_concepts:
    #                 # Vrai positif
    #                 if not equal_prop:
    #                     confusion_matrix[concept]['TP'] += 1
    #                 else:
    #                     if (confusion_matrix[concept]['TP'] + confusion_matrix[concept]['FN']) < min_instances.get(concept, float('inf')):
    #                         confusion_matrix[concept]['TP'] += 1
    #             else:
    #                 # Faux positif
    #                 if not equal_prop:
    #                     confusion_matrix[concept]['FP'] += 1
    #                 else:
    #                     if (confusion_matrix[concept]['FP'] + confusion_matrix[concept]['TN']) < min_instances.get(concept, float('inf')):
    #                         confusion_matrix[concept]['FP'] += 1
    #         else:  # Prédiction = 0
    #             if concept in instance_concepts:
    #                 # Faux négatif
    #                 if not equal_prop:
    #                     confusion_matrix[concept]['FN'] += 1
    #                 else:
    #                     if (confusion_matrix[concept]['TP'] + confusion_matrix[concept]['FN']) < min_instances.get(concept, float('inf')):
    #                         confusion_matrix[concept]['FN'] += 1
    #             else:
    #                 # Vrai négatif
    #                 if not equal_prop:
    #                     confusion_matrix[concept]['TN'] += 1
    #                 else:
    #                     if (confusion_matrix[concept]['FP'] + confusion_matrix[concept]['TN']) < min_instances.get(concept, float('inf')):
    #                         confusion_matrix[concept]['TN'] += 1
    
    # # Calcul de la F1-score et de l'accuracy pour chaque concept
    # for concept, stats in confusion_matrix.items():
    #     TP = stats['TP']
    #     FP = stats['FP']
    #     FN = stats['FN']
    #     TN = stats['TN']
    #     denominator = (2 * TP + FP + FN)
    #     f1 = (2 * TP / denominator) if denominator > 0 else 0.0
    #     stats['f1'] = f1
    #     total = TP + FP + FN + TN
    #     acc = (TP + TN) / total if total > 0 else 0.0
    #     stats['acc'] = acc
    
    # return confusion_matrix
    return data_df