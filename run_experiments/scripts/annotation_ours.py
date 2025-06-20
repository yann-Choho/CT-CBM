##### Augment train dataset
# the difference with test and val is that they used the umap created while augmented train


import sys
sys.path.append('./scripts')
sys.path.append('./run_experiments/')
sys.path.append('./run_experiments/models')
sys.path.append('./run_experiments/data')

import os
import json
import torch
import joblib
import pandas as pd

from concepts_bank_utils import (
    prepare_data, save_results, process_extracted_topics, 
    create_macro_concepts_pipeline, create_macro_concepts_pipeline_v2, create_dataloader
)



##### Augment train dataset
# the difference with test and val is that they used the umap created while augmented train


def create_train_augmented_dataframe(df, config, discovery_model, discovery_tokenizer, n_cluster, save_dir):
    """
    Crée un DataFrame augmenté avec des concepts macro et le sauvegarde.

    Args:
        df (pd.DataFrame): DataFrame initial contenant les colonnes 'text' et 'label'.
        config (Config): Configuration contenant les paramètres comme le device.
        discovery_model (Model): Modèle pour découvrir des concepts dans les textes.
        discovery_tokenizer (Tokenizer): Tokenizer pour le modèle de découverte.
        save_dir (str): Répertoire pour sauvegarder les résultats intermédiaires.

    Returns:
        pd.DataFrame: DataFrame augmenté filtré.
    """
    print("--------------initialize_concept_bank ------------")

    # Préparer les données initiales
    df = prepare_data(df)

    # Extraire les sujets à partir du modèle de découverte
    df_with_topics = process_extracted_topics(df, discovery_model, discovery_tokenizer, device=config.device)
    df_with_topics.to_csv(f'{save_dir}/df_with_topics_v2.csv', index = False)
    print(f"Results saved to {save_dir}/df_with_topics_v2.csv")
    
    # Pipeline pour créer des macro-concepts
    print('------------create_macro_concepts_pipeline -----------')
    create_macro_concepts_pipeline(
        df_path=f"{save_dir}/df_with_topics_v2.csv",
        save_path=save_dir,
        discovery_model=discovery_model,
        discovery_tokenizer=discovery_tokenizer,
        n_clusters=n_cluster,
        model_name='all-mpnet-base-v2',
        config=config
    )

    # Charger le DataFrame augmenté
    df_aug = pd.read_csv(f"{save_dir}/df_with_topics_v4.csv")

    # Compter les occurrences de chaque concept
    dummy_columns = df_aug.filter(like="dummy_")
    concept_counts = (dummy_columns == 1).sum()
    concept_counts.index = concept_counts.index.str.replace('dummy_', '')  # Supprimer le préfixe 'dummy_'
    concept_counts_dict = concept_counts.to_dict()

    # Filtrer les concepts (désactivé dans la nouvelle logique)
    filtered_concept_counts = concept_counts_dict  # Garde tous les concepts

    # Garder uniquement les colonnes pertinentes dans le DataFrame
    filtered_columns = ['dummy_' + k for k in filtered_concept_counts.keys()]
    df_aug_filtered = df_aug[['text', 'label'] + filtered_columns]

    # Sauvegarder le DataFrame augmenté
    df_aug_filtered_path = os.path.join(save_dir, "df_aug_filtered.csv")
    df_aug_filtered.to_csv(df_aug_filtered_path, index=False)
    print(f"DataFrame augmenté sauvegardé à : {df_aug_filtered_path}")

    return df_aug_filtered


def create_dataloader_from_dataframe(df_aug_filtered, embedder_tokenizer, config, save_dir, dataset = None):
    """
    Crée un DataLoader à partir d'un DataFrame augmenté et le sauvegarde.

    Args:
        df_aug_filtered (pd.DataFrame): DataFrame augmenté filtré.
        embedder_tokenizer (Tokenizer): Tokenizer pour créer des embeddings des textes.
        config (Config): Configuration contenant les paramètres max_len et batch_size.
        save_dir (str): Répertoire pour sauvegarder le DataLoader.

    Returns:
        DataLoader: DataLoader PyTorch basé sur le DataFrame augmenté.
    """
    if dataset == None:
        print('Please give the dataset parameters = train, test or val')
    else:
        # Créer un DataLoader à partir du DataFrame filtré
        train_loader_aug = create_dataloader(df_aug_filtered, embedder_tokenizer, config.max_len, config.batch_size)
    
        # Sauvegarder le DataLoader (données seulement, pas l'instance PyTorch complète)
        dataloader_save_path = os.path.join(save_dir, f"{dataset}_loader_aug.pth")
        torch.save(
            {
                "dataset": df_aug_filtered.to_dict(),
                "config": {
                    "max_len": config.max_len,
                    "batch_size": config.batch_size
                }
            },
            dataloader_save_path
        )
        print(f"DataLoader sauvegardé à : {dataloader_save_path}")

    return train_loader_aug

##### Augment test and val dataset
def create_val_or_test_augmented_dataframe(dataset, df, discovery_model, discovery_tokenizer, config, save_dir):
    """
    Prépare les données et génère un DataFrame augmenté avec des concepts macro.

    Args:
        dataset (str): Type de dataset ('train', 'val', ou 'test').
        df (pd.DataFrame): DataFrame source à traiter.
        discovery_model (Model): Modèle pour la découverte de concepts.
        discovery_tokenizer (Tokenizer): Tokenizer associé au modèle de découverte.
        config (Config): Configuration avec device, max_len, batch_size, etc.
        save_dir (str): Répertoire pour sauvegarder les fichiers intermédiaires et finaux.

    Returns:
        pd.DataFrame: DataFrame augmenté avec des concepts macro.
    """
    print(f"--- Traitement et génération du DataFrame augmenté pour : {dataset} ---")

    # Préparer les données
    print("Préparation des données...")
    df_prepared = prepare_data(df)

    # Découverte des sujets avec le modèle
    print("Application du modèle de découverte des sujets...")
    df_with_topics = process_extracted_topics(df_prepared, discovery_model, discovery_tokenizer, device=config.device)
    topics_file_path = f"{save_dir}/df_with_topics_v2_{dataset}.csv"
    df_with_topics.to_csv(topics_file_path, index=False)
    print(f"Fichier des sujets sauvegardé : {topics_file_path}")

    # Charger UMAP et DBSCAN
    print("Chargement des modèles UMAP et DBSCAN...")
    reducer = joblib.load(f'{save_dir}/umap_model.pkl')
    clusterer = joblib.load(f'{save_dir}/dbscan_model.pkl')

    # Appliquer le pipeline des concepts macro
    print("Application du pipeline de concepts macro...")
    create_macro_concepts_pipeline_v2(
        df_path=topics_file_path,
        save_path=save_dir,
        model_name='all-mpnet-base-v2',
        config=config,
        reducer=reducer,
        clusterer=clusterer,
        dataset=dataset
    )

    # Charger le DataFrame augmenté
    augmented_file_path = f"{save_dir}/df_with_topics_v4_{dataset}.csv"
    df_augmented = pd.read_csv(augmented_file_path)
    print(f"DataFrame augmenté chargé depuis : {augmented_file_path}")

    return df_augmented



# fonction for all process
def process_and_save_augmentation(train_df, 
                                  val_df,
                                  test_df,
                                  config,
                                  discovery_model,
                                  discovery_tokenizer,
                                  embedder_tokenizer,
                                  save_dir = '',
                                  n_cluster = 100,
                                 ):
    # verify direcory
    os.makedirs(save_dir, exist_ok=True)
    
    # launch the augmentation of data
    # train
    df_aug_filtered_movie_train = create_train_augmented_dataframe(df = train_df,
                                                        config = config,
                                                        discovery_model = discovery_model,
                                                        discovery_tokenizer = discovery_tokenizer,
                                                        n_cluster = n_cluster,
                                                        save_dir = save_dir
                                                      )
    
    
    notre_loader_train = create_dataloader_from_dataframe(df_aug_filtered = df_aug_filtered_movie_train,
                                     embedder_tokenizer = embedder_tokenizer,
                                     config = config,
                                     save_dir=save_dir,
                                     dataset = 'train'
                                    )
    # val
    # df_aug_filtered_movie_val  = create_val_or_test_augmented_dataframe(
    #     dataset='val',
    #     df=val_df,
    #     discovery_model=discovery_model,
    #     discovery_tokenizer=discovery_tokenizer,
    #     config=config,
    #     save_dir=save_dir
    # )
    
    # create_dataloader_from_dataframe(df_aug_filtered = df_aug_filtered_movie_val,
    #                                  embedder_tokenizer = embedder_tokenizer,
    #                                  config = config,
    #                                  save_dir=save_dir,
    #                                  dataset = 'val'
    #                                 )
    
    # test
    df_aug_filtered_movie_test  = create_val_or_test_augmented_dataframe(
        dataset='test',
        df=test_df,
        discovery_model=discovery_model,
        discovery_tokenizer=discovery_tokenizer,
        config=config,
        save_dir=save_dir
    )
    
    create_dataloader_from_dataframe(df_aug_filtered = df_aug_filtered_movie_test,
                                     embedder_tokenizer = embedder_tokenizer,
                                     config = config,
                                     save_dir=save_dir,
                                     dataset = 'test'
                                    )
    print('all augmentation done !!!')
    

##### AUTOMATION ALL-IN-SCRIPT-FILE #####

def launch_our_annotation(model_name = 'google/gemma-2-9b-it',
                          hf_access_token = None, 
                          train_df = None, 
                          val_df = None,
                          test_df = None,
                          config = None,
                          save_dir = 'dbfs/concept_xai/CBM/results_movies/concepts_discovery_try_code',
                          n_cluster = 100
                        ):
    # import the PLM model and tokenizer and bottleneck layer
    from concepts_discovery_utils import load_model
    
    # import SLM
    discovery_model, discovery_tokenizer = load_model(model_name, hf_access_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discovery_model.to(device)

    from models.utils import load_model_and_tokenizer
    embedder_model, embedder_tokenizer, ModelXtoCtoY_layer, classifier = load_model_and_tokenizer(config)

    process_and_save_augmentation(train_df, 
                                  val_df,
                                  test_df,
                                  config,
                                  discovery_model,
                                  discovery_tokenizer,
                                  embedder_tokenizer,
                                  save_dir = 'dbfs/concept_xai/CBM/results_movies/concepts_discovery_try_code',
                                  n_cluster = 100,
                                )


