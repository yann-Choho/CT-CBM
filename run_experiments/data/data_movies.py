import torch
import pandas as pd
from models.utils import load_model_and_tokenizer
from torch.utils.data import Dataset, DataLoader
from config_movies import Config
import json
import os

def prepare_movies_data(config):
    """
    Prépare et retourne les DataLoader pour le dataset xxx xx.
    
    Cette fonction effectue les opérations suivantes :
      - Détermine le répertoire de sauvegarde en fonction de config.infra.
      - Charge les données prétraitées depuis des fichiers pickle si disponibles,
        sinon effectue le prétraitement (limitation du nombre d'exemples par classe,
        séparation en train/validation/test) et sauvegarde les DataFrame en pickle.
      - Charge le tokenizer via load_model_and_tokenizer.
      - Crée une classe customisée OurCustomDataset.
      - Instancie les datasets et retourne les DataLoader pour l'entraînement,
        la validation et le test.
    
    Paramètres :
      - config : objet de configuration contenant notamment
          - config.max_len (int) : longueur maximale pour le tokenizing
          - config.infra (str) : infrastructure (ex: "DATABRICKS" ou autre)
          - config.batch_size (int) : taille du batch pour le DataLoader
          - ... (d'autres paramètres éventuellement requis par load_model_and_tokenizer)
    
    Retourne :
      - train_loader, val_loader, test_loader, train_df, val_df, test_df : les DataLoader et dataframe respectifs.
    """
    max_len = config.max_len
    infra = config.infra
    
    if infra == "DATABRICKS":
        # Chemins des données
        DATASET_PATH = "/dbfs/mnt/ekixai/main/data/Interpretability/concept_xai/experiment/dataset/movies/"  # this line change when we go to A100
        RESULTS_PATH  = "/dbfs/mnt/ekixai/main/data/Interpretability/concept_xai/experiment/results_movies/concepts_discovery" # this line change when we go to A100
        save_path_dictionary = "/dbfs/mnt/ekixai/main/data/Interpretability/concept_xai/experiment/results_movies/concepts_discovery" # this line change when we go to A100
    else:
        DATASET_PATH = "/home/bhan/Yann_CBM/Launch/dbfs/dataset/movies/"  # this line change when we go to A100
        RESULTS_PATH  = "/home/bhan/Yann_CBM/Launch/dbfs/results_movies/concepts_discovery" # this line change when we go to A100
        save_path_dictionary = "/home/bhan/Yann_CBM/Launch/dbfs/results_movies/concepts_discovery" # this line change when we go to A100    
    
    os.makedirs(save_path_dictionary, exist_ok=True)
    
    # Fichiers de sauvegarde en pkl
    train_file = f"{save_path_dictionary}/train_data.pkl"
    val_file = f"{save_path_dictionary}/val_data.pkl"
    test_file = f"{save_path_dictionary}/test_data.pkl"
    
    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        print("Chargement des fichiers sauvegardés...")
        train_df = pd.read_pickle(train_file)
        val_df = pd.read_pickle(val_file)
        test_df = pd.read_pickle(test_file)
    else:
        print("Prétraitement des données...")
        train_path = f"{DATASET_PATH}/train_data.txt"
        test_path = f"{DATASET_PATH}/test_data_solution.txt"
    
        train_data = pd.read_csv(train_path, sep=":::", names=["TITLE", "GENRE", "DESCRIPTION"], engine="python")
        test_data = pd.read_csv(test_path, sep=":::", names=["TITLE", "GENRE", "DESCRIPTION"], engine="python")
    
        train_data.rename(columns={"DESCRIPTION": "text", "GENRE": "label"}, inplace=True)
        test_data.rename(columns={"DESCRIPTION": "text", "GENRE": "label"}, inplace=True)
    
        train_data["label"] = train_data["label"].astype(str).str.strip()
        train_data["text"] = train_data["text"].astype(str)
        test_data["label"] = test_data["label"].astype(str).str.strip()
        test_data["text"] = test_data["text"].astype(str)
    
        caption_to_number = {"documentary": 0, "comedy": 1, "horror": 2, "western": 3}
        with open(f"{save_path_dictionary}/dictionary_movies.json", "w") as f:
            json.dump(caption_to_number, f)
    
        train_data = train_data[train_data["label"].isin(caption_to_number.keys())]
        test_data = test_data[test_data["label"].isin(caption_to_number.keys())]
        train_data["label"] = train_data["label"].map(caption_to_number)
        test_data["label"] = test_data["label"].map(caption_to_number)
    
        train_data_limited = train_data.groupby("label", group_keys=False).apply(lambda x: x.sample(n=1000, random_state=42))
        remaining_data = train_data.drop(train_data_limited.index)
    
        train_df = train_data_limited[["text", "label"]]
        val_df = remaining_data[["text", "label"]]
        test_df = test_data[["text", "label"]]
    
        train_df.to_pickle(train_file)
        val_df.to_pickle(val_file)
        test_df.to_pickle(test_file)
        print("Données sauvegardées.")
    
    _m, tokenizer, _, _c = load_model_and_tokenizer(config)
    
    class OurCustomDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
    
        def __len__(self):
            return len(self.texts)
    
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
    
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
    
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "label": torch.tensor(label, dtype=torch.long),
            }
    
    train_dataset = OurCustomDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len)
    val_dataset = OurCustomDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_len)
    test_dataset = OurCustomDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    print("Dataloaders prêts.")

    return train_loader, test_loader, val_loader, train_df, val_df, test_df 