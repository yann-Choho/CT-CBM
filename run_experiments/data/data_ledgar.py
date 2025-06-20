import torch
import pandas as pd
from models.utils import load_model_and_tokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import json
import os

def prepare_ledgar_data(config):
    """
    Prépare et retourne les DataLoader pour le dataset LEDGAR.
    
    Cette fonction effectue les opérations suivantes :
      - Détermine le répertoire de sauvegarde en fonction de config.infra.
      - Charge les données depuis le dataset "ledgar" de LexGLUE via Hugging Face.
      - Ne conserve que les exemples appartenant aux 5 labels les plus représentés dans le split d'entraînement.
      - (Optionnel) Limite le nombre d'exemples par label pour l'entraînement.
      - Sauvegarde les DataFrames en format pickle et le mapping des labels.
      - Charge le tokenizer via load_model_and_tokenizer.
      - Crée une classe customisée OurCustomDataset.
      - Instancie les datasets et retourne les DataLoader pour l'entraînement, la validation et le test.
    
    Paramètres :
      - config : objet de configuration contenant notamment :
          - config.max_len (int) : longueur maximale pour le tokenizing,
          - config.infra (str) : infrastructure (ex: "DATABRICKS" ou autre),
          - config.batch_size (int) : taille du batch pour le DataLoader,
          - ... d'autres paramètres éventuellement requis par load_model_and_tokenizer.
    
    Retourne :
      - train_loader, val_loader, test_loader, train_df, val_df, test_df.
    """
    max_len = config.max_len
    infra = config.infra

    # Définir le chemin de sauvegarde en fonction de l'infrastructure
    if infra == "DATABRICKS":
        save_path = "/dbfs/mnt/ekixai/main/data/Interpretability/concept_xai/experiment/results_ledgar/concepts_discovery"
    else:
        save_path = "/home/bhan/Yann_CBM/Launch/dbfs/results_ledgar/concepts_discovery"
    os.makedirs(save_path, exist_ok=True)

    # Chemins des fichiers sauvegardés en pickle
    train_file = os.path.join(save_path, "train_data.pkl")
    val_file = os.path.join(save_path, "val_data.pkl")
    test_file = os.path.join(save_path, "test_data.pkl")

    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        print("Chargement des fichiers sauvegardés...")
        train_df = pd.read_pickle(train_file)
        val_df = pd.read_pickle(val_file)
        test_df = pd.read_pickle(test_file)
    else:
        print("Chargement et prétraitement des données LEDGAR depuis Hugging Face...")
        dataset = load_dataset("lex_glue", "ledgar")

        train_df = pd.DataFrame(dataset['train'])
        val_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        
        # Identifier les 5 labels les plus représentés dans le split d'entraînement
        # top_labels = train_df['label'].value_counts().nlargest(100).index.tolist()
        top_labels = [2, 85, 89, 88]


        # Conserver uniquement les exemples correspondant aux top 5 labels pour tous les splits
        train_df = train_df[train_df['label'].isin(top_labels)]
        val_df = val_df[val_df['label'].isin(top_labels)]
        test_df = test_df[test_df['label'].isin(top_labels)]
        
        # Optionnel : pour le training set, limiter le nombre d'exemples par label (ici 1000 par label)
        train_df = train_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=1000, random_state=42) if len(x) >= 1000 else x
        )
        
        # Créer un mapping : de l'étiquette d'origine au nouvel identifiant (0 à 4)
        label_to_id = {label: i for i, label in enumerate(sorted(top_labels))}
        
        # Sauvegarder le mapping dans un fichier JSON
        with open(os.path.join(save_path, "label_mapping.json"), 'w') as f:
            json.dump(label_to_id, f)

        # Remapper les labels dans les DataFrames
        train_df["label"] = train_df["label"].map(label_to_id)
        val_df["label"] = val_df["label"].map(label_to_id)
        test_df["label"] = test_df["label"].map(label_to_id)

        # Sauvegarder les DataFrames pour réutilisation ultérieure
        train_df.to_pickle(train_file)
        val_df.to_pickle(val_file)
        test_df.to_pickle(test_file)
        print("Données sauvegardées.")

    # Charger le tokenizer via la fonction utilitaire
    _, tokenizer, _, _ = load_model_and_tokenizer(config)

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
                return_tensors="pt"
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long)
            }

    # Instancier les datasets pour chaque split
    train_dataset = OurCustomDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len)
    val_dataset = OurCustomDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_len)
    test_dataset = OurCustomDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len)

    # Créer les DataLoader correspondants
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, val_loader, test_loader, train_df, val_df, test_df
