def load_fc_prepare_data(dataset):
    
    if dataset == 'movies':
        from data_movies import prepare_movies_data as prepare_data
    elif dataset == 'agnews':
        from data_agnews import prepare_agnews_data as prepare_data
    elif dataset == 'dbpedia':
        from data_dbpedia import prepare_dbpedia_data as prepare_data
    elif dataset == 'medical':
        from data_medical import prepare_medical_data as prepare_data
    elif dataset == 'n24':
        from data_n24 import prepare_n24_data as prepare_data
    else:
        raise ValueError("Entrez un nom de dataset valide parmi ['movies','agnews','dbpedia','medical', 'n24']")
    
    return prepare_data

import pandas as pd
import json

def prepare_data_from_csv(annotation, config, return_test = False):
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

        # TEST FOR C3M
        df_aug_test = pd.read_csv(f"{config.SAVE_PATH_CONCEPTS}/df_with_topics_v4_test_C3M.csv")
        
        df_aug_test['text'] = df_aug_test['text'].astype(str).str.strip()
        if(df_aug_test['label'].dtype != int):
            df_aug_test['label'] = df_aug_test["label"].astype(str).str.strip()
                
        if(df_aug_test['label'].dtype != int):
            df_aug_test = df_aug_test[df_aug_test["label"].isin(caption_to_number.keys())]
            df_aug_test["label"] = df_aug_test["label"].map(caption_to_number)
                
        # clean column names
        df_aug_test.columns = ["dummy_"+col.replace("\n", "").strip() if col in columns_C3M else col.replace("\n", "").strip() for col in df_aug_test]
        
        # clean types to int for "missing values"
        for col in [col for col in df_aug_test.columns if (df_aug_test[col].dtype == 'O') and (col !='text') and (col!='label')]:
            df_aug_test[col] = df_aug_test[col].apply(lambda x: int(x) if str(x).isdigit() else 0)

    
    elif annotation == 'our_annotation':
        # 1) Charger utilitaire de nettoyage
        from concepts_bank_utils import clean_concept_name

        # 2) Charger le CSV ‘our_annotation’
        path_csv = f"{config.SAVE_PATH_CONCEPTS}/df_with_topics_v4.csv"
        df_aug_train = pd.read_csv(path_csv)

        # 3) Nettoyer les noms de colonnes (concepts)
        # df_aug_train.rename(columns=lambda x: clean_concept_name(x), inplace=True)

        df_aug_test = pd.read_csv(f"{config.SAVE_PATH_CONCEPTS}/df_with_topics_v4_test.csv")
        # df_aug_test.rename(columns=lambda x: clean_concept_name(x), inplace=True)

    elif annotation == 'combined_annotation':
        # --- i) Prétraitement C3M (mêmes étapes que ci-dessus) ---
        path_csv = f"{config.SAVE_PATH_CONCEPTS}/df_with_topics_v4_C3M.csv"
        df_aug_train = pd.read_csv(path_csv)
                
        # import json file
        with open(f"{config.SAVE_PATH_CONCEPTS}/dictionary_{config.DATASET}.json", "r") as f:
            caption_to_number = json.load(f)
                
        if(df_aug_train['label'].dtype != int):
            df_aug_train = df_aug_train[df_aug_train["label"].isin(caption_to_number.keys())]
            df_aug_train["label"] = df_aug_train["label"].map(caption_to_number)
                
        # determine total number of concepts
        columns_CBM = [col for col in df_aug_train.drop(columns=['Unnamed: 0','text','label']) if 'dummy' in col]
        columns_C3M = [col for col in df_aug_train.drop(columns=['Unnamed: 0','text','label']) if not 'dummy' in col]
        n_concepts = len(columns_C3M)
        
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
        
        # df_aug_train_final.drop(columns = )
        for df in [df_aug_train, df_aug_train_1]:
            df['text'] = df['text'].astype(str).str.strip().str.lower()
            df['label'] = df['label'].astype(str).str.strip()
        
        # Supprimer les doublons sur ['text', 'label'] dans chaque DataFrame
        df_aug_train = df_aug_train.drop_duplicates(subset=['text', 'label'])
        df_aug_train_1 = df_aug_train_1.drop_duplicates(subset=['text', 'label'])
                
        df_aug_train_final = df_aug_train.merge(df_aug_train_1, on=['text', 'label'], how='inner')
        
        # Reconvertir la colonne 'label' en int
        df_aug_train_final['label'] = df_aug_train_final['label'].astype(int)
        
        df_aug_train = df_aug_train_final
    else:
        raise ValueError("Entrez un type d’annotation valide parmi ['C3M','our_annotation']")

    if (annotation == 'C3M' or annotation == 'our_annotation') and return_test == True :
        return df_aug_train, df_aug_test
    else:
        return df_aug_train