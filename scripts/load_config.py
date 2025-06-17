def load_config(model_name,dataset):
    # --- A) Sélection et instanciation de la config ---
    if model_name == 'bert-base-uncased':
        if dataset == 'movies':
            from config_movies import Config as Config_movies
            config = Config_movies()
        elif dataset == 'agnews':
            from config_agnews import Config as Config_agnews
            config = Config_agnews()
        elif dataset == 'dbpedia':
            from config_dbpedia import Config as Config_dbpedia
            config = Config_dbpedia()
        elif dataset == 'medical':
            from config_medical import Config as Config_medical
            config = Config_medical()
        else:
            raise ValueError("Entrez un nom de dataset valide parmi ['movies','agnews','dbpedia','medical']")
    elif model_name == 'deberta-large':
        if dataset == 'movies':
            from config_movies_deberta import Config as Config_movies
            config = Config_movies()
        elif dataset == 'agnews':
            from config_agnews_deberta import Config as Config_agnews
            config = Config_agnews()
        elif dataset == 'dbpedia':
            from config_dbpedia_deberta import Config as Config_dbpedia
            config = Config_dbpedia()
        elif dataset == 'medical':
            from config_medical_deberta import Config as Config_medical
            config = Config_medical()
        else:
            raise ValueError("Entrez un nom de dataset valide parmi ['movies','agnews','dbpedia','medical']")
    elif model_name == 'gemma':
        if dataset == 'movies':
            from config_movies_gemma import Config as Config_movies
            config = Config_movies()
        elif dataset == 'agnews':
            from config_agnews_gemma import Config as Config_agnews
            config = Config_agnews()
        elif dataset == 'dbpedia':
            from config_dbpedia_gemma import Config as Config_dbpedia
            config = Config_dbpedia()
        elif dataset == 'medical':
            from config_medical_gemma import Config as Config_medical
            config = Config_medical()
        else:
            raise ValueError("Entrez un nom de dataset valide parmi ['movies','agnews','dbpedia','medical']")
    else:
        raise ValueError("Entrez un nom de modèle valide parmi ['bert-base-uncased','deberta-large', 'gemma']")

    return config