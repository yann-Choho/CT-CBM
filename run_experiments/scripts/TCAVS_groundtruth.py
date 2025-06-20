import os 
import sys

import numpy as np
import torch 
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import json

from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from scipy.spatial.distance import euclidean

from torch.utils.data import DataLoader, Subset
from TCAVS_utils import stratified_subset_dataloader

import time

class TCAV():
    """ TCVAS class for ranking concepts importance by assigning a sensitivity score to each concept. 
    
    Args:
    - concepts: List of concepts to explain.
    - device: Device to use for computations.
    - batch_size: Batch size for DataLoader.
    - verbose: If True, print additional information.
    - svm_params_dict: Dictionary containing parameters for the SVM classifier.
    - num_classes: Number of classes in the dataset.
    
    Attributes:
    - model: The model to use for explanations.
    - tokenizer: The tokenizer to use for explanations.
    - max_len: Maximum length of input sequences.
    - seed: Random seed for reproducibility.
    - concepts: List of concepts to explain.
    - batch_size: Batch size for DataLoader.
    - cavs: Dictionary containing the concept-attribute vectors.
    - verbose: If True, print additional information.
    
    Methods:
    - fit: Fit the explainer to the data.
    - get_gradient: Get the gradient of the model with respect to the embedding.
    - get_embeddings: Get the embeddings of the data.
    - learn_cav: Learn the concept-attribute vector for a given concept.
    - save_cavs_to_file: Save the concept-attribute vectors to a file.
    - load_cavs_from_file: Load the concept-attribute vectors from a file.
    - get_gradients: Get the gradients of the model with respect to the embedding.
    - get_gradients_per_class: Get the gradients of the model with respect to the embedding for a given class.
    - calculate_tcav_scores: Calculate the TCAV scores for each concept and class.
    - most_important_concepts: Calculate the most important concepts.
    - plot_scores: Plot the TCAV scores
    """

    def __init__(self, concepts, baseline_model, embedder_tokenizer, batch_size=64, verbose=False, svm_params_dict=None, config = None, train_loader = None, val_loader = None, test_loader = None):
        super().__init__()
        self.embedder_model = baseline_model.embedder_model  # Utiliser le modèle chargé
        self.embedder_tokenizer = embedder_tokenizer  # Utiliser le tokenizer chargé
        self.seed = 42
        self.concepts = concepts
        self.device = config.device
        self.batch_size = batch_size
        self.cavs = {}
        self.verbose = verbose
        # default paramters in sklearn library
        if not svm_params_dict:
            self.svm_params_dict = {
                'alpha': 0.0001,
                'max_iter': 1000,
                'tol': 0.001,
                'class_weight': 'balanced',
            }
        self.score_by_class = {}
        self.sorted_concepts_macro_concepts = None
        self.config = config
        self.use_cls_token = config.use_cls_token
        
        # besoin d'un baseline classifier pour calculer les gradients
        self.baseline_model = baseline_model
        
    def fit(self, dataloader, layer_num = -1, linear_classifier = True, balanced=True, closest=False):
        """ Fit the explainer to the data.
        
            Args:
            - dataloader: DataLoader containing the data.
            - classifier: The (SVM) classifier to use for explanations .            
            Returns:
            - Update the cav vector for each concept."""
        self.concept_accuracies = {}
        self.concept_f1score = {}
        for concept in self.concepts:
            cav_, accuracy_, f1score_ = self.learn_cav(dataloader, concept, layer_num, linear_classifier, balanced, closest)
            self.cavs[concept] = cav_
            self.concept_accuracies[concept] = accuracy_
            self.concept_f1score[concept] = f1score_

        file_path = f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}/"
        os.makedirs(f"{file_path}", exist_ok=True)
        with open(file_path+'cavs_svm.json', 'w') as f:
            json.dump({key: value.tolist() for key, value in self.cavs.items()}, f, indent=4)
        with open(file_path+'cavs_acc.json', 'w') as f:
            json.dump(self.concept_accuracies, f, indent=4)
        with open(file_path+'cavs_f1.json', 'w') as f:
            json.dump(self.concept_f1score, f, indent=4)
        print(f"Concepts saved to {file_path}")

        return self.cavs, self.concept_accuracies, self.concept_f1score
         
    def get_embeddings_and_labels(self, dataloader, concept, layer_num=-1, use_cls_token=True):
        """ get the embedding and the label of the concept to avoid dataloader sampling strategy to disrupt everything"""
        all_embeddings = []
        all_labels = []
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.embedder_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                if use_cls_token:
                    if(layer_num == -1):
                        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
                    else:
                        pooled_output = outputs.hidden_states[layer_num][:, 0, :]  # CLS token
                else:
                    if(layer_num == -1):
                        pooled_output = outputs.last_hidden_state.mean(1) # Mean pooling
                    else:
                        pooled_output = outputs.hidden_states[layer_num].mean(1) # Mean pooling         
                # outputs[0] correspond à outputs.last_hidden_state en fait
            all_embeddings.append(pooled_output.cpu().numpy())
            all_labels.append(batch[concept].numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return embeddings, labels
        
    @staticmethod
    def undersample_dataloader(dataloader, concept, balanced=True, closest=False):
        from collections import Counter
        import numpy as np
        from torch.utils.data import DataLoader, Subset

        # Accéder au dataset depuis le DataLoader
        dataset = dataloader.dataset

        # Initialiser les listes pour stocker les labels et les indices correspondants
        labels = []
        indices = []

        # Parcourir le dataset pour extraire les labels et les indices
        for idx in range(len(dataset)):
            sample = dataset[idx]
            label = sample[concept]
            labels.append(label)
            indices.append(idx)

        labels = np.array(labels)
        indices = np.array(indices)

        # Compter le nombre d'instances pour chaque classe
        class_counts = Counter(labels)

        # Trouver la taille de la classe minoritaire
        min_class_count = min(class_counts.values())

        # Trouver le label correspondant à la classe minoritaire
        min_class_label = [label for label, count in class_counts.items() if count == min_class_count][0]

        # Initialiser la liste des indices à conserver
        retain_indices = []

        # Appliquer l'undersampling
        for class_label, count in class_counts.items():
            class_indices = indices[labels == class_label]

            if count == min_class_count:
                # Garder tous les indices de la classe minoritaire
                retain_indices.extend(class_indices)

            else:

                if(closest):
                    #distances = [sum([euclidean(dataset[idx]['input_ids'], dataset[idx_topic]['input_ids']) for idx_topic in indices[labels == min_class_label]]) for idx in class_indices]
                    embeddings = [dataset[idx_topic]['input_ids'] for idx_topic in indices[labels == min_class_label]]
                    mean_topic_embed = torch.mean(torch.stack(embeddings).float(), dim=0)
                    distances = [euclidean(dataset[idx]['input_ids'], mean_topic_embed) for idx in class_indices]
                    ranked_class_indices = class_indices[np.argsort(distances)]

                # Sélectionner aléatoirement un nombre égal d'indices dans la classe majoritaire
                if(balanced):
                    sampled_indices = np.random.choice(class_indices, size=min_class_count, replace=False)
                    if(closest):
                        sampled_indices = ranked_class_indices[:min_class_count]
                else:
                    sampled_indices = np.random.choice(class_indices, size=class_counts[0], replace=False)
                retain_indices.extend(sampled_indices)

        # Créer un nouveau DataLoader en utilisant les indices sous-échantillonnés
        undersampled_dataset = Subset(dataset, retain_indices)
        undersampled_loader = DataLoader(undersampled_dataset, batch_size=dataloader.batch_size, shuffle=True)

        return undersampled_loader



    def learn_cav(self, dataloader, concept, layer_num=-1, linear_classifier=True, 
                  balanced=True, closest=False, logistic_classifier=False, test_dataloader=None):
        """
        Apprend le CAV pour un concept donné en entraînant un classifieur sur les embeddings extraits
        d'un dataloader (après undersampling) et évalue sa performance sur un dataloader de test (si fourni).
        Les performances (train et test) sont sauvegardées dans un fichier JSON.
        
        Args:
            dataloader (DataLoader): DataLoader pour l'entraînement.
            concept (str): Nom du concept pour lequel apprendre le CAV.
            layer_num (int): Indice de la couche à utiliser pour extraire les embeddings.
            linear_classifier (bool): Si True, utilise SGDClassifier.
            balanced (bool): Paramètre pour l'undersampling.
            closest (bool): Option pour l'undersampling basé sur la distance.
            logistic_classifier (bool): Si True, utilise LogisticRegression.
            test_dataloader (DataLoader, optionnel): DataLoader pour évaluer les performances du classifieur.
            
        Returns:
            cav (np.array): Le vecteur CAV appris.
            accuracy (float): Précision du classifieur sur l'ensemble de test.
            f1score (float): f1-score du classifieur sur l'ensemble de test.
        """
        # Trouver l'indice du concept actuel dans la liste des concepts (si nécessaire)
        concept_idx = self.concepts.index(concept)
    
        # Compter les éléments avant undersampling
        original_labels = [batch[concept].numpy() for batch in dataloader]
        original_labels = np.concatenate(original_labels)
        unique, counts = np.unique(original_labels, return_counts=True)
        print(f'Avant undersampling: {dict(zip(unique, counts))}')
    
        # Appliquer l'undersampling sur le dataloader d'entraînement
        undersampled_dataloader = self.undersample_dataloader(dataloader, concept=concept, 
                                                              balanced=balanced, closest=closest)
    
        # Extraire les embeddings et les labels après undersampling pour l'entraînement
        embeddings, labels = self.get_embeddings_and_labels(undersampled_dataloader, 
                                                              concept=concept, 
                                                              layer_num=layer_num, 
                                                              use_cls_token=self.use_cls_token)
                        
        # Compter les éléments après undersampling
        undersampled_labels = [batch[concept].numpy() for batch in undersampled_dataloader]
        undersampled_labels = np.concatenate(undersampled_labels)
        unique, counts = np.unique(undersampled_labels, return_counts=True)
        print(f'Après undersampling: {dict(zip(unique, counts))}')
    
        # Vérifier que le problème est binaire
        if len(set(labels)) > 2:
            raise NotImplementedError('Les CAVs sont définis pour des problèmes binaires.')
    
        # Récupérer les ensembles d'entraînement et de test
        if test_dataloader is not None:
            # Utiliser le dataloader de test fourni pour l'évaluation
            train_embeddings, train_labels = embeddings, labels
            test_embeddings, test_labels = self.get_embeddings_and_labels(test_dataloader, 
                                                                           concept=concept, 
                                                                           layer_num=layer_num, 
                                                                           use_cls_token=self.use_cls_token)
        else:
            # Fractionner aléatoirement l'ensemble d'entraînement en train/test
            train_embeddings, test_embeddings, train_labels, test_labels = \
                train_test_split(embeddings, labels, test_size=0.2, random_state=self.seed, stratify=labels)
    
        # Affichage de la répartition dans les ensembles de train et test
        unique, counts = np.unique(train_labels, return_counts=True)
        print(f'Train set: {dict(zip(unique, counts))}')
        unique, counts = np.unique(test_labels, return_counts=True)
        print(f'Test set: {dict(zip(unique, counts))}')
    
        # Choix du classifieur
        if linear_classifier:
            lm = linear_model.SGDClassifier(**self.svm_params_dict)
        elif logistic_classifier:
            lm = linear_model.LogisticRegression()
        else:
            lm = SVC(kernel='rbf', class_weight='balanced')
        
        # Entraînement du classifieur
        lm.fit(train_embeddings, train_labels)
        
        # Prédictions sur l'ensemble d'entraînement et de test
        train_predictions = lm.predict(train_embeddings)
        test_predictions = lm.predict(test_embeddings)
        
        # Calcul des métriques pour l'ensemble d'entraînement
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_f1score = f1_score(train_labels, train_predictions, average='binary')
        train_conf_matrix = confusion_matrix(train_labels, train_predictions).tolist()
        train_report = classification_report(train_labels, train_predictions, output_dict=True)
        
        # Calcul des métriques pour l'ensemble de test
        test_accuracy = accuracy_score(test_labels, test_predictions)
        test_f1score = f1_score(test_labels, test_predictions, average='binary')
        test_conf_matrix = confusion_matrix(test_labels, test_predictions).tolist()
        test_report = classification_report(test_labels, test_predictions, output_dict=True)
        
        # Affichage du rapport de classification pour le test
        print("Rapport de classification (Test):")
        print(classification_report(test_labels, test_predictions))
        
        # Sauvegarde des performances dans un dictionnaire
        performance_dict = {
            "train": {
                "accuracy": train_accuracy,
                "f1score": train_f1score,
                "confusion_matrix": train_conf_matrix,
                "classification_report": train_report
            },
            "test": {
                "accuracy": test_accuracy,
                "f1score": test_f1score,
                "confusion_matrix": test_conf_matrix,
                "classification_report": test_report
            }
        }
        
        # Sauvegarde dans un fichier JSON avec le maximum de détails
        performance_file = f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}/performance_{concept}.json"
        os.makedirs(os.path.dirname(performance_file), exist_ok=True)
        with open(performance_file, 'w') as f:
            json.dump(performance_dict, f, indent=4)
        
        # Extraction du CAV selon le type de classifieur utilisé
        if linear_classifier or logistic_classifier:
            cav = -1 * lm.coef_[0]
            cav = cav / np.linalg.norm(cav)
        else:
            print("Impossible d'obtenir les CAVs avec ce type de classifieur, un vecteur nul est retourné.")
            cav = np.zeros(768)
        
        if self.verbose:
            print(f'Learned CAV for concept: {concept}')
            print(f'{list(labels).count(1)} (concept) vs {list(labels).count(0)} (others)')
            print(f'\tTrain Accuracy: {train_accuracy * 100:.1f}% - Test Accuracy: {test_accuracy * 100:.1f}%')
            print(f'\tTrain f1-score: {train_f1score * 100:.1f}% - Test f1-score: {test_f1score * 100:.1f}%')
            print(f'\tTrain confusion matrix: {train_conf_matrix}')
            print(f'\tTest confusion matrix: {test_conf_matrix}')
            print()
    
        return cav, test_accuracy, test_f1score



    def save_cavs_to_file(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.cavs, f)

    def load_cavs_from_file(self, path = None):
        """
        Charge les CAVs depuis un fichier JSON et les stocke dans self.cavs.
        Retourne le dictionnaire des CAVs.
        """
        if path == None:
            file_path = f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}/cavs_{self.config.cavs_type}.json"
            with open(file_path, 'r') as f:
                cavs = json.load(f)
            print("cavs loaded at", file_path)
        else:
            file_path = path
            with open(file_path, 'r') as f:
                cavs = json.load(f)
            print("cavs loaded at", file_path)
        # Si nécessaire, convertir les listes en tableaux numpy
        for key, value in cavs.items():
            cavs[key] = np.array(value)
        
        self.cavs = cavs
        return self.cavs
    
    def get_pooled_output(self, input_ids, attention_mask):
        # with torch.no_grad():
        outputs = self.embedder_model(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_cls_token:
            pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            pooled_output = outputs.last_hidden_state.mean(1)  # Mean pooling
        return pooled_output

    def get_gradients_per_class(self, input_ids, attention_mask, class_idx=0):
        """
        Calcule le gradient par rapport à `final_output` pour une classe donnée pour une seule entrée.

        Args:
        - classifier: Le modèle à utiliser pour le calcul du gradient.
        - input_ids: Les IDs d'entrée de la phrase (une seule entrée).
        - attention_mask: Le masque d'attention de la phrase (une seule entrée).
        - class_idx: L'indice de la classe pour laquelle calculer le gradient.

        Returns:
        - gradients: Les gradients calculés par rapport à `final_output`.
        """
        classifier = self.baseline_model.classifier
        classifier.eval()  # Mettre le modèle en mode évaluation
        input_ids = input_ids.to(self.device)  # Déplacer les inputs sur le bon device
        attention_mask = attention_mask.to(self.device)

        pooled_output = self.get_pooled_output(input_ids, attention_mask)
        pooled_output = torch.autograd.Variable(pooled_output, requires_grad=True)
        pooled_output.retain_grad()

        linear_output = classifier(pooled_output)
        target_output = linear_output[:, class_idx]  # Obtenir la sortie pour la classe spécifiée

        classifier.zero_grad()

        target_output.backward()
        
        gradients = pooled_output.grad.clone().detach().cpu().numpy()
        # print("Weights:", classifier[0].weight[0])
        # print("Gradients:", pooled_output.grad)

        return gradients


    # @staticmethod
    def partition_dataloader_by_labels(self, main_dataloader):
        """ Partitionne un DataLoader en plusieurs DataLoaders en fonction des labels des échantillons.
        Args:
            main_dataloader (DataLoader): Le DataLoader principal à partitionner.
        
        Returns:
            dict: Un dictionnaire de DataLoaders partitionnés par label.
        """
        # Récupérer les indices des échantillons pour chaque classe
        indices_by_label = {}
        
        for idx, batch in enumerate(main_dataloader):
            # On suppose que batch est un dictionnaire contenant 'label'
            labels = batch['label']
            
            for i, label in enumerate(labels):
                if label.item() not in indices_by_label:
                    indices_by_label[label.item()] = []
                indices_by_label[label.item()].append(idx * main_dataloader.batch_size + i)

        # Créer un dictionnaire pour stocker les DataLoaders partitionnés
        partitioned_loaders = {}

        for label, indices in indices_by_label.items():
            subset = Subset(main_dataloader.dataset, indices)
            
            # Créer un DataLoader pour le sous-ensemble
            partitioned_loaders[label] = DataLoader(subset, batch_size=main_dataloader.batch_size, shuffle=True)

        return partitioned_loaders



    def calculate_tcav_scores(self, dataloader=None, 
                              use_subset=True, 
                              selection_method="all", 
                              measure_time=False, 
                              bootstrap=False, 
                              bootstrap_runs=5):
        """
        Calcule les scores TCAV pour chaque concept en utilisant un DataLoader, basé sur le ground truth.
        
        Args:
          - dataloader: DataLoader contenant les données de test.
          - use_subset: Si True, utilise un sous-échantillon (10%) du dataloader.
          - selection_method: Méthode de sélection des exemples parmi :
                "all"    -> tous les exemples,
                "random" -> un échantillon aléatoire de 10%,
                "active" -> uniquement les exemples où le concept est activé (extraction de la colonne correspondante).
          - measure_time: Si True, mesure le temps de calcul de chaque run.
          - bootstrap: Si True, effectue plusieurs runs pour bootstrapping.
          - bootstrap_runs: Nombre de runs de bootstrapping (par défaut 5).
          
        Retourne un dictionnaire final avec pour chaque concept :
          - la moyenne du score TCAV,
          - la variance,
          - la fréquence d'activation dans le dataloader.
          
        La fonction sauvegarde également deux fichiers pickle :
          - scores_by_concept_ground_truth.pkl : les scores par concept,
          - sensitivity_matrix.pkl : la matrice (liste par concept) avec le détail par exemple.
        """
        
        if dataloader is None:
            print("Aucun DataLoader fourni.")
            return None
    
        # Option de sous-échantillonnage sur le dataloader
        if use_subset:
            print("Utilisation d'un sous-échantillon pour réduire le temps de calcul")
            dataloader = stratified_subset_dataloader(dataloader, fraction=0.1)
        
        # Construire une liste de tous les exemples pour faciliter la sélection
        examples = []
        for batch in dataloader:
            batch_size = len(batch["input_ids"])
            for i in range(batch_size):
                # Pour chaque exemple, récupérer les entrées et le label
                example = {
                    "input_ids": batch["input_ids"][i].unsqueeze(0),  # forme (1, seq_len)
                    "attention_mask": batch["attention_mask"][i].unsqueeze(0),
                    "label": batch["label"][i].item()
                }
                # Pour chaque concept, la clé est directement présente (les colonnes dummy_ ont été renommées)
                example["concepts"] = {}
                for concept in self.concepts:
                    if concept in batch:
                        # On suppose que la valeur est un tenseur scalaire (0 ou 1)
                        example["concepts"][concept] = batch[concept][i].item()
                    else :
                        print(f"concept {concept} doesn't exist ") 
                examples.append(example)
        total_examples = len(examples)
        
        # Calculer la fréquence d'activation pour chaque concept
        concept_frequency = {concept: 0 for concept in self.concepts}
        if total_examples > 0:
            for ex in examples:
                for concept in self.concepts:
                    # On considère le concept activé si la valeur est > 0
                    if ex["concepts"].get(concept, 0) > 0:
                        concept_frequency[concept] += 1
            # for concept in self.concepts:
            #     concept_frequency[concept] /= total_examples
        
        # Initialiser les structures pour le bootstrapping
        bootstrap_scores = {concept: [] for concept in self.concepts}
        bootstrap_times = []
        bootstrap_matrices = []  # stocke la sensibilité par exemple pour le dernier run
    
        runs = bootstrap_runs if bootstrap else 1
    
        for run in range(runs):
            print(f"---- run number {run} ------")
            if measure_time:
                start_time = time.time()
            
            # Sélectionner les exemples selon la méthode choisie
            if selection_method == "all":
                selected_examples = examples
            elif selection_method == "random":
                sample_size = max(1, int(0.1 * total_examples))
                selected_examples = random.sample(examples, sample_size)
            elif selection_method == "active":
                # Pour "active", nous traitons chaque concept séparément.
                # Ici, on démarre avec tous les exemples, puis pour chaque concept on filtrera.
                selected_examples = examples
            else:
                print(f"Selection method '{selection_method}' non reconnu, utilisation de 'all'.")
                selected_examples = examples
            
            # Initialiser les structures pour stocker les sensibilités et la matrice détaillée pour ce run
            run_sensitivities = {concept: [] for concept in self.concepts}
            sensitivity_matrix = {concept: [] for concept in self.concepts}
            
            # Parcourir les exemples sélectionnés
            for ex in tqdm(selected_examples, desc="example", unit='line') :
                input_ids = ex["input_ids"].to(self.device)
                attention_mask = ex["attention_mask"].to(self.device)
                label = ex["label"]
                
                # Calculer les gradients par rapport au label réel
                grads = self.get_gradients_per_class(input_ids, attention_mask, class_idx=label)
                
                for concept in self.concepts:
                    # Pour la méthode "active", ne traiter que l'exemple si le concept est activé (valeur > 0)
                    if selection_method == "active" and ex["concepts"].get(concept, 0) <= 0:
                        continue
                    
                    # Calculer la sensibilité en prenant le produit scalaire entre le CAV et les gradients
                    sensitivity = np.dot(self.cavs[concept], grads.flatten())
                    run_sensitivities[concept].append(sensitivity)
                    
                    # Décoder le texte s'il existe un tokenizer associé
                    if hasattr(self, "embedder_tokenizer") and self.embedder_tokenizer is not None:
                        text = self.embedder_tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    else:
                        text = "<texte indisponible>"
                    sensitivity_matrix[concept].append({
                        "text": text,
                        "sensitivity": sensitivity,
                        "label": label
                    })
            
            # Calcul du score TCAV pour le run courant : proportion d'exemples avec sensibilité positive
            run_scores = {}
            for concept, sens_list in run_sensitivities.items():
                if len(sens_list) > 0:
                    score = len([s for s in sens_list if s > 0]) / len(sens_list)
                else:
                    score = 0
                run_scores[concept] = score
                bootstrap_scores[concept].append(score)
            
            if measure_time:
                end_time = time.time()
                bootstrap_times.append(end_time - start_time)
            
            bootstrap_matrices.append(sensitivity_matrix)
        
        # Calculer la moyenne et la variance des scores sur les runs de bootstrapping
        final_scores = {}
        for concept, scores in bootstrap_scores.items():
            avg_score = np.mean(scores)
            score_variance = np.var(scores)
            final_scores[concept] = {
                "avg_score": avg_score,
                "variance": score_variance,
                "frequency": concept_frequency.get(concept, None)
            }
        
        if measure_time and bootstrap_times:
            avg_time = np.mean(bootstrap_times)
            print(f"Temps de calcul moyen sur {runs} run(s) : {avg_time:.4f} secondes")
        
        # Sauvegarder les résultats
        save_path = f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}"
        os.makedirs(save_path, exist_ok=True)
        with open(f"{save_path}/scores_by_concept_ground_truth.pkl", 'wb') as f:
            pickle.dump(final_scores, f)
        with open(f"{save_path}/sensitivity_matrix.pkl", 'wb') as f:
            # Sauvegarde la matrice du dernier run
            pickle.dump(bootstrap_matrices[-1], f)
        
        return final_scores
    
    
    def plot_scores(self, save_path):
        """
        Trace les scores TCAV par catégorie et les sauvegarde sous forme d'image.

        Args:
        - score_by_class: Dictionnaire des scores TCAV par classe.
        - save_path: Chemin pour sauvegarder le graphique.
        - model_name: Le nom du modèle.
        """
        # Extraire les catégories et les valeurs
        categories = list(self.score_by_class[0].keys())
        indices = list(self.score_by_class.keys())
        values = np.array([[self.score_by_class[idx].get(cat, 0) for cat in categories] for idx in indices])

        # Créer des labels personnalisés pour chaque catégorie en fonction des éléments trouvés
        category_labels = []
        for cat in categories:
            elements = [f"{elem}" for elem in self.score_by_class[0].keys()]
            label = f"{cat} ({', '.join(elements)})"
            category_labels.append(label)

        # Définir la largeur des barres
        bar_width = 0.2
        # Calculer les positions des barres pour chaque catégorie
        positions = [np.arange(len(indices))]
        for i in range(1, len(categories)):
            positions.append([x + bar_width for x in positions[i - 1]])

        # Créer le graphique en barres non empilées
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, label in enumerate(category_labels):
            ax.bar(positions[i], values[:, i], width=bar_width, edgecolor='grey', label=label)

        # Ajouter des labels, des titres et des légendes
        ax.set_xlabel('Index', fontweight='bold')
        ax.set_ylabel('Scores', fontweight='bold')
        ax.set_title('Scores par catégorie et par index')
        ax.set_xticks([r + bar_width for r in range(len(indices))])
        ax.set_xticklabels(indices)
        ax.legend()

        # Afficher et sauvegarder le graphique
        plt.tight_layout()
        plt.savefig(f"{save_path}/blue_checkpoints/{config.model_name}/cavs/{config.cavs_type}/scores_by_class.png")
        plt.show()

    def save_tcv_ranker(self, file_path):
        """
        Sauvegarde l'objet TCAV avec tous ses attributs dans un fichier pickle.

        Args:
        - file_path: Chemin du fichier où l'objet sera sauvegardé.
        """
        # Sauvegarder l'objet entier dans un fichier
        # os.makedirs(f"{file_path}", exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"TCAV object saved to {file_path}")
    
    @staticmethod
    def load_tcv_ranker(file_path):
        """
        Charge un objet TCAV sauvegardé à partir d'un fichier pickle.

        Args:
        - file_path: Chemin du fichier d'où l'objet sera chargé.
        
        Returns:
        - L'objet TCAV chargé.
        """
        # Charger l'objet depuis un fichier
        with open(file_path, 'rb') as f:
            loaded_tcv_ranker = pickle.load(f)
        print(f"TCAV object loaded from {file_path}")
        return loaded_tcv_ranker