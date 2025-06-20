import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm  
import numpy as np
from tabulate import tabulate


from models.utils import ElasticNetLinearLayer

class ProjectionModel(nn.Module):
    def __init__(self, embedder_model, embedder_tokenizer, cavs_path_or_concepts_list, config, use_cavs_file=True, use_cls_token=True, device='cpu', classifier=None, linear_layer = None):
        super(ProjectionModel, self).__init__()
        self.embedder_model = embedder_model
        self.embedder_tokenizer = embedder_tokenizer
        self.device = device
        self.use_cavs_file = use_cavs_file
        self.use_cls_token = use_cls_token
        self.config = config

        self.concepts_name = []
        self.cavs = torch.tensor([]).to(self.device)  # Tensor vide pour commencer

        if self.use_cavs_file and cavs_path_or_concepts_list:
            self.concepts_name, self.cavs = self.load_cavs(cavs_path_or_concepts_list)
        elif cavs_path_or_concepts_list:  # Si on a des concepts, alors on les charge
            self.concepts_name, self.cavs = self.generate_cavs_from_words(cavs_path_or_concepts_list)

        self.embedder_model.to(self.device)
        self.embedder_model.eval()
        self.classifier = classifier
        self.alternate_save = False
        self.strategy = None # 'random', 'tcavs', 'lig', 'frequences'

        # new
        self.linear_layer = linear_layer
        self.model_name = config.model_name
        
    # OK
    def get_pooled_output(self, input_ids, attention_mask):
        """ get the embedding of the input text"""
        # with torch.no_grad()
        outputs = self.embedder_model(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_cls_token:
            pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            pooled_output = outputs.last_hidden_state.mean(1)  # Mean pooling
        return pooled_output

      
    def load_cavs(self, cavs_path):
        """ Load the dictionary (concept_name, vector) from a JSON file"""
        import json
        with open(cavs_path, 'r') as f:
            cavs_dict = json.load(f)
        
        self.cavs_dict = cavs_dict
            
    def cavs_selection(self):
        """ Selection des concepts dans le dictionnaire"""

        selected_cavs_dict = {name : vector for name, vector in self.cavs_dict.items() if name in self.concepts_name}
        if len(selected_cavs_dict) == 0:
            print("No concepts found in the dictionary. Please add concepts first.")
        else:
            cavs = torch.tensor(np.stack(list(selected_cavs_dict.values())), dtype=torch.float32).to(self.device) #only the vectors into tensor
            self.cavs = cavs
        print("Number of concepts loaded : ", len(selected_cavs_dict))    
        


    # OK : specific to projection modle
    def generate_cavs_from_words(self, concepts_list):
        """ TODO: manage this function"""
        inputs = self.embedder_tokenizer(concepts_list, return_tensors='pt', padding=True, truncation=True).to(self.device)
        # TODO: voir un beau jour si max len peut avoir un impact sur ça
        with torch.no_grad():
            outputs = self.embedder_model(**inputs)
            if self.use_cls_token:
                pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
            else:
                pooled_output = outputs.last_hidden_state.mean(1)  # Mean pooling
        
        concepts_name = concepts_list
        cavs = pooled_output.to(self.device)

        return concepts_name, cavs
    
    # OK : specific to projection modle
    def add_new_concepts(self, new_concepts_list):
        # Cette méthode permet d'ajouter de nouveaux concepts à tout moment
        new_concepts_name, new_cavs = self.generate_cavs_from_words(new_concepts_list)
        
        # Mise à jour des concepts et CAVs
        self.concepts_name.extend(new_concepts_name)
        if self.cavs.numel() == 0:  # Si cavs est vide
            self.cavs = new_cavs
        else:
            self.cavs = torch.cat((self.cavs, new_cavs), dim=0)

    # OK : specific to projection modle
    def compute_dist(self, emb, cavs):  
        """ Calcul de la distance entre les concepts et les embeddings 
        cavs : vecteurs de norme 1
        """      
        margins = torch.matmul(cavs, emb.T)
        return margins.T

    # OK
    def get_projections(self, input_ids, attention_mask):
        self.embedder_model.eval()

        if len(self.concepts_name) == 0:  # Si aucun concept n'est encore défini
            raise ValueError("No concepts available. Add concepts first.")
            
        with torch.no_grad():
            outputs = self.embedder_model(input_ids=input_ids, attention_mask=attention_mask)
            if self.use_cls_token:
                pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
            else:
                pooled_output = outputs.last_hidden_state.mean(1)  # Mean pooling

            projections = self.compute_dist(pooled_output, self.cavs)
        return projections
    
    # NEW
    def forward_projection_CBM(self, input_ids, attention_mask):
        projection = self.get_projections(input_ids, attention_mask)
        if self.classifier is not None:
            predictions = self.classifier(projection)
            return predictions
        else : 
            print("""classification head not defined : so return the projection outputs,
               use load method to load a classification head before continuing""")

    def forward(self, input_ids, attention_mask):
        pooled_output = self.get_pooled_output(input_ids, attention_mask)
        linear_output = self.linear_layer(pooled_output)

        projection_output = self.forward_projection_CBM(input_ids, attention_mask)
        final_output = linear_output + projection_output

        XtoC_output = self.get_projections(input_ids, attention_mask)

        return final_output, XtoC_output, pooled_output
    
    # NEW
    def forward_residual_layer (self, input_ids, attention_mask):
        """ forward on the résidal layer for for the LIG """
        pooled_output = self.get_pooled_output(input_ids, attention_mask)
        linear_output = self.linear_layer(pooled_output)

        projection_output = self.forward_projection_CBM(input_ids, attention_mask)
        final_output = linear_output + projection_output

        XtoC_output = self.get_projections(input_ids, attention_mask)

        return linear_output

    # NEW
    def forward_per_concept(self, input_ids, attention_mask, concept_index, threshold=0.5, mode='soft', abs_score=False):
        """
        Prédit la présence (score entre 0 et 1) d'un seul concept à partir des embeddings.
        
        Args:
            input_ids: Les identifiants des tokens (batch_size, sequence_length).
            attention_mask: La mask d'attention associée (batch_size, sequence_length).
            concept_index: L'indice du concept à prédire parmi les CAVs.
            threshold: Seuil pour la classification binaire en mode 'hard'.
            mode: 'soft' pour retourner un score, 'hard' pour retourner 0 ou 1.
            abs_score: Si True, applique la valeur absolue à concept_scores avant la sigmoïde.
        
        Returns:
            predictions: Un tenseur de dimension (batch_size,) contenant des scores entre 0 et 1,
                        représentant la probabilité que le concept soit présent.
                        Si mode = 'hard', renvoie directement 0 ou 1.
        """
        # Calcul des projections pour tous les concepts
        projections = self.get_projections(input_ids, attention_mask)  # shape: (batch_size, num_concepts)
        print('projections', projections)
        
        # Vérification que l'indice du concept est valide
        if concept_index < 0 or concept_index >= projections.shape[1]:
            raise ValueError(f"L'indice du concept doit être entre 0 et {projections.shape[1] - 1}")
        
        
        # Extraction du score pour le concept choisi
        concept_scores = projections[:, concept_index]
        if abs_score:
            concept_scores = torch.abs(concept_scores)
        print('concept_scores', concept_scores)
    
        print('sigmoid all concepts', torch.sigmoid(projections))

        # Passage par la fonction sigmoïde pour obtenir une probabilité (pour un problème binaire)
        predictions = torch.sigmoid(concept_scores)
        print('predictions', predictions)
        
        if mode == 'hard':
            predictions = torch.where(predictions > threshold, torch.tensor(1.0), torch.tensor(0.0))
        
        return predictions
    
    def reset_classifier(self):
        """Réinitialise les poids de la couche classifier (ElasticNetLinearLayer)."""
        if self.classifier is not None:
            self.classifier.reset_parameters()

    def train_model(self, train_loader, val_loader, num_epochs):
        # Vérifier si les CAVs (Concepts) sont chargés
        if len(self.concepts_name) == 0 or self.cavs.numel() == 0:
            raise ValueError("No concepts or CAVs available. Please add concepts before training.")
        
        # cavs selection
         

        # Initialiser ou réinitialiser la couche classifier
        if self.classifier is None:
            self.classifier = ElasticNetLinearLayer(self.cavs.shape[0], self.config.num_labels, alpha=self.config.alpha, l1_ratio=self.config.l1_ratio).to(self.device)
        else:
            self.classifier = ElasticNetLinearLayer(self.cavs.shape[0], self.config.num_labels, alpha=self.config.alpha, l1_ratio=self.config.l1_ratio).to(self.device)
            self.reset_classifier()  # Réinitialiser si déjà initialisé

        optimizer = optim.SGD(self.classifier.parameters(), lr=0.01, momentum=0.9)

        start_time = time.time()
        best_acc_score = 0
        kept_epoch = 0
        
        for epoch in range(num_epochs):
            self.classifier.train()
            for batch in tqdm(train_loader, desc=f"Training epoch {epoch + 1}", unit="batch"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # print(f"Labels shape: {labels.shape}, Labels: {labels}")

                projections = self.get_projections(input_ids, attention_mask)
                outputs = self.classifier(projections)
                # print(f"Outputs shape: {outputs.shape}")

                loss = self.classifier.elasticnet_loss(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate on validation data

            val_result = self.evaluate_model(val_loader, verbose=False, metrics_on_concepts=False)
            val_accuracy = val_result["task_global_accuracy"]
            val_f1 = val_result["task_global_macro_f1_score"]
            val_cls_acc = val_result["accuracies_per_class"]
            val_cls_f1 = val_result["f1_scores_per_class"]

            print(f"Epoch {epoch + 1}: Val Acc = {val_accuracy} Val Macro F1 = {val_f1} Val Cls Acc = {val_cls_acc} Val Cls F1 {val_cls_f1} " )
                  
            # keep the best model on th e validation set
            if val_accuracy > best_acc_score:
                kept_epoch = epoch     
                best_acc_score = val_accuracy
                self.save_model()

        total_duration = time.time() - start_time
        print(f"Total training duration: {total_duration:.2f} seconds")
        print("best model on val data found at epoch", kept_epoch)

        return self.classifier

    def evaluate_model(self, data_loader, verbose=False, metrics_on_concepts=False):
        """
        Évalue la performance sur la tâche principale et, si activé, sur les prédictions de concepts.
        Pour chaque concept, on applique la sigmoïde puis un arrondi pour obtenir des prédictions binaires.
        On suppose que, pour chaque concept, le batch contient une clé (le nom du concept) avec ses labels.
        """
        self.embedder_model.eval()
        self.linear_layer.eval()
        if self.classifier is not None:
            self.classifier.eval()
        
        total_correct = 0
        predict_labels = []
        true_labels = []
        
        # Dictionnaires pour accumuler les labels prédits et réels pour chaque concept
        concept_pred_labels = {concept: [] for concept in self.concepts_name}
        concept_true_labels = {concept: [] for concept in self.concepts_name}
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = self.forward(input_ids, attention_mask)
                final_output = outputs[0]      # sortie principale
                XtoC_output = outputs[1]       # projections sur les concepts
                
                preds = torch.argmax(final_output, dim=1)
                total_correct += (preds == labels).sum().item()
                predict_labels.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
                if metrics_on_concepts:
                    # Pour chaque concept, on applique sigmoïde et on arrondit (seuil 0.5)
                    for i, concept in enumerate(self.concepts_name):
                        concept_preds = torch.round(torch.sigmoid(XtoC_output[:, i])).cpu().numpy()
                        # On suppose que le batch contient une clé portant le nom du concept pour les labels réels.
                        if concept in batch:
                            concept_true = batch[concept].cpu().numpy()
                        else:
                            # Sinon, on peut par exemple considérer des labels nuls
                            concept_true = np.zeros_like(concept_preds)
                        concept_pred_labels[concept].extend(concept_preds.tolist())
                        concept_true_labels[concept].extend(concept_true.tolist())
        
        # Métriques pour la tâche principale
        true_labels_np = np.array(true_labels)
        predict_labels_np = np.array(predict_labels)
        accuracy = total_correct / len(data_loader.dataset)
        num_labels = len(np.unique(true_labels_np))
        
        macro_f1_scores = []
        accuracies_per_class = {}
        f1_scores_per_class = {}
        for i in range(num_labels):
            macro_f1 = f1_score(true_labels_np == i, predict_labels_np == i, average="macro")
            macro_f1_scores.append(macro_f1)
            
            class_pred = (predict_labels_np == i)
            class_true = (true_labels_np == i)
            if np.sum(class_true) > 0:
                class_accuracy = np.sum(class_pred & class_true) / np.sum(class_true)
            else:
                class_accuracy = 0
            accuracies_per_class[i] = class_accuracy
            
            f1_scores_per_class[i] = f1_score(true_labels_np == i, predict_labels_np == i, average="macro")
        
        mean_macro_f1 = np.mean(macro_f1_scores)
        
        results = {
            "task_global_accuracy": accuracy,
            "task_global_macro_f1_score": mean_macro_f1,
            "accuracies_per_class": accuracies_per_class,
            "f1_scores_per_class": f1_scores_per_class,
        }
        
        # Évaluation sur les concepts si activée dans la config et demandée
        if self.config.eval_concepts and metrics_on_concepts:
            concept_accuracies = {}
            concept_f1_scores = {}
            for concept in self.concepts_name:
                acc = accuracy_score(concept_true_labels[concept], concept_pred_labels[concept])
                f1_sc = f1_score(concept_true_labels[concept], concept_pred_labels[concept], average="macro")
                concept_accuracies[concept] = acc
                concept_f1_scores[concept] = f1_sc
            
            mean_concept_accuracy = np.mean(list(concept_accuracies.values()))
            mean_concept_f1_score = np.mean(list(concept_f1_scores.values()))
            
            results.update({
                "concept_accuracies": concept_accuracies,
                "concept_f1_scores": concept_f1_scores,
                "mean_concept_accuracy": mean_concept_accuracy,
                "mean_concept_f1_score": mean_concept_f1_score
            })
        
        if verbose:
            print(f"Evaluation : Task Accuracy = {accuracy*100:.2f}%, Macro F1 = {mean_macro_f1*100:.2f}%")
            print("Accuracy par classe :", accuracies_per_class)
            print("F1 par classe :", f1_scores_per_class)
            if self.config.eval_concepts and metrics_on_concepts:
                print("Métriques sur les concepts :")
                for concept in self.concepts_name:
                    print(f"  {concept}: Accuracy = {results['concept_accuracies'][concept]*100:.2f}%, F1 = {results['concept_f1_scores'][concept]*100:.2f}%")
        
        return results

    def run(self, train_loader, val_loader, test_loader, num_epochs = None, verbose=False, meta_on_concept=False):
        """
        Exécute l'entraînement puis évalue le modèle sur les ensembles d'entraînement, validation et test.
        Retourne un dictionnaire compilant les métriques obtenues.
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        # Entraînement du modèle
        # Ici, on suppose que self.train_model est déjà implémenté et qu'il entraîne la partie classifier.
        self.train_model(train_loader, val_loader, num_epochs=num_epochs)
        
        # Évaluation avec évaluation complète (tâche principale et concepts) sur l'ensemble d'entraînement
        train_metrics = self.evaluate_model(train_loader, verbose=verbose, metrics_on_concepts=True)
        # Pour validation et test, on peut ne pas calculer les métriques sur les concepts si non nécessaires
        val_metrics = self.evaluate_model(val_loader, verbose=verbose, metrics_on_concepts=False)
        test_metrics = self.evaluate_model(test_loader, verbose=verbose, metrics_on_concepts=False)
        
        run_info = {
            "train_acc": train_metrics["task_global_accuracy"],
            "val_acc": val_metrics["task_global_accuracy"],
            "test_acc": test_metrics["task_global_accuracy"],
            "f1_train": train_metrics["task_global_macro_f1_score"],
            "f1_val": val_metrics["task_global_macro_f1_score"],
            "f1_test": test_metrics["task_global_macro_f1_score"],
            "cls_acc_train": train_metrics["accuracies_per_class"],
            "cls_acc_val": val_metrics["accuracies_per_class"],
            "cls_acc_test": test_metrics["accuracies_per_class"],
            "cls_f1_train": train_metrics["f1_scores_per_class"],
            "cls_f1_val": val_metrics["f1_scores_per_class"],
            "cls_f1_test": test_metrics["f1_scores_per_class"]
        }
        
        if verbose:
            table = [
                ['Metric', 'Train', 'Validation', 'Test'],
                ['Global Accuracy', f"{run_info['train_acc']*100:.2f}%", f"{run_info['val_acc']*100:.2f}%", f"{run_info['test_acc']*100:.2f}%"],
                ['Global Macro F1', f"{run_info['f1_train']*100:.2f}%", f"{run_info['f1_val']*100:.2f}%", f"{run_info['f1_test']*100:.2f}%"],
                ['Class Accuracies', str(run_info['cls_acc_train']), str(run_info['cls_acc_val']), str(run_info['cls_acc_test'])],
                ['Class F1 Scores', str(run_info['cls_f1_train']), str(run_info['cls_f1_val']), str(run_info['cls_f1_test'])]
            ]
            print(tabulate(table, headers="firstrow", tablefmt="grid"))
        
        if meta_on_concept:
            return run_info, {"task_concepts_perf": train_metrics.get("concept_accuracies", None)}
        return run_info

   # ---------------------------------- ADDED METHOD FOR RESIDUAL
    # NEW
    def reset_residual_layer(self):
        """Reset the residual layer (linear_layer).
        TODO : EVALUTE the need to do this
        """
        if isinstance(self.linear_layer, torch.nn.Linear):
            self.linear_layer.reset_parameters()
        else:
            # TODO: For other types of layers (ReLU, etc.), adjust initialization accordingly
            for layer in self.linear_layer.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def train_residual_layer(self, train_loader, num_epochs):
        print("-------- Training the residual layer ----------")   

        # Reset the residual layer (linear_layer) before each iteration
        self.reset_residual_layer()

        self.embedder_model.eval()
        self.classifier.eval() # only the classifier in the CBM part is trainable here

        self.residual_optimizer = torch.optim.Adam(self.linear_layer.parameters(), lr=self.config.lr)
    
        for epoch in range(num_epochs):
            self.linear_layer.train()

            train_accuracy = 0
            # concept_train_accuracy = 0
            predict_labels = np.array([])
            true_labels = np.array([])

            all_pool_outputs = []

            for batch in tqdm(train_loader, desc="Train", unit="batch"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label = batch["label"].to(self.device)

                self.linear_layer.train()
                
                final_output, XtoC_output, pooled_output = self.forward(input_ids, attention_mask)

                all_pool_outputs.append(pooled_output)

                loss = self.linear_layer.ridge_loss(final_output, label)

                self.residual_optimizer.zero_grad()
                loss.backward()
                self.residual_optimizer.step()

                predictions = torch.argmax(final_output, axis=1)
                train_accuracy += torch.sum(predictions == label).item()
                predict_labels = np.append(predict_labels, predictions.cpu().numpy())
                true_labels = np.append(true_labels, label.cpu().numpy())
                
            train_accuracy /= len(train_loader.dataset)
            num_labels = len(np.unique(true_labels))

            macro_f1_scores = []
            for label in range(num_labels):
                label_pred = np.array(predict_labels) == label
                label_true = np.array(true_labels) == label
                macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score = np.mean(macro_f1_scores)

            print(f"Epoch {epoch + 1}: Train Acc = {train_accuracy * 100} train Macro F1 = {mean_macro_f1_score * 100}")
            
            if self.alternate_save == True:
                torch.save(self.linear_layer.state_dict(), f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/Our_CBM_projection/{self.model_name}_residual_layer_projection_strategy_{self.strategy}_{self.iteration}.pth") 
            else: 
                torch.save(self.linear_layer.state_dict(), f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/ProjectionCBM/{self.model_name}_residual_layer_projection_strategy.pth")

    def evaluate_model_with_residual(self, data_loader):
        self.embedder_model.eval()
        self.classifier.eval()
        self.linear_layer.eval()
        
        val_loss = 0
        val_accuracy = 0
        concept_val_accuracy = 0
        predict_labels = np.array([])
        true_labels = np.array([])

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Validation", unit="batch"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label = batch["label"].to(self.device)

                final_output, XtoC_output, _ = self.forward(input_ids, attention_mask)
                
                loss = self.linear_layer.ridge_loss(final_output, label)
                val_loss += loss.item()

                predictions = torch.argmax(final_output, axis=1)
                val_accuracy += torch.sum(predictions == label).item()
                predict_labels = np.append(predict_labels, predictions.cpu().numpy())
                true_labels = np.append(true_labels, label.cpu().numpy())

            val_loss /= len(data_loader)
            val_accuracy /= len(data_loader.dataset)

            num_labels = len(np.unique(true_labels))
            macro_f1_scores = []
            for label in range(num_labels):
                label_pred = np.array(predict_labels) == label
                label_true = np.array(true_labels) == label
                macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score = np.mean(macro_f1_scores)

            print(f"Validation Loss: {val_loss}")
            print(f"Validation Acc: {val_accuracy * 100}, Validation Macro F1: {mean_macro_f1_score * 100}")

        return val_loss, val_accuracy #, mean_macro_f1_score


   # ---------------------------------- Load and save now
    def save_model(self):
        """ Only save the classifier head of the trained projection CBM"""
        if self.alternate_save:
            os.makedirs(f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/Our_CBM_projection", exist_ok=True)
            torch.save(self.classifier.state_dict(), f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/Our_CBM_projection/classifier_{self.strategy}_{self.iteration}.pth")
        else:
            os.makedirs(f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/ProjectionCBM", exist_ok=True)
            torch.save(self.classifier.state_dict(), f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/ProjectionCBM/classifier.pth")


    def load_model(self):
        """ Only load the classifier head of the trained projection CBM"""

        self.classifier = ElasticNetLinearLayer(self.cavs.shape[0], self.config.num_labels, alpha=0.01, l1_ratio=0.5).to(self.device)

        if self.alternate_save:
            self.classifier.load_state_dict(torch.load(f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/Our_CBM_projection/classifier_{self.strategy}_{self.iteration}.pth", map_location=self.device))
            if self.linear_layer is not None:
                self.linear_layer.load_state_dict(torch.load(f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/Our_CBM_projection/{self.model_name}_residual_layer_projection_strategy_{self.strategy}_{self.iteration}.pth"))
        else:
            self.classifier.load_state_dict(torch.load(f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/ProjectionCBM/classifier.pth", map_location=self.device))
            if self.linear_layer is not None:
                self.linear_layer.load_state_dict(torch.load(f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/ProjectionCBM/{self.model_name}_residual_layer_projection_strategy.pth"))

        self.classifier.eval()
        if self.linear_layer is not None:
            self.linear_layer.eval()        
        return self.classifier



