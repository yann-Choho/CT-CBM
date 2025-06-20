import re
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
from tqdm import tqdm

class Config:
    """defining the following parameters:
    - model_name: the name of the model to use
    - dim: the dimension of the model
    - max_len: the maximum length of the input
    - batch_size: the batch size
    - lambda_XtoC: the weight of the auxiliary loss
    - is_aux_logits: whether to use auxiliary logits
    - num_labels: the number of labels
    - num_epochs: the number of epochs
    - num_each_concept_classes: the number of classes for each concept
    - data_type: the type of data to use
    - device: the device to use
    """

    annotation = "C3M" # "C3M", "our_annotation"

    infra = "A100" # "A100" , "DATABRICKS"
    mode = 'joint'
    model_name = 'gemma' # 'bert-base-uncased'/'deberta-large'/'roberta-large'/
    # 'roberta-base'/'deberta-large'/'deberta-base'/'modern-bert-base'/'modern-bert-large'
    # 'gemma'
    if model_name == 'gemma':
        dim = 2304   # it is for gemma 2B 
    else:
        dim = 1024 if re.search(r'large', model_name, re.IGNORECASE) else 768
    max_len = 256
    # 128 pour dbpedia et agnews 
    # 512 health et movie genre  
    batch_size = 8
    lambda_XtoC = 0.5     # parameter for the loss of the joint strategy
    is_aux_logits = False
    num_epochs = 10
    num_each_concept_classes = 1 # fait exprès pour garder la logique du code CBM_template_model


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    expand_dim = 0
    num_concept_labels = 4
    seed = 42
    n_concept_initial = 1  # nombre de concepts initial dans la pipeline v3

    DATASET = "medical" #"medical"/"dbpedia"/"agnews"/ "movies" 

    num_concept_labels = 4
    
    if DATASET == "dbpedia":
        num_labels = 6
    elif DATASET == "agnews":
        num_labels = 4
    elif DATASET == "movies":
        num_labels = 4
    elif DATASET == "medical":
        num_labels = 5    

    # storage params
    if infra == "DATABRICKS":
        DATASET_PATH = f"/dbfs/mnt/ekixai/main/data/Interpretability/concept_xai/experiment/dataset/{DATASET}/"
        SAVE_PATH  = f"/dbfs/mnt/ekixai/main/data/Interpretability/concept_xai/experiment/results_{DATASET}/"
        SAVE_PATH_CONCEPTS = f"{SAVE_PATH}concepts_discovery"
    else:
        DATASET_PATH = f"/home/bhan/Yann_CBM/Launch/dbfs/dataset/{DATASET}/"
        SAVE_PATH  = f"/home/bhan/Yann_CBM/Launch/dbfs/results_{DATASET}/"
        SAVE_PATH_CONCEPTS = f"{SAVE_PATH}concepts_discovery"        

    criterion = nn.CrossEntropyLoss()
    use_cls_token = True  

    lr = 0.001 # learning rate

    #--------------------------  SPARSITY PARAMETERS TO TUNE
    
    # ElasticNetLinearLayer net parameter 
    alpha = 0.01
    l1_ratio = 0.5
    # Ridge Parameters
    l2_lambda = 0.01

    #--------------------------- OTHER PARAMS BUT IMPORTANT -------
    use_sigmoid = False
    use_relu = False
    eval_concepts = True  # Par défaut, activer le calcul des métriques sur les concepts
    top_n = 1 # use to selected the most important word after LIG : for residual layer and concept too (we can separate if necessary) 
    #top n is linked to concepts_discovery_utils  import extract_target_words, create_context_window, load_model, run_concepts_discovery

    if use_sigmoid == True:
        sigmoid_or_relu_state = 'sigmoid' 
    elif use_relu == True:
        sigmoid_or_relu_state = 'relu'
    else :
        sigmoid_or_relu_state = 'linearity' 

    dropout = 0.1  # default value in cb llm too
    projection = 256 # default value in cb llm too

    cavs_type_arg = "mean" # "mean" , "svm"
    agg_mode ="abs"
    agg_scope ="all"
    #---------------------------  TCAV PARAMETERS
    # TCAV parameters à ajouter comme hyperparamètres d'experience
    cavs_type = 'mean'
    # layer_num = -1
    # use_cls_token = True
    # linear_classifier = True
    # balanced=True
    # closest=False
    

