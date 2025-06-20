from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import defaultdict

def get_example(dataloader, device):
    """ Get an example from the dataloader 
    
    Args:
    dataloader: torch.utils.data.DataLoader
    device: torch.device

    Returns:
    inputs_exemple: torch.Tensor
    """
    for batch in dataloader:
        for i in range(batch['input_ids'].shape[0]):
            train_inputs_exemple, train_masks_exemple, y_train_exemple = batch['input_ids'][i].to(device),batch['attention_mask'][i].to(device),batch['label'][i].to(device)
            break
    return train_inputs_exemple, train_masks_exemple, y_train_exemple
    
def eval_attributions(model, tokenizer, train_inputs, train_masks, y_train, layer, mode='normal', n_class = 5):
    # TODO : rendre ne n_class dependant du model (ou du y_train au pire)
    # n_class = y_train.max().item() + 1  #depend on y_train here
    lig = LayerIntegratedGradients(model, layer)

    outputs = []
    predictions = []
    attributions = []
    attributions_sum = []
    attributions_target = []
    deltas = []
    tokens = []
    texts = []

    for i in range(train_inputs.shape[0]):

        if(mode=='normal'):
            nloops = 1
        elif(mode=='all'):
            nloops = n_class
        else:
            ValueError('mode is not normal or all')
        
        output = model(train_inputs[i:i+1], train_masks[i:i+1])

        prediction = torch.argmax(output, dim=1)
        true_label = y_train[i]
        for j in range(nloops):

            if(mode=='normal'):
                target = true_label
            elif(mode=='all'):
                target = torch.tensor([j], device=model.device)
            else:
                ValueError('mode is not normal or all')

            attribution, delta = lig.attribute(inputs=(train_inputs[i:i+1],train_masks[i:i+1]), target=target, return_convergence_delta=True,
            n_steps=50)
            # normalize attributions: sum over last layer dim, to get attributions for each token
            # TODO: voir si mettre les ponctuation en 0 affecte negativement la pipeline
            attribution_sum = attribution.sum(dim=-1).squeeze(0)
            attribution_sum = attribution_sum / torch.norm(attribution_sum)

            token = [token for token in tokenizer.convert_ids_to_tokens(train_inputs[i]) if token!='[PAD]']
            text = ' '.join([token for token in tokenizer.convert_ids_to_tokens(train_inputs[i]) if token!='[PAD]']).replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')

            outputs.append(output)
            predictions.append(prediction)
            attributions.append(attribution)
            attributions_sum.append(attribution_sum)
            attributions_target.append(target)
            deltas.append(delta)
            tokens.append(token)
            texts.append(text)

    return torch.stack(outputs), torch.stack(predictions), torch.stack(attributions), torch.stack(attributions_sum), torch.stack(attributions_target), torch.stack(deltas),tokens,texts

def get_attributions(model, tokenizer, train_inputs, train_masks, y_train, layer, mode='normal'):

    outputs, predictions, attributions, attributions_sum, attributions_target, deltas, tokens, texts = eval_attributions(model, tokenizer, train_inputs, train_masks, y_train, layer=layer, mode=mode)

    return outputs, predictions, attributions, attributions_sum, attributions_target, deltas, tokens, texts

def display_attributions(model, tokenizer, train_inputs, train_masks, y_train, layer, mode='normal', per_token=True):

    outputs, predictions, attributions, attributions_sum, attributions_target, deltas, tokens, texts = eval_attributions(model,tokenizer, train_inputs, train_masks, y_train, layer=layer, mode=mode)

    vis = []


    for i in range(outputs.shape[0]):


        vis.append(viz.VisualizationDataRecord(
                            attributions_sum[i].flatten()[0:len(tokens[i]),:] if per_token else attributions_sum[i].flatten(), # word_attributions
                            torch.max(torch.softmax(outputs, dim=-1),dim=-1).values.flatten()[i].item(), # pred prob
                            torch.argmax(outputs, dim=-1).flatten()[i].item(), # pred label
                            y_train[i].item() if (mode=='normal') else y_train[int(i/n_class)].item(), # true_label
                            attributions_target[i].item(), # attr label
                            attributions_sum[i].sum(), # attr score
                            tokens[i], # raw_input_ids
                            deltas[i].item() # convergence_score
    ))

    _ = viz.visualize_text(vis)

def aggregate_attributions(tokens, attributions):
    """
    Agrège les attributions pour chaque mot complet à partir des sous-mots.
    
    Args:
    tokens (list of str): La liste des tokens (sous-mots) d'un texte.
    attributions (tensor): Le tenseur des attributions des tokens.

    Returns:
    list of tuple: Une liste de tuples où chaque tuple contient un mot complet et son attribution agrégée.
    """
    word_attributions = []
    current_word = ""
    current_attr = 0
    token_count = 0

    for token, attr in zip(tokens, attributions):
        if token.startswith("##"):
            # Si le token est un sous-mot, on l'ajoute au mot courant
            current_word += token[2:]
            current_attr += attr.item()
            token_count += 1
        else:
            # Si le token n'est pas un sous-mot, on ajoute le mot courant et son attribution agrégée à la liste
            if current_word:
                word_attributions.append((current_word, current_attr / token_count))
            # On initialise le nouveau mot courant
            current_word = token
            current_attr = attr.item()
            token_count = 1

    # Ajouter le dernier mot à la liste
    if current_word:
        word_attributions.append((current_word, current_attr / token_count))

    return word_attributions

def assign_scores_to_aggregated_tokens(aggregated_tokens):
    punctuations = {',', '.', '!', '?', ';', ':', '[SEP]', '[CLS]',
                    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'à', 'en', 'au', 'aux', 'avec', 'par', 'pour', 'dans', 
    'sur', 'sous', 'entre', 'après', 'avant', 'mais', 'ou', 'donc', 'car', 'ni', 'ne', 'que', 'qui', 'quoi', 'dont', 
    'où', 'ce', 'cet', 'cette', 'ces', 'ça', 'cela', 'ci', 'là', 'me', 'te', 'se', 'nous', 'vous', 'ils', 'elles', 
    'on', 'il', 'elle', 'est', 'sont', 'étaient', 'été', 'ai', 'a', 'as', 'avais', 'avez', 'avons', 'étions', 'sera', 
    'seront', 'serais', 'serait', 'soit', 'être', 'avoir', 'fais', 'fait', 'faisons', 'faire', 'es', 'était', 'avait', 
    'de', 'du', 'au', 'aux', 'aussi', 'bien', 'comme', 'plus', 'moins', 'trop', 'très', 'tout', 'tous', 'toutes', 'si',
    # English stopwords
    'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
    'up', 'down', 'in', 'out', 'on', 'off', 'again', 'further', 'then', 'once', 'here', 'there', 
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
    'should', 'now', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'he', 'him', 'his', 'her', 'she', 'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who', 
    'whom', 'this', 'that', 'these', 'those', 'am', 'isn', 'aren', 'wasn', 'weren', 'hasn', 'haven', 'hadn', 'doesn', 
    'didn', 'won', 'wouldn', 'don', 'doesn', 'shouldn', 'can'}
    new_aggregated_tokens = []

    for token, attribution in aggregated_tokens:
        # if token.startswith('##') or token in punctuations:
        if token in punctuations:
            new_aggregated_tokens.append((token, 0))
        else:
            new_aggregated_tokens.append((token, attribution))
    
    return new_aggregated_tokens

def batch_generator(inputs, masks, targets, batch_size):
    for i in range(0, len(inputs), batch_size):
        yield inputs[i:i + batch_size], masks[i:i + batch_size], targets[i:i + batch_size]

def process_batch(residual_model, tokenizer, batch_inputs, batch_masks, batch_targets):
    outputs_, predictions_, attributions_, attributions_sum_, attributions_target_, deltas_, tokens_, texts_ = get_attributions(
        residual_model.forward_residual_layer,
        tokenizer,
        batch_inputs, 
        batch_masks, 
        batch_targets, 
        layer=residual_model.embedder_model.embeddings.word_embeddings,
        #TODO: rendre flexible l'ajout du nom du model : 'embedder_model' pour rendre au format package 
        mode='normal'
    )
    return outputs_, predictions_, attributions_, attributions_sum_, attributions_target_, deltas_, tokens_, texts_

def process_data_in_batches(residual_model, tokenizer, inputs, masks, targets, batch_size = 1, example_index=None):
    """ Process the data in batches and return the texts and word attributions. 
    
    Args:
    residual_model: The residual model to use for attribution.
    tokenizer: The tokenizer to use for tokenization.
    inputs: The input tensors.
    masks: The mask tensors.
    targets: The target tensors.
    batch_size: The batch size to use for processing.
    
    Returns:
    list of str: The list of texts.
    list of list of tuple: The list of word attributions.
    """
    all_texts = []
    all_word_attributions = []

    # Utiliser tqdm pour la barre de progression
    for batch_inputs, batch_masks, batch_targets in tqdm(batch_generator(inputs, masks, targets, batch_size), 
                                                          total=len(inputs) // batch_size + 1, 
                                                          desc="Processing Batches"):

        outputs_, predictions_, attributions_, attributions_sum_,attributions_target_, deltas_, tokens_, texts_ = process_batch(residual_model, tokenizer, batch_inputs, batch_masks, batch_targets)
        
        for i in range(len(tokens_)):
            word_attributions = aggregate_attributions(tokens_[i], attributions_sum_[i])
            word_attributions = assign_scores_to_aggregated_tokens(word_attributions)
            all_texts.append(texts_[i])
            all_word_attributions.append(word_attributions)
            
        # Libération de la mémoire pour chaque batch
        del batch_inputs, batch_masks, batch_targets, attributions_, attributions_sum_, tokens_, texts_
        torch.cuda.empty_cache()
    return all_texts, all_word_attributions

def example_attribution(residual_model, tokenizer, input_tensor, mask_tensor, target_tensor):
    """Process a single example and return the text and word attributions.
    
    Args:
    residual_model: The residual model to use for attribution.
    tokenizer: The tokenizer to use for tokenization.
    input_tensor: The input tensor for a single example.
    mask_tensor: The mask tensor for a single example.
    target_tensor: The target tensor for a single example.
    
    Returns:
    str: The text of the example.
    list of tuple: The list of word attributions.
    """
    # Traiter l'exemple unique
    outputs_, predictions_, attributions_, attributions_sum_, attributions_target_, deltas_, tokens_, texts_ = process_batch(
        residual_model, tokenizer, input_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), target_tensor.unsqueeze(0)
    )
    
    # Traiter le premier élément (et unique élément) obtenu
    word_attributions = aggregate_attributions(tokens_[0], attributions_sum_[0])
    word_attributions = assign_scores_to_aggregated_tokens(word_attributions)

    # Visualisation
    vis = []
    per_token = False
    vis.append(viz.VisualizationDataRecord(
        attributions_sum_[0].flatten(),  # word_attributions
        torch.max(torch.softmax(outputs_, dim=-1), dim=-1).values.flatten()[0].item(),  # pred prob
        predictions_[0],  # pred label
        target_tensor.item(),  # true_label
        attributions_target_[0].item(),  # attr label
        attributions_sum_[0].sum().item(),  # attr score
        tokens_[0],  # raw_input_ids
        deltas_[0].item()  # convergence_score
    ))

    _ = viz.visualize_text(vis)

    return texts_[0], word_attributions


def split_dataloader(main_dataloader, n_splits=20, batch_size=8, verbose = False):
    """
    Divise un DataLoader principal en plusieurs DataLoaders (sans répétition), 
    tout en respectant la répartition des labels dans chaque sous-ensemble.
    
    Args:
    - main_dataloader (DataLoader): Le DataLoader principal contenant l'ensemble des données.
    - n_splits (int): Nombre de DataLoaders à générer. Par défaut, 20.
    - batch_size (int): Taille des batchs dans les DataLoaders générés. Par défaut, 8.

    Returns:
    - dataloaders (list): Liste des DataLoaders générés.
    """
    
    # Récupérer le dataset à partir du DataLoader principal
    dataset = main_dataloader.dataset
    
    # Extraire les labels du dataset
    labels = np.array([dataset[i]['label'].item() for i in range(len(dataset))])
    
    # Trouver les indices de chaque classe de labels
    label_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    
    # Diviser les indices par label en n_splits parts égales
    split_indices = [[] for _ in range(n_splits)]
    for label, indices in label_indices.items():
        np.random.shuffle(indices)  # Mélange les indices de chaque classe
        
        # Diviser les indices de cette classe en n_splits parts égales
        split_size = len(indices) // n_splits
        remainder = len(indices) % n_splits
        
        start = 0
        for i in range(n_splits):
            end = start + split_size + (1 if i < remainder else 0)  # Distribue également le reste
            split_indices[i].extend(indices[start:end])
            start = end
    
    # Créer les DataLoaders pour chaque split sans répétition
    dataloaders = []
    for indices in split_indices:
        subset = Subset(dataset, indices)  # Créer un subset basé sur les indices
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)  # Ajuste batch_size si nécessaire
        dataloaders.append(dataloader)
    
    # Optionnel : Vérifier que tous les indices sont utilisés et qu'il n'y a pas de répétition
    all_used_indices = [idx for split in split_indices for idx in split]
    assert len(all_used_indices) == len(dataset), "Il y a des exemples non utilisés ou des doublons !"
    assert len(set(all_used_indices)) == len(all_used_indices), "Il y a des doublons d'indices !"

    from collections import Counter

    # Fonction pour compter les labels dans un DataLoader
    def count_labels_in_dataloader(dataloader_):
        label_counts = Counter()
        
        for batch in dataloader_:
            labels = batch['label'].numpy()  # Extraire les labels du batch
            label_counts.update(labels)      # Mettre à jour le compteur de labels
        # print(label_counts)

    if verbose:
        for dataloader in dataloaders:
            count_labels_in_dataloader(dataloader)
    return dataloaders

