

##### Compute of concept score ######

def get_annotation_CB_LLM(text_list, concept, model = "all-mpnet-base-v2"):
    """ Compute cosine similarity score for each concept and each text"""

    if model == "all-mpnet-base-v2":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-mpnet-base-v2")

    sentences1 = [concept]
    sentences2 = text_list

    # Compute embeddings for both lists
    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)

    # Compute cosine similarities
    similarities = model.similarity(embeddings1, embeddings2)
    return (similarities)


##### ACC Filtering ######
def mapping_concept_label(task, label, concept, concept_data):
    label_list = concept_data[0][task][label]
    if concept in label_list:
        return(1)
    else:
        return(0)
