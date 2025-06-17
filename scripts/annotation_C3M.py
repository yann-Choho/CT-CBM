import torch

preprompt_generic = "According to the following text: '"
preprompt_movie_synopsis = "According to the following text: '"
answer_start = "The feature called: '"
concept1 = "urban development"
concept2 = "cultural heritage"
concept3 = "human cognition"
concept4 = "neuroscience"

concepts_list_1 = "As cities expand and populations grow, there is a growing tension between development and the need to preserve historical landmarks. Citizens and authorities often clash over the balance between progress and cultural heritage."
micro_concept_1 = "[urban development, cultural heritage, conflict]"


concepts_list_2 = "Recent breakthroughs in neuroscience are shedding light on the complexities of human cognition. Researchers are particularly excited about the potential to better understand decision-making processes and emotional regulation in the brain."
micro_concept_2 = "[neuroscience, human cognition, decision-making, emotional regulation]"


def get_annotation(text, concept, model, tokenizer, preprompt = preprompt_generic, answer_start = answer_start, 
                   text_generic_1 = concepts_list_1, text_generic_2 = concepts_list_2,
                   concept_generic_1 = concept1, concept_generic_2 = concept2, concept_generic_3 = concept3, concept_generic_4 = concept4):


    messages = [{"role": "user", "content": preprompt  + text_generic_1 + "', is the feature '" + concept_generic_1 + "'" + " **detected** or **missing**?"}]
    messages.append({"role": "assistant", "content": "**detected**" +  "<eos>"})  
    
    messages.append({"role": "user", "content":preprompt  + text_generic_1 +  "', the feature '" + concept_generic_4 + "'" + " **detected** or **missing**?"})
    messages.append({"role": "assistant", "content": "**missing**" + "<eos>"})  
    
    messages.append({"role": "user", "content":preprompt  + text_generic_2 +  "', the feature '" + concept_generic_2 + "'" + " **detected** or **missing**?"})
    messages.append({"role": "assistant", "content": "**missing**" + "<eos>"})
    
    messages.append({"role": "user", "content":preprompt  + text_generic_2 +  "', the feature '" + concept_generic_3 + "'" + " **detected** or **missing**?"})
    messages.append({"role": "assistant", "content": "**detected**" + "<eos>"})  
        
    # Création du message avec la liste de concepts
    messages.append({"role": "user", "content":preprompt  + text + "', is the feature called: '" + concept + "'" + " **detected** or **missing**?"})
    #Guider la réponse
    messages.append({"role": "assistant", "content": "**"})

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    input_ids = torch.reshape(input_ids[0][:-2], (1, input_ids[0][:-2].shape[0]))

    # print(messages)
    outputs = model.generate(input_ids, max_new_tokens=2)

    return(tokenizer.decode(outputs[0][len(input_ids[0]):]))