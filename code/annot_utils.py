#!/usr/bin/python3
# -*- coding: UTF-8 -*-
#
# annot_utils.py
#
# Note that Python 3 has processed better the UTF8 characters.
# 
# Leonardo Campillos-Llanos (UAM & CSIC)
# 2019-2022
#
#########################################################################

import re
import json
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_space(EntsList):
    
    ''' Remove white spaces or new lines in predicted entities '''
    
    FinalList = []
    
    for item in EntsList:
        ent = item['word']
        # default value
        finalItem = item
        # Remove space at the beginning of the string
        if ent.startswith(" "):
            finalItem = {'entity_group': item['entity_group'], 'word': item['word'][1:], 'start': item['start'], 'end': item['end']}
        if ent.startswith("\n"):
            finalItem = {'entity_group': item['entity_group'], 'word': item['word'][1:], 'start': item['start']+1, 'end': item['end']} #'score': item['score'],
        # Remove spaces at the end of the string
        if ent.endswith("\s") or ent.endswith("\t") or ent.endswith("\n"):
            finalWord = re.sub("(\s+|\t+|\n+)$", "", finalItem['word'])
            # Update offsets
            new_end = int(finalItem['start']) + len(finalWord)
            finalItem = {'entity_group': finalItem['entity_group'], 'word': finalWord, 'start': finalItem['start'], 'end': new_end}
        # Remove "\n" in the middle of the string
        if "\n" in finalItem['word']:
            index = finalItem['word'].index("\n")
            finalWord = finalItem['word'][:index]
            new_end = int(finalItem['start']) + len(finalWord)
            finalItem = {'entity_group': finalItem['entity_group'], 'word': finalWord, 'start': finalItem['start'], 'end': new_end}
        # Update list of dictionaries
        if finalItem['word']!='':
            FinalList.append(finalItem)
            
    return FinalList


def annotate_sentence(string, annotation_model, tokenizer_model, device):

    ''' Predict entities in sentence with ROBERTa neural classifier. '''

    tokenized = tokenizer_model(string, return_offsets_mapping=True)

    input_ids = tokenizer_model.encode(string, return_tensors="pt")

    tokens = tokenizer_model.convert_ids_to_tokens(tokenized["input_ids"])

    tokens = [tokenizer_model.decode(tokenized['input_ids'][i]) for i, token in enumerate(tokens)]

    offsets = tokenized["offset_mapping"]

    word_ids = tokenized.word_ids()

    outputs = annotation_model(input_ids.to(device)).logits

    predictions = torch.argmax(outputs, dim=-1)

    TagNames = [annotation_model.config.id2label[i] for i in annotation_model.config.id2label]

    preds = [TagNames[p] for p in predictions[0].cpu().numpy()]

    index2tag = {idx: tag for idx, tag in enumerate(TagNames)}

    tag2index = {tag: idx for idx, tag in enumerate(TagNames)}

    labels = []
    for pred in preds:
        labels.append(tag2index[pred])

    previous_word_idx = None
    label_ids = []

    for i, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(labels[i])
        previous_word_idx = word_idx

    labels_final = [index2tag[l] if l != -100 else "IGN" for l in label_ids]

    return tokens, offsets, word_ids, label_ids, labels_final


def postprocess_entities(DataList):

    ''' 
        Postprocess and aggregate annotated entities that are subwords from BERT / RoBERTa model.
        E.g. "auto", "medic", "arse" -> "automedicarse"
    '''

    Tokens = DataList[0]
    Offsets = DataList[1]
    Word_ids = DataList[2]
    Label_ids = DataList[3]
    Labels = DataList[4]

    Entities = []

    prev_label = ""

    for i, k in enumerate(Word_ids):
        if Word_ids != None:
            label = Labels[i]
            if label == 'O':
                prev_label = label
                continue
            elif label == 'IGN' and Tokens[i] != "</s>":  # use the previous label
                label = prev_label
                # if previous label is not 'O', update tokens and offsets
                if prev_label != 'O' and prev_label != "" and len(Entities) > 0:
                    LastEntity = Entities[len(Entities) - 1]
                    new_word = LastEntity['word'] + Tokens[i]
                    new_end = Offsets[i][1]
                    Entities[len(Entities) - 1]['word'] = new_word
                    Entities[len(Entities) - 1]['end'] = new_end
                prev_label = label
            else:
                # start of entity
                bio = label[:2]
                tag = label[2:]
                if bio == "B-":
                    # If entity is a contiguous subword, merge it with previous entity
                    if not (Tokens[i].startswith(" ")) and not (Tokens[i].startswith("\n")) and (
                        len(Entities) > 0) and ((Entities[len(Entities) - 1]['end']) == (Offsets[i][0])):
                        LastEntity = Entities[len(Entities) - 1]
                        new_word = LastEntity['word'] + Tokens[i]
                        new_end = Offsets[i][1]
                        Entities[len(Entities) - 1]['word'] = new_word
                        Entities[len(Entities) - 1]['end'] = new_end
                    # If entity is not a subword
                    else:
                        Entities.append(
                            {
                                "entity_group": tag,
                                "word": Tokens[i],
                                "start": Offsets[i][0],
                                "end": Offsets[i][1],
                            }
                        )
                elif bio == "I-" and len(Entities) > 0:
                    if prev_label != 'O':  # update tokens and offsets
                        # if previous token is space or hyphen
                        LastEntity = Entities[len(Entities) - 1]
                        new_word = LastEntity['word'] + Tokens[i]
                        new_end = Offsets[i][1]
                        Entities[len(Entities) - 1]['word'] = new_word
                        Entities[len(Entities) - 1]['end'] = new_end
                    else:
                        Entities.append(
                            {
                                "entity_group": tag,
                                "word": Tokens[i],
                                "start": Offsets[i][0],
                                "end": Offsets[i][1],
                            }
                        )

                prev_label = label

    return Entities


def update_offsets(List, offset, text):
    
    ''' Updates offsets of annotated entities according to given position in paragraph '''
    
    NewList = []
    
    for dictionary in List:
        
        start_old = dictionary['start']
        end_old = dictionary['end']
        entity = dictionary['word']
        new_start = int(start_old) + int(offset)
        new_end = int(end_old) + int(offset)
        dictionary['start'] = new_start
        dictionary['end'] = new_end
        
        # Validate offsets with original text
        candidate = text[new_start:new_end]
        if (entity == candidate):
            NewList.append(dictionary)
        else:
            corrected = False
            # Correct offsets
            # 1 different offset
            new_start = new_start + 1
            new_end = new_end + 1
            dictionary['start'] = new_start
            dictionary['end'] = new_end
            candidate = text[new_start:new_end]

            if (entity == candidate):
                NewList.append(dictionary)
                corrected = True

            if (corrected == False):
                # 2 different offsets
                new_start = new_start + 1
                new_end = new_start + len(entity)
                dictionary['start'] = new_start
                dictionary['end'] = new_end
                candidate = text[new_start:new_end]
                if (entity == candidate):
                    corrected = True
                    NewList.append(dictionary)
                else:
                    # Try to get offsets from original text
                    try:
                        new_start, new_end = re.search(re.escape(entity),text).span()
                        dictionary['start'] = new_start
                        dictionary['end'] = new_end
                        NewList.append(dictionary)
                        print("Check offsets of entity: %s" % (entity))
                    except:
                        print("Error in offsets of entity: %s" % (entity))

    return NewList


def annotate_sentences_with_model(SentencesList,text_string,model,tokenizer,device):

    ''' Given a list of sentences, and given a transformer model, 
    annotate sentences and yield a list of hashes with data of annotated entities '''

    offset = 0

    HashList = []

    for sentence in SentencesList:

        if not (sentence.text.isspace()):

            EntsList = annotate_sentence(sentence.text, model, tokenizer, device)

            EntsList = remove_space(postprocess_entities(EntsList))

            # Change offsets
            if offset != 0:
                EntsList = update_offsets(EntsList, offset, text_string)

            last_token = sentence[-1]
            last_token_offset = int(last_token.idx) + int(len(last_token))
            offset = last_token_offset

            HashList.append(EntsList)
        else:
            offset = offset + 1

    # Merge list of lists
    HashList = [item for sublist in HashList for item in sublist]

    return HashList




def convert2json(EntityHash):

    ''' Convert a hash of entities to json format '''

    jsonEntities = []

    for i in EntityHash:
        EntDict = {"entity_group": EntityHash[i]['label'], "word": EntityHash[i]['ent'],
                   "start": EntityHash[i]['start'], "end": EntityHash[i]['end']}
        jsonEntities.append(EntDict)
    
    return json.dumps(jsonEntities)
   
    