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

def remove_space(EntsList):
    
    ''' Function to remove space before predicted entities '''
    
    FinalList = []
    
    for item in EntsList:
        ent = item['word']
        # default value
        finalItem = item
        # Remove space at the beginning of the string
        if ent.startswith(" "):
            finalItem = {'entity_group': item['entity_group'], 'score': item['score'], 'word': item['word'][1:], 'start': item['start'], 'end': item['end']}

        # Remove spaces at the end of the string
        if ent.endswith("\s") or ent.endswith("\t") or ent.endswith("\n"):
            finalWord = re.sub("(\s+|\t+|\n+)$", "", finalItem['word'])
            # Update offsets
            new_end = int(finalItem['start']) + len(finalWord)
            finalItem = {'entity_group': finalItem['entity_group'], 'score': finalItem['score'], 'word': finalWord, 'start': finalItem['start'], 'end': new_end}
        # Remove "\n" in the middle of the string
        if "\n" in finalItem['word']:
            index = finalItem['word'].index("\n")
            finalWord = finalItem['word'][:index]
            new_end = int(finalItem['start']) + len(finalWord)
            finalItem = {'entity_group': finalItem['entity_group'], 'score': finalItem['score'], 'word': finalWord, 'start': finalItem['start'], 'end': new_end}
        # Update list of dictionaries
        if finalItem['word']!='':
            FinalList.append(finalItem)
            
    return FinalList


def aggregate_subword_entities(DictList, string):

    ''' Postprocess and aggregate annotated entities that are subwords from BERT model
        E.g. "auto", "medic", "arse" -> "automedicarse"
    '''
    
    Aggregated = []

    AuxDict = {}

    # Sort list of dictionaries in reverse order according to offsets
    ReverseDictList = [item for item in reversed(DictList)]

    for i, Dict in enumerate(ReverseDictList):

        word = Dict['word']
        start = Dict['start']

        prev_char = start - 1

        end = Dict['end']

        # check if the character previous to the starting offset is a white space or a punctuation sign
        # if it is not a white space or a punctuation sign
        if (string[prev_char]) not in [" ", "/", "\n", "\r", "?", "!"]:  # TODO: check if add more punctuation characters

            # Get data from previous entity saved in AuxDict
            if len(AuxDict) > 0:
                # Check if entity type matches
                if (Dict['entity_group'] == AuxDict['entity_group']):
                    # if continuous entity
                    if (AuxDict['start']==Dict['end']):
                        new_word = word + AuxDict['word']
                        new_start = start
                        new_end = AuxDict['end']
                        AuxDict = {'entity_group': Dict['entity_group'], 'word': new_word, 'start': new_start, 'end': new_end}
                        # if hyphen as first character of sentence (not to be merged with previous entity)
                        if ((string[prev_char]) == "-") and (prev_char != 0) and (string[prev_char - 1] == "\n"):
                            Aggregated.append(AuxDict)
                            AuxDict = {}
                    else:
                        Aggregated.append(AuxDict)
                        AuxDict = Dict
                        # If it is the last entity (not to be reconstructed)
                        if ((i + 1) == len(ReverseDictList)):
                            Aggregated.append(AuxDict)
                else:
                    # if continuous entities, they are merged using the label of the starting entity
                    if (AuxDict['start'] == Dict['end']):
                        new_word = word + AuxDict['word']
                        new_start = start
                        new_end = AuxDict['end']
                        AuxDict = {'entity_group': Dict['entity_group'], 'word': new_word, 'start': new_start, 'end': new_end}
                        Aggregated.append(AuxDict)
                    else:
                        Aggregated.append(AuxDict)
                    AuxDict = Dict
                    # If it is the last entity (not to be reconstructed)
                    if ((i + 1) == len(ReverseDictList)):
                        Aggregated.append(AuxDict)
            else:
                AuxDict = {'entity_group': Dict['entity_group'], 'word': word, 'start': start, 'end': end}
                # If it is the first entity (not to be reconstructed)
                if ((i + 1) == len(ReverseDictList)):
                    Aggregated.append(AuxDict)
                    AuxDict = {}

        else:

            if len(AuxDict) > 0:
                # Check if end offset of previous word is next to start offset of present word
                prev_start = AuxDict['start']
                if (prev_start != end):
                    # If parenthesis character as part of an entity (check if same label)
                    char = AuxDict['word'][0]
                    if (char == "(") and (Dict['entity_group'] == AuxDict['entity_group']):
                        new_word = Dict['word'] + " " + AuxDict['word']
                        FinalDict = {'entity_group': Dict['entity_group'], 'word': new_word, 'start': Dict['start'], 'end': AuxDict['end']}
                    else:
                        # If not the same label, do not aggregate
                        Aggregated.append(AuxDict)
                        FinalDict = {'entity_group': Dict['entity_group'], 'word': Dict['word'], 'start': Dict['start'], 'end': Dict['end']}
                    Aggregated.append(FinalDict)
                    AuxDict = {}
                else:
                    # if character is an opening parenthesis, check to merge it with previous item
                    if word == ("(") and (i < len(ReverseDictList)) and (Dict['entity_group'] == AuxDict['entity_group']):
                        AuxDict = {'entity_group': Dict['entity_group'], 'word': Dict['word'] + AuxDict['word'], 'start': Dict['start'], 'end': AuxDict['end']}
                    #elif (Dict['entity_group'] != AuxDict['entity_group']): # Esto causa "ruido": ej. "S + tent", "interferir + ía"
                        Aggregated.append(AuxDict)
                        AuxDict = Dict
                        # If it is the last entity (not to be reconstructed)
                        if ((i + 1) == len(ReverseDictList)):
                            Aggregated.append(AuxDict)
                    else:
                        FinalDict = {'entity_group': Dict['entity_group'], 'word': Dict['word'] + AuxDict['word'], 'start': Dict['start'], 'end': AuxDict['end']}
                        Aggregated.append(FinalDict)
                        AuxDict = {}
            else:
                Aggregated.append(Dict)

    # Reverse again before returning results
    FinalAggregated = [item for item in reversed(Aggregated)]

    return FinalAggregated


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


def annotate_sentences_with_model(SentencesList,text_string,model):

    ''' Given a list of sentences, and given a transformer model, 
    annotate sentences and yield a list of hashes with data of annotated entities '''

    offset = 0

    HashList = []

    for sentence in SentencesList:

        if not (sentence.text.isspace()):
            # Predict entities with ROBERTa neural classifier
            EntsList = remove_space(model(sentence.text))

            EntsList = aggregate_subword_entities(EntsList, sentence.text)

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
        EntDict = {'entity_group': EntityHash[i]['label'], 'word': EntityHash[i]['ent'],
                   'start': EntityHash[i]['start'], 'end': EntityHash[i]['end']}
        jsonEntities.append(EntDict)

    return jsonEntities
   
    