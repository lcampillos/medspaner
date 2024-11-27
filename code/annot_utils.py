#!/usr/bin/python3
# -*- coding: UTF-8 -*-
#
# annot_utils.py
#
# Note that Python 3 has processed better the UTF8 characters.
# 
# Leonardo Campillos-Llanos (UAM & CSIC)
# 2019-2024
#
#########################################################################

import re
import json
import lexicon_tools
from lexicon_tools import *
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # If CUDA is needed
device = torch.device("cpu")


def normalize(string):
    
    ''' Normalize characters that cause issues with BPE tokenized characters '''
    
    string = re.sub('ª', 'a', string)
    string = re.sub('º', 'o', string)
    string = re.sub('²', '2', string)
    string = re.sub('µL', 'm#L', string)
    string = re.sub('µg', 'm#g', string)
    string = re.sub('µm', 'm#m', string)
    # non-standard, hidden white space character
    string = re.sub(" ","_", string)
    # Superscript mark (∧)
    string = re.sub("∧","ᶺ", string)
    # Other
    string = re.sub("x´","x'", string)
    return string
    
                        
def normalize_back(Hash):
    
    ''' Normalize to the original string those characters that were normalized due to BPE tokenization problems '''
    
    word = Hash['word']
    
    # 1o -> 1º
    normalized_o = re.search(r"\do\b", word)
    if normalized_o:
        word = re.sub("(\d)o", r"\1º", word)
        Hash['word'] = word
    
    # 1a -> 1ª
    normalized_a = re.search(r"\b\da\b", word)
    if normalized_a:
        word = re.sub("(\d)a", r"\1ª", word)
        Hash['word'] = word
    
    # 1ah -> 1ªh
    normalized_ha = re.search(r"\b\dah\b", word)
    if normalized_ha:
        word = re.sub("(\d)ah", r"\1ªh", word)
        Hash['word'] = word
    
    # Tª (Temperatura)
    normalized_Ta = re.search(r"\bTa\b", word)
    if normalized_Ta:
        word = re.sub("a", "ª", word)
        Hash['word'] = word

    # ºC (uppercase)
    normalized_oC = re.search(r"\doC\b", word)
    if normalized_oC:
        word = re.sub("(\d)oC", r"\1ºC", word)
        Hash['word'] = word

    # ºc (lowercase)
    normalized_oC = re.search(r"\doc\b", word)
    if normalized_oC:
        word = re.sub("(\d)oc", r"\1ºc", word)
        Hash['word'] = word
        
    # m² 
    normalized_m2 = re.search(r"m2\b", word)
    if normalized_m2:
        word = re.sub("m2", "m²", word)
        Hash['word'] = word

    # µL 
    normalized_ml = re.search(r"m#L", word)
    if normalized_ml:
        word = re.sub("m#L", "µL", word)
        Hash['word'] = word
    
    # µg 
    normalized_mg = re.search(r"m#g", word)
    if normalized_mg:
        word = re.sub("m#g", "µg", word)
        Hash['word'] = word

    # µm 
    normalized_mm = re.search(r"m#m", word)
    if normalized_mm:
        word = re.sub("m#m", "µm", word)
        Hash['word'] = word
    
    # non-standard, hidden white space character
    normalized_hws = re.search("_", word)
    if normalized_hws:
        word = re.sub("_", " ", word)
        Hash['word'] = word
        
    # Superindex mark (∧)
    normalized_sp = re.search("ᶺ", word)
    if normalized_sp:
        word = re.sub("ᶺ", "∧", word)
        Hash['word'] = word
        
    # Other
    normalized_apost = re.search("x'", word)
    if normalized_apost:
        word = re.sub("x'", "x´", word)
        Hash['word'] = word
    
    return Hash
                        

def remove_nested_entity(EntsOut,EntsIn):

    ''' 
        Remove nested entity if:
        a) same label of outer entity: e.g. "dolor" in "dolor de cabeza"
        b) same span as another nested entity, except Negated/Speculated/Contraindicated: 
        e.g. "alteraciones" DISO and CONC 
    '''
        
    # First, remove nested entity if same label of outer entity
    for i in EntsOut:
        outLabel = EntsOut[i]['label']
        outStart = EntsOut[i]['start']
        outEnd = EntsOut[i]['end']
        outEnt = EntsOut[i]['ent']
        for j in EntsIn.copy(): 
            inLabel = EntsIn[j]['label']
            if outLabel == inLabel:
                inStart = EntsIn[j]['start']
                inEnd = EntsIn[j]['end']
                inEnt = EntsIn[j]['ent']
                if (int(inStart) <= int(outStart)) and (int(inEnd) <= int(outEnd)) and (int(outEnd) > int(inStart))  and (int(inEnd) > int(outStart)) and (inEnt in outEnt):
                    EntsIn.pop(j, None)
    
    LabelsToKeep = ["Negated", "Speculated", "Contraindicated"]
    
    # Second, remove nested entities with same span, except assertion labels 
    ToDelete = []
    # Auxiliar hash
    AuxInEnts = EntsIn
    for i in EntsIn:
        inEnt = EntsIn[i]['ent']
        inStart = EntsIn[i]['start']
        inEnd = EntsIn[i]['end']
        inLabel = EntsIn[i]['label']
        for j in AuxInEnts.copy():
            AuxInEnt = AuxInEnts[j]['ent']
            AuxInStart = AuxInEnts[j]['start']
            AuxInEnd = AuxInEnts[j]['end']
            AuxInLabel = AuxInEnts[j]['label']
            if EntsIn[i] != AuxInEnts[j]:
                if (inEnt in AuxInEnt):
                    if (inLabel not in LabelsToKeep):
                        if (int(inStart) <= int(AuxInStart)) and (int(inEnd) >= int(AuxInEnd)):
                            ToDelete.append(EntsIn[i])
                        elif (int(inStart) >= int(AuxInStart)) and (int(inEnd) <= int(AuxInEnd)):
                            ToDelete.append(EntsIn[i])
                    # if both nested entities have assertion labels, keep only one
                    elif ((inLabel and AuxInLabel) in LabelsToKeep):
                        if (int(inStart) <= int(AuxInStart)) and (int(inEnd) >= int(AuxInEnd)):
                            ToDelete.append(EntsIn[i])
                        elif (int(inStart) >= int(AuxInStart)) and (int(inEnd) <= int(AuxInEnd)):
                            ToDelete.append(EntsIn[i])

    FinalEntsIn = {}
    
    for i in EntsIn:
        if EntsIn[i] not in ToDelete:
            FinalEntsIn[len(FinalEntsIn)+1] = EntsIn[i]
        
    return FinalEntsIn

                    
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
    
    string = normalize(string)
    
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
                    if not (Tokens[i].startswith(" ")) and not (Tokens[i].startswith("\n")) and (len(Entities) > 0) and ((Entities[len(Entities) - 1]['end']) == (Offsets[i][0])):
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
        # Normalize back if needed
        dictionary = normalize_back(dictionary)
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
                        corrected = True
                    except:
                        pass
                    if corrected == False:
                        try:
                            # Try normalization if not found previously
                            entity = normalize(entity)
                            new_start, new_end = re.search(re.escape(entity),text).span()
                            dictionary['start'] = new_start
                            dictionary['end'] = new_end
                            dictionary['word'] = entity
                            NewList.append(dictionary)
                            print("Check offsets of entity: %s" % (entity))
                            corrected = True
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
            EntsList = [ normalize_back(EntHash) for EntHash in EntsList ]
            
            # Change offsets
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


def codeAttribute(Hash):
    
    '''
    Code attribute and value of annotated entity.
    '''
    
    Assertion = ['Negated', 'Speculated']
    Experiencer = ['Family_member', 'Patient', 'Other']
    Event_temp = ['Future', 'History_of', 'Hypothetical']
    Attribute = ['Age', 'Contraindicated']
    
    FinalHash = {}
    
    Saved = []
    
    for i in Hash:
        s = Hash[i]['start']
        e = Hash[i]['end']
        label = Hash[i]['label']
        if label in Assertion:
            # Check offsets
            for k in FinalHash.copy():
                start = FinalHash[k]['start']
                end = FinalHash[k]['end']
                tag = FinalHash[k]['label'] 
                if (start == s and end == e):
                    FinalHash[k]['assertion'] = label
                    Saved.append(Hash[i])
        elif label in Experiencer:
            # Check offsets
            for k in FinalHash.copy():
                start = FinalHash[k]['start']
                end = FinalHash[k]['end']
                tag = FinalHash[k]['label']
                if (start == s and end == e):
                    FinalHash[k]['experiencer'] = label
                    Saved.append(Hash[i])
        elif label in Event_temp:
            # Check offsets
            for k in FinalHash.copy():
                start = FinalHash[k]['start']
                end = FinalHash[k]['end']
                tag = FinalHash[k]['label']
                if (start == s and end == e):
                    FinalHash[k]['event_temp'] = label
                    Saved.append(Hash[i])
        elif label in Attribute:
            # Check offsets
            for k in FinalHash.copy():
                start = FinalHash[k]['start']
                end = FinalHash[k]['end']
                tag = FinalHash[k]['label']
                if (start == s and end == e):
                    FinalHash[k]['attribute'] = label
                    Saved.append(Hash[i])
        elif Hash[i] not in Saved:
            Saved.append(Hash[i])
            FinalHash[len(FinalHash)+1] = Hash[i]

    # Rest of unprocessed entities
    for i in Hash:
        if Hash[i] not in Saved:
            label = Hash[i]['label']
            ent = Hash[i]['ent']
            start = Hash[i]['start']
            end = Hash[i]['end']
            found = False
            if label in Assertion or label in Experiencer or label in Event_temp or label in Attribute:
                # Check if missing attribute
                for j in FinalHash.copy():
                    ent2 = FinalHash[j]['ent']
                    if (ent == ent2) and (start == FinalHash[j]['start']) and (end == FinalHash[j]['end']):
                        found = True
                        if (label in Assertion) and (label != FinalHash[j]['label']):
                            FinalHash[j]['assertion'] = label
                        elif (label in Experiencer) and (label != FinalHash[j]['label']):
                            FinalHash[j]['experiencer'] = label
                        elif (label in Event_temp) and (label != FinalHash[j]['label']):
                            FinalHash[j]['event_temp'] = label
                        elif (label in Attribute) and (label != FinalHash[j]['label']):
                            FinalHash[j]['attribute'] = label
                if found == False:
                    FinalHash[len(FinalHash)+1] = Hash[i]
    
    return FinalHash


def convert2brat(Hash,FileName,LexiconData,SourceData,Source):

    ''' Convert a hash of entities to BRAT format 
        LexiconData, SourceData and Source are optional parameters (if normalization is selected).
        Source must be either "umls" or "snomed".
    '''
    
    n_comm = 0
    n_att = 0

    for i in Hash:
        # T#  Annotation  Start   End String
        print("T{}\t{} {} {}\t{}".format(i, Hash[i]['label'], Hash[i]['start'], Hash[i]['end'], Hash[i]['ent']),file=FileName)
        # Attributes with format: A#	AttributeType T# Value
        if 'assertion' in Hash[i].keys():
            n_att += 1
            print("A{}\tAssertion T{} {}".format(n_att, i, Hash[i]['assertion']),file=FileName)
        if 'event_temp' in Hash[i].keys():
            n_att += 1
            print("A{}\tStatus T{} {}".format(n_att, i, Hash[i]['event_temp']),file=FileName)
        if 'experiencer' in Hash[i].keys():
            n_att += 1
            print("A{}\tExperiencer T{} {}".format(n_att, i, Hash[i]['experiencer']),file=FileName)
        if 'attribute' in Hash[i].keys():
            n_att += 1
            if Hash[i]['attribute'] == 'Age':
                print("A{}\tPopulation_data T{} {}".format(n_att, i, Hash[i]['attribute']),file=FileName)    
            if Hash[i]['attribute'] == 'Contraindicated':
                print("A{}\tAssertion T{} {}".format(n_att, i, Hash[i]['attribute']),file=FileName)    
        # Print UMLS codes in additional comment
        if LexiconData and SourceData:
            CUIsList = get_codes_from_lexicon(Hash[i]['ent'], Hash[i]['label'], LexiconData)
            if (CUIsList):
                if (Source == "umls"):
                    # Complete normalization data of UMLS CUIs
                    CUIsList = complete_norm_data(CUIsList,SourceData)
                    n_comm += 1
                    codes_string = " | ".join(CUIsList)
                    print("#{}	AnnotatorNotes T{}	{}".format(n_comm,i,codes_string),file=FileName)
                elif (Source == "snomed"):
                    # Complete normalization data of SNOMED
                    CUIsList = complete_snomed_code(CUIsList,SourceData)
                    if len(CUIsList)>0:
                        n_comm += 1
                        codes_string = " | ".join(CUIsList)
                        print("#{}	AnnotatorNotes T{}	{}".format(n_comm,i,codes_string),file=FileName)
                elif (Source == "omop"):
                    # Complete normalization data of OMOP
                    CUIsList = complete_omop_code(CUIsList,SourceData)
                    if len(CUIsList)>0:
                        n_comm += 1
                        codes_string = " | ".join(CUIsList)
                        print("#{}	AnnotatorNotes T{}	{}".format(n_comm,i,codes_string),file=FileName)                


def convert2json(EntityHash, LexiconData, SourceData, Source):

    ''' Convert a hash of entities to json format 
        LexiconData, SourceData and Source are optional parameters (if normalization is selected).
        Source must be either "umls" or "snomed"
     '''

    jsonEntities = []

    for i in EntityHash:
    
        EntDict = {"entity_group": EntityHash[i]['label'], 
                    "word": EntityHash[i]['ent'],
                    "start": EntityHash[i]['start'], 
                    "end": EntityHash[i]['end']}
        
        if 'assertion' in EntityHash[i].keys():
            EntDict["assertion"] = EntityHash[i]["assertion"]
        elif 'experiencer' in EntityHash[i].keys():
            EntDict["experiencer"] = EntityHash[i]["experiencer"]
        elif 'event_temp' in EntityHash[i].keys():
            EntDict["status"] = EntityHash[i]["event_temp"]
        elif 'attribute' in EntityHash[i].keys():
            EntDict["attribute"] = EntityHash[i]["attribute"]            
        jsonEntities.append(EntDict)
    
    if (LexiconData and SourceData):
        for entityData in jsonEntities:
            CUIsList = get_codes_from_lexicon(entityData['word'],entityData['entity_group'], LexiconData)
            if (CUIsList): 
                if (Source == "umls"):
                    # Complete normalization data of UMLS CUIs
                    CUIsList = complete_norm_data(CUIsList,SourceData)
                    codes_string = " | ".join(CUIsList)                    
                    entityData['umls'] = codes_string
                elif (Source == "snomed"):
                    # Complete normalization data of SNOMED
                    CUIsList = complete_snomed_code(CUIsList,SourceData)
                    if len(CUIsList)>0:
                        codes_string = " | ".join(CUIsList)                    
                        entityData['snomed'] = codes_string
                elif (Source == "omop"):
                    # Complete normalization data of OMOP
                    CUIsList = complete_omop_code(CUIsList,SourceData)
                    if len(CUIsList)>0:
                        codes_string = " | ".join(CUIsList)
                        entityData['omop'] = codes_string
    
    return jsonEntities
   
    