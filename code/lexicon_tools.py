#!/usr/bin/python3
# -*- coding: UTF-8 -*-
#
# lexicon_tools.py
#
#
#########################################################################

import re
import os
import sys
import pickle
from spacy_tokenizer import *
import textsearch
from textsearch import TextSearch


POSData = {}


def merge_dicts(D1, D2):
    
    '''
    Update the contents of Dictionary 1 with those of Dictionary 2
    Repeated values are removed, and dictionary keys are rearranged.
    Items are sorted according to starting offset.
    '''
    
    AuxDict = {}
    
    if len(D1)>0:
        for i in D1:
            if D1[i] not in AuxDict.values():
                AuxDict[len(AuxDict)+1] = D1[i]
    
    if len(D2)>0:
        for j in D2:
            if D2[j] not in AuxDict.values():
                AuxDict[len(AuxDict)+1] = D2[j]
    
    # Sort according to starting offset
    FinalDict = {}
    
    k=1
    for n in sorted(AuxDict.items(), key=lambda n_v: n_v[1]['start']):
        FinalDict[k]=n[1]
        k+=1
    
    return FinalDict


def read_lexicon(dictFilename):
    
    '''
    Reads lexicon file into dictionary
    Arguments:
    - Path to lexicon data in MedLexSp format
    '''

    LexiconData = {}
    POSData = {}   
    
    # Entity data and labels
    if ".pickle" in dictFilename:
        lexicon_pickle = dictFilename
    else:
        lexicon_pickle = re.sub(".[^.]+?$",".pickle",str(dictFilename))
    LexiconDataFile = open(lexicon_pickle, 'rb')
    LexiconData = pickle.load(LexiconDataFile)
    
    # POS data
    POSData_pickle = re.sub("MedLexSp","MedLexSpPOS",dictFilename)
    POSDataFile = open(POSData_pickle, 'rb')
    POSData = pickle.load(POSDataFile)
    
    return LexiconData, POSData


def remove_substring_overlap(text, Hash):
    
    '''
    Removes overlapping substring inside a matched entity using TextSearch. 
    This problem only happens if the substring is a prefix or a suffix,
    and preceeded or followed by a Latin 1 character:
        e.g. "ma" (ANAT) and "ana" (CHEM, PROC) in "mañana" or "maíz"
        (No problem in "llama" or "mano")
    '''
    
    # List of Latin-1 characters (lower and uppercase)
    L1_chars = ['ñ', 'Ñ', 'á', 'Á', 'é', 'É', 'í', 'Í', 'ó', 'Ó', 'ú', 'Ú', 'à', 'À', 'è', 'È', 'ì', 'Ì', 'ò', 'Ò', 'ù', 'Ù', 'ä', 'Ä', 'ë', 'Ë', 'ï', 'Ï', 'ö', 'Ö', 'ü', 'Ü', 'â', 'Â', 'ê', 'Ê', 'î', 'Î', 'ô', 'Ô', 'û', 'Û', "ç", 'Ç']
    
    Keys_to_delete = []
    
    for Entity in Hash:
        s = Hash[Entity]['start']
        e = Hash[Entity]['end']
        skip = False
        # Check entity offsets in original text, and boundary characters
        string_length = len(text)
        # The substring is a prefix followed by Latin-1 character ("ma" in "mañana")
        # String index of previous and following characters
        i_prev = s-1
        if (i_prev >= 0):
            # Get previous character
            prev_char = text[i_prev]
            # Check if Latin-1 character
            if prev_char in L1_chars:
                # Do not use entity
                Keys_to_delete.append(Entity)
                continue
        if (e < string_length):
            # Get next character 
            next_char = text[e]
            # Check if Latin-1 character
            if next_char in L1_chars:
                Keys_to_delete.append(Entity)
                # Do not use entity
                continue        
    FinalHash = {}
    
    for Key in Hash:
        if Key not in Keys_to_delete:
            FinalHash[len(FinalHash)] = Hash[Key]
    
    return FinalHash
    

def remove_overlap(Hash):
    
    '''
    Removes overlapping entities
       e.g. cardiaco, infarto cardiaco => infarto cardiaco
    '''
    
    InnerEnts = []
    ToDelete = []

    Dict={}

    # Sort dictionary of dictionaries according to starting offsets; save in another sorted dictionary:
    i = 1
    for k in sorted(Hash.items(), key=lambda k_v: k_v[1]['start']):
        Dict[i] = k[1]
        i += 1
        
    for k in Dict:
        n = len(Dict)
        # Compare from the last entity back to the first
        while n > k:
            ToCompare1 = Dict[k]['ent']
            ToCompare2 = Dict[n]['ent']
            Start_ToCompare1 = Dict[k]['start']
            Start_ToCompare2 = Dict[n]['start']
            End_ToCompare1 = Dict[k]['end']
            End_ToCompare2 = Dict[n]['end']
            Tag_ToCompare1 = Dict[k]['label']
            Tag_ToCompare2 = Dict[n]['label']
            # To avoid comparing an entity with itself
            if (Dict[k] != Dict[n]):
                if ((Start_ToCompare1 >= Start_ToCompare2) and (End_ToCompare1 <= End_ToCompare2)) or ((Start_ToCompare2 >= Start_ToCompare1) and (End_ToCompare2 <= End_ToCompare1)):
                    # Añadido lo siguiente para evitar error cuando la entidad tiene caracteres como "+" (ej. "Mg++" o "Mg++ serico")
                    ToCompare1 = re.escape(ToCompare1)
                    ToCompare2 = re.escape(ToCompare2)
                    m1 = re.search(ToCompare1, ToCompare2)
                    m2 = re.search(ToCompare2, ToCompare1)
                    if (m1) or (m2) or (ToCompare1 in ToCompare2) or (ToCompare2 in ToCompare1):
                        span1 = End_ToCompare1 - Start_ToCompare1
                        span2 = End_ToCompare2 - Start_ToCompare2
                        if (span2 > span1):
                            # If the label is the same as in the outer entity, the entity is removed
                            if (Tag_ToCompare1==Tag_ToCompare2):
                                ToDelete.append(Dict[k])
                            # If the label is different, the nested entity is kept
                            else:
                                InnerEnts.append(Dict[k])
                        elif (span1 > span2):
                            # If the label is the same as in the outer entity, the entity is removed
                            if (Tag_ToCompare1 == Tag_ToCompare2):
                                ToDelete.append(Dict[n])
                            # If the label is different, the nested entity is kept
                            else:
                                InnerEnts.append(Dict[n])
                        # In case that an entity can have 2 types of labels: e.g. insulina, CHEM and PROC
                        elif span2 == span1:
                            if Dict[n] not in InnerEnts:
                                InnerEnts.append(Dict[n])
            n = n - 1

    FinalEntities = {}

    for k in Dict:
        Ent = Dict[k]
        if (Ent not in ToDelete) and (Ent not in InnerEnts):
            FinalEntities[len(FinalEntities) + 1] = Ent

    # Nested entities (inner tags) are returned as a dictionary of dictionaries:
    # Check again because of entities with multiple labels: "vitamina C" (PROC & CHEM), "vitamina" (PROC & CHEM)
    n = 0
    NestedEnts = {}
    for ent in InnerEnts:
        if ent not in ToDelete and ent not in NestedEnts.values():
            n+=1
            NestedEnts[n] = ent

    return FinalEntities, NestedEnts


def apply_lexicon(text,LexiconData,*nest):
    
    '''
    Applies the lexicon to annotate the input text
    
    Arguments:
    - Text string
    - Dictionary data compiled in pickle object
    - Nesting annotation argument (optional)
    '''

    #print("Preannotating with lexicon...")

    Entities = {}

    n_ents = 0

    if (nest):
        EntsList = LexiconData.find_overlapping(text)
    else:
        EntsList = LexiconData.findall(text)
        
    for EntDict in EntsList:
        # We need to process several annotated labels in ambiguous entities
        for n in range(len(EntDict['norm'])):
            n_ents += 1
            # Entities[n_ents] = {"start": EntDict['start'], "end": EntDict['end'], "ent": EntDict['match'], "label": EntDict['norm'][n]}
            Entities[n_ents] = {"start": EntDict['start'], "end": EntDict['end'], "ent": EntDict['match'], "label": EntDict['norm'][n][0]}

    FinalEntities = {}

    SortedEntities={}

    # Sort in ascending starting offset
    # Rearrange entities starting with same offset so that the entity with longest scope is the only tagged
    for i,k in enumerate(sorted(Entities, key=lambda k: Entities[k]["start"], reverse=False)):
        SortedEntities[i] = Entities[k]
    
    FinalEntities,NestedEntities = remove_overlap(SortedEntities)
    
    # Remove substring overlap ("ma" in "por la mañana")
    FinalEntities = remove_substring_overlap(text,FinalEntities)
    NestedEntities = remove_substring_overlap(text,NestedEntities)
   
    #print("Finished preannotation")
    return FinalEntities,NestedEntities


def get_codes_from_lexicon(entity,label,LexiconData):
    
    '''
    Get CUI data from lexicon using entity and semantic label.
    Data format in lexicon:
        {'dolor': [('DISO', ['C0030193', 'C0234238']), ('PHYS', ['C0018235'])]
    Data format of LexiconData match (TextSearch library):
        {'norm': [('DISO', ['C0030193', 'C0234238']), ('PHYS', ['C0018235']), 'exact': False, 'match': 'dolor', 'case': 'lower', 'start': 0, 'end': 6}
    '''
    
    Data = LexiconData.findall(entity)

    # Apply exact match to avoid mapping of "signo" and "signo de infección"
    if (Data):
        for Dict in Data:
            if (Dict['match']) == entity:
                TuplesList = Dict['norm']
                for Tuple in TuplesList:
                    if (Tuple[0]) == label:
                        return sorted(Tuple[1])
    return None


def complete_norm_data(List,UMLSDataDict):
    
    '''
    Given a list of CUIs (List), return the data of preferred term and semantic type in the UMLS.
    Data is a file with a dictionary in pickle format.
    E.g. "abdomen"
        [C0000726, C1281594]
    Return:
        ["C0000726; Abdomen; Body Location or Region", "C1281594; Entire abdomen; Body Part, Organ, or Organ Component"]
    '''
        
    CUIList = []
    
    for cui in List:
        # Default value
        data = cui
        # Complete full data, only if available
        if cui in UMLSDataDict.keys():
            data = cui + "; " + UMLSDataDict[cui]['term'] + "; " + UMLSDataDict[cui]['sty']
        CUIList.append(data)
    
    return CUIList


def complete_snomed_code(List,SCTSPADict):
    
    '''
    Given a list of CUIs (List), return the code data of SNOMED-CT Spanish version.
    Data is a file with a dictionary in pickle format.
    E.g. "hipertensión"
        [C0020538]
    Return:
        ["hipertensión arterial", "38341003"]
    '''
        
    CUIList = []
    
    for cui in List:
        if cui in SCTSPADict.keys():
            data = SCTSPADict[cui]['term'] + "; " + SCTSPADict[cui]['code']
            CUIList.append(data)
    
    return CUIList


def complete_omop_code(List,OMOPDict):
    
    '''
    Given a list of CUIs (List), return the code data of OMOP.
    Data is a file with a dictionary in pickle format.
    
    List = [C0072980]
    
    OMOPDict = { C0072980: {'codes': ['19034726', '4348083'], 'terms': ['sirolimus', 'Sirolimus']}, ... }
    
    E.g. "sirolimus"
        C0072980
    Return:
        ["sirolimus; 19034726" | Sirolimus; 4348083"]
    '''

    OMOPCodesList = []
    
    for cui in List:
        if cui in OMOPDict.keys():
            # Convert codes list to string
            TermList = OMOPDict[cui]['terms']
            CodesList = OMOPDict[cui]['codes']
            # Map each term with the corresponding code
            Mappings = zip(TermList,CodesList)
            data = ""
            for i,item in enumerate(Mappings):
                if i == 0:
                    data = item[0] + "; " + item[1]
                else:
                    data = data + " | " + item[0] + "; " + item[1]
            OMOPCodesList.append(data)
    
    return OMOPCodesList


def add_label_to_token(EntitiesDict,TokensDict):
    
    '''
    Appends data of BIO labels to each word-dictionary of a dictionary of tokens.
    
    - TokensDict: a dictionary of tokens:
        {0: {'token': 'Métodos', 'lemma': 'Métodos', 'pos': 'NOUN', 'tag': 'NOUN__Gender=Masc|Number=Plur', 'shape': 'Xxxxx', 'start': 0, 'end': 7},
        1: {'token': ':', 'lemma': ':', 'pos': 'PUNCT', 'tag': 'PUNCT__PunctType=Colo', 'shape': ':', 'start': 7, 'end': 8},
        2: {'token': 'resultados', 'lemma': 'resultado', 'pos': 'NOUN', 'tag': 'NOUN__Gender=Masc|Number=Plur', 'shape': 'xxxx', 'start': 26, 'end': 36}, ...
    
    - EntitiesDict: a dictionary of annotated entities with offsets:

       {1: {'start': 0, 'end': 7, 'ent': 'Métodos', 'label': 'CONC'}, 2: {'start': 26, 'end': 36, 'ent': 'Métodos', 'label': 'CONC'} ...}

    The output is a dictionary of tokens with the label for each token:

        {0: {'token': 'Métodos', 'lemma': 'Métodos', 'pos': 'NOUN', 'tag': 'NOUN__Gender=Masc|Number=Plur', 'shape': 'Xxxxx', 'start': 0, 'end': '7', 'label': 'B-CONC'},
        1: {'token': ':', 'lemma': ':', 'pos': 'PUNCT', 'tag': 'PUNCT__PunctType=Colo', 'shape': ':', 'start': 7, 'end': '8', 'label': 'O'}, ...   
    '''
    
    for i in TokensDict:
        start = TokensDict[i]['start']
        end = TokensDict[i]['end']
        Data = TokensDict[i]
        # Check if existing label; otherwise, initialize to "O"
        if 'label' not in Data:            
            Data.update({'label': "O"})
        for j in EntitiesDict:
            s_label = EntitiesDict[j]['start']
            e_label = EntitiesDict[j]['end']
            if (start!=None) and (end!=None) and (int(start)==int(s_label)) and (int(end) == int(e_label)):
                label = EntitiesDict[j]['label']
                Data.update({'label': label})

            # Check and correct tokenization mismatch in words with "-":
            # 80: {'token': 'SARS', 'lemma': 'SARS', 'pos': 'PROPN', 'tag': 'PROPN___', 'shape': 'XXXX', 'start': 459, 'end': 463},
            # 81: {'token': '-', 'lemma': '-', 'pos': 'PUNCT', 'tag': 'PUNCT__PunctType=Dash', 'shape': '-', 'start': 463, 'end': 464},
            # 82: {'token': 'CoV-2', 'lemma': 'CoV-2', 'pos': 'PROPN', 'tag': 'PROPN___', 'shape': 'XxX-d', 'start': 464, 'end': 469}
            # ENT: {'start': 459, 'end': 469, 'ent': 'SARS-CoV-2', 'label': 'LIVB'}
            elif (i>1) and (i<len(TokensDict)) and (TokensDict[i]['lemma']=='-'):
                if (int(TokensDict[i-1]['start'])==int(s_label)) and (int(TokensDict[i+1]['end'])==int(e_label)):
                    label = EntitiesDict[j]['label']
                    # Previous entity
                    TokensDict[i-1]['label']=label
                    # Current entity
                    Data.update({'label': label})
                    # Next entity
                    TokensDict[i+1]['label']=label
                    # skip checking next entity
                    continue
            # Check and correct tokenization mismatch in cases such as "160mg", "80mL", "150µg", "3g" (without space; Spacy tokenizes in 2 items):
            # FIXED ALSO: "/sem" ('por semana'); "/día" ('al día') CONFIRM THAT IT DOES NOT CAUSE NOISE
            # 17: {'token': '/', 'lemma': '/', 'pos': 'PUNCT', 'tag': 'PUNCT___', 'shape': '/', 'start': 59, 'end': 60},
            # 18: {'token': 'sem', 'lemma': 'sem', 'pos': 'NOUN', 'tag': 'NOUN__Gender=Masc|Number=Sing', 'shape': 'xxx', 'start': 60, 'end': 63}
            # ENT: {'ent': '/sem', 'start': 59, 'end': 63, 'label': 'Frequency'}
            elif (i>0) and (i<len(TokensDict)) and ((TokensDict[i]['token']=='mg') or (TokensDict[i]['token']=='g') or (TokensDict[i]['token']=='µg') or (TokensDict[i]['token']=='ml') or (TokensDict[i]['token']=='mmol') or (TokensDict[i]['token']=='sem') or (TokensDict[i]['token']=='día')):
                if TokensDict[i-1]['start']!=None and TokensDict[i]['end']!=None and s_label!=None and e_label!=None and (int(TokensDict[i-1]['start'])==int(s_label)) and (int(TokensDict[i]['end'])==int(e_label)):
                    label = EntitiesDict[j]['label']
                    # Previous entity
                    TokensDict[i-1]['label']=label
                    # Current entity                    
                    Data.update({'label': label})
                    # skip checking next entity
                    continue

    return TokensDict


def read_exceptions_list(FILENAME):
    
    ''' Reads file with list of exceptions to process. '''
    
    ExceptionsHash = {}
    
    # Read and process the exceptions
    with open(FILENAME, 'r', newline='') as f:
        n = 0
        Lines = f.readlines()
        for line in Lines:
            line = line.strip()
            if "#" not in line and line != '':
                pattern_string, finalLabel = re.search("([^\|]+)\|([^\|]+)", line).group(1, 2)
                if pattern_string and finalLabel:
                    PatternList = pattern_string.split()
                    TuplesList = []
                    for pattern in PatternList:
                        lemma, label = re.search("(.+)/(.+)", pattern).group(1, 2)
                        if lemma and label:
                            n += 1
                            Tuple = (lemma, label)
                            TuplesList.append(Tuple)
                            # Use a tuple, lists are not allowed as dictionary keys
                            ExceptionsHash[n] = {'pattern': TuplesList, 'finalLabel': finalLabel}

    return ExceptionsHash


def remove_entities(HashEntities, TokensDict, ExceptionsDict, label_type, NestedEnts):
    
    '''
    Remove entities belonging to specific entity types defined in an exception list.
    Entities included in NestedEnts that have a label not to be excluded, are not deleted but replaced with that semantic label (e.g. "contaminación", PHEN -> DISO)
    The parameters are:
       - HashEntities: a dictionary with annotated entities in the following format:
            HashEntities = {0: {'start': 0, 'end': 8, 'ent': 'insulina', 'label': 'CHEM'}, ...}
       - TokensDict: a hash with tokens (obtained with Spacy), in the following format:
            TokensDict = {1: {'token': 'Métodos', 'lemma': 'Métodos', 'pos': 'NOUN', 'tag': 'NOUN__Gender=Masc|Number=Plur', 'shape': 'Xxxxx', 'start': 0, 'end': '7'}, {2: ... } ... }
       - ExceptionsDict: a hash with defined exceptions, in the following format:
            ExceptionsDict = {'pattern': [('de', 'O'), ('base', 'CHEM')], 'finalLabel': 'O'}
       - label_type: the label of default layer ('label') or label of layer 2 ('label2') for nested labels
       - NestedDataHash: a hash with annotated entities inside the scope of wider entities, in the following format:
            {1: {'start': 0, 'end': 8, 'ent': 'insulina', 'label': 'PROC'}, ... }
    '''
    
    CleanedEntities = {}

    # Save entity, new labels, and offsets
    NewEntityLabels = {}
    
    # Save labels to be changed regardless of the patterns or the lemma ("ANY-LEMMA")
    LabelsToRemove = []

    # Add semantic labels to tokens
    TokensDict = add_label_to_token(HashEntities,TokensDict)
    k = 0
    for i in TokensDict:
        
        found = False
        # Data of current entity
        label = TokensDict[i]['label']
        lemma = TokensDict[i]['lemma'].lower()
        token = TokensDict[i]['token']
        # Check current and previous entity (except 1st token to avoid error)
        if (i > 0) and (i < len(TokensDict)):
            # Data of previous entity
            label_prev = TokensDict[i - 1][label_type]
            lemma_prev = TokensDict[i - 1]['lemma'].lower()
            for k in ExceptionsDict:
                patternList = ExceptionsDict[k]['pattern']
                finalLabel = ExceptionsDict[k]['finalLabel']
                # first entity in pattern
                lemma_pat_prev = patternList[0][0]
                tag_prev = patternList[0][1]
                # Save for later processing
                if ((patternList[0][0] == 'ANY-LEMMA') and (finalLabel == 'O')) and tag_prev not in LabelsToRemove:
                    LabelsToRemove.append(tag_prev)
                # Process 2-grams exceptions
                if (len(patternList) == 2) and (found == False):
                    # second entity in pattern
                    lemma_pat = patternList[1][0]
                    tag = patternList[1][1]
                    if (found == False) and ((lemma_pat_prev == lemma_prev) and (lemma_pat == lemma) and (
                        tag_prev == label_prev) and (tag == label)):
                        found = True
                        CleanedEntities[i] = TokensDict[i]
                        CleanedEntities[i - 1] = TokensDict[i - 1]
                        if finalLabel != "O":
                            if label_type == 'label':
                                # Comprobar que no se borran entidades anidadas
                                NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                         'start': CleanedEntities[i]['start'],
                                                                         'end': CleanedEntities[i]['end'],
                                                                         'label': finalLabel, 'label2': 'O'}
                            else:
                                NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                         'start': CleanedEntities[i]['start'],
                                                                         'end': CleanedEntities[i]['end'],
                                                                         'label': CleanedEntities[i]['label'],
                                                                         'label2': finalLabel}
                        else:
                            if label_type == 'label':
                                NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                         'start': CleanedEntities[i]['start'],
                                                                         'end': CleanedEntities[i]['end'],
                                                                         'label': 'O', 'label2': 'O'}
                            else:
                                NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                         'start': CleanedEntities[i]['start'],
                                                                         'end': CleanedEntities[i]['end'],
                                                                         'label': CleanedEntities[i]['label'],
                                                                         'label2': 'O'}
                        CleanedEntities[i - 1][label_type] = finalLabel
                        CleanedEntities[i][label_type] = finalLabel
                # Process 1-gram exceptions
                if (len(patternList) == 1) and (found == False):
                    if (found == False) and ((lemma_pat_prev == lemma) and (tag_prev == label)) or ((lemma_pat_prev == "ANY-LEMMA") and (tag_prev == label)):
                        found = True
                        CleanedEntities[i] = TokensDict[i]
                        # CASE 1: There are nested entities
                        if len(NestedEnts) > 0:
                            
                            # If an entity in NestedEnts does not occur in the exception list, its label is used
                            for e in NestedEnts:
                                nested_ent = NestedEnts[e]['ent']
                                nested_ent_label = NestedEnts[e]['label']
                                # nested_ent_label: label of the inner entity
                                # tag_prev: label to be excluded, as defined in the exceptions list
                                # label: label currently annotated, to evaluate if to be changed or not
                                if (token == nested_ent) and (label != nested_ent_label) and (nested_ent_label != tag_prev) and (label != tag_prev):
                                    CleanedEntities[i][label_type] = nested_ent_label
                                    if label_type == 'label':
                                        NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                                 'start': CleanedEntities[i]['start'],
                                                                                 'end': CleanedEntities[i]['end'],
                                                                                 'label': nested_ent_label,
                                                                                 'label2': 'O'}
                                    else:
                                        NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                                 'start': CleanedEntities[i]['start'],
                                                                                 'end': CleanedEntities[i]['end'],
                                                                                 'label': CleanedEntities[i]['label'],
                                                                                 'label2': nested_ent_label}
                                    break
                                else:
                                    CleanedEntities[i][label_type] = 'O'
                                    if finalLabel != "O":
                                        if label_type == 'label':
                                            NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                                     'start': CleanedEntities[i]['start'],
                                                                                     'end': CleanedEntities[i]['end'],
                                                                                     'label': finalLabel, 
                                                                                     'label2': 'O'}
                                        else:
                                            NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                                     'start': CleanedEntities[i]['start'],
                                                                                     'end': CleanedEntities[i]['end'],
                                                                                     'label': CleanedEntities[i]['label'], 
                                                                                     'label2': finalLabel}
                                    else:
                                        if label_type == 'label':
                                            NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                                     'start': CleanedEntities[i][
                                                                                         'start'],
                                                                                     'end': CleanedEntities[i]['end'],
                                                                                     'label': 'O', 'label2': 'O'}
                                        else:
                                            NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                                     'start': CleanedEntities[i][
                                                                                         'start'],
                                                                                     'end': CleanedEntities[i]['end'],
                                                                                     'label': CleanedEntities[i][
                                                                                         'label'], 'label2': 'O'}
                                    CleanedEntities[i][label_type] = finalLabel
                                    break
                        # CASE 2: There are not nested entities
                        else:
                            
                            CleanedEntities[i][label_type] = 'O'
                            if finalLabel != "O":
                                if label_type == 'label':
                                    NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                             'start': CleanedEntities[i]['start'],
                                                                             'end': CleanedEntities[i]['end'],
                                                                             'label': finalLabel, 'label2': 'O'}
                                else:
                                    NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                             'start': CleanedEntities[i]['start'],
                                                                             'end': CleanedEntities[i]['end'],
                                                                             'label': CleanedEntities[i]['label'],
                                                                             'label2': finalLabel}
                            else:
                                if label_type == 'label':
                                    NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                             'start': CleanedEntities[i]['start'],
                                                                             'end': CleanedEntities[i]['end'],
                                                                             'label': 'O', 'label2': 'O'}
                                else:
                                    NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                             'start': CleanedEntities[i]['start'],
                                                                             'end': CleanedEntities[i]['end'],
                                                                             'label': CleanedEntities[i]['label'],
                                                                             'label2': 'O'}
                            CleanedEntities[i][label_type] = finalLabel
                            break
        # Check 1st token
        else:
            for k in ExceptionsDict:
                patternList = ExceptionsDict[k]['pattern']
                finalLabel = ExceptionsDict[k]['finalLabel']
                if (len(patternList) == 1):                    
                    lemma_pat = patternList[0][0]
                    tag = patternList[0][1]
                    if (found == False) and ((lemma_pat == lemma) and (tag == label)) or (
                        (lemma_pat == "ANY-LEMMA") and (label == tag)):
                        found = True
                        CleanedEntities[i] = TokensDict[i]
                        # CASE 1: There are nested entities
                        if len(NestedEnts) > 0:
                            # If an entity in NestedEnts does not occur in the exception list, its label is used
                            for e in NestedEnts:
                                nested_ent = NestedEnts[e]['ent']
                                nested_ent_label = NestedEnts[e]['label']
                                if (token == nested_ent) and (label != nested_ent_label) and (
                                    nested_ent_label != tag) and (label != tag):
                                    CleanedEntities[i][label_type] = nested_ent_label
                                    if label_type == 'label':
                                        NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                                 'start': CleanedEntities[i]['start'],
                                                                                 'end': CleanedEntities[i]['end'],
                                                                                 'label': nested_ent_label,
                                                                                 'label2': 'O'}
                                    else:
                                        NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                                 'start': CleanedEntities[i]['start'],
                                                                                 'end': CleanedEntities[i]['end'],
                                                                                 'label': CleanedEntities[i]['label'],
                                                                                 'label2': nested_ent_label}
                                    break
                                else:
                                    CleanedEntities[i][label_type] = 'O'
                                    if finalLabel != "O":
                                        if label_type == 'label':
                                            NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                                     'start': CleanedEntities[i][
                                                                                         'start'],
                                                                                     'end': CleanedEntities[i]['end'],
                                                                                     'label': finalLabel, 'label2': 'O'}
                                        else:
                                            NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                                     'start': CleanedEntities[i][
                                                                                         'start'],
                                                                                     'end': CleanedEntities[i]['end'],
                                                                                     'label': CleanedEntities[i][
                                                                                         'label'], 'label2': finalLabel}
                                        CleanedEntities[i][label_type] = finalLabel
                                    else:
                                        NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                                 'start': CleanedEntities[i]['start'],
                                                                                 'end': CleanedEntities[i]['end'],
                                                                                 'label': 'O', 'label2': finalLabel}
                        # CASE 2: There are not nested entities
                        else:
                            CleanedEntities[i][label_type] = 'O'
                            if finalLabel != "O":
                                if label_type == 'label':
                                    NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                             'start': CleanedEntities[i]['start'],
                                                                             'end': CleanedEntities[i]['end'],
                                                                             'label': finalLabel, 'label2': 'O'}
                                else:
                                    NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                             'start': CleanedEntities[i]['start'],
                                                                             'end': CleanedEntities[i]['end'],
                                                                             'label': CleanedEntities[i]['label'],
                                                                             'label2': finalLabel}
                            else:
                                if label_type == 'label':
                                    NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                             'start': CleanedEntities[i]['start'],
                                                                             'end': CleanedEntities[i]['end'],
                                                                             'label': 'O', 'label2': 'O'}
                                else:
                                    NewEntityLabels[len(NewEntityLabels)] = {'ent': CleanedEntities[i]['token'],
                                                                             'start': CleanedEntities[i]['start'],
                                                                             'end': CleanedEntities[i]['end'],
                                                                             'label': CleanedEntities[i]['label'],
                                                                             'label2': 'O'}
                            CleanedEntities[i][label_type] = finalLabel
                            break

        if (found == False):
            CleanedEntities[i] = TokensDict[i]
    
    FinalHash = {}

    # Change the old labels in the original hash to return (only if offsets match, to avoid tokenization errors)
    Ents2Change = {}
    AuxList = []
    for Ent in HashEntities:
        s = HashEntities[Ent]['start']
        e = HashEntities[Ent]['end']
        for NewEnt in NewEntityLabels:
            s_new = NewEntityLabels[NewEnt]['start']
            e_new = NewEntityLabels[NewEnt]['end']
            new_label = NewEntityLabels[NewEnt]['label']
            new_label2 = NewEntityLabels[NewEnt]['label2']
            if (s == s_new) and (e == e_new) and (HashEntities[Ent]['ent'] == NewEntityLabels[NewEnt]['ent']):
                
                Hash = {'start': s, 'end': e, 'ent': HashEntities[Ent]['ent'], 'label': new_label, 'label2': new_label2, 'hashkey': Ent}
                # Avoid repeated items to remove
                if Hash not in Ents2Change.values():
                    Ents2Change[len(Ents2Change)] = Hash
                    AuxList.append(Hash)
    
    # Change entities in the original hash using the hash keys, in reverse order:
    for item in sorted(Ents2Change.items(), key=lambda item: item[1]['hashkey'], reverse=True):
        # if new label is 'O', remove key
        hashkey = item[1]['hashkey']
        if (item[1]['label'] == 'O'):
            HashEntities.pop(hashkey)
        else:
            HashEntities[hashkey]['label'] = item[1]['label']
    
    # Remove in the remaining multiword entities the labels defined in exception lists regardless of patterns ("ANY-LEMMA")
    # e.g. "metabolismo celular", PHYS, is not removed using any pattern, but needs to be removed using "ANY-LEMMA" 
    for k in HashEntities.copy():
        if HashEntities[k]['label'] in LabelsToRemove:
            del HashEntities[k]
    
    # Remove duplicated values 
    FinalHashEntities = {}
    n = 0
    for k in HashEntities:
        if HashEntities[k] not in FinalHashEntities.values():
            n+=1
            FinalHashEntities[n] = HashEntities[k]
    
    return FinalHashEntities


def main(arg1, *arg2):
    with open(sys.input, 'r', newline = '') as f:
        t = f.read()
        Entities = apply_lexicon(t)
        # Remove entities defined in an exception list (optional parameter)
        if (args.exc):
            Entities = remove_entities(Entities,args.exc)

#
#
#############
#
# Main class
#


if __name__ == '__main__':
    main(args.input)
    
    