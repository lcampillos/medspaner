#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# (Indicate encoding as 'iso-8859-1' or 'utf-8')
#
# add_bio_label.py
#
# Usage:
#
#   python add_bio_label.py <FILE>
#
#
# The input file has tabular format:
#
#       Token   Lemma   POS  ...
#       El	El	DET	
#       paciente	paciente	NOUN	
#       fue	ser	AUX	
#       intervenido	intervenir	VERB	
#       ...
#
# BIO (Begin, Inside, Outside) labels are appended as a column:
#
#       Token   Lemma   POS     Label...
#       El	El	DET	    O
#       paciente	paciente	NOUN	   B-LIVB
#       fue	ser	AUX	O
#       intervenido	intervenir	VERB	B-PROC
#       ...
#
# 
# Leonardo Campillos 
# InterTalentumUAM (2019-2021)
#
#########################################################################

import re
import os
import sys
import spacy

# Compare offsets to detect overlapping entities
def compare_offsets(start,end,TuplesList):
    for Tuple in TuplesList:
        # Offsets of previous processed entities
        prev_e = Tuple[1]
        # Compare offsets of previous entities and entities to be processed
        if (start < prev_e):
            return True
        else:
            return False
        
# Given a dictionary of dictionaries with entities, offsets and labels, returns a representation in BIO scheme
# Input:
#   { 1: { ent: "diálisis peritoneal", start: 95, end: 114, label: "DISO"}, ...}
# Output:
#   { 1: { ent: "diálisis", start: 95, end: 103, label: "B-DISO" }, 2: { ent: "peritoneal", start: 105, end: 114, label: "I-DISO" } ... }
#
def offset2bio(EntsDict):

    BIOEnts={}

    label_id = 0

    Offsets = []
    
    for i in EntsDict:
        ent = EntsDict[i]["ent"]
        label = EntsDict[i]["label"]
        start = EntsDict[i]["start"]
        end = EntsDict[i]["end"]
        # Preprocess multi-word entities that overlap
        # e.g. "bloqueo del nervio", "nervio isquiático" => only keep "bloqueo del nervio"
        # Here we save tuples of start and end offsets, to compare and detect overlapping
        mw = re.search(" ",ent)
        if mw:
            start = EntsDict[i]["start"]
            end = EntsDict[i]["end"]
            # Compare offsets with those of previous entities
            # If offset is inside a previous entity, the ovelapping entity is skipped
            if (compare_offsets(start,end,Offsets) == True):
                continue
            else:
                Tuple = (start, end)
                Offsets.append(Tuple)
        SplitEnts0 = ent.split(" ")
        SplitEnts = []
        # Include also punctuation characters that may be annotated (e.g. commas); the split methods deletes commas
        # Note that "7," needs to be split, but not "57,1"
        for ent in SplitEnts0:
            punct = re.search("[≥≤><,=]",ent)
            if punct:
                char = re.search("([≥≤><,=])",ent).group(1)
                w = re.sub(char,"",ent)
                # if punctuation at the beginning of the entity ("≥8")
                if (punct.span()[0] == 0):                    
                    SplitEnts.append(char)
                    SplitEnts.append(w)                    
                # if punctuation at the end of the entity ("7,")
                elif (punct.span()[0] == (len(ent)-1)):
                    SplitEnts.append(w)
                    SplitEnts.append(char)
                # if punctuation in the middle of the entity ("57,1"), do not split
                else:
                    SplitEnts.append(ent)
            else:
                SplitEnts.append(ent)
        label_id += 1
        if (len(SplitEnts)==1) and (("B-") not in label):
            # This can be changed to U- (label for unique entity)
            BIOEnts[label_id] = {'ent': ent, 'start': start, 'end': end, 'label': "B-" + label}
        else:
            prev_word_offset = 0
            for j,word in enumerate(SplitEnts):
                if (j==0) and (("B-") not in label):
                    tag = "B-" + label
                    end = int(start) + int(len(word))
                    prev_word_offset = end
                    BIOEnts[label_id] = { "ent": word, "label": tag, "start": start, "end": end}
                    #print("ESTO 0 1",label_id,BIOEnts[label_id])
                elif (("I-") not in label) and (("B-") not in label):
                    label_id += 1
                    tag = "I-" + label
                    # Offset of posterior punctuation characters (e.g. "7,") or anterior punctuation characters (e.g. "<7", ">=18")
                    # (there is no separating white space)
                    if (word in [","]) or (BIOEnts[label_id-1]['ent'] in ["≥", "≤", ">", "<", "="]):
                        start = prev_word_offset
                    # Rest of words (separated by white space)
                    else:
                        start = prev_word_offset + 1
                    end = int(start) + int(len(word))
                    prev_word_offset = end
                    BIOEnts[label_id] = {"ent": word, "label": tag, "start": start, "end": end}
                    #print("ESTO 1",label_id,BIOEnts[label_id])
    #print(BIOEnts)
    # This step is to fix tokenization issues with spaCy: "mg/ml" => "mg", "/", "ml"
    BIOEntsFinal = {}

    n = 0
    for k in BIOEnts:
        ent = BIOEnts[k]['ent']
        # Slash with measurement units (do not use numbers because it causes errors in dates: e.g. 1/1/2001)
        to_split = re.search("([moµμgl]+)\/([moµμkgl]+)",ent)
        # Slash with "día", "semana"...
        to_split2 = re.search("(ampollas?|ui|vez|veces)\/(semana|sem|d[ií]a)", ent)
        # Percent character
        to_split3 = re.search("([^\/]+)%$", ent)
        # Slash with "/12 horas"... => Esto da error con fechas
        # to_split4 = re.search("\/([0-9]+)", ent)
        if to_split:
            part1 = to_split.group(1)
            part1_start = BIOEnts[k]['start']
            part1_end = int(part1_start) + int(len(part1))
            n += 1
            BIOEntsFinal[n] = {"ent": part1, "label": BIOEnts[k]['label'], "start": part1_start, "end": part1_end}
            n += 1
            slash_end = int(part1_end) + 1
            BIOEntsFinal[n] = {"ent": "/", "label": BIOEnts[k]['label'], "start": part1_end, "end": slash_end}
            part2 = to_split.group(2)
            part2_end = BIOEnts[k]['end']
            n += 1
            BIOEntsFinal[n] = {"ent": part2, "label": BIOEnts[k]['label'], "start": slash_end, "end": part2_end}
        elif to_split2:
            part1 = to_split2.group(1)
            part1_start = BIOEnts[k]['start']
            part1_end = int(part1_start) + int(len(part1))
            n += 1
            BIOEntsFinal[n] = {"ent": part1, "label": BIOEnts[k]['label'], "start": part1_start, "end": part1_end}
            n += 1
            part2 = "/"+ to_split2.group(2)
            part2_end = BIOEnts[k]['end']
            n += 1
            BIOEntsFinal[n] = {"ent": part2, "label": BIOEnts[k]['label'], "start": part1_end, "end": part2_end}
        elif to_split3:
            part1 = to_split3.group(1)
            part1_start = BIOEnts[k]['start']
            part1_end = int(part1_start) + int(len(part1))
            n += 1
            BIOEntsFinal[n] = {"ent": part1, "label": BIOEnts[k]['label'], "start": part1_start, "end": part1_end}
            n += 1
            char_end = int(part1_end) + 1
            BIOEntsFinal[n] = {"ent": "%", "label": BIOEnts[k]['label'], "start": part1_end, "end": char_end}
            '''
            # Esto da error con fechas "1/1/2001"
            elif to_split4:
                number = to_split4.group(1)
                slash_start = BIOEnts[k]['start']
                slash_end = int(slash_start) + 1
                n += 1
                BIOEntsFinal[n] = {"ent": "/", "label": BIOEnts[k]['label'], "start": slash_start, "end": slash_end}
                n += 1
                number_end = int(slash_end) + int(len(number))
                BIOEntsFinal[n] = {"ent": number, "label": BIOEnts[k]['label'], "start": slash_end, "end": number_end}
            '''
        else:
            n += 1
            BIOEntsFinal[n] = BIOEnts[k]

    #print("BIOEntsFinal",BIOEntsFinal)

    return BIOEntsFinal


# Appends data of BIO labels to each word-dictionary of a dictionary of tokens
# TokensDict is a dictionary of tokens:
#
#   {0: {'token': 'Métodos', 'lemma': 'Métodos', 'pos': 'NOUN', 'tag': 'NOUN__Gender=Masc|Number=Plur', 'shape': 'Xxxxx', 'start': 0, 'end': 7},
#   1: {'token': ':', 'lemma': ':', 'pos': 'PUNCT', 'tag': 'PUNCT__PunctType=Colo', 'shape': ':', 'start': 7, 'end': 8},
#   2: {'token': 'resultados', 'lemma': 'resultado', 'pos': 'NOUN', 'tag': 'NOUN__Gender=Masc|Number=Plur', 'shape': 'xxxx', 'start': 26, 'end': 36}, ...
#
# EntitiesDict is a dictionary of annotated entities with offsets:
#
#   {1: {'start': 0, 'end': 7, 'ent': 'Métodos', 'label': 'B-CONC'}, 2: {'start': 26, 'end': 36, 'ent': 'Métodos', 'label': 'B-CONC'} ...}
#
# Output is a dictionary of tokens with the labeled value for each token:
#
#   {0: {'token': 'Métodos', 'lemma': 'Métodos', 'pos': 'NOUN', 'tag': 'NOUN__Gender=Masc|Number=Plur', 'shape': 'Xxxxx', 'start': 0, 'end': '7', 'label': 'B-CONC'},
#   1: {'token': ':', 'lemma': ':', 'pos': 'PUNCT', 'tag': 'PUNCT__PunctType=Colo', 'shape': ':', 'start': 7, 'end': '8', 'label': 'O'}, ...
#
def add_label(TokensDict, EntitiesDict):
    
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
                # print("AQUI",Data)
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
                    # Change B- to B- + I- if needed
                    if "B-" in label:
                        label =  re.sub("B-","I-",label)
                    Data.update({'label': label})
                    # Next entity
                    TokensDict[i+1]['label']=label
                    # skip checking next entity
                    continue
            # Check and correct tokenization mismatch in cases such as "160mg", "80mL", "150µg", "3g" (without space; Spacy tokenizes in 2 items):
            # FIXED ALSO: "/sem" ('por semana'); "/día" ('al día') CONFIRM THAT IT DOES NOT CAUSE NOISE
            # 17: {'token': '/', 'lemma': '/', 'pos': 'PUNCT', 'tag': 'PUNCT___', 'shape': '/', 'start': 59, 'end': 60},
            # 18: {'token': 'sem', 'lemma': 'sem', 'pos': 'NOUN', 'tag': 'NOUN__Gender=Masc|Number=Sing', 'shape': 'xxx', 'start': 60, 'end': 63}
            # ENT: {'ent': '/sem', 'start': 59, 'end': 63, 'label': 'B-FREQ'}
            elif (i>0) and (i<len(TokensDict)) and ((TokensDict[i]['token']=='mg') or (TokensDict[i]['token']=='g') or (TokensDict[i]['token']=='µg') or (TokensDict[i]['token']=='ml') or (TokensDict[i]['token']=='mmol') or (TokensDict[i]['token']=='sem') or (TokensDict[i]['token']=='día')):
                if TokensDict[i-1]['start']!=None and TokensDict[i]['end']!=None and s_label!=None and e_label!=None and (int(TokensDict[i-1]['start'])==int(s_label)) and (int(TokensDict[i]['end'])==int(e_label)):
                    label = EntitiesDict[j]['label']
                    # Previous entity
                    TokensDict[i-1]['label']=label
                    # Change B- to B- + I- if needed
                    if "B-" in label:
                        label =  re.sub("B-","I-",label)
                    # Current entity                    
                    Data.update({'label': label})
                    # skip checking next entity
                    continue


    return TokensDict


# Appends data of BIO labels in 2nd layer to each word-dictionary of a dictionary of tokens
# TokensDict is a dictionary of tokens with labels (layer 1):
#
#   {0: {'token': 'Métodos', 'lemma': 'Métodos', 'pos': 'NOUN', 'tag': 'NOUN__Gender=Masc|Number=Plur', 'shape': 'Xxxxx', 'start': 0, 'end': '7', 'label': 'B-CONC'},
#
# EntitiesDict is a dictionary of tokens with a new value for token labels in layer 2
#
#   {1: {'start': 3, 'end': 8, 'ent': 'radio', 'label': 'B-CONC'}, 2: {'start': 3, 'end': 8, 'ent': 'radio', 'label': 'B-PHEN'}, ...
#
# Output:
#
#   {0: {'token': 'Métodos', 'lemma': 'Métodos', 'pos': 'NOUN', 'tag': 'NOUN__Gender=Masc|Number=Plur', 'shape': 'Xxxxx', 'start': 0, 'end': '7', 'label': 'B-CONC', label2': 'O'},
#    1: {'token': ':', 'lemma': ':', 'pos': 'PUNCT', 'tag': 'PUNCT__PunctType=Colo', 'shape': ':', 'start': 7, 'end': 8, 'label': 'O', 'label2': 'O'} ...}
#
def add_label_layer2(TokensDict, EntitiesDict):

    # Save previous label to check error
    prev_label = ""
    for i in TokensDict:
        start = TokensDict[i]['start']
        end = TokensDict[i]['end']
        Data = TokensDict[i]
        Data.update({'label2': "O"})
        label1 = TokensDict[i]['label'] 
        for j in EntitiesDict:
            s_label = EntitiesDict[j]['start']
            e_label = EntitiesDict[j]['end']
            label2 = EntitiesDict[j]['label']
            
            if (start!=None) and (end!=None) and (int(start)==int(s_label)) and (int(end) == int(e_label)):
                # Label 2 is added only if different annotations, and if previous entity is not "O" followed by "I-" (to avoid error of sequence such as "O I-PROC")
                label2_no_bio = re.sub("(B|I)-","",label2)
                label1_no_bio = re.sub("(B|I)-", "", label1)
                i_label = re.search("I\-",label2)
                if (label1_no_bio!=label2_no_bio):
                    if (i_label) and (prev_label=="O"):
                        continue
                    else:
                        Data.update({'label2': label2})
            prev_label = label2
        
    return TokensDict


# Given a dictionary of entities with Begin, Inside and Out (BIO) information, it removes the BIO labels for merging offsets and printing to BRAT format
#
#   Input:
#       {1: {'start': 3, 'end': 8, 'ent': 'radio', 'label': 'B-CHEM'}, 2: {'start': 13, 'end': 18, 'ent': 'brazo', 'label': 'B-ANAT'}, ...}
#
#   Output:
#       {1: {'start': 3, 'end': 8, 'ent': 'radio', 'label': 'CHEM'}, 2: {'start': 13, 'end': 18, 'ent': 'brazo', 'label': 'ANAT'}, ...}
#
def bio2offset(EntitiesDict):
    FinalEntities={}
    i=0
    last_i=0
    tag = "O" # Default value
    # Entities annotated in layer 1
    prev_label = ""
    prev_end_span = ""
    for k in EntitiesDict:
        label = EntitiesDict[k]['label']
        b_label = re.search("B-",label)
        i_label = re.search("I-", label)
        end = EntitiesDict[k]['end']
        if b_label:
            i += 1
            last_i = i
            start = EntitiesDict[k]['start']
            ent = EntitiesDict[k]['token']
            tag = re.sub("B-","",EntitiesDict[k]['label'])
            FinalEntities[i] = { 'start': start, 'end': end, 'ent': ent, 'label':tag }
        # Check if previous label is not "O", to avoid erroneous sequence "O I-PROC"
        elif i_label and (prev_label!= "O"):
            i_tag = re.sub("I-","",EntitiesDict[k]['label'])
            if (i_tag==tag):
                FinalEntities[last_i]['end']=end
                ent = EntitiesDict[k]['token']
                # Check spans of each word in the multiword entity
                if (EntitiesDict[k-1]['end']==EntitiesDict[k]['start']):
                    ent_rest = FinalEntities[last_i]['ent'] + ent
                else:
                    ent_rest = FinalEntities[last_i]['ent'] + " " + ent
                FinalEntities[last_i]['ent'] = ent_rest
        prev_label = label
        #prev_end_span = end

    # Nested entities annotated in layer 2 (if available)
    prev_label = ""
    for k in EntitiesDict:
        try:
            label2 = EntitiesDict[k]['label2']
            b_label2 = re.search("B-",label2)
            i_label2 = re.search("I-", label2)
            if b_label2:
                i += 1
                last_i = i
                start = EntitiesDict[k]['start']
                end = EntitiesDict[k]['end']
                ent = EntitiesDict[k]['token']
                tag = re.sub("B-","",EntitiesDict[k]['label2'])
                FinalEntities[i] = { 'start': start, 'end': end, 'ent': ent, 'label':tag }
            # Check if previous label is not "O", to avoid erroneous sequence "O I-PROC"
            elif i_label2 and (prev_label!= "O"):
                i_tag = re.sub("I-","",EntitiesDict[k]['label2'])
                if (i_tag==tag):
                    end = EntitiesDict[k]['end']
                    FinalEntities[last_i]['end']=end
                    ent = EntitiesDict[k]['token']
                    ent_rest = FinalEntities[last_i]['ent'] + " " + ent
                    FinalEntities[last_i]['ent'] = ent_rest
            prev_label = label2
        except:
            break

    return FinalEntities


def convert_to_conll(EntitiesHash, TokensHash, *NestedEntsHash):
    
    '''
    Converts to CONLL format and Begin, Inside and Out (BIO) scheme
    Arguments:
    - Dictionary of entities with the following format:
        EntitiesHash = {1: {'start': 0, 'end': 18, 'ent': 'sangrado digestivo', 'label': 'DISO'}, 2: ... }
    - Dictionary of tokens obtained with Spacy tokenizer:
        TokensHash = {0: {'token': 'sangrado', 'lemma': 'sangrado', 'pos': 'NOUN', 'tag': 'NOUN', 'shape': 'xxxx', 'start': 0, 'end': 8},
                      1: {'token': 'digestivo', 'lemma': 'digestivo', 'pos': 'NOUN', 'tag': 'ADJ', 'shape': 'xxxx', 'start': 9, 'end': 18}...} 
    - (Optional argument) Dictionary of nested entities with the following format:
        NestedEntsHash = {1: {'start': 9, 'end': 18, 'ent': 'digestivo', 'label': 'ANAT'}, ...}
    '''

    BIOlabels = offset2bio(EntitiesHash)
    ResultsCONLL = add_label(TokensHash, BIOlabels)
    
    # If output also nested entities
    if (NestedEntsHash):
        NestedBIOlabels = offset2bio(NestedEntsHash[0])
        ResultsCONLL = add_label_layer2(TokensHash, NestedBIOlabels)
    
    return ResultsCONLL

