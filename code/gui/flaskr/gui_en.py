#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask, request, render_template, Markup

# Import annotation modules
import sys

sys.path.append("..")  # Adds higher directory to python modules path.

import re

import argparse

import spacy
import spacy_tokenizer
from spacy_tokenizer import *

import lexicon_tools
from lexicon_tools import *

import transformers
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer, pipeline

import pickle


# Function to remove space before predicted entities
def remove_space(EntsList):
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
        if (string[prev_char]) not in [" ", "/", "\n", "\r", "?", "!"]:  # TODO: check if add more punctuation characters

            # Get data from previous entity saved in AuxDict
            if len(AuxDict) > 0:
                new_word = word + AuxDict['word']
                new_start = start
                new_end = AuxDict['end']
                AuxDict = {'entity_group': Dict['entity_group'], 'word': new_word, 'start': new_start, 'end': new_end}
                # if hyphen as first character of sentence (not to be merged with previous entity)
                if ((string[prev_char]) == "-") and (prev_char != 0) and (string[prev_char - 1] == "\n"):
                    Aggregated.append(AuxDict)
                    AuxDict = {}
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
                    FinalDict = {'entity_group': Dict['entity_group'], 'word': Dict['word'], 'start': Dict['start'],
                                 'end': Dict['end']}
                else:
                    FinalDict = {'entity_group': Dict['entity_group'], 'word': Dict['word'] + AuxDict['word'],
                                 'start': Dict['start'], 'end': AuxDict['end']}
                Aggregated.append(FinalDict)
                AuxDict = {}
            else:
                Aggregated.append(Dict)

    # Reverse again before returning results
    FinalAggregated = [item for item in reversed(Aggregated)]

    return FinalAggregated


# Load the previously trained Transformers model using full path (no relative)
model_checkpoint = "/Users/Leonardo 1/Documents/Trabajo/nn-workspace/BERT-2022/transformers/token-classification/roberta-es-clinical-trials-umls-7sgs-ner"

# Transformers tokenizer  
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Token classifier
# aggregation_strategy="simple" -> errors in out of vocabulary words, which are split in subwords (e.g. "auto - medi - carse")
# aggregation_strategy="first" -> errors with entities wrongly tagged, very noisy 
umls_token_classifier = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple", tokenizer=tokenizer)


# list of exceptions and patterns to change according to task
EXCEPTIONS_LIST = "/Users/Leonardo 1/Documents/Trabajo/InterTalentumUAM2018/WP2/WP2-1/patterns/list_except.txt"

# Read exceptions and save to hash
ExceptionsDict = {}

# Read and process the exceptions
with open(EXCEPTIONS_LIST, 'r', newline='') as f:
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
                        ExceptionsDict[n] = {'pattern': TuplesList, 'finalLabel': finalLabel}


def update_offsets(List,offset,text):
    
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
        if (entity==candidate):
            NewList.append(dictionary)
        else:
            # Correct offsets
            new_start = new_start + 1
            new_end = new_end + 1
            dictionary['start'] = new_start
            dictionary['end'] = new_end
            candidate = text[new_start:new_end]
            if (entity==candidate):
                NewList.append(dictionary)
            else:
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


def remove_overlap_gui(Hash):
    '''
    Remove overlapped entities to be displayed on GUI:
        e.g. cardiaco, infarto cardiaco => infarto cardiaco
    Note that overlapped entities are output to the downloadable file
    '''

    InnerEnts = []
    ToDelete = []

    Dict = {}

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
                if ((Start_ToCompare1 >= Start_ToCompare2) and (End_ToCompare1 <= End_ToCompare2)) or (
                    (Start_ToCompare2 >= Start_ToCompare1) and (End_ToCompare2 <= End_ToCompare1)):
                    # Escape entities to avoid errors in characters such as "+", "(", etc.
                    ToCompare1 = re.escape(ToCompare1)
                    ToCompare2 = re.escape(ToCompare2)
                    m1 = re.search(ToCompare1, ToCompare2)
                    m2 = re.search(ToCompare2, ToCompare1)
                    if (m1) or (m2) or (ToCompare1 in ToCompare2) or (ToCompare2 in ToCompare1):
                        span1 = End_ToCompare1 - Start_ToCompare1
                        span2 = End_ToCompare2 - Start_ToCompare2
                        if (span2 > span1):
                            # If the label is the same as in the outer entity, the entity is removed
                            if (Tag_ToCompare1 == Tag_ToCompare2):
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
                # Overlapping entities (e.g. "trastorno de conducta" y "conducta suicida")
                elif (Start_ToCompare1 <= Start_ToCompare2) and (End_ToCompare1 <= End_ToCompare2) and (Start_ToCompare2 <= End_ToCompare1):
                    ToDelete.append(Dict[n])
            n = n - 1

    FinalEntities = {}

    for k in Dict:
        Ent = Dict[k]
        if (Ent not in ToDelete) and (Ent not in InnerEnts):
            FinalEntities[len(FinalEntities) + 1] = Ent

    # Nested entities (inner tags) are returned as a dictionary of dictionaries:
    # For entities with same offsets and multiple labels, output only one: "vitamina C" (PROC & CHEM), "vitamina" (PROC & CHEM)
    n = 0
    NestedEnts = {}
    for ent in InnerEnts:
        if ent not in ToDelete and ent not in NestedEnts.values():
            n += 1
            NestedEnts[n] = ent

    return FinalEntities, NestedEnts


def get_codes_from_lexicon_gui(entity, label, LexiconData, TuplesList):
    '''
    Get CUI data from lexicon using entity and semantic label.
    Data format in lexicon:
        {'dolor': [('DISO', ['C0030193', 'C0234238']), ('PHYS', ['C0018235'])]
    Data format of LexiconData match (TextSearch library):
        {'norm': [('DISO', ['C0030193', 'C0234238']), ('PHYS', ['C0018235']), 'exact': False, 'match': 'dolor', 'case': 'lower', 'start': 0, 'end': 6}
    '''

    Data = LexiconData.findall(entity)

    CUIsList = []

    # Apply exact match to avoid mapping of "signo" and "signo de infección"
    if (Data):
        for Dict in Data:
            if (Dict['match']) == entity:
                TuplesList = Dict['norm']
                for Tuple in TuplesList:
                    if (Tuple[0]) == label:
                        return sorted(Tuple[1])

    return None


def translate_label(label):
    '''
    Translate labels to be displayed
    '''

    LabelDict = {
        'ACTI': "Activity",
        'ANAT': "Anatomy",
        'CHEM': "Chemical or pharmacological substance",
        'CONC': "Concept",
        'DEVI': "Medical device",
        'DISO': "Pathologic condition",
        'GENE': "Gene",
        'LIVB': "Living Being",
        'OBJC': "Object",
        'OCCU': "Occupation or profession",
        'ORGA': "Organization",
        'PROC': "Procedure",
        'PHEN': "Phenomenon",
        'PHYS': 'Physiology',
        # Time expressions
        'Age': "Age",
        'Date': "Date",
        'Duration': "Duration",
        'Frequency': "Frecuency",
        'Time': 'Time or part of the day',
        # Drug attributes
        'Contraindicated': "Contraindicated",
        'Dose': "Dose, concentration or strength",
        'Form': "Dosage form",
        'Route': 'Administration route',
        # Negation and uncertainty
        'Neg_cue': 'Negation cue',
        'Negated': 'Negated',
        'Spec_cue': 'Speculation cue',
        'Speculated': 'Speculated'
    }
    return LabelDict[label]


def EntsDict2html(text, Hash, LexiconData, Nested, UMLSDataDict):
    '''
    Format text and convert dictionary of annotated entities to HTML.
    The Nested argument is optional, only when nested entities were annotated.
    '''

    # Converts string to list of characters
    stringChars = list(text)

    # Sort hash in reverse offset order
    SortedHash = {r: Hash[r] for r in sorted(Hash, key=Hash.get('start'), reverse=True)}

    for id_entity in SortedHash:

        start = SortedHash[id_entity]['start']
        end = SortedHash[id_entity]['end']
        ent = SortedHash[id_entity]['ent']
        label = SortedHash[id_entity]['label']
        # Convert integer to string
        id_entity = str(id_entity)

        norm = request.form.getlist("norm")
        if (norm):
            try:
                TuplesList = LexiconData.findall(ent)[0]['norm']
                CUIsList = get_codes_from_lexicon_gui(ent, label, LexiconData, TuplesList)
                if (CUIsList):
                    # Complete normalization data of UMLS CUIs
                    CUIsList = complete_norm_data(CUIsList, UMLSDataDict)
                    codes_string = " | ".join(CUIsList)
                    # HTML tag to write in string
                    open_tag = "<span class=\"" + label + "\" " + "title=\"" + translate_label(
                        label) + " - UMLS: " + codes_string + "\" id=\"flat" + id_entity + "\">"
                else:
                    # HTML tag to write in string
                    open_tag = "<span class=\"" + label + "\" " + "title=\"" + translate_label(
                        label) + "\" id=\"flat" + id_entity + "\">"
            except:
                # For temporal, negation or drug entities not in dictionary (cause errors)
                # HTML tag to write in string
                open_tag = "<span class=\"" + label + "\" " + "title=\"" + translate_label(
                    label) + "\" id=\"flat" + id_entity + "\">"

            # Search for nested entities
            if Nested and (len(Nested) > 0):
                # Sort hash of nested entities in reverse offset order
                SortedNested = {r: Nested[r] for r in sorted(Nested, key=Nested.get('start'), reverse=True)}
                for NestedEnt in SortedNested:

                    start_nested = SortedNested[NestedEnt]['start']
                    end_nested = SortedNested[NestedEnt]['end']
                    # Check if offsets are inside those of outer entity
                    if (start_nested >= start) and (end_nested <= end):
                        ent_nested = SortedNested[NestedEnt]['ent']
                        label_nested = SortedNested[NestedEnt]['label']
                        if (norm):
                            try:
                                TuplesList = LexiconData.findall(ent_nested)[0]['norm']
                                CUIsList = get_codes_from_lexicon_gui(ent_nested, label_nested, LexiconData, TuplesList)
                                if (CUIsList):
                                    # Complete normalization data of UMLS CUIs
                                    CUIsList = complete_norm_data(CUIsList, UMLSDataDict)
                                    codes_string = " | ".join(CUIsList)
                                    # HTML tag to write in string
                                    tag_nested = "<span class=\"" + label_nested + "\" " + "title=\"" + translate_label(
                                        label_nested) + " - UMLS: " + codes_string + "\" id=\"inner" + id_entity + "\">" + ent_nested + "</span>"
                                else:
                                    # HTML tag to write in string
                                    tag_nested = "<span class=\"" + label_nested + "\" " + "title=\"" + translate_label(
                                        label_nested) + "\" id=\"inner" + id_entity + "\">" + ent_nested + "</span>"
                            except:

                                tag_nested = "<span class=\"" + label_nested + "\" " + "title=\"" + translate_label(
                                    label_nested) + "\" id=\"inner" + id_entity + "\">" + ent_nested + "</span>"

                        # Converts inner string to list of characters and replace with inner tags using offsets
                        NestedStringChars = list(ent)

                        WordBoundaryList = [",", ";", ".", "?", "¿", ")", "(", " ", "-", "[", "]", "¡", "!", "%", "/",
                                            ":", "+", "<", ">", "≥", "≤"]

                        # Calculate offset position in inner entity (cannot use full string offsets)
                        start_inner = start_nested - start
                        end_inner = start_inner + len(ent_nested)


                        # Check if the inner entity is full word; otherwise, do not output
                        # Exceptions are negated, speculated or contraindicated entities (can be the full word)
                        if label_nested not in ['Negated', 'Speculated', 'Contraindicated']:
                            # Beginning of string
                            if ((start_inner == 0) and not (end_inner == len(ent))):
                                try:
                                    post_char = NestedStringChars[end_inner]
                                    if (post_char) not in WordBoundaryList:
                                        break
                                except:
                                    break
                            # End of string
                            elif ((end_inner == len(ent)) and not (start_inner == 0)):
                                try:
                                    prev_char = NestedStringChars[start_inner - 1]
                                    if (prev_char) not in WordBoundaryList:
                                        break
                                except:
                                    break
                            # Middle of string
                            else:
                                try:
                                    prev_char = NestedStringChars[start_inner - 1]
                                    post_char = NestedStringChars[end_inner]
                                    if ((post_char not in WordBoundaryList) or (prev_char not in WordBoundaryList)):
                                        break
                                except:
                                    break

                        NestedStringChars[start_inner:end_inner] = tag_nested

                        # Join again characters to string
                        ent = "".join(NestedStringChars)
        else:

            # HTML tag to write in string
            open_tag = "<span class=\"" + label + "\" " + "title=\"" + translate_label(
                label) + "\" id=\"flat" + id_entity + "\">"
            # Search for nested entities
            if Nested and (len(Nested) > 0):
                # Sort hash of nested entities in reverse offset order
                SortedNested = {r: Nested[r] for r in sorted(Nested, key=Nested.get('start'), reverse=True)}
                for NestedEnt in SortedNested:
                    start_nested = SortedNested[NestedEnt]['start']
                    end_nested = SortedNested[NestedEnt]['end']
                    # Check if offsets are inside those of outer entity
                    if (start_nested >= start) and (end_nested <= end):
                        ent_nested = SortedNested[NestedEnt]['ent']
                        label_nested = SortedNested[NestedEnt]['label']
                        tag_nested = "<span class=\"" + label_nested + "\" " + "title=\"" + translate_label(
                            label_nested) + "\" id=\"inner" + id_entity + "\">" + ent_nested + "</span>"

                        # Converts inner string to list of characters and replace with inner tags using offsets
                        NestedStringChars = list(ent)
                        WordBoundaryList = [",", ";", ".", "?", "¿", ")", "(", " ", "-", "[", "]", "¡", "!", "%", "/",
                                            ":", "+", "<", ">", "≥", "≤"]

                        # Calculate offset position in inner entity (cannot use full string offsets)
                        start_inner = start_nested - start
                        end_inner = start_inner + len(ent_nested)

                        # Check if the inner entity is full word; otherwise, do not output
                        # Exceptions are negated, speculated or contraindicated entities (can be the full word)
                        if label_nested not in ['Negated', 'Speculated', 'Contraindicated']:
                            # Beginning of string
                            if ((start_inner == 0) and not (end_inner == len(ent))):
                                try:
                                    post_char = NestedStringChars[end_inner]
                                    if (post_char) not in WordBoundaryList:
                                        break
                                except:
                                    break
                            # End of string
                            elif ((end_inner == len(ent)) and not (start_inner == 0)):
                                try:
                                    prev_char = NestedStringChars[start_inner - 1]
                                    if (prev_char) not in WordBoundaryList:
                                        break
                                except:
                                    break
                            # Middle of string
                            elif ((start_inner > 0) and (end_inner < len(ent))):
                                try:
                                    prev_char = NestedStringChars[start_inner - 1]
                                    post_char = NestedStringChars[end_inner]
                                    if ((post_char not in WordBoundaryList) or (prev_char not in WordBoundaryList)):
                                        break
                                except:
                                    break

                        NestedStringChars[start_inner:end_inner] = tag_nested
                        # Join again characters to string
                        ent = "".join(NestedStringChars)

        close_tag = "</span>"
        # Final replacement in original string
        stringChars[start:end] = open_tag + ent + close_tag

    # Join again characters to string
    string = "".join(stringChars)

    return string


app = Flask(__name__)


@app.route('/gui_en', methods=['GET', 'POST'])
def gui_tf():
    if request.method == 'POST':

        # Save all flat entities
        AllFlatEnts = {}

        # Save all nested entities
        AllNestedEnts = {}

        # Save the annotated UMLS entities with lexicon or neural model
        Entities = {}

        # Get text from input form
        text = request.form['text']
        
        # Removes first white character of line
        text = text.lstrip()

        # Replace "\r" to "\n" throughtout the string (COMPROBAR QUE NO DA ERROR)
        text = re.sub("\r","",text)

        # If normalization data is needed, load file
        norm = request.form.getlist("norm")
        UMLSData = {}
        if (norm):
            DataFile = open("../lexicon/umls_data.pickle", 'rb')
            UMLSData = pickle.load(DataFile)

        # Split text into sentences
        offset = 0

        Sentences = sentences_spacy(text)

        LexiconData, POSData = read_lexicon("../lexicon/MedLexSp.pickle")

        # Text needs to be tokenized for recognizing time expressions, negation or drug scheme
        Tokens = tokenize_spacy_text(text, POSData)

        # Annotation of nested entities
        nest = request.form.getlist("nest")

        # If usage of lexicon
        lex = request.form.getlist("lex")
        if (lex):
            #  If annotation of nested entities
            if (nest):
                print("Annotating UMLS entities (flat and nested) with lexicon...")
                Entities, NestedEnts = apply_lexicon(text, LexiconData, nest)
            else:
                # Annotate and extract entities
                print("Annotating UMLS entities using lexicon...")
                Entities, NestedEnts = apply_lexicon(text, LexiconData)

            AllFlatEnts = merge_dicts(AllFlatEnts,Entities)
            AllNestedEnts = merge_dicts(AllNestedEnts, NestedEnts)

        # if annotation of UMLS entities with neural model
        neu = request.form.getlist("neu")
        if (neu):

            # Usage of transformers neural model
            print("Annotating using transformers neural model for UMLS entities...")

            Output = annotate_sentences_with_model(Sentences,text,umls_token_classifier)

            # Change format to:
            #   Entities = {1: {'start': 3, 'end': 11, 'ent': 'COVID-19', 'label': 'DISO'}, 2: ... }
            for i, Ent in enumerate(Output):
                Entities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'], 'label': Ent['entity_group']}

            AllFlatEnts = merge_dicts(AllFlatEnts,Entities)

        # Annotation of temporal expressions
        temp = request.form.getlist("temp")
        if (temp):

            print("Annotating using transformers neural model for temporal entities...")
            # Load the previously trained Transformers model using full path (no relative)
            temp_model_checkpoint = "/Users/Leonardo " \
                                    "1/Documents/Trabajo/nn-workspace/BERT-2022/transformers/token-classification" \
                                    "/roberta-es-clinical-trials-temporal-ner"

            # Transformers tokenizer
            tokenizer = AutoTokenizer.from_pretrained(temp_model_checkpoint)

            # Token classifier
            temp_token_classifier = pipeline("token-classification", model=temp_model_checkpoint,
                                             aggregation_strategy="simple", tokenizer=tokenizer)

            TempOutput = annotate_sentences_with_model(Sentences, text, temp_token_classifier)

            # Save the annotated entities with the final format
            TempEntities = {}

            # Change format
            for i, Ent in enumerate(TempOutput):
                TempEntities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'],
                                   'label': Ent['entity_group']}

            # Merge all entities
            AllFlatEnts = merge_dicts(AllFlatEnts, TempEntities)


        # Annotation of drug features
        drg = request.form.getlist("drg")
        if (drg):

            print("Annotating using transformers neural model for drug information...")
            # Load the previously trained Transformers model using full path (no relative)
            medic_attr_model_checkpoint = "/Users/Leonardo " \
                                          "1/Documents/Trabajo/nn-workspace/BERT-2022/transformers/token-classification" \
                                          "/roberta" \
                                          "-es-clinical-trials-medic-attr-ner"

            # Transformers tokenizer
            tokenizer = AutoTokenizer.from_pretrained(medic_attr_model_checkpoint)

            # Token classifier
            medic_attr_token_classifier = pipeline("token-classification", model=medic_attr_model_checkpoint,
                                                   aggregation_strategy="simple", tokenizer=tokenizer)

            MedicAttrOutput = annotate_sentences_with_model(Sentences, text, medic_attr_token_classifier)

            # Save the annotated entities with the final format
            MedicAttrEntities = {}

            # Change format to:
            #   Entities = {1: {'start': 3, 'end': 11, 'ent': 'COVID-19', 'label': 'DISO'}, 2: ... }
            for i, Ent in enumerate(MedicAttrOutput):
                MedicAttrEntities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'],
                                        'label': Ent['entity_group']}

            # Merge all entities
            AllFlatEnts = merge_dicts(AllFlatEnts, MedicAttrEntities)


        # Annotation of entities expressing negation
        neg = request.form.getlist("neg")
        if (neg):

            # Load the previously trained Transformers model using full path (no relative)
            print("Annotating using transformers neural model for negation and speculation...")
            neg_spec_model_checkpoint = "/Users/Leonardo " \
                                        "1/Documents/Trabajo/nn-workspace/BERT-2022/transformers/token-classification" \
                                        "/roberta" \
                                        "-es-clinical-trials-neg-spec-ner"

            # Transformers tokenizer
            tokenizer = AutoTokenizer.from_pretrained(neg_spec_model_checkpoint)

            # Token classifier
            neg_spec_token_classifier = pipeline("token-classification", model=neg_spec_model_checkpoint,
                                                 aggregation_strategy="simple", tokenizer=tokenizer)

            NegSpecOutput = annotate_sentences_with_model(Sentences, text, neg_spec_token_classifier)

            # Save the annotated entities with the final format
            NegSpecEntities = {}

            # Change format
            for i, Ent in enumerate(NegSpecOutput):
                NegSpecEntities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'],
                                      'label': Ent['entity_group']}

            # Merge all entities
            AllFlatEnts = merge_dicts(AllFlatEnts, NegSpecEntities)

        # Remove entities defined in an exception list (it could be selected with a parameter: (exc))
        AllFlatEnts = remove_entities(AllFlatEnts, Tokens, ExceptionsDict, 'label', AllNestedEnts)
        # If nested entities, remove from layer 2 the entities defined in an exception list
        #  This is needed to change CHEM to ANAT in "sangrado [digestivo]"
        NestedEntsCleaned = {}
        if (nest):
            NestedEntsCleaned = remove_entities(AllNestedEnts, Tokens, ExceptionsDict, 'label', AllNestedEnts)

        # Text showed in screen
        # Flat entities
        EntitiesNoOverlap, NestedEntities = remove_overlap_gui(AllFlatEnts)

        # Remove nested entities with same offset, to avoir error in gui
        NestedEntsCleaned = merge_dicts(NestedEntities, NestedEntsCleaned)
        NestedEntsCleaned, NestedEntities = remove_overlap_gui(NestedEntsCleaned)

        text = Markup(EntsDict2html(text, EntitiesNoOverlap, LexiconData, NestedEntsCleaned, UMLSData))
            

        # Annotated entities to download in BRAT format
        annot = ""
        n_comm = 0
        
        if (nest):
            # Merge both dictionaries of nesting (outer) and nested entities to output in BRAT format
            AllFinalEntities = merge_dicts(AllFlatEnts,AllNestedEnts)
        else:
            AllFinalEntities = AllFlatEnts

        for i in AllFinalEntities:

            # T#  Annotation  Start   End String
            line = "T{}\t{} {} {}\t{}".format(i, AllFinalEntities[i]['label'], AllFinalEntities[i]['start'], AllFinalEntities[i]['end'],
                                              AllFinalEntities[i]['ent'])
            annot = annot + line + "\n"
            # If normalization to UMLS CUIs
            norm = request.form.getlist("norm")
            if (norm):

                try:
                    TuplesList = LexiconData.findall(AllFinalEntities[i]['ent'])[0]['norm']

                    CUIsList = get_codes_from_lexicon_gui(AllFinalEntities[i]['ent'], AllFinalEntities[i]['label'], LexiconData,
                                                          TuplesList)

                    if (CUIsList):
                        # Complete normalization data of UMLS CUIs
                        CUIsList = complete_norm_data(CUIsList, UMLSData)
                        n_comm += 1
                        codes_string = " | ".join(CUIsList)
                        line = "#{}	AnnotatorNotes T{}	{}".format(n_comm, i, codes_string)
                        annot = annot + line + "\n"
                except:
                    # For temporal, negation or other entities without a CUI in lexicon
                    pass

    else:

        text = ""
        annot = ""

    return render_template('gui_en.html', results=text, ann_data=annot)


if __name__ == '__main__':
    app.run(debug=True)
