#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask, request, render_template, Markup

# Import annotation modules
import sys

sys.path.append("../")  # Adds higher directory to python modules path.

import re
import argparse

import spacy
import spacy_tokenizer
from spacy_tokenizer import *

import lexicon_tools
from lexicon_tools import *

import annot_utils
from annot_utils import *

import pickle

# Deep learning libraries
import transformers
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer, pipeline
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# UMLS entities
# Load the previously trained Transformers model using full path (no relative)
model_checkpoint = "../models/roberta-es-clinical-trials-umls-7sgs-ner"

# Transformers tokenizer  
umls_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# UMLS entities token classifier
umls_token_classifier = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# Temporal information entities
# Load the previously trained Transformers model using full path (no relative)
temp_model_checkpoint = "../models/roberta-es-clinical-trials-temporal-ner"

# Transformers tokenizer
temp_tokenizer = AutoTokenizer.from_pretrained(temp_model_checkpoint)

# Token classifier
temp_token_classifier = AutoModelForTokenClassification.from_pretrained(temp_model_checkpoint)

# Medication information
# Load the previously trained Transformers model using full path (no relative)
medic_attr_model_checkpoint = "../models/roberta-es-clinical-trials-medic-attr-ner"

# Transformers tokenizer
medic_attr_tokenizer = AutoTokenizer.from_pretrained(medic_attr_model_checkpoint)

# Token classifier
medic_attr_token_classifier = AutoModelForTokenClassification.from_pretrained(medic_attr_model_checkpoint)

# Miscellaneous medical entities
# Load the previously trained Transformers model using full path (no relative)
misc_ents_model_checkpoint = "../models/roberta-es-clinical-trials-misc-ents-ner"

# Transformers tokenizer
misc_ents_tokenizer = AutoTokenizer.from_pretrained(misc_ents_model_checkpoint)

# Token classifier
misc_ents_token_classifier = AutoModelForTokenClassification.from_pretrained(misc_ents_model_checkpoint)

# Negation and speculation
# Load the previously trained Transformers model using full path (no relative)
neg_spec_model_checkpoint = "../models/roberta-es-clinical-trials-neg-spec-ner"

# Transformers tokenizer
neg_spec_tokenizer = AutoTokenizer.from_pretrained(neg_spec_model_checkpoint)

# Token classifier
neg_spec_token_classifier = AutoModelForTokenClassification.from_pretrained(neg_spec_model_checkpoint)

# Experiencer and temporality attributes
# Load the previously trained Transformers model using full path (no relative)
attributes_model_checkpoint = "../models/roberta-es-clinical-trials-attributes-ner"

# Transformers tokenizer
attributes_tokenizer = AutoTokenizer.from_pretrained(attributes_model_checkpoint)

# Token classifier
attributes_token_classifier = AutoModelForTokenClassification.from_pretrained(attributes_model_checkpoint)


# list of exceptions and patterns to change according to task
EXCEPTIONS_LIST = "../patterns/list_except.txt"

# Read exceptions and save to hash
ExceptionsDict = read_exceptions_list(EXCEPTIONS_LIST)
    

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

                if ((Start_ToCompare1 >= Start_ToCompare2) and (End_ToCompare1 <= End_ToCompare2)) or ((Start_ToCompare2 >= Start_ToCompare1) and (End_ToCompare2 <= End_ToCompare1)):

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
                            # Keep only entities expressing assertion, experiencer or event temporality
                            if (Dict[n] not in InnerEnts) and (Tag_ToCompare2=='Negated'):
                                InnerEnts.append(Dict[n])
                            elif (Dict[n] not in InnerEnts) and (Tag_ToCompare2=='Speculated'):
                                InnerEnts.append(Dict[n])
                            elif (Dict[n] not in InnerEnts) and (Tag_ToCompare2=='Contraindicated'):
                                InnerEnts.append(Dict[n])
                            elif (Dict[n] not in InnerEnts) and (Tag_ToCompare2=='Future'):
                                InnerEnts.append(Dict[n])
                            elif (Dict[n] not in InnerEnts) and (Tag_ToCompare2=='History_of'):
                                InnerEnts.append(Dict[n])
                            elif (Dict[n] not in InnerEnts) and (Tag_ToCompare2=='Patient'):
                                InnerEnts.append(Dict[n])
                            elif (Dict[n] not in InnerEnts) and (Tag_ToCompare2=='Family_member'):
                                InnerEnts.append(Dict[n])
                            elif (Dict[n] not in InnerEnts) and (Tag_ToCompare2=='Other'):
                                InnerEnts.append(Dict[n])
                            elif (Dict[n] not in InnerEnts):
                                ToDelete.append(Dict[n])
                # Overlapping entities (e.g. "trastorno de conducta" y "conducta suicida")
                elif (Start_ToCompare1 <= Start_ToCompare2) and (End_ToCompare1 <= End_ToCompare2) and (Start_ToCompare2 <= End_ToCompare1):                    
                    # if not a juxtaposed entity
                    if (Start_ToCompare2 != End_ToCompare1):
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


def convert2brat_gui(Hash,LexiconData,UMLSData):

    ''' Convert a hash of entities to BRAT format. 
        LexiconData and UMLSData are optional parameters (if normalization is selected).
        Variant of the convert2brat function, used for the GUI.
    '''
    
    n_comm = 0
    n_att = 0

    annot = ""

    for i in Hash:
        # T#  Annotation  Start   End String
        line = "T{}\t{} {} {}\t{}".format(i, Hash[i]['label'], Hash[i]['start'], Hash[i]['end'], Hash[i]['ent'])
        annot = annot + line + "\n"
        # Attributes with format: A#	AttributeType T# Value
        if 'assertion' in Hash[i].keys():
            n_att += 1
            line = "A{}\tAssertion T{} {}".format(n_att, i, Hash[i]['assertion'])
            annot = annot + line + "\n"
        if 'event_temp' in Hash[i].keys():
            n_att += 1
            line = "A{}\tStatus T{} {}".format(n_att, i, Hash[i]['event_temp'])
            annot = annot + line + "\n"
        if 'experiencer' in Hash[i].keys():
            n_att += 1
            line = "A{}\tExperiencer T{} {}".format(n_att, i, Hash[i]['experiencer'])
            annot = annot + line + "\n"
        if 'attribute' in Hash[i].keys():
            n_att += 1
            if Hash[i]['attribute'] == 'Age':
                line = "A{}\tPopulation_data T{} {}".format(n_att, i, Hash[i]['attribute']) 
                annot = annot + line + "\n"  
            if Hash[i]['attribute'] == 'Contraindicated':
                line = "A{}\tAssertion T{} {}".format(n_att, i, Hash[i]['attribute'])
                annot = annot + line + "\n"
        # Print UMLS codes in additional comment
        if LexiconData and UMLSData:
            CUIsList = get_codes_from_lexicon(Hash[i]['ent'], Hash[i]['label'], LexiconData)
            if (CUIsList):
                # Complete normalization data of UMLS CUIs
                CUIsList = complete_norm_data(CUIsList,UMLSData)
                n_comm += 1
                codes_string = " | ".join(CUIsList)
                line = "#{}	AnnotatorNotes T{}	{}".format(n_comm,i,codes_string)
                annot = annot + line + "\n"
    
    return annot


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
        'ACTI': "Actividad",
        'ANAT': "Anatomía",
        'CHEM': "Sustancia química o farmacológica",
        'CONC': "Concepto",
        'DEVI': "Instrumento médico",
        'DISO': "Patología",
        'GENE': "Gen",
        'LIVB': "Ser vivo",
        'OBJC': "Objeto",
        'OCCU': "Ocupación o profesión",
        'ORGA': "Organización",
        'PROC': "Procedimiento",
        'PHEN': "Fenómeno",
        'PHYS': 'Fisiología',
        # Time expressions
        'Age': "Edad",
        'Date': "Fecha / punto temporal indefinido",
        'Duration': "Duración",
        'Frequency': "Frecuencia",
        'Time': 'Hora o parte del día',
        # Medical drug information
        'Contraindicated': "Contraindicado",
        'Dose': "Dosis o concentración",
        'Form': "Forma de presentación",
        'Route': 'Vía de administración',
        # Miscellaneous medical entities
        'Food': "Alimento o bebida",
        'Observation': "Observación/Hallazgo",
        'Quantifier_or_Qualifier': "Calificador o cuantificador",
        'Result_or_Value': 'Resultado o valor',
        # Negation and uncertainty
        'Neg_cue': 'Marca de negación',
        'Negated': 'Negado',
        'Spec_cue': 'Marca de especulación',
        'Speculated': 'Especulado',
        # Attributes
        'Patient': 'Paciente',
        'Family_member': 'Miembro de la familia',
        'Other': 'Otro tipo de persona',
        'History_of': 'Antecedente médico',
        'Future': 'Evento futuro'
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
                    #open_tag = "<span class=\"" + label + "\" " + "data-title=\"" + translate_label(label) + " - UMLS: " + codes_string + "\" id=\"flat" + id_entity + "\">"
                    open_tag = "<mark class=\"" + label + "\" " + "data-title=\"" + translate_label(label) + " - UMLS: " + codes_string + "\" id=\"flat" + id_entity + "\">"
                else:
                    # HTML tag to write in string
                    #open_tag = "<span class=\"" + label + "\" " + "data-title=\"" + translate_label(label) + "\" id=\"flat" + id_entity + "\">"
                    open_tag = "<mark class=\"" + label + "\" " + "data-title=\"" + translate_label(label) + "\" id=\"flat" + id_entity + "\">"
            except:
                # For temporal, negation or drug entities not in dictionary (cause errors)
                # HTML tag to write in string
                #open_tag = "<span class=\"" + label + "\" " + "data-title=\"" + translate_label(label) + "\" id=\"flat" + id_entity + "\">"
                open_tag = "<mark class=\"" + label + "\" " + "data-title=\"" + translate_label(label) + "\" id=\"flat" + id_entity + "\">"

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
                                    #tag_nested = "<span class=\"" + label_nested + "\" " + "data-title=\"" + translate_label(label_nested) + " - UMLS: " + codes_string + "\" id=\"inner" + id_entity + "\">" + ent_nested + "</span>"
                                    tag_nested = "<mark class=\"" + label_nested + "\" " + "data-title=\"" + translate_label(label_nested) + " - UMLS: " + codes_string + "\" id=\"inner" + id_entity + "\">" + ent_nested + "<span class=\"inner_title\">" + label_nested + "</span></mark>"
                                else:
                                    # HTML tag to write in string
                                    #tag_nested = "<span class=\"" + label_nested + "\" " + "data-title=\"" + translate_label(label_nested) + "\" id=\"inner" + id_entity + "\">" + ent_nested + "</span>"
                                    tag_nested = "<mark class=\"" + label_nested + "\" " + "data-title=\"" + translate_label(label_nested) + "\" id=\"inner" + id_entity + "\">" + ent_nested + "<span class=\"inner_title\">" + label_nested + "</span></mark>"
                            except:

                                #tag_nested = "<span class=\"" + label_nested + "\" " + "data-title=\"" + translate_label(label_nested) + "\" id=\"inner" + id_entity + "\">" + ent_nested + "</span>"
                                tag_nested = "<mark class=\"" + label_nested + "\" " + "data-title=\"" + translate_label(label_nested) + "\" id=\"inner" + id_entity + "\">" + ent_nested + "<span class=\"inner_title\">" + label_nested + "</span></mark>"

                        # Converts inner string to list of characters and replace with inner tags using offsets
                        NestedStringChars = list(ent)

                        WordBoundaryList = [",", ";", ".", "?", "¿", ")", "(", " ", "-", "[", "]", "¡", "!", "%", "/",
                                            ":", "+", "<", ">", "≥", "≤"]

                        # Calculate offset position in inner entity (cannot use full string offsets)
                        start_inner = start_nested - start
                        end_inner = start_inner + len(ent_nested)


                        # Check if the inner entity is full word; otherwise, do not output
                        # Exceptions are entities expressing assertion, event temporality or experiencer (can be the full word)
                        if label_nested not in ['Negated', 'Speculated', 'Contraindicated', 'Future', 'History_of', 'Patient', 'Family_member', 'Other']:
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
            #open_tag = "<span class=\"" + label + "\" " + "data-title=\"" + translate_label(label) + "\" id=\"flat" + id_entity + "\">"
            open_tag = "<mark class=\"" + label + "\" " + "data-title=\"" + translate_label(label) + "\" id=\"flat" + id_entity + "\">"
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
                        #tag_nested = "<span class=\"" + label_nested + "\" " + "data-title=\"" + translate_label(label_nested) + "\" id=\"inner" + id_entity + "\">" + ent_nested + "</span>"
                        tag_nested = "<mark class=\"" + label_nested + "\" " + "data-title=\"" + translate_label(label_nested) + "\" id=\"inner" + id_entity + "\">" + ent_nested + "<span class=\"inner_title\">" + label_nested + "</span></mark>"

                        # Converts inner string to list of characters and replace with inner tags using offsets
                        NestedStringChars = list(ent)
                        WordBoundaryList = [",", ";", ".", "?", "¿", ")", "(", " ", "-", "[", "]", "¡", "!", "%", "/",
                                            ":", "+", "<", ">", "≥", "≤"]

                        # Calculate offset position in inner entity (cannot use full string offsets)
                        start_inner = start_nested - start
                        end_inner = start_inner + len(ent_nested)

                        # Check if the inner entity is full word; otherwise, do not output
                        # Exceptions are entities expressing assertion, event temporality or experiencer (can be the full word)
                        if label_nested not in ['Negated', 'Speculated', 'Contraindicated', 'Future', 'History_of', 'Patient', 'Family_member', 'Other']:
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

        #close_tag = "</span>"
        title_tag = "<span class=\"inner_title\">" + label + "</span>"
        close_tag = "</mark>"
        # Final replacement in original string
        #stringChars[start:end] = open_tag + ent + close_tag
        stringChars[start:end] = open_tag + ent + title_tag + close_tag

    # Join again characters to string
    string = "".join(stringChars)

    return string


app = Flask(__name__)


@app.route('/gui', methods=['GET', 'POST'])
def gui():
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

        # Replace "\r" to "\n" throughtout the string
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

            Output = annotate_sentences_with_model(Sentences, text, umls_token_classifier, umls_tokenizer, device)

            # Change format to:
            #   Entities = {1: {'start': 3, 'end': 11, 'ent': 'COVID-19', 'label': 'DISO'}, 2: ... }
            for i, Ent in enumerate(Output):
                Entities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'], 'label': Ent['entity_group']}

            AllFlatEnts = merge_dicts(AllFlatEnts,Entities)
            # In case of overlap
            AllFlatEnts, NestedEntities = remove_overlap(AllFlatEnts)
            AllNestedEnts = merge_dicts(AllNestedEnts, NestedEntities)

        # Annotation of temporal expressions
        temp = request.form.getlist("temp")
        if (temp):

            print("Annotating using transformers neural model for temporal entities...")
            
            TempOutput = annotate_sentences_with_model(Sentences, text, temp_token_classifier, temp_tokenizer, device)

            # Save the annotated entities with the final format
            TempEntities = {}

            # Change format
            for i, Ent in enumerate(TempOutput):
                TempEntities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'], 'label': Ent['entity_group']}
            
            # Merge all entities
            AllFlatEnts = merge_dicts(AllFlatEnts, TempEntities)

        # Annotation of drug features
        drg = request.form.getlist("drg")
        if (drg):

            print("Annotating using transformers neural model for drug information...")

            MedicAttrOutput = annotate_sentences_with_model(Sentences, text, medic_attr_token_classifier, medic_attr_tokenizer, device)

            # Save the annotated entities with the final format
            MedicAttrEntities = {}

            # Change format to:
            #   Entities = {1: {'start': 3, 'end': 11, 'ent': 'COVID-19', 'label': 'DISO'}, 2: ... }
            for i, Ent in enumerate(MedicAttrOutput):
                MedicAttrEntities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'], 'label': Ent['entity_group']}

            # Merge all entities
            AllFlatEnts = merge_dicts(AllFlatEnts, MedicAttrEntities)

        # Annotation of miscellaneous medical entities
        misc = request.form.getlist("misc")
        if (misc):

            print("Annotating using transformers neural model for miscellaneous medical entities...")

            MiscOutput = annotate_sentences_with_model(Sentences, text, misc_ents_token_classifier, misc_ents_tokenizer, device)

            # Save the annotated entities with the final format
            MiscEntities = {}

            # Change format
            for i, Ent in enumerate(MiscOutput):
                MiscEntities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'], 'label': Ent['entity_group']}

            # Merge all entities
            AllFlatEnts = merge_dicts(AllFlatEnts, MiscEntities)
            
        # Annotation of entities expressing negation and speculation
        neg = request.form.getlist("neg")
        if (neg):

            print("Annotating using transformers neural model for negation and speculation...")

            NegSpecOutput = annotate_sentences_with_model(Sentences, text, neg_spec_token_classifier, neg_spec_tokenizer, device)

            # Save the annotated entities with the final format
            NegSpecEntities = {}

            # Change format
            for i, Ent in enumerate(NegSpecOutput):
                NegSpecEntities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'], 'label': Ent['entity_group']}

            # Merge all entities
            AllFlatEnts = merge_dicts(AllFlatEnts, NegSpecEntities)

        # Annotation of experiencer and event temporality attributes
        att = request.form.getlist("att")
        if (att):

            print("Annotating using transformers neural model for experiencer and temporality attributes...")
            
            AttrOutput = annotate_sentences_with_model(Sentences, text, attributes_token_classifier, attributes_tokenizer, device)

            # Save the annotated entities with the final format
            Attributes = {}

            # Change format
            for i, Ent in enumerate(AttrOutput):
                Attributes[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'], 'label': Ent['entity_group']}
            
            # Merge all entities
            AllFlatEnts = merge_dicts(AllFlatEnts, Attributes)

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

        # Remove nested entities with same offset, to avoid error in gui
        NestedEntsCleaned = merge_dicts(NestedEntities, NestedEntsCleaned)
        NestedEntsCleaned, NestedEntities = remove_overlap_gui(NestedEntsCleaned)
        
        # Remove nested entities with same label for gui
        NestedEntsCleaned = remove_nested_entity(EntitiesNoOverlap, NestedEntsCleaned)
        text = Markup(EntsDict2html(text, EntitiesNoOverlap, LexiconData, NestedEntsCleaned, UMLSData))
            

        # Annotated entities to download in BRAT format
        annot = ""
        n_comm = 0
        
        if (nest):
            # Merge both dictionaries of nesting (outer) and nested entities to output in BRAT format
            AllFinalEntities = merge_dicts(AllFlatEnts,AllNestedEnts)
        else:
            AllFinalEntities = AllFlatEnts

        FinalHash = codeAttribute(AllFinalEntities)

        if (norm):
            annot = convert2brat_gui(FinalHash,LexiconData,UMLSData)
        else:
            annot = convert2brat_gui(FinalHash,None,None)

    else:

        text = ""
        annot = ""

    return render_template('gui.html', results=text, ann_data=annot)


if __name__ == '__main__':
    app.run(debug=True)
