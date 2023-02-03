#!/usr/bin/python3
# -*- coding: UTF-8 -*-
#
# medspaner.py
#
# Medical Semantic python-assisted Named Entity Recognizer using dictionary and pre-trained transformer models.
#
# Usage:
#
#   python medspaner.py -conf <CONFIG_FILE> -input <FILE>
# 
#
#########################################################################

import re
import os
import sys
import spacy
import annot_utils
from annot_utils import *
import argparse
import spacy_tokenizer
from spacy_tokenizer import *
import lexicon_tools
from lexicon_tools import *
import pickle

# Deep learning utilities
# Transformers
import transformers
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer, pipeline
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # If CUDA is needed
device = torch.device("cpu")

# Parser for config file
import configparser
 
# Command line arguments
parser = argparse.ArgumentParser(description='Given a text file, annotate it with semantic labels')
parser.add_argument('-conf', required=False, help='use configuration file to parse arguments (optional)')
parser.add_argument('-drg', required=False, default=False, action='store_true', help='annotate drug features such as dose (not annotated by default)')
parser.add_argument('-exc', required=False, default=False, help='use a list of exceptions of entity types not to be annotated, or word patterns to re-label')
parser.add_argument('-input', required=True, default=False, help='specify file to annotate')
parser.add_argument('-lex', required=False, help='specify pickle object dictionary for preannotation (neural model is used by default); indicate using: -lex "lexicon/MedLexSp.pickle"')
parser.add_argument('-neu', required=False, default=True, action='store_true', help='specify to annotate UMLS entities with neural model (used by default)')
parser.add_argument('-neg', required=False, default=False, action='store_true', help='annotate entities expressing negation and uncertainty (not annotated by default)')
parser.add_argument('-nest', required=False, default=False, action='store_true', help='annotate inner or nested entities inside wider entities (not annotated by default)')
parser.add_argument('-norm', required=False, default=False, action='store_true', help='normalize entities and output terminology codes from UMLS (optional)')
parser.add_argument('-out', required=False, default="ann", help='specify output to JSON format ("json") or BRAT ann format ("ann", default value)')
parser.add_argument('-temp', required=False, default=False, action='store_true', help='annotate temporal expressions (not annotated by default)')

args = parser.parse_args()


def main(arguments):

    config = configparser.ConfigParser()

    # Use configuration parameters from file; if there is not config file, use command line arguments
    if (arguments.conf):
        
        config.read(arguments.conf)
        for arg in (config['config']):
            if config['config'][arg].lower() in ["true", "false"]:
                # Use .getboolean() to parse True / False values as boolean types (not strings)
                setattr(args, arg, config['config'].getboolean(arg.lower()))
            else:
                setattr(args, arg, config['config'][arg])

    # Validate arguments
    arguments.out = arguments.out.lower()
    if ((arguments.out) != "ann") and ((arguments.out) != "json"):
        print('Please, provide the output format: "json" or "ann" (BRAT annotated file)')
        parser.print_help()
        sys.exit()
    
    # If normalization data is needed, load file
    if (arguments.norm):
        DataFile = open("lexicon/umls_data.pickle", 'rb')
        UMLSData = pickle.load(DataFile)

    # If use of an exception list to remove specific entities 
    if (arguments.exc):
        
        # Read exceptions and save to hash
        ExceptionsDict = read_exceptions_list(arguments.exc)

    text = arguments.input
    
    # Load lexicon data
    LexiconData, POSData = read_lexicon("lexicon/MedLexSp.pickle")

    Tokens = tokenize_spacy(text,POSData)

    with open(arguments.input,'r', newline = '') as f:

        # Save all flat entities
        AllFlatEnts = {}
        
        # Save all nested entities
        AllNestedEnts = {}

        text = f.read()

        # Split text into sentences
        Sentences = sentences_spacy(text)
            
        # Use lexicon
        if (arguments.lex):

            #  If annotation of nested entities        
            if (arguments.nest):
                print("Annotating UMLS entities (flat and nested) with lexicon...")
                Entities,NestedEnts = apply_lexicon(text,LexiconData,arguments.nest)               
            else:
                print("Annotating UMLS entities with lexicon...")
                Entities,NestedEnts = apply_lexicon(text,LexiconData)
            
            AllFlatEnts = merge_dicts(AllFlatEnts,Entities)
            AllNestedEnts = merge_dicts(AllNestedEnts, NestedEnts)

        # Use transformers model to annotate UMLS entities
        if (arguments.neu):

            print("Annotating using transformers neural model for UMLS entities...")
            # Load the previously trained Transformers model using full path (no relative)
            umls_model_checkpoint = "models/roberta-es-clinical-trials-umls-7sgs-ner"
    
            # Transformers tokenizer
            tokenizer = AutoTokenizer.from_pretrained(umls_model_checkpoint)
    
            # Token classifier for UMLS entities
            umls_token_classifier = AutoModelForTokenClassification.from_pretrained(umls_model_checkpoint)
    
            Output = annotate_sentences_with_model(Sentences,text,umls_token_classifier,tokenizer,device)
    
            # Save the annotated entities with the final format
            Entities = {}
    
            # Change output format
            for i, Ent in enumerate(Output):
                Entities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'], 'label': Ent['entity_group']}
            
            AllFlatEnts = merge_dicts(AllFlatEnts,Entities)
            # In case of overlap
            AllFlatEnts, NestedEntities = remove_overlap(AllFlatEnts)
            AllNestedEnts = merge_dicts(AllNestedEnts, NestedEntities)

        # Annotation of temporal expressions
        if (arguments.temp):

            print("Annotating using transformers neural model for temporal entities...")
            # Load the previously trained Transformers model
            temp_model_checkpoint = "models/roberta-es-clinical-trials-temporal-ner"

            # Transformers tokenizer
            tokenizer = AutoTokenizer.from_pretrained(temp_model_checkpoint)

            # Token classifier
            temp_token_classifier = AutoModelForTokenClassification.from_pretrained(temp_model_checkpoint)

            TempOutput = annotate_sentences_with_model(Sentences,text,temp_token_classifier,tokenizer,device)

            # Save the annotated entities with the final format
            TempEntities = {}

            # Change format
            for i, Ent in enumerate(TempOutput):
                TempEntities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'], 'label': Ent['entity_group']}

            # Merge all entities
            AllFlatEnts = merge_dicts(AllFlatEnts, TempEntities)


        # Annotation of drug features
        if (arguments.drg):

            print("Annotating using transformers neural model for drug information...")
            # Load the previously trained Transformers model
            medic_attr_model_checkpoint = "models/roberta-es-clinical-trials-medic-attr-ner"

            # Transformers tokenizer
            tokenizer = AutoTokenizer.from_pretrained(medic_attr_model_checkpoint)

            # Token classifier
            medic_attr_token_classifier = AutoModelForTokenClassification.from_pretrained(medic_attr_model_checkpoint)

            MedicAttrOutput = annotate_sentences_with_model(Sentences, text, medic_attr_token_classifier,tokenizer,device)

            # Save the annotated entities with the final format
            MedicAttrEntities = {}

            # Change format to:
            #   Entities = {1: {'start': 3, 'end': 11, 'ent': 'COVID-19', 'label': 'DISO'}, 2: ... }
            for i, Ent in enumerate(MedicAttrOutput):
                MedicAttrEntities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'],'label': Ent['entity_group']}

            # Merge all entities
            AllFlatEnts = merge_dicts(AllFlatEnts, MedicAttrEntities)

        # Annotation of entities expressing negation and uncertainty
        if (arguments.neg):

            # Load the previously trained Transformers model
            print("Annotating using transformers neural model for negation and speculation...")
            neg_spec_model_checkpoint = "models/roberta-es-clinical-trials-neg-spec-ner"

            # Transformers tokenizer
            tokenizer = AutoTokenizer.from_pretrained(neg_spec_model_checkpoint)

            # Token classifier
            neg_spec_token_classifier = AutoModelForTokenClassification.from_pretrained(neg_spec_model_checkpoint)

            NegSpecOutput = annotate_sentences_with_model(Sentences, text, neg_spec_token_classifier,tokenizer,device)

            # Save the annotated entities with the final format
            NegSpecEntities = {}

            # Change format
            for i, Ent in enumerate(NegSpecOutput):
                NegSpecEntities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'],'label': Ent['entity_group']}

            # Merge all entities
            AllFlatEnts = merge_dicts(AllFlatEnts, NegSpecEntities)

        # Remove entities defined in an exception list
        if (arguments.exc):
            AllFlatEnts = remove_entities(AllFlatEnts, Tokens, ExceptionsDict, 'label', AllNestedEnts)

            # If nested entities and use of exceptions list, remove from layer 2 the entities to remove
            # This is needed to change CHEM to ANAT in "sangrado [digestivo]"
            if (arguments.nest):
                NestedEntsCleaned = remove_entities(AllNestedEnts, Tokens, ExceptionsDict, 'label', AllNestedEnts)
                AllNestedEnts = merge_dicts(AllNestedEnts, NestedEntsCleaned)

    
    # Output to BRAT ("ann") format
    if (arguments.out == "ann"):
        # Open output file to print
        outFileName = re.sub("txt$", "ann", arguments.input)
        with open(outFileName, 'w', encoding='utf8') as out:
            n_comm = 0

            if (arguments.nest):
                # Merge both dictionaries of nesting (outer) and nested entities to output in BRAT format
                AllFinalEntities = merge_dicts(AllFlatEnts,AllNestedEnts)
                # Remove entities with same label as wider entities (e.g. "dolor" in "dolor de muelas")
                AllFinalEntities, AllNestedEntities = remove_overlap(AllFinalEntities)
                AllFinalEntities = merge_dicts(AllFinalEntities, AllNestedEntities)
            else:
                AllFinalEntities = AllFlatEnts
            
            for i in AllFinalEntities:
                print("T{}\t{} {} {}\t{}".format(i, AllFinalEntities[i]['label'], AllFinalEntities[i]['start'], AllFinalEntities[i]['end'], AllFinalEntities[i]['ent']),file=out)
                # Print UMLS codes in additional comment
                if (arguments.norm):
                    CUIsList = get_codes_from_lexicon(AllFinalEntities[i]['ent'], AllFinalEntities[i]['label'], LexiconData)
                    if (CUIsList):
                        # Complete normalization data of UMLS CUIs
                        CUIsList = complete_norm_data(CUIsList,UMLSData)
                        n_comm += 1
                        codes_string = " | ".join(CUIsList)
                        print("#{}	AnnotatorNotes T{}	{}".format(n_comm,i,codes_string),file=out)

    elif (arguments.out == "json"):

        AllFinalEntities = merge_dicts(AllFlatEnts,AllNestedEnts)
        # Remove entities with same label as wider entities (e.g. "dolor" in "dolor de muelas")
        AllFinalEntities, AllNestedEntities = remove_overlap(AllFinalEntities)
        AllFinalEntities = merge_dicts(AllFinalEntities, AllNestedEntities)
        jsonEntities = convert2json(AllFinalEntities)
        # Normalization to UMLS CUIs
        if (arguments.norm):
            for entityData in jsonEntities:
                CUIsList = get_codes_from_lexicon(entityData['word'],entityData['entity_group'], LexiconData)
                if (CUIsList):
                    # Complete normalization data of UMLS CUIs
                    CUIsList = complete_norm_data(CUIsList,UMLSData)
                    codes_string = " | ".join(CUIsList)                    
                    entityData['umls'] = codes_string
            
        print(json.dumps(jsonEntities, indent=4))

    print("Done!")

#
#
#############
#
# Main class
#

if __name__ == '__main__':
    main(args)
    
