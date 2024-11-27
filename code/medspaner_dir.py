#!/usr/bin/python3
# -*- coding: UTF-8 -*-
#
# medspaner_dir.py
#
# Medical Semantic python-assisted Named Entity Recognizer using dictionary and pre-trained transformer models.
#
# Usage:
#
#   python medspaner_dir.py -conf <CONFIG_FILE> -input <FOLDER>
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parser for config file
import configparser

# Command line arguments
parser = argparse.ArgumentParser(description='Given a text file, annotate it with semantic labels')
parser.add_argument('-att', required=False, default=False, action='store_true', help='annotate entity attributes (experiencer and event temporality)')
parser.add_argument('-conf', required=False, help='use configuration file to parse arguments (optional)')
parser.add_argument('-drg', required=False, default=False, action='store_true', help='annotate drug features such as dose (not annotated by default)')
parser.add_argument('-exc', required=False, default=False, help='use a list of exceptions of entity types not to be annotated, or word patterns to re-label')
parser.add_argument('-input', required=True, default=False, help='specify file to annotate')
parser.add_argument('-lex', required=False, help='specify pickle object dictionary for preannotation (neural model is used by default); indicate using: -lex "lexicon/MedLexSp.pickle"')
parser.add_argument('-misc', required=False, default=False, action='store_true', help='annotate miscellaneous medical entities (observations, qualifiers and results/values)')
parser.add_argument('-neu', required=False, default=True, action='store_true', help='specify to annotate UMLS entities with neural model (used by default)')
parser.add_argument('-neg', required=False, default=False, action='store_true', help='annotate entities expressing negation and uncertainty (not annotated by default)')
parser.add_argument('-nest', required=False, default=False, action='store_true', help='annotate inner or nested entities inside wider entities (not annotated by default)')
parser.add_argument('-norm', required=False, default=False, help='normalize entities and output terminology codes (optional); indicate "umls" (default), "omop" or "snomed"')
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
        arguments.norm = arguments.norm.lower()
        if ((arguments.norm) != "umls") and ((arguments.norm) != "snomed") and ((arguments.norm) != "omop"):
            print('Please, provide the normalization source: "umls", "omop" or "snomed"')
            parser.print_help()
            sys.exit()
        if (arguments.norm == "umls"):
            DataFile = open("lexicon/umls_data.pickle", 'rb')
            UMLSData = pickle.load(DataFile)
        elif (arguments.norm == "snomed"):
            DataFile = open("lexicon/sctspa_norm.pickle", 'rb')
            SCTSPAData = pickle.load(DataFile)
        elif (arguments.norm == "omop"):
            DataFile = open("lexicon/omop_norm.pickle", 'rb')
            OMOPData = pickle.load(DataFile)

    # If use of an exception list to remove specific entities 
    if (arguments.exc):
        
        # Read exceptions and save to hash
        ExceptionsDict = read_exceptions_list(arguments.exc)

    text = arguments.input
    
    # Load lexicon data
    LexiconData, POSData = read_lexicon("lexicon/MedLexSp.pickle")

    # Load the previously trained transformers model
    if (arguments.neu):

        # Load the model for UMLS entities using full path
        umls_model_checkpoint = "models/roberta-es-clinical-trials-umls-7sgs-ner"

        # Transformers tokenizer
        umls_tokenizer = AutoTokenizer.from_pretrained(umls_model_checkpoint)

        # Token classifier for UMLS entities
        umls_token_classifier = AutoModelForTokenClassification.from_pretrained(umls_model_checkpoint)

    if (arguments.temp):

        # Load the previously trained model for temporal entities
        temp_model_checkpoint = "models/roberta-es-clinical-trials-temporal-ner"

        # Transformers tokenizer
        temp_tokenizer = AutoTokenizer.from_pretrained(temp_model_checkpoint)

        # Token classifier
        temp_token_classifier = AutoModelForTokenClassification.from_pretrained(temp_model_checkpoint)

    if (arguments.drg):

        # Load the previously trained transformers model for medication information
        medic_attr_model_checkpoint = "models/roberta-es-clinical-trials-medic-attr-ner"

        # Transformers tokenizer
        drg_tokenizer = AutoTokenizer.from_pretrained(medic_attr_model_checkpoint)

        # Token classifier
        medic_attr_token_classifier = AutoModelForTokenClassification.from_pretrained(medic_attr_model_checkpoint)

    if (arguments.misc):        
    
        # Load the previously trained Transformers model
        misc_model_checkpoint = "models/roberta-es-clinical-trials-misc-ents-ner"

        # Transformers tokenizer
        tokenizer = AutoTokenizer.from_pretrained(misc_model_checkpoint)

        # Token classifier
        misc_token_classifier = AutoModelForTokenClassification.from_pretrained(misc_model_checkpoint)

    if (arguments.neg):

        # Load the previously trained transformers model for negation / speculation
        neg_spec_model_checkpoint = "models/roberta-es-clinical-trials-neg-spec-ner"

        # Transformers tokenizer
        neg_spec_tokenizer = AutoTokenizer.from_pretrained(neg_spec_model_checkpoint)

        # Token classifier
        neg_spec_token_classifier = AutoModelForTokenClassification.from_pretrained(neg_spec_model_checkpoint)

    if (arguments.att):
                        
        # Load the previously trained Transformers model for attributes                        
        att_model_checkpoint = "models/roberta-es-clinical-trials-attributes-ner"

        # Transformers tokenizer
        tokenizer = AutoTokenizer.from_pretrained(att_model_checkpoint)

        # Token classifier
        att_token_classifier = AutoModelForTokenClassification.from_pretrained(att_model_checkpoint)

    # Check that input is a folder, not a file
    if os.path.isfile(arguments.input):
        print("Error: The input is a file, not a directory.")
        sys.exit()

    for base, dirs, files in os.walk(arguments.input):
        # Annotate all files in folder
        for file in files:
            text_file = re.search(".txt$", file)
            if file and text_file:
                file_path = os.path.join(base, file)
                print("Processing %s ..." % file_path)

                with open(file_path,'r', newline = '') as f:

                    # Save all flat entities
                    AllFlatEnts = {}

                    # Save all nested entities
                    AllNestedEnts = {}

                    # Tokenize with Spacy
                    Tokens = tokenize_spacy(file_path, POSData)

                    text = f.read()

                    # Split text into sentences
                    Sentences = sentences_spacy(text)

                    # Use lexicon
                    if (arguments.lex):

                        #  If annotation of nested entities
                        if (arguments.nest):
                            print("Annotating nested entities with lexicon...")
                            Entities,NestedEnts = apply_lexicon(text,LexiconData,arguments.nest)
                        else:
                            print("Annotating with lexicon...")
                            Entities,NestedEnts = apply_lexicon(text,LexiconData)

                        AllFlatEnts = merge_dicts(AllFlatEnts,Entities)
                        AllNestedEnts = merge_dicts(AllNestedEnts, NestedEnts)

                    # Use transformers model to annotate UMLS entities
                    if (arguments.neu):

                        print("Annotating using transformers neural model for UMLS entities...")

                        Output = annotate_sentences_with_model(Sentences,text,umls_token_classifier, umls_tokenizer, device)

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

                        TempOutput = annotate_sentences_with_model(Sentences,text,temp_token_classifier, temp_tokenizer, device)

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

                        MedicAttrOutput = annotate_sentences_with_model(Sentences, text, medic_attr_token_classifier, drg_tokenizer, device)

                        # Save the annotated entities with the final format
                        MedicAttrEntities = {}

                        # Change format to:
                        #   Entities = {1: {'start': 3, 'end': 11, 'ent': 'COVID-19', 'label': 'DISO'}, 2: ... }
                        for i, Ent in enumerate(MedicAttrOutput):
                            MedicAttrEntities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'], 'label': Ent['entity_group']}

                        # Merge all entities
                        AllFlatEnts = merge_dicts(AllFlatEnts, MedicAttrEntities)

                    # Annotation of miscellaneous medical entities (observations, qualifiers and results/values)
                    if (arguments.misc):
                                 
                        print("Annotating using transformers neural model for miscellaneous medical entities...")                       
                                       
                        MiscOutput = annotate_sentences_with_model(Sentences,text,misc_token_classifier,tokenizer,device)

                        # Save the annotated entities with the final format
                        MiscEntities = {}

                        # Change format
                        for i, Ent in enumerate(MiscOutput):
                            MiscEntities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'], 'label': Ent['entity_group']}

                        # Merge all entities
                        AllFlatEnts = merge_dicts(AllFlatEnts, MiscEntities)

                    # Annotation of entities expressing negation and uncertainty
                    if (arguments.neg):

                        print("Annotating using transformers neural model for negation and speculation...")

                        NegSpecOutput = annotate_sentences_with_model(Sentences, text, neg_spec_token_classifier, neg_spec_tokenizer, device)

                        # Save the annotated entities with the final format
                        NegSpecEntities = {}

                        # Change format
                        for i, Ent in enumerate(NegSpecOutput):
                            NegSpecEntities[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'],
                                                    'label': Ent['entity_group']}

                        # Merge all entities
                        AllFlatEnts = merge_dicts(AllFlatEnts, NegSpecEntities)

                    # Annotation of experiencer and event temporality attributes
                    if (arguments.att):
                        
                        print("Annotating using transformers neural model for experiencer and event temporality attributes...")
                       
                        AttOutput = annotate_sentences_with_model(Sentences,text,att_token_classifier,tokenizer,device)

                        # Save the annotated entities with the final format
                        Attributes = {}

                        # Change format
                        for i, Ent in enumerate(AttOutput):
                            Attributes[i] = {'start': Ent['start'], 'end': Ent['end'], 'ent': Ent['word'], 'label': Ent['entity_group']}

                        # Merge all entities
                        AllFlatEnts = merge_dicts(AllFlatEnts, Attributes)

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
                    outFileName = re.sub("txt$", "ann", file_path)
                    with open(outFileName, 'w', encoding='utf8') as outFile:
                        n_comm = 0
                        if (arguments.nest):
                            # Merge both dictionaries of nesting (outer) and nested entities to output in BRAT format
                            AllFinalEntities = merge_dicts(AllFlatEnts,AllNestedEnts)
                            # Remove entities with same label as wider entities (e.g. "dolor" in "dolor de muelas")
                            AllFinalEntities, AllNestedEntities = remove_overlap(AllFinalEntities)
                            AllFinalEntities = merge_dicts(AllFinalEntities, AllNestedEntities)
                        else:
                            AllFinalEntities = AllFlatEnts

                        FinalHash = codeAttribute(AllFinalEntities)

                        if (arguments.norm):
                            if (arguments.norm=="umls"):
                                convert2brat(FinalHash,outFile,LexiconData,UMLSData,arguments.norm)
                            elif (arguments.norm=="snomed"):
                                convert2brat(FinalHash,outFile,LexiconData,SCTSPAData,arguments.norm)
                            elif (arguments.norm=="omop"):
                                convert2brat(FinalHash,outFile,LexiconData,OMOPData,arguments.norm)
                        else:
                            convert2brat(FinalHash,outFile,None,None,None)

                elif (arguments.out == "json"):

                    AllFinalEntities = merge_dicts(AllFlatEnts,AllNestedEnts)
                    # Remove entities with same label as wider entities (e.g. "dolor" in "dolor de muelas")
                    AllFinalEntities, AllNestedEntities = remove_overlap(AllFinalEntities)
                    AllFinalEntities = merge_dicts(AllFinalEntities, AllNestedEntities)
                    
                    FinalHash = codeAttribute(AllFinalEntities)
        
                    if (arguments.norm):
                        if (arguments.norm=="umls"):
                            jsonEntities = convert2json(FinalHash,LexiconData,UMLSData,arguments.norm)
                        elif (arguments.norm=="snomed"):
                            jsonEntities = convert2json(FinalHash,LexiconData,SCTSPAData,arguments.norm)
                        elif (arguments.norm=="omop"):
                            jsonEntities = convert2json(FinalHash,LexiconData,OMOPData,arguments.norm)
                    else:
                        jsonEntities = convert2json(FinalHash,None,None,None)
  
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
    
