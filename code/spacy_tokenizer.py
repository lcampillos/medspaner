#!/usr/bin/python3
# -*- coding: UTF-8 -*-
#
# tokenize_spacy.py
#
# Usage:
#
#   tokenize_spacy.py <FILE>
#
# Note that Python 3 has processed better the UTF8 characters.
# 
# Leonardo Campillos-Llanos 
# 2019-2022
#
#########################################################################

import re
import os
import sys
import spacy
import argparse
import requests


def format_pos_name(POS_label_name,predicted_POS):

    ''' Given the name of a part-of-speech tag in MedLexSp dictionary, changes the tag name according to Universal Dependencies used in Spacy / Stanza. 
        In case several tags are possible, the part-of-speech prediction is used to disambigate. 
        E.g. "ADJ;N" (tag name in MedLexSp) and "ADJ" (predicted tag) => output "ADJ". 
        E.g. "N;NPR" -> "NOUN" or "PROPN", "ADJ;N" -> "ADJ" or "NOUN"
        MedLexSp category "AFF" has not an equivalent category in Spacy / Stanza.
    '''
    # keys are MedLexSp PoS codes, values are Spacy / Stanza labels
    POSFormat = {'ADJ': 'ADJ', 'ADV': 'ADV', 'N': 'NOUN', 'PREP': 'ADP', 'V': 'VERB', 'art': 'DET', 'NPR': 'PROPN'}

    if ((POS_label_name == "ADJ;ADV") and (predicted_POS == "ADV")):
        return POSFormat['ADV']
    elif ((POS_label_name == "ADJ;ADV") and (predicted_POS == "ADJ")):
        return POSFormat['ADJ']
    elif ((POS_label_name == "N;NPR") and (predicted_POS == "NOUN")):
        return POSFormat['N']
    elif ((POS_label_name == "N;NPR") and (predicted_POS == "PROPN")):
        return POSFormat['NPR']
    elif ((POS_label_name == "ADJ;N") and (predicted_POS == "ADJ")):
        return POSFormat['ADJ']
    elif ((POS_label_name == "ADJ;N") and (predicted_POS == "NOUN")):
        return POSFormat['N']
    else:
        return POSFormat[POS_label_name]


def get_pos_from_lexicon(word,predicted_POS,POSDict):
    
    ''' Get Part-of-Speech (PoS) category from MedLexSp lexicon. Use default Spacy PoS if not available. '''
    
    try:
        word = word.lower()
        if POSDict[word]:
            TuplesList = POSDict[word]
            # Look up the dictionary using the PoS tag, if several categories are possible: "curva": [('ADJ', 'curvo'), ('N', 'curva')]
            if len(TuplesList)>1:
                # Default value (in case the following step fails)
                POS = TuplesList[0][0]
                lemma = TuplesList[0][1]
                # Take the lemma according to PoS predicted by Stanza/Spacy
                for Tuple in TuplesList:
                    POS = Tuple[0]
                    if format_pos_name(POS, predicted_POS) == predicted_POS:
                        lemma = Tuple[1]
                        return format_pos_name(POS,predicted_POS), lemma
            else:
                POS=TuplesList[0][0]
                lemma=TuplesList[0][1]
                
            return format_pos_name(POS, predicted_POS),lemma
    except:
        return None


def tokenize_spacy(filename,POSDict):
    
    ''' Opens a file and tokenizes it with Spacy '''

    #nlp = spacy.load('es_core_news_sm') # Small size model
    nlp = spacy.load('es_core_news_md') # Medium size model

    with open(filename, 'r', encoding='utf8', newline = '') as f: 
        #print("Tokenizing with Spacy...")
        original_text = f.read()        
        # lowercase input (better results for pos tagging and normalization)
        text = original_text.lower()
        doc = nlp(text)
        Tokens={}
        for i,token in enumerate(doc):
            # Starting offset: token.idx
            # Ending offset:
            end = int(token.idx) + int(len(token.text))
            # Get POS category from MedLexSp lexicon; if not available, use Spacy POS
            pos = token.pos_
            lemma = token.lemma_
            if get_pos_from_lexicon(token.text,pos,POSDict):
                pos, lemma = get_pos_from_lexicon(token.text,pos,POSDict)
            else:
                pos = token.pos_
            # Get the original text (not lowercased)
            orig_token = original_text[token.idx:end]
            Tokens[i] = { 'token': orig_token, 'lemma': lemma, 'pos': pos, 'tag': token.tag_, 'dep': token.dep_, 'shape': token.shape_, 'is_alpha': token.is_alpha, 'start': token.idx, 'end': end }
        return Tokens


def tokenize_spacy_text(text,POSDict):

    ''' Given a text from an opened file, tokenizes it with Spacy '''

    nlp = spacy.load('es_core_news_md') # Medium size model

    original_text = text

    # lowercase input (better results for pos tagging and normalization)
    text = original_text.lower()

    doc = nlp(text)
    
    Tokens={}
    
    for i,token in enumerate(doc):
        end = int(token.idx) + int(len(token.text))
        # Get POS category from MedLexSp lexicon; if not available, use Spacy POS
        pos = token.pos_
        lemma = token.lemma_
        if get_pos_from_lexicon(token.text,pos,POSDict):
            pos, lemma = get_pos_from_lexicon(token.text,pos,POSDict)
        else:
            pos = token.pos_
        # Get the original text (not lowercased)
        orig_token = original_text[token.idx:end]
        Tokens[i] = { 'token': orig_token, 'lemma': lemma, 'pos': pos, 'tag': token.tag_, 'shape': token.shape_, 'start': token.idx, 'end': end }
        
    return Tokens


def sentences_spacy(text):
    
    ''' Split text into sentences (returns a list) '''
    
    #nlp = spacy.load('es_core_news_sm') # Small size model
    nlp = spacy.load('es_core_news_md') # Medium size model

    doc = nlp(text)
    
    Sentences = [sentence for sentence in doc.sents]
    
    return Sentences


def main(filename):
    Tokens = tokenize_spacy(filename)

