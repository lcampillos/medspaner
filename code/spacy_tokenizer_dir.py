#!/usr/bin/python3
# -*- coding: UTF-8 -*-
#
# tokenize_spacy.py
#
# Usage:
#
#   tokenize_spacy_dir.py <FOLDER-PATH>
#
#
# The output file has format:
#
#       Token   Lemma   POS  LingTags   Shape
#       El	El	DET	DET__Definite=Def|Gender=Masc|Number=Sing|PronType=Art	Xx
#       paciente	paciente	NOUN	NOUN__Number=Sing	xxxx
#       fue	ser	AUX	AUX__Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	xxx
#       intervenido	intervenir	VERB	VERB__Gender=Masc|Number=Sing|Tense=Past|VerbForm=Part	xxxx
#       ...
#
# Note that Python 3 has processed better the UTF8 characters.
# 
# Leonardo Campillos 
# 2019
#
#########################################################################

import re
import os
import sys
import spacy


# Get Part-of-Speech category from MedLexSp lexicon
# It can be improved to get lemma
def get_pos_from_lexicon(word,POSDict):
    # Unify POS format to Spacy; keys are MedLexSp PoS codes, values are Spacy labels
    POSFormat = {'N': 'NOUN', 'PREP': 'ADP', 'V': 'VERB', 'art': 'DET', 'NPR': 'PROPN', 'N;NPR': 'NOUN', 'ADJ;N': 'NOUN'}
    try:
        if POSDict[word]:
            #POS = POSDict[word][0]
            TuplesList = POSDict[word]
            # Arreglar, hay >2600 casos de >1 categoría: técnica (N, ADJ;N), ('ADJ', 'abdominal'), ('N', 'abdominal')
            # Ahora se toma por defecto la primera tupla
            POS=TuplesList[0][0]
            lemma=TuplesList[0][1]
            return POSFormat[POS],lemma
    except:
        return None


def tokenize_spacy(filename,POSDict):

    nlp = spacy.load('es')

    with open(filename, 'r', encoding='utf8', newline = '') as f:
        print("Tokenizing with Spacy...")
        original_text = f.read()
        # lowercase input (better results for pos tagging and normalization)
        text = original_text.lower()
        doc = nlp(text)
        Tokens={}
        for i,token in enumerate(doc):
            # Starting offset: token.idx
            # Ending offset:
            end = str(int(token.idx) + int(len(token.text)))
            # Get POS category from MedLexSp lexicon; if not available, use Spacy POS
            if get_pos_from_lexicon(token.text,POSDict):
                pos, lemma = get_pos_from_lexicon(token.text,POSDict)
            else:
                pos = token.pos_
                lemma = token.lemma_
            # Get the original text (not lowercased)
            orig_token = original_text[token.idx:end]
            Tokens[i] = { 'token': orig_token, 'lemma': token.lemma_, 'pos': token.pos_, 'tag': token.tag_, 'shape': token.shape_, 'start': token.idx, 'end': end }
        return Tokens

def main(args):
    Tokens = tokenize_spacy(args)

#
#
#############
#
# Main class
#

# this means that if this script is executed, then
# main() will be executed
if __name__ == '__main__':
    main(args)