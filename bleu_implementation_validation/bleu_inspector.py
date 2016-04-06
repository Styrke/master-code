
references = [
"We must bear in mind the Community as a whole.",
"But we must reach agreement on what to put in this constitution.",
"This does not mean that everything has to happen at once.",
"Vote",
"It cannot serve as a basis for the establishment of a European constitution.",
"The advantages are already there; they are visible and everyone stands to gain.",
"We can help countries catch up, but not by putting their neighbours on hold.",
"Several years ago in Copenhagen we defined the requisite criteria.",
"Our support and commitment continue - to a large extent, for our own sake.",
"The environment is the area which will constitute a stumbling block in the negotiations.",
"They are simply not entitled to chide this Baltic country like a stupid schoolboy." ]

candidates_identical = [
"We must bear in mind the Community as a whole.",
"But we must reach agreement on what to put in this constitution.",
"This does not mean that everything has to happen at once.",
"Vote",
"It cannot serve as a basis for the establishment of a European constitution.",
"The advantages are already there; they are visible and everyone stands to gain.",
"We can help countries catch up, but not by putting their neighbours on hold.",
"Several years ago in Copenhagen we defined the requisite criteria.",
"Our support and commitment continue - to a large extent, for our own sake.",
"The environment is the area which will constitute a stumbling block in the negotiations.",
"They are simply not entitled to chide this Baltic country like a stupid schoolboy." ]

candidates_single_word = [
"We must KEEP in mind the Community as a whole.",
"But we SHOULD reach agreement on what to put in this constitution.",
"This does not mean that ALL has to happen at once.",
"CHOOSE",
"It cannot serve as a basis for the FORMATION of a European constitution.",
"The advantages are already there; they are visible and everyone stands to ADVANCE.",
"We can help countries KEEP up, but not by putting their neighbours on hold.",
"MANY years ago in Copenhagen we defined the requisite criteria.",
"Our BACKING and commitment continue - to a large extent, for our own sake.",
"The environment is the area which will constitute a stumbling block in the DISCUSSIONS.",
"They are simply not entitled to CRITICIZE this Baltic country like a stupid schoolboy." ]

candidates_many_to_one = [
"We must bear in mind the Community ALTOGETHER.",
"But we must NEGOTIATE on what to put in this constitution.",
"This does not mean that everything has to happen INSTANTLY.",
"CHOOSE",
"It cannot BE a basis for the establishment of a European constitution.",
"The advantages are CLEAR; they are visible and everyone stands to gain.",
"We can help countries catch up, but not by OBSTRUCTING their neighbours.",
"AGES ago in Copenhagen we defined the requisite criteria.",
"Our ENGAGEMENT continue - to a large extent, for our own sake.",
"The environment is the area which will constitute a HINDRANCE in the negotiations.",
"They are simply not entitled to chide this Baltic country like a FOOL." ]

candidates_multiple_differences = [
"Bear in mind the Community as a whole.",
"But we must AGREE on what to put in the constitution.",
"It does not mean that everything has to happen at the same time.",
"CHOICE",
"It cannot BE a basis for the CREATION of a European constitution.",
"The advantages are already PRESENT; they are CLEAR and everyone stands to gain.",
"We can help countries KEEP up, but DEFINITELY not by PUSHING their neighbours BACKWARDS.",
"MANY years ago in Copenhagen we AGREED UPON the requisite criteria.",
"Our COMMITMENT continue - MOSTLY for our own sake.",
"The environment will constitute a HINDRANCE in the DISCUSSIONS.",
"They SIMPLY HAVE NO RIGHT to chide this Baltic country like a FOOL." ]

candidates_different = [
"This is not correct.",
"This is not correct.",
"This is not correct.",
"This is not correct.",
"This is not correct.",
"This is not correct.",
"This is not correct.",
"This is not correct.",
"This is not correct.",
"This is not correct.",
"This is not correct." ]

import sys,os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.performancemetrics import corpus_bleu as custom_corpus_bleu
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu

BASE_NAME = 'bleu_comparison_'
EXT       = '.dat'
IDENTICAL = 'identical'
SINGLE    = 'single_word'
MANY_ONE  = 'many_to_one'
MULTIPLE  = 'multiple'
DIFFERENT = 'different'

def calc(c, r):
    return (custom_corpus_bleu([c], [r]), nltk_corpus_bleu([c],[r]))

def perform(cs, rs, filename):
    with open(filename, 'w') as f:
        print("candidate,reference,custom,nltk", file=f)
        for c, r in zip(cs, rs):
            print('"{0}","{1}",{2},{3}'.format(c, r, *calc(c, r)), file=f)
        print("wrote comparison scores", filename, "...")

perform(candidates_identical, references, BASE_NAME + IDENTICAL + EXT)
perform(candidates_single_word, references, BASE_NAME + SINGLE + EXT)
perform(candidates_many_to_one, references, BASE_NAME + MANY_ONE + EXT)
perform(candidates_multiple_differences, references, BASE_NAME + MULTIPLE + EXT)
perform(candidates_different, references, BASE_NAME + DIFFERENT + EXT)
