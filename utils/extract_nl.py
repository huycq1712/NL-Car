import nltk
import re
import pprint
from nltk import Tree
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')


patterns = """
    NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
    PP: {<IN><NP>}               # Chunk prepositions followed by NP
    VP: {<VB.*><TO|RB|RP>?<VP|NP|PP>?} # Chunk verbs and their arguments
    """
NPChunker = nltk.RegexpParser(patterns)

def return_a_list_of_NPs(sentence):
    nps = []  # an empty list in which to NPs will be stored.
    tree = NPChunker.parse(sentence)
    for subtree in tree.subtrees():
         if subtree.label() == 'NP':
            t = subtree
            t = ' '.join(word for word, tag in t.leaves())
            nps.append(t)
            
    return nps
def return_a_list_of_VPs(sentence):
    nps = []  # an empty list in which to NPs will be stored.
    tree = NPChunker.parse(sentence)
    for subtree in tree.subtrees():
        if subtree.label() == 'VP':
            t = subtree
            t = ' '.join(word for word, tag in t.leaves())
            nps.append(t)
            
    return nps
def return_a_list_of_PPs(sentence):
    nps = []  # an empty list in which to NPs will be stored.
    tree = NPChunker.parse(sentence)
    for subtree in tree.subtrees():
        if subtree.label() == 'PP':
            t = subtree
            t = ' '.join(word for word, tag in t.leaves())
            nps.append(t)
            
    return nps

def prepare_text(input):
   # tokenized_sentence = nltk.sent_tokenize(sample_text)
    tokenized_words = [nltk.word_tokenize(input)]
    tagged_words = [nltk.pos_tag(word) for word in tokenized_words]
    word_tree = [NPChunker.parse(word) for word in tagged_words]
    return word_tree


if __name__ == '__main__':
    sample_text = """
    A red SUV car drove through an intersection.    
    """
    sample_text = prepare_text(sample_text)
    print(return_a_list_of_NPs(sample_text))
    