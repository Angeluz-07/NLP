import nltk
import string
import gensim
from itertools import chain
from gensim import models, corpora

"""
 Minimal topic modelling using gensim library

 i  : transform to lower case
 ii  : tokenize documents(i.e. text corpus or paragraphs) to words
 iii : remove non-alpha 
 iv : remove stopwords
 v : frequency distribution of words
 vi : build LDA and LSI models
"""

#flatten a list of lists of tokens
def flatten_list(l):
    return list(chain.from_iterable(l))

def lower_case(string): 
    return string.lower()

NUM_TOPICS = 5
EN_STOP_WORDS = nltk.corpus.stopwords.words('english')

#random sentences courtesy from https://randomwordgenerator.com/sentence.php
random_sentences = [
    "She was too short to see over the fence.",
    "Joe made the sugar cookies; Susan decorated them.",
    "She folded her handkerchief neatly.",
    "We need to rent a room for our party.",
    "They got there early, and they got really good seats.",
    "If you like tuna and tomato sauce- try combining the two. It’s really not as bad as it sounds.",
    "Where do random thoughts come from?",
    "I currently have 4 windows open up… and I don’t know why.",
    "Writing a list of random sentences is harder than I initially thought it would be.",
    "Check back tomorrow; I will see if the book has arrived."
]

docs_tokenized = []
for s in random_sentences:    
    l = lower_case(s)
    l = nltk.word_tokenize(l)
    l = [ x for x in l if x.isalpha() ]
    l = [ x for x in l if x not in EN_STOP_WORDS ]
    #print(l)
    #print()
    docs_tokenized.append(l)


#Frequency distribution analysis
all_words = flatten_list(docs_tokenized)
fdist = nltk.FreqDist(all_words)
print("word freq dist: ")
for w,f in fdist.most_common(10):
	print(f'{w} : {f}')

#Topic modelling using gensim https://nlpforhackers.io/topic-modeling/ , https://monkeylearn.com/topic-analysis/
#Build a dictionary - associate word to numeric id
dictionary = corpora.Dictionary(docs_tokenized)

#Transform collection of text to numerical form
corpus = [dictionary.doc2bow(text) for text in docs_tokenized]

#Build the LDA model
lda_model = models.LdaModel(corpus = corpus, num_topics = NUM_TOPICS, id2word = dictionary)

# Build the LSI model
lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
 
print("LDA model: ")
for idx in range(NUM_TOPICS):
	topic_summary = ' '.join([ v[0] for v in lda_model.show_topic(idx,5) ])
	print("Topic #%s : %s " % (idx, topic_summary))
		
print("=" * 20)

print("LSI model: ")
for idx in range(NUM_TOPICS):
	topic_summary = ' '.join([ v[0] for v in lsi_model.show_topic(idx,5) ])
	print("Topic #%s : %s " % (idx, topic_summary))
		
print("=" * 20)