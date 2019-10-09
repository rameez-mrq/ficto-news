# Parsing the Newspaper
# %%
from newspaper import Article
import statistics
import nltk
import string
import pickle
from nltk.tag import StanfordPOSTagger

# %%
# ADD YOUR PATH AND MODEL FILENAME
path_to_jar = "/home/sam/Dropbox/fictonews/dependencies/stanford-postagger.jar"
model_filename = '/home/sam/Dropbox/fictonews/dependencies/english-left3words-distsim.tagger'
# Features Used
features_pos = ["adverb/adjective", "adverb/noun", "adverb/pronoun",
                "adjective/verb", "adjective/pronoun", "noun/verb", "noun/pronoun", "verb/pronoun"]
# %%


def features(text):
    # POS-Tagging
    tagged = StanfordPOSTagger(model_filename=model_filename, path_to_jar=path_to_jar,
                               encoding='utf8', verbose=False, java_options='-mx3000m')
    classified_word = tagged.tag(nltk.word_tokenize(text))
    text_postags = []
    for index_classified in classified_word:
        text_postags.append(index_classified[1])
    freq_pos = nltk.FreqDist(text_postags)
    adverb, adjective, noun, pronoun, verb = 0, 0, 0, 0, 0
    for index_freq in freq_pos.most_common(len(freq_pos)):
        if index_freq[0] in ["RB", "RBR", "RBS"]:
            adverb += index_freq[1]
        elif index_freq[0] in ["JJ", "JJR", "JJS"]:
            adjective += index_freq[1]
        elif index_freq[0] in ["NN", "NNS", "NNP", "NNPS"]:
            noun += index_freq[1]
        elif index_freq[0] in ["PRP", "PRP$"]:
            pronoun += index_freq[1]
        elif index_freq[0] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
            verb += index_freq[1]
    X_test = []
    X_test.extend((adverb/adjective, adverb/noun, adverb/pronoun, adjective /
                   verb, adjective/pronoun, noun/verb, noun/pronoun, verb/pronoun))
    return X_test


def predict_result(X_test):
    with open("dependencies/logisticregression.pickle", "rb") as f:
        model = pickle.load(f)
    score = model.predict([X_test])
    score2 = model.predict_proba([X_test])
    # s2=score2
    score2 = score2.tolist()
    proba = []
    for i in score2[0]:
        j = round(i*100, 2)
        proba.append(j)
    if score == 0:
        result = ('FICTION', proba[0])
    else:
        result = ('NON FICTION', proba[1])

    return result


# print(predict_text(text))
# %%
if __name__ == "__main__":
    while True:
        url = input("Enter the link to be parsed and analysed : ")
        # url = "https://timesofindia.indiatimes.com/india/china-snubs-imran-says-resolve-jk-bilaterally/articleshow/71496416.cms"
        article = Article(url, language="en")
        article.download()
        article.parse()
        text = article.text
        # print (text) #uncomment this to check the input parsed text
        x = features(text)
        results = predict_result(x)
        print ("\n The writing style of the given article resembles to >>>>__{}__<<<< with a probability of >>>>__{}__<<<< \n".format(results[0], results[1]))
