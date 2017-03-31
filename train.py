import pandas as pd
import nltk
import pickle
import random
from nltk.corpus import stopwords

custom_stopwords = ['crookedhillary', 'kaine', 'trump2016', 'trumppence2016', 'votetrump', 'iacaucus', 'tedcruz', 'ted', 'cruz', 'bernie', 'sanders', 'trumppence16', 'jeb', 'bush', 'lyin', 'gopdebate', "lindsey", "graham"]

# Read tweets and label with "trump" and "staff"
tweets = pd.read_json("data/condensed_2016.json")
tweets['year'] = 2016
for year in [2017]:
   temptweets = pd.read_json("data/condensed_" + str(year) + ".json")
   temptweets['year'] = year
   tweets = tweets.append(temptweets)
   
tweets['type'] = ['trump' if source =='Twitter for Android' else 'staff' for source in tweets['source']] 

# Condense tweets down to simplest components
train_tweets = [] 
test_tweets = []
for(index, row) in tweets.iterrows():
    if row.is_retweet == False:
        if row['year'] == 2017:
            test_tweets.append(( nltk.word_tokenize(row['text'].lower()), row['type']))
        else:
            train_tweets.append(( nltk.word_tokenize(row['text'].lower()), row['type']))

# Remove stopwords
train_tweets = [( [ word for word in tweet[0] if word not in stopwords.words('english') and word not in custom_stopwords], tweet[1]) for tweet in train_tweets]

def get_word_features(tweets):

    all_words = []
    for(words, sentiments) in tweets:
        all_words.extend(words)
    
    wordlist = nltk.FreqDist(all_words)
    wordlist = wordlist.most_common()
    word_features = [word[0] for word in wordlist]
    return word_features
    
word_features = get_word_features(train_tweets)
word_features = word_features[:500]



def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(' + word + ')'] = (word in document_words)
    return features
    
training_set = nltk.classify.apply_features(extract_features, train_tweets)
test_set = nltk.classify.apply_features(extract_features, test_tweets)

print("beginning training of trainer")
classifier = nltk.NaiveBayesClassifier.train(training_set)

print("saving classifier")
file = open('classify/trump_classifier.pickle', 'wb')
pickle.dump(classifier, file, -1)
file.close()
file = open('classify/trump_classifier_features.pickle', 'wb')
pickle.dump(word_features, file, -1)
file.close()

print("accuracy test")
print(nltk.classify.accuracy(classifier, test_set))