import pandas as pd
import nltk
import pickle
import random
from nltk.corpus import stopwords

# Read tweets and label with "trump" and "staff"
tweets = pd.read_json("data/condensed_2016.json")
for year in [2017]:
   tweets = tweets.append(pd.read_json("data/condensed_" + str(year) + ".json"))
tweets['type'] = ['trump' if source =='Twitter for Android' else 'staff' for source in tweets['source']] 

# Condense tweets down to simplest components
filtered_tweets = []    
for(index, row) in tweets.iterrows():
    if row.is_retweet == False:
        filtered_tweets.append(( nltk.word_tokenize(row['text'].lower()), row['type']))

# Remove stopwords
filtered_tweets = [( [ word for word in tweet[0] if word not in stopwords.words('english')], tweet[1]) for tweet in filtered_tweets]

# Shuffle tweets
random.shuffle(filtered_tweets);

# Split dataset into training and regualr 
train_tweets = filtered_tweets[(len(filtered_tweets) / 2):]
test_tweets = filtered_tweets[:(len(filtered_tweets) / 2)]

def get_word_features(tweets):

    all_words = []
    for(words, sentiments) in tweets:
        all_words.extend(words)
    
    wordlist = nltk.FreqDist(all_words)
    wordlist = wordlist.most_common()
    word_features = [word[0] for word in wordlist]
    return word_features
    
word_features = get_word_features(train_tweets)
word_features = word_features[:100]

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
file = open('trump_classifier.pickle', 'wb')
pickle.dump(classifier, file, -1)
file.close()
file = open('trump_classifier_features.pickle', 'wb')
pickle.dump(word_features, file, -1)
file.close()

print("accuracy test")
print(nltk.classify.accuracy(classifier, test_set))