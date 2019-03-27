
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import json, re, os
from nltk.corpus import words
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from statistics import mean, median

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  


# pre-processing
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text)
    text = re.sub('@[^\s]+','', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +',' ', text)
    return text.strip()

	
# Extract sub-folders and JSON files
dirPath = 'K:/NIU/sem3/BIG Data/Project/alt/altmetric_clean_sample.tar/altmetric_clean_sample/altmetric_clean_sample/';

json_files = [pos_json for pos_json in os.listdir(dirPath)]
def getListOfFiles(arrDir):
    # create a list of file and sub directories 
    # names in the given directory 
    allFiles = []
    count = 0
    for subdirName in arrDir:
        listOfFile = os.listdir(dirPath + '/' + subdirName)
        # Iterate over all the entries
        
        if count < 15000:
            for entry in listOfFile:
                # Create full path
                fullPath = dirPath + '/' + subdirName + '/' + entry
                with open(fullPath) as file:
                    json_obj = json.load(file)
                    if count > 15000:
                        break
                    if 'twitter' in json_obj['counts']:
                        if 'selected_quotes' in json_obj:
                            allFiles.append(fullPath)
                            count += 1
        else:
            break
    
    return allFiles

list_of_files = getListOfFiles(json_files)

# Extract all the words from the selected quotes
contents = []
bag_of_words = []
posts_count = []
unique_keys = []
key_value_obj = []
words_count = []
preprocessed_str = ""

for file in list_of_files:
    with open(file) as json_file:
        json_obj = json.load(json_file)
        str1 = '. '.join(json_obj['selected_quotes'])
        preprocessed_str = preprocess_text(str1)
        if preprocessed_str != "":
			# pre-processing
            contents.append(preprocess_text(str1))
            vectorizer = CountVectorizer()
            bag_of_words = vectorizer.fit_transform(contents).todense()
            words_frequency = [(word, bag_of_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            dict_words = dict(words_frequency)
            contents = []
            key_value_obj.append(dict_words)
            for key in dict_words.keys():
                if key not in stop_words and key not in unique_keys:
                    unique_keys.append(key)
            posts_count.append(json_obj['counts']['twitter']['posts_count'])


# filter out non-english words
features = []
for key in unique_keys:
    if key in words.words():
        features.append(key)

# prepare the data
count = 0
for record in key_value_obj:
    arr = []
    for key in features:
        if key in record:
            arr.append(record[key])
        else:
            arr.append(0)
    arr.append(posts_count[count])
    words_count.append(arr)
    count += 1
    
features.append('twitter_post_counts')

# storing the data to csv
df = pd.DataFrame(words_count, columns=features)
df.to_csv("K:\\NIU\\sem3\\BIG Data\\Project\\alt_data_final.csv")

# reading from csv
df = pd.read_csv("K:\\NIU\\sem3\\BIG Data\\Project\\alt_data_final.csv")
df = df.drop('Unnamed: 0', axis=1)

# Get Top 10 popular words - highest word frequency
sum_per_column = {}
for i in range(1, len(df.columns)-2):
    if sum(df.iloc[:,i]) > 10 :
        sum_per_column[df.columns[i]] = sum(df.iloc[:,i])
        
from operator import itemgetter
d=sorted(sum_per_column.items(),reverse=True, key=itemgetter(1))

# Plot bar chart for top 10 words
plt.bar(range(len(d[:10])), [val[1] for val in d[:10]], align='center')
plt.xticks(range(len(d[:10])), [val[0] for val in d[:10]])
plt.xticks(rotation=70)
plt.title('')
plt.xlabel('Features')
plt.ylabel('Frequency')

# Splitting Test and Train data
X, y = df.iloc[:, :-2], df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = StandardScaler()

# Fit on training set only.
scaler.fit(X_train)

# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#PCA set variance
pca = PCA(.90)
pca.fit(X_train)
pca.n_components_

# PCA Transform
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# LogisticRegression
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(X_train, y_train)
logistic_acc = logisticRegr.score(X_test, y_test)
print('{:.2f}'.format(logistic_acc))

# KNN
knn = KNeighborsClassifier(n_neighbors=6)  
knn.fit(X_train, y_train)
knn_acc = knn.score(X_test, y_test)
print('{:.2f}'.format(knn_acc))



