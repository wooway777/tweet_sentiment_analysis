# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:47:46 2017

@author: Wei
"""

import csv
import string
import numpy as np
import math
import scipy.sparse as sps
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Question 1.1
num_data = 0
sentiments = []
raw_tweets = []

with open('training.1600000.processed.noemoticon.csv', 'rb') as csvTrainingFile:
    global num_data
    num_data = 0
    trainingReader = csv.reader(csvTrainingFile)
    for row in trainingReader:
        #index = str(num_data)
        if row[0] == '0':
            sentiments.append(-1)
        if row[0] == '4':
            sentiments.append(1)
        raw_tweets.append(row[5])
        num_data += 1
    print "Finished reading", num_data, "lines of data."
    
csvTrainingFile.close()

# Question 1.2
file_in = open('stopwords.txt', 'r')
stopwords = file_in.readlines()
file_in.close()
cleaned_tweets = []
rep_cleaned_tweets = []
for i in range(len(stopwords)):
    stopwords[i] = stopwords[i].strip()
for tweet in raw_tweets:
    tweet = tweet.lower()#1
    new_tweet = []
    for word in tweet.split():
        if 'http' in word or 'www.' in word or '.com' in word or '.edu' in word or '.org' in word:#2
            pass#new_tweet += 'URL'
        elif word[0] == '@':#5
            pass#new_tweet += 'AT-USER'
        else:#3
            temp_word = word.translate(None, string.punctuation)
            if temp_word not in stopwords:#no stopwords
                if any(char.isdigit() for char in temp_word) == False:
                    new_tweet.append(temp_word)
        if ',' in word:#till first coma
            break
    #new_tweet = new_tweet.translate(None, string.punctuation)#4
    #print new_tweet
    rep_cleaned_tweets.append(new_tweet)
    cleaned_tweets.append(np.unique(new_tweet).tolist())
    
for i in range(len(rep_cleaned_tweets)):
    rep_cleaned_tweets[i] = ' '.join(rep_cleaned_tweets[i])    
      
# Question 1.3
all_words = []
for tweet in cleaned_tweets:
    for word in tweet:
        all_words.append(word)
all_words = np.unique(all_words).tolist()

max_length = len(all_words)
#all_features = sps.lil_matrix((len(cleaned_tweets), max_length))

vectorizer = CountVectorizer(vocabulary=all_words, decode_error='ignore')
train_features = vectorizer.fit_transform(rep_cleaned_tweets)
#sps.save_npz('train_bigram.npz', train_features)

#Part 1.6
num_data = 0
test_sentiments = []
raw_tweets = []
with open('testdata.manual.2009.06.14.csv', 'rb') as csvTestFile:
    global num_data
    num_data = 0
    testReader = csv.reader(csvTestFile)
    for row in testReader:
        #index = str(num_data)
        if row[0] == '0':
            test_sentiments.append(-1)
        if (row[0] == '4') or (row[0] == '2'):
            test_sentiments.append(1)
            
        raw_tweets.append(row[5])
        num_data += 1
    print "Finished reading", num_data, "lines of data."
    
csvTestFile.close()


# Question 1.2 --- For part 1.6
file_in = open('stopwords.txt', 'r')
stopwords = file_in.readlines()
file_in.close()
cleaned_tweets = []
rep_cleaned_tweets = []
for i in range(len(stopwords)):
    stopwords[i] = stopwords[i].strip()
for tweet in raw_tweets:
    tweet = tweet.lower()#1
    new_tweet = []
    for word in tweet.split():
        if 'http' in word or 'www.' in word or '.com' in word or '.edu' in word or '.org' in word:#2
            pass#new_tweet += 'URL'
        elif word[0] == '@':#5
            pass#new_tweet += 'AT-USER'
        else:#3
            temp_word = word.translate(None, string.punctuation)
            if temp_word not in stopwords:#no stopwords
                if any(char.isdigit() for char in temp_word) == False:
                    new_tweet.append(temp_word)
        if ',' in word:#till first coma
            break
    #new_tweet = new_tweet.translate(None, string.punctuation)#4
    #print new_tweet
    rep_cleaned_tweets.append(new_tweet)
    cleaned_tweets.append(np.unique(new_tweet).tolist())
    
for i in range(len(rep_cleaned_tweets)):
    rep_cleaned_tweets[i] = ' '.join(rep_cleaned_tweets[i])    
      
# Question 1.3 ---For part 1.6

vectorizer = CountVectorizer(vocabulary=all_words, decode_error='ignore')
test_features = vectorizer.fit_transform(rep_cleaned_tweets)
        
# Question 1.4
lamda = 0.001
T = 500
B = 500
wt = np.zeros(train_features.shape[1])
error_rates = np.zeros(T)
for j in range(T):
    t = j+1
    At_plus = np.zeros(train_features.shape[1])
    a_v = np.random.randint(1, train_features.shape[0], B )
    #print a_v
    #Create At+
    count = 0
    for i in a_v:
        inner = np.inner(train_features.getrow(i).toarray(), wt) * sentiments[i]
        if (inner < 1):
            #At_plus.append(train_features.getrow(i).toarray() * sentiments[i])
            At_plus = np.add(At_plus, train_features.getrow(i).toarray() * sentiments[i])
        #print sentiments[i], inner
        if inner < 0:
            count += 1
            
    #error_rate = float(count)/float(B)
    #error_rates.append(error_rate)
    error_rates[j] = float(count)/float(B)
    
    #sum_At = np.sum (At_plus, axis=0)
    eta_t = 1/(t*lamda)
    nabla_t = lamda*wt - eta_t/B*At_plus
    wt_prime = wt - eta_t*nabla_t
    wt = wt_prime*min(1,((1/math.sqrt(lamda))/np.linalg.norm(wt_prime)))    

plt.plot(error_rates, 'ro')
plt.show()

#Question 1.5
lamda = 0.01
T = 5000
B = 200
w_adagrad = np.ones(train_features.shape[1])
w_adagrad = np.multiply(0.1, w_adagrad)
Gt = np.multiply(0, w_adagrad)
error_rates = np.zeros(T)
error_rates_test = np.zeros(T/10)
test_index = 0

for j in range(T):
    t = j+1
    At_plus = np.zeros(train_features.shape[1])
    a_v = np.random.randint(1, train_features.shape[0], B )
    
    count = 0
    for i in a_v:
        inner = np.inner(train_features.getrow(i).toarray(), w_adagrad) * sentiments[i]
        if (inner < 1):
            At_plus = np.add(At_plus, train_features.getrow(i).toarray() * sentiments[i])
        if inner < 0:
            count += 1
            
    if (t % 10 == 1):
        test_error = 0
        for m in range(498):
            some = test_features.getrow(m).toarray()
            inner = np.inner(some[0], w_adagrad) * test_sentiments[m]
            if inner < 0:
                test_error += 1
        error_rates_test[test_index] = float(test_error)/float(498)
        test_index += 1
        
    error_rates[j] = float(count)/float(B)
    
    #sum_At = np.sum (At_plus, axis=0)
    eta_t = 1/(t*lamda)
    nabla_t = lamda*w_adagrad - eta_t/B*At_plus
    
    #outer = np.outer(nabla_t, np.transpose(nabla_t)) <- causes a memory error
    #diag = outer.diagonal()
    diag = np.square(nabla_t)
    #Gt = np.add(Gt, np.sqrt(diag))
    Gt = np.add(Gt, diag)
    GtR = np.sqrt(Gt)
    GtR = np.reciprocal(GtR)
    #GtR = np.invert(Gt)
    w_adagrad = w_adagrad - eta_t * np.multiply(GtR, nabla_t)
'''
lamda = 0.01
T = 500
B = 500
w_adagrad = np.ones(train_features.shape[1])
w_adagrad = np.multiply(0.1, w_adagrad)
Gt = np.multiply(0, w_adagrad)
error_rates = np.zeros(T)
error_rates_test = np.zeros(200)
test_index = 0

for j in range(T):
    t = j+1
    At_plus = np.zeros(train_features.shape[1])
    a_v = np.random.randint(1, train_features.shape[0], B )
    
    count = 0
    for i in a_v:
        inner = np.inner(train_features.getrow(i).toarray(), w_adagrad) * sentiments[i]
        if (inner < 1):
            At_plus = np.add(At_plus, train_features.getrow(i).toarray() * sentiments[i])
        if inner < 0:
            count += 1
            
    if (t % 10 == 1):
        test_error = 0
        for m in range(498):
            some = test_features.getrow(m).toarray()
            inner = np.inner(some[0], w_adagrad) * test_sentiments[m]
            if inner < 0:
                test_error += 1
        error_rates_test[test_index] = float(test_error)/float(498)
        test_index += 1
        
    error_rates[j] = float(count)/float(B)
    
    #sum_At = np.sum (At_plus, axis=0)
    eta_t = 1/(t*lamda)
    nabla_t = lamda*w_adagrad - eta_t/B*At_plus
    
    #outer = np.outer(nabla_t, np.transpose(nabla_t)) <- causes a memory error
    #diag = outer.diagonal()
    diag = np.square(nabla_t)
    #Gt = np.add(Gt, np.sqrt(diag))
    Gt = np.add(Gt, diag)
    GtR = np.sqrt(Gt)
    GtR = np.reciprocal(GtR)
    #GtR = np.invert(Gt)
    w_adagrad = w_adagrad - eta_t * np.multiply(GtR, nabla_t)
'''    
"""            
#Question 1.5
lamda = 0.001
T = 2000
B = 200
wt = np.ones(train_features.shape[1])
wt = np.multiply(0.1, wt)
Gt = np.multiply(0, wt)
error_rates = np.zeros(T)
error_rates_test = np.zeros(200)
test_index = 0
for j in range(T):
    t = j+1
    At_plus = np.zeros(train_features.shape[1])
    a_v = np.random.randint(1, train_features.shape[0], B )
    #print a_v
    #Create At+
    count = 0
    for i in a_v:
        inner = np.inner(train_features.getrow(i).toarray(), wt) * sentiments[i]
        if (inner < 1):
            #At_plus.append(train_features.getrow(i).toarray() * sentiments[i])
            At_plus = np.add(At_plus, train_features.getrow(i).toarray() * sentiments[i])
        #print sentiments[i], inner
        if inner < 0:
            count += 1
            
    #error_rate = float(count)/float(B)
    #error_rates.append(error_rate)
    error_rates[j] = float(count)/float(B)
    
    #sum_At = np.sum (At_plus, axis=0)
    eta_t = 1/(t*lamda)
    nabla_t = lamda*wt - eta_t/B*At_plus
    
    #outer = np.outer(nabla_t, np.transpose(nabla_t)) <- causes a memory error
    #diag = outer.diagonal()
    diag = np.square(nabla_t)
    Gt = np.add(Gt, np.sqrt(diag))
    GtR = np.reciprocal(Gt)
    #GtR = np.invert(Gt)
    wt = wt - eta_t * np.multiply(GtR, nabla_t)
    
    #wt_prime = wt - eta_t*nabla_t
    #wt = wt_prime*min(1,((1/math.sqrt(lamda))/np.linalg.norm(wt_prime)))    

plt.plot(error_rates, 'ro')
plt.show()


for j in range(T):
    t = j+1
    At_plus = np.zeros(train_features.shape[1])
    a_v = np.random.randint(1, train_features.shape[0], B )
    #print a_v
    #Create At+
    count = 0
    for i in a_v:
        inner = np.inner(train_features.getrow(i).toarray(), wt) * sentiments[i]
        if (inner < 1):
            #At_plus.append(train_features.getrow(i).toarray() * sentiments[i])
            At_plus = np.add(At_plus, train_features.getrow(i).toarray() * sentiments[i])
        #print sentiments[i], inner
        if inner < 0:
            count += 1
    
    if (t % 10 == 0):
        test_error = 0
        for m in range(498):
            some = test_features.getrow(m).toarray()
            inner = np.inner(some[0], wt) * test_sentiments[m]
            if inner < 0:
                test_error += 1
        error_rates_test[test_index] = float(test_error)/float(498)
        test_index += 1
           
    #error_rate = float(count)/float(B)
    #error_rates.append(error_rate)
    error_rates[j] = float(count)/float(B)
    
    #sum_At = np.sum (At_plus, axis=0)
    eta_t = 1/(t*lamda)
    nabla_t = lamda*wt - eta_t/B*At_plus
    wt_prime = wt - eta_t*nabla_t
    wt = wt_prime*min(1,((1/math.sqrt(lamda))/np.linalg.norm(wt_prime)))    

plt.plot(error_rates)
plt.plot(error_rates_test)
plt.show()
"""