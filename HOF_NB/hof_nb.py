import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split   #split data into train and test
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#import data and clean all rows that have NA values
batting_post = "~/Desktop/DataAnalysis_Udemy/BaseballAnalysis/baseballdatabank-master/core/Batting.csv"
batting_post_df = pd.read_csv(batting_post)[['playerID', 'yearID', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB',  'CS',  'BB', 'SO']].dropna()
batting_post_df['twoB'] = batting_post_df['2B']
batting_post_df['threeB'] = batting_post_df['3B']
batting_post_df = batting_post_df.drop(['2B', '3B'], axis=1)
hof = "~/Desktop/DataAnalysis_Udemy/BaseballAnalysis/baseballdatabank-master/core/HallOfFame.csv"
hof_df = pd.read_csv(hof)[['playerID','inducted']].dropna()

#change inducted column to use integers so that max() returns if they were every inducted or not
hof_df['inducted'] = hof_df['inducted'].map({'Y': 1, 'N': 0})
# hof_df['inducted'] = hof_df['inducted'].eq('Y').mul(1)

#need to get dataframe with playerIDs, batting stats, and if they have either been inducted or not, since they could've been
#on multiple ballots
def hof_or_not (lst):
  if max(lst) == 1:
    return 1
  else:
    return 0

hof_df = hof_df[hof_df['inducted'] == hof_df.groupby(['playerID'])['inducted'].transform(hof_or_not)]
hof_df = hof_df.drop_duplicates()

#NOW I HAVE A DATAFRAME IN HOF_DF WITH THE PLAYERIDS AND IF THEY HAVE EVER BEEN INDUCTED TO THE HOF OR NOT, TIME TO MERGE & DROP YEAR
merged_df = hof_df.merge(batting_post_df, on=['playerID'], how='inner').drop(labels=['yearID'], axis=1)

#need to average out all columns of statistics, since there are multiple years and this will get a better overall picture of the 
#player's career
merged_df = merged_df.groupby(['playerID'], as_index=False).mean()
merged_df = merged_df.drop(labels=['playerID'], axis=1)

#need to put all columns into binary format for NaiveBayes to work
merged_df_copy = merged_df
le = preprocessing.LabelEncoder()
inducted_cat = le.fit_transform(merged_df_copy.inducted)
G_cat = le.fit_transform(merged_df_copy.G)
AB_cat = le.fit_transform(merged_df_copy.AB)
R_cat = le.fit_transform(merged_df_copy.R)
H_cat = le.fit_transform(merged_df_copy.H)
twoB_cat = le.fit_transform(merged_df_copy.twoB)
threeB_cat = le.fit_transform(merged_df_copy.threeB)
HR_cat = le.fit_transform(merged_df_copy.HR)
RBI_cat = le.fit_transform(merged_df_copy.RBI)
SB_cat = le.fit_transform(merged_df_copy.SB)
CS_cat = le.fit_transform(merged_df_copy.CS)
BB_cat = le.fit_transform(merged_df_copy.BB)
SO_cat = le.fit_transform(merged_df_copy.SO)
#initialize the encoded categorical columns
merged_df_copy['inducted_cat'] = inducted_cat
merged_df_copy['G_cat'] = G_cat
merged_df_copy['AB_cat'] = AB_cat
merged_df_copy['R_cat'] = R_cat
merged_df_copy['H_cat'] = H_cat
merged_df_copy['twoB_cat'] = twoB_cat
merged_df_copy['threeB_cat'] = threeB_cat
merged_df_copy['HR_cat'] = HR_cat
merged_df_copy['RBI_cat'] = RBI_cat
merged_df_copy['SB_cat'] = SB_cat
merged_df_copy['CS_cat'] = CS_cat
merged_df_copy['BB_cat'] = BB_cat
merged_df_copy['SO_cat'] = SO_cat
merged_df_copy = merged_df_copy.drop(['inducted', 'G','AB', 'R', 'H', 'twoB', 'threeB', 'HR', 'RBI', 'SB',  'CS',  'BB', 'SO'], axis=1)

#now need to standardize data
features = ['G_cat','AB_cat', 'R_cat', 'H_cat', 'twoB_cat', 'threeB_cat', 'HR_cat', 'RBI_cat', 'SB_cat',  'CS_cat',  'BB_cat', 'SO_cat']
scaled_features = {}
for feature in features:
  mean, std = merged_df_copy[feature].mean(), merged_df_copy[feature].std()
  scaled_features[feature] = [mean, std]
  merged_df_copy.loc[:, feature] = (merged_df_copy[feature]-mean)/std

#split data into training and testing sets
features = merged_df_copy.values[...,1:]
target = merged_df_copy.values[:,:1]
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.33, random_state = 10)

#train model
clf = GaussianNB()
clf.fit(features_train, target_train.ravel())
prediction = clf.predict(features_test)

#get accuracy
accuracy = accuracy_score(target_test, prediction, normalize=True)
#0.6890756302521008 - not very good but it's my first try 

#simple test to see whether it picks up an obvious example of an all star, which it does.
#ISSUE WITH THIS TEST: DATA ISN'T NORMALIZED, NEED TO TAKE THAT INTO ACCOUNT WHEN WRITING INTERACTIVE PART
# ethan_test = [[7,28,3,6,0,0,0,2,2,0,2,3]]
# prediction1 = clf.predict(ethan_test)
# print (prediction1)


#function for interactive.py
def make_prediction(test_data_dict):
  adj_feats = {
  'G':'G_cat',
  'AB':'AB_cat',
  'R':'R_cat',
  'H':'H_cat',
  'twoB':'twoB_cat',
  'threeB':'threeB_cat',
  'HR':'HR_cat',
  'RBI':'RBI_cat',
  'SB':'SB_cat',
  'CS':'CS_cat',
  'BB':'BB_cat',
  'SO':'SO_cat'
  }
  test_data_df = pd.DataFrame(test_data_dict, index=[0])
  #standardize input data
  for feature in adj_feats.keys():
    test_data_df[feature] = (test_data_df[feature]-scaled_features[adj_feats[feature]][0])/scaled_features[adj_feats[feature]][1]

  prediction = clf.predict(test_data_df.values)
  print (prediction)

  # prediction = clf.predict(test_data_df.values)
  # print (prediction)
