import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split   #split data into train and test
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#import data and clean all rows that have NA values
batting_post = "~/Desktop/DataAnalysis_Udemy/BaseballAnalysis/baseballdatabank-master/core/BattingPost.csv"
batting_post_df = pd.read_csv(batting_post)[['playerID', 'yearID', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB',  'CS',  'BB', 'SO']].dropna()

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

print (merged_df.loc[merged_df['playerID'] == 'rootch01'])
# print (merged_df.head())
