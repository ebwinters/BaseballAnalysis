from hof_nb import make_prediction
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

data = input("Please input a comma deliniated list of values corresponding to the correct rows in the BattingPost.csv table:\n")
data_lst = data.split(",")
data_lst = list(map(float, data_lst))
data_lst_dict = {
  'G':data_lst[0],
  'AB':data_lst[1],
  'R':data_lst[2],
  'H':data_lst[3],
  'twoB':data_lst[4],
  'threeB':data_lst[5],
  'HR':data_lst[6],
  'RBI':data_lst[7],
  'SB':data_lst[8],
  'CS':data_lst[9],
  'BB':data_lst[10],
  'SO':data_lst[11]
}

make_prediction(data_lst_dict)

# print(test_data_df_copy)
# make_prediction(data_lst_dict)