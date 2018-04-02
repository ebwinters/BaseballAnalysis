import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

people = "~/Desktop/DataAnalysis_Udemy/BaseballAnalysis/baseballdatabank-master/core/People.csv"
batting = "~/Desktop/DataAnalysis_Udemy/BaseballAnalysis/baseballdatabank-master/core/Batting.csv"

people_df = pd.read_csv(people)
batting_df = pd.read_csv(batting)

'''ISSUE: for players in the batting table, there are rows with batting data for each year they played. Need to get mean HRs 
for all years played and put it into a series to use later on.'''

def get_correct_hr_data(df):
  #grouping primarily by playerID, and only keep playerID and homerun columns
  df = df.groupby(by='playerID', as_index=False)['playerID', 'HR']
  #sum the total amount of homeruns for each player, since there might be different values for each season
  df = df.sum()
  #returns df with only playeerIDs and total number of homeruns over career
  return df

hr_df = get_correct_hr_data(batting_df)

# merge both tables to get playerID/homeruns and height/weight in same table
merged_df = people_df.merge(hr_df, on='playerID', how='inner')

#can't correlate 3 columns, so need to calculate BMI to use pearson's r
merged_df['BMI'] = (merged_df['weight']/(merged_df['height'] * merged_df['height']) * 703)
'''pearson's r = 0.10976, some positive correlation'''

data_by_height_weight = merged_df.groupby(['height', 'weight'], as_index=False).mean()
#scale bubbles properly by normalizing
scaled_bubbles = data_by_height_weight['HR']-data_by_height_weight['HR'].mean()/data_by_height_weight['HR'].std()
plt.scatter(data_by_height_weight['height'], data_by_height_weight['weight'], s=scaled_bubbles)
plt.ylabel('Weight (lb)')
plt.xlabel('Height (in)')
plt.show()