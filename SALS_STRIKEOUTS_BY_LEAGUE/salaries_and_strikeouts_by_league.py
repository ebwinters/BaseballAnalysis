import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

salaries = "~/Desktop/DataAnalysis_Udemy/BaseballAnalysis/baseballdatabank-master/core/Salaries.csv"
salaries_df = pd.read_csv(salaries)

pitching = "~/Desktop/DataAnalysis_Udemy/BaseballAnalysis/baseballdatabank-master/core/Pitching.csv"
pitching_df = pd.read_csv(pitching)

#step 1: average salaries per year for players over all years played where player_ids are in the pitcher table
#drop all rows with one or more NA values
pitching_df = pitching_df.dropna(thresh=1)
salaries_df = salaries_df.dropna(thresh=1)
#entry for each person in the salaries table with an entry if they were in the AL and if they were in the NL, with avg salary for those
#years in that league
sum_strikeouts_per_league = pitching_df.groupby(by=['lgID', 'playerID'], as_index=False)['SO'].sum()

average_salraies_df = salaries_df.groupby(by=['lgID', 'playerID'], as_index=False)['salary'].mean()
average_salraies_df['avg_sal'] = average_salraies_df['salary']
average_salraies_df = average_salraies_df.drop(columns=['salary'])
#now we have a dataframe we can merge with average salary

#only way to get salaries for stricly pitchers is to merge
merged_df = average_salraies_df.merge(pitching_df, on=['playerID', 'lgID'], how='inner')
#df has entries only for pitchers with two rows for each payer max, which is their average salary in AL and NL respectively
merged_df = merged_df.groupby(by=['lgID', 'playerID'], as_index=False)['avg_sal'].mean()

#step 2: get number strikeouts for those pitchers
#now df has league id's, player ids for each league id, average salery for each league id, and strikeouts per pitcher in each league
#for only AL and NL
merged_df = merged_df.merge(sum_strikeouts_per_league, on=['playerID', 'lgID'], how='inner')
merged_df['lgID'] = merged_df[(merged_df.lgID == 'AL') | (merged_df.lgID == 'NL')]

#fit striekouts to x axis by grouping them by groups of 20
def myround(x, base=50):
    return int(base * round(float(x)/base))

merged_df['SO'] = merged_df['SO'].apply(myround)

#step 3: plot it
