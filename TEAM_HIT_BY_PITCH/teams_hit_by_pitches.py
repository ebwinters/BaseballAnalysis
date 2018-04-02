import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load in data
teams = "~/Desktop/DataAnalysis_Udemy/BaseballAnalysis/baseballdatabank-master/core/Teams.csv"
teams_df = pd.read_csv(teams)

odds_by_team_id = {}

def get_odds_hit_by_pitch(team_id):
  #only get teams with team_id, and drop any columns with no data (probably from early years of baseball)
  df = teams_df.loc[teams_df['teamID'] == team_id].dropna()
  df = df.groupby(by='teamID', as_index=False)['teamID', 'HBP', 'AB'].sum()
  #add to dictionary to use later in plotting
  odds_by_team_id[team_id] = float(df['HBP']/df['AB'])

team_id_list = [
  'ARI',
  'ATL',
  'BAL',
  'BOS',
  'CHA',
  'CHN',
  'CIN',
  'CLE',
  'COL',
  'DET',
  'HOU',
  'KCA',
  'LAA',
  'LAN',
  'MIA',
  'MIL',
  'MIN',
  'NYA',
  'NYN',
  'OAK',
  'PHI',
  'PIT',
  'SDN',
  'SEA',
  'SFN',
  'SLN',
  'TBA',
  'TEX',
  'TOR',
  'WAS'
]

for team_id in team_id_list:
  get_odds_hit_by_pitch(team_id)

barchart = sns.barplot(x=list(odds_by_team_id.keys()), y=list(odds_by_team_id.values()), palette='deep')
barchart.set(xlabel='Team', ylabel='% Chance hit by pitch')
barchart.tick_params(labelsize=5)
plt.show(barchart)
