import pandas as pd

d15 = pd.read_csv('../data/marathon_results_2015.csv')
d16 = pd.read_csv('../data/marathon_results_2016.csv')
d17 = pd.read_csv('../data/marathon_results_2017.csv')

d15['Year'] = 2015
d16['Year'] = 2016
d17['Year'] = 2017
df = pd.concat([d15,d16,d17])

df['Stage1'] = df['10K'] - df['5K']
df['Stage2'] = df['15K'] - df['10K']
df['Stage3'] = df['20K'] - df['15K']
df['Stage4'] = df['25K'] - df['20K']
df['Stage5'] = df['30K'] - df['25K']
df['Stage6'] = df['35K'] - df['30K']
df['Stage7'] = df['40K'] - df['35K']
df['Stage8'] = df['Official Time'] - df['40K']

df['Fitness'] = 

df = df['Age','M/F','Country','Fitness','Stage1','Stage2','Stage3','Stage4','Stage5','Stage6','Stage7','Stage8']

