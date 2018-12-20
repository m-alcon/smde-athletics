import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn import preprocessing

run_one_hot = False

def time_to_sec(t):
    t = t.split(':')
    if len(t) == 3:
        return int(t[0])*3600 + int(t[1])*60 + int(t[2])
    print(t)
    return t[0]

def erase_F(s):
    if 'F' in s:
        s = s[1:]
    return s

def before_one_hot(col):
    ohe = preprocessing.OneHotEncoder(sparse=False)
    ohe_col = np.transpose(ohe.fit_transform(col.values.reshape(len(col),1)))
    return ohe_col

def preprocess():

    d15 = pd.read_csv('../data/marathon_results_2015.csv')
    d16 = pd.read_csv('../data/marathon_results_2016.csv')
    d17 = pd.read_csv('../data/marathon_results_2017.csv')

    d15['Year'] = 2015
    d16['Year'] = 2016
    d17['Year'] = 2017
    df = pd.concat([d15,d16,d17], ignore_index=True)
    df = df.replace('-', np.nan)
    df = df.dropna(subset=['Bib','Age','M/F','Country','5K','10K','15K','20K','25K','30K','35K','40K','Official Time'])

    for colname in ['5K','10K','15K','20K','25K','30K','35K','40K','Official Time']:
        df[colname] = [time_to_sec(x) for x in df[colname]]

    df['Stage0'] = df['5K']
    df['Stage1'] = df['10K'].sub(df['5K'])
    df['Stage2'] = df['15K'].sub(df['10K'])
    df['Stage3'] = df['20K'].sub(df['15K'])
    df['Stage4'] = df['25K'].sub(df['20K'])
    df['Stage5'] = df['30K'].sub(df['25K'])
    df['Stage6'] = df['35K'].sub(df['30K'])
    df['Stage7'] = df['40K'].sub(df['35K'])
    df['Stage8'] = df['Official Time'].sub(df['40K'])

    df['Fitness'] = [int(erase_F(x)) for x in df['Bib']]
    nmax = max(df['Fitness'])
    fitness_threshold = 23000
    df['Fitness'] = [(1000*x)/fitness_threshold if x<fitness_threshold else 1000 for x in df['Fitness']]
    #print(max(df['Fitness']))
    #print(min(df['Fitness']))

    # Label Encoder
    le = preprocessing.LabelEncoder()
    gender_cat = list(df['M/F'].astype('category').cat.categories)
    country_cat = list(df['Country'].astype('category').cat.categories)
    df['Gender'] = le.fit_transform(df['M/F'])
    copy_country = df['Country']
    df['Country'] = le.fit_transform(df['Country'])

    # OneHot Encoder
    if run_one_hot:
        gender_oh = before_one_hot(df['Gender'])
        country_oh = before_one_hot(df['Country'])
        not_eq_counter = 0
        for i,c in enumerate(gender_cat):
            df[c] = gender_oh[i]
            if sum(df[c]) != sum(df['M/F'] == c):
                not_eq_counter += 1
        print('OneHot Gender differences: %d'%not_eq_counter)
        not_eq_counter = 0
        for i,c in enumerate(country_cat):
            df[c] = country_oh[i]
            if sum(df[c]) != sum(copy_country == c):
                not_eq_counter += 1
        print('OneHot Country differences: %d'%not_eq_counter)
        final_columns = ['Year','Age']+ gender_cat + country_cat + ['Stage0',
        'Stage1','Stage2','Stage3','Stage4','Stage5','Stage6','Stage7','Stage8']

        return df[final_columns]
    else:
        return df[['Year','Age','Gender','Country','Fitness','Stage0','Stage1',
                'Stage2','Stage3','Stage4','Stage5','Stage6','Stage7','Stage8']]



def add_race_info(df):
    dei = pd.read_csv('../data/races_info.csv')

    # Label Encoder
    le = preprocessing.LabelEncoder()
    weather_cat = list(dei['Weather'].astype('category').cat.categories)
    copy_weather = dei['Weather']
    dei['Weather'] = le.fit_transform(dei['Weather'])

    if run_one_hot:
        weather_oh = before_one_hot(dei['Weather'])
        not_eq_counter = 0
        for i,c in enumerate(weather_cat):
            dei[c] = weather_oh[i]
            if sum(dei[c]) != sum(copy_weather == c):
                not_eq_counter += 1
        dei = dei.drop(['Weather'],axis=1)
        print('OneHot Weather differences: %d'%not_eq_counter)

    return pd.merge(df,dei)

def mile_to_km(x):
    return(x*1.609344)

def show_elevation_plot(x,y,spl):
    plt.plot(x, y, 'ro', ms=5)
    xs = np.linspace(0, 42, 1000)
    plt.plot(xs, spl(xs), 'g', lw=3, alpha=0.7)
    plt.show()

def preprocess_elevation():
    de = pd.read_csv('../data/elevation_changes.csv')
    de['Km'] = de['Mileage'].apply(lambda x: x*1.609344)
    de['Elevation'] = de['Elevation'].apply(lambda x: x*0.3048)
    previous,act = 0,5.0
    spl = InterpolatedUnivariateSpline(de['Km'],de['Elevation'],k=3)
    #show_elevation_plot(de['Km'],de['Elevation'],spl)

    dp = pd.DataFrame(columns=['Stage','Elevation'])
    dp['Stage'] = ['Stage0','Stage1','Stage2','Stage3','Stage4',
            'Stage5','Stage6','Stage7','Stage8']
    ant,elevation = spl(0),[]
    for i in [5,10,15,20,25,30,35,40,42]:
        elevation.append(spl(i)-ant)
        ant = spl(i)
    dp['Elevation'] = elevation
    return dp

if __name__ == '__main__':
    df = preprocess()
    df = add_race_info(df)
    df.to_csv('../data/final_marathon.csv',index=False)
    dpec = preprocess_elevation()
    dpec.to_csv('../data/final_elevation_changes.csv',index=False)
