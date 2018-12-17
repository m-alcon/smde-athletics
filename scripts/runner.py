import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

class Runner:
    __coefs = {}
    __intercepts = {}
    #__models = {}
    to_predict_names = ['Stage0','Stage1','Stage2','Stage3','Stage4',
            'Stage5','Stage6','Stage7','Stage8']

    unused_columns = ['Year']

    def __init__(self,runners_data, elevation_data):
        self.runners_data = runners_data
        self.elevation_data = elevation_data
        self.total_time = -1
        self.__fit()

    def __fit(self):
        for name in self.to_predict_names:
            X = self.runners_data.drop(self.to_predict_names,axis=1)
            X = X.drop(self.unused_columns,axis=1)
            elevation = self.elevation_data[self.elevation_data['Stage']==name]['Elevation']
            X['Elevation'] = float(elevation)
            model = linear_model.LinearRegression()
            self.__coefs[name] = model.fit(X,self.runners_data[name]).coef_
            self.__intercepts[name] = model.fit(X,self.runners_data[name]).intercept_
            #self.__models[name] = model
            print(X.columns.values.tolist())
            print(self.__coefs[name])

    def predict(self,stage,data):
        elevation = self.elevation_data[self.elevation_data['Stage']==stage]['Elevation']
        data = np.append(data,elevation)
        #print(self.__models[stage].predict([data]))
        #print(np.sum(np.array(self.__coefs[stage]) * np.array(data)) + self.__intercepts[stage])
        return np.sum(np.array(self.__coefs[stage]) * np.array(data)) + self.__intercepts[stage]

    def run_marathon(self,data):
        total_time = 0
        for stage in self.to_predict_names:
            total_time += self.predict(stage,data.values)
        self.total_time = total_time[0]
        #self.total_time += np.random.normal(0.5,0.5)
        return self.total_time

if __name__ == '__main__':
    runners_data_path = pd.read_csv('../data/final_marathon.csv')
    train, test = train_test_split(runners_data_path,
                        test_size=0.005, random_state=42)
    elevation_data_path = pd.read_csv('../data/final_elevation_changes.csv')
    runner = Runner(train,elevation_data_path)
    X = test.drop(Runner.to_predict_names, axis=1)
    X = X.drop(Runner.unused_columns,axis=1)
    y = test[Runner.to_predict_names]
    for i in range(len(X)):
        pred = runner.run_marathon(X.iloc[i])
        print(pred, sum(y.iloc[i]))



