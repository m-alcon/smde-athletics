import pandas as pd
from sklearn import linear_model

class Runner:
    __models = {}

    def __init__(self,runners_data_path, elevation_data_path):
        self.runners_data = pd.read_csv(runners_data_path)
        self.elevation_data = pd.read_csv(elevation_data_path)
        self.to_predict_names = ['Stage0','Stage1','Stage2','Stage3','Stage4',
            'Stage5','Stage6','Stage7','Stage8']
        self.__fit()

    def __fit(self):
        for name in self.to_predict_names:
            X = self.runners_data.drop(self.to_predict_names,axis=1)
            elevation = self.elevation_data[self.elevation_data['Stage']==name]['Elevation']
            X['Elevation'] = float(elevation)
            self.__models[name] = linear_model.LinearRegression()
            self.__models[name].fit(X,self.runners_data[name])
    def predict(self,stage,data):
        return self.__models[stage].predict(data)

if __name__ == '__main__':
    runners_data_path = '../data/final_marathon.csv'
    elevation_data_path = '../data/final_elevation_changes.csv'
    runner = Runner(runners_data_path,elevation_data_path)


