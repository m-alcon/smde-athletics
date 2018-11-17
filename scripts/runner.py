import pandas as pd
from sklearn import linear_model

class Runner:
    __models = {}

    def __init__(self,runners_data,race_data):
        self.runners_data = runners_data
        self.race_data = race_data
        self.to_predict_names = ['Stage1','Stage2','Stage3','Stage4',
            'Stage5','Stage6','Stage7','Stage8']
        self.__fit()

    def __fit():
        for name in self.to_predict_names:
            self.__models[name] = linear_model.LinearRegression()
            self.__models[name].fit(
                self.runners_data.drop(self.to_predict_names,axis=1),
                self.runners_data[name]
            )
    def predict(stage,data):
        return self.__models[stage].predict(data)

if __name__ == '__main__':
    runners_data = pd.read_csv('../data/processed_marathon.csv')
    race_data = pd.read_csv('../data/races_info.csv')
    runner = Runner(runners_data,race_data)


