import pandas as pd
import numpy as np
import runner

if __name__ == '__main__':

    train = pd.read_csv('../data/final_marathon_train.csv')
    test = pd.read_csv('../data/final_marathon_test.csv')

    #elevation_data_path = pd.read_csv('../data/final_elevation_changes.csv')
    runner_obj = runner.Runner(train)
    X = test.drop(runner.Runner.to_predict_names, axis=1)
    y = test[runner.Runner.to_predict_names]
    res = 0
    pred_res = 0
    for i in range(len(X)):
        pred = runner_obj.run_marathon(X.iloc[i].values)
        pred_res += pred
        res += sum(y.iloc[i])
        # print(abs(pred-sum(y.iloc[i])))
        print(pred, sum(y.iloc[i]), abs(pred-sum(y.iloc[i])))
    # print('Total Test instances',len(X))
    # print('Total difference',abs(res-pred_res))
