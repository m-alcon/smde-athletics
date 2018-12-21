import pandas as pd
import numpy as np
import runner

class Experiment(object):

    n = 10
    t = 3.250

    def __init__ (self,low,high,runner):
        self.low = low
        self.high = high
        self.runner = runner
        self.table_2k = None
        self.table_yates = None
        self.data_interval = {'table': [], 'mean': 0, 'std': 0, 'h': 0}

    def confidence_interval(self):
        all_runs = []
        for i in range(self.n):
            all_runs.append(self.runner.run_marathon([self.low[0],self.low[1],self.low[2]]))
        self.data_interval['mean'] = np.mean(all_runs)
        self.data_interval['std'] = np.std(all_runs)
        self.data_interval['h'] = self.t * np.std(all_runs)/np.sqrt(self.n)
        self.data_interval['table'] = pd.DataFrame(data={'Result': all_runs})
        self.data_interval['interval'] = (self.data_interval['mean']-self.data_interval['h'],
                                        self.data_interval['mean']+self.data_interval['h'])

    def _mean_running_repetitions (self,data_runner):
        all_runs = []
        for i in range(self.n):
            all_runs.append(self.runner.run_marathon([data_runner[0],data_runner[1],data_runner[2]]))
        return np.mean(all_runs)

    def configuration_2k(self):
        answers = []
        answers.append(self._mean_running_repetitions(
            [self.low[0],self.low[1],self.low[2]]))
        answers.append(self._mean_running_repetitions(
            [self.high[0],self.low[1],self.low[2]]))
        answers.append(self._mean_running_repetitions(
            [self.low[0],self.high[1],self.low[2]]))
        answers.append(self._mean_running_repetitions(
            [self.high[0],self.high[1],self.low[2]]))
        answers.append(self._mean_running_repetitions(
            [self.low[0],self.low[1],self.high[2]]))
        answers.append(self._mean_running_repetitions(
            [self.high[0],self.low[1],self.high[2]]))
        answers.append(self._mean_running_repetitions(
            [self.low[0],self.high[1],self.high[2]]))
        answers.append(self._mean_running_repetitions(
            [self.high[0],self.high[1],self.high[2]]))
        self.table_2k = pd.DataFrame()
        self.table_2k['Age'] = [self.low[0],self.high[0],self.low[0],self.high[0],self.low[0],self.high[0],self.low[0],self.high[0]]
        self.table_2k['Gender'] = [self.low[1],self.low[1],self.high[1],self.high[1],self.low[1],self.low[1],self.high[1],self.high[1]]
        self.table_2k['Fitness'] = [self.low[2],self.low[2],self.low[2],self.low[2],self.high[2],self.high[2],self.high[2],self.high[2]]
        self.table_2k['Results'] = answers
        return answers

    def _generate_effects (self,answers):
        res = [None]*8
        res[0] = answers[0] + answers[1]
        res[1] = answers[2] + answers[3]
        res[2] = answers[4] + answers[5]
        res[3] = answers[6] + answers[7]
        res[4] = answers[1] - answers[0]
        res[5] = answers[3] - answers[2]
        res[6] = answers[5] - answers[4]
        res[7] = answers[7] - answers[6]
        return res

    def yates_algorithm (self,answers):
        self.table_yates = pd.DataFrame()
        c1 = self._generate_effects(answers)
        self.table_yates['(1)'] = c1
        c2 = self._generate_effects(c1)
        self.table_yates['(2)'] = c2
        c3 = self._generate_effects(c2)
        self.table_yates['(3)'] = c3
        effect = [x/4 for x in c3]
        effect[0] = effect[0]/2
        self.table_yates['Effect'] = effect
        return effect

    def run(self):
        print('################### CONFIDENCE INTERVAL ###################')
        self.confidence_interval()
        print('mean:',self.data_interval['mean'])
        print('std:',self.data_interval['std'])
        print('h:',self.data_interval['h'])
        print('interval:',self.data_interval['interval'])
        print(self.data_interval['table'].to_latex(),end='\n\n')

        print('################### 2K CONFIGURATION ###################')
        answers = self.configuration_2k()
        print(self.table_2k.to_latex(),end='\n\n')

        print('################### YATES ###################')
        self.yates_algorithm(answers)
        print(self.table_yates.to_latex())

if __name__ == '__main__':
    data = pd.read_csv('../data/final_marathon.csv')

    new_runner = runner.Runner(data)
    low = [data['Age'].min(),data['Gender'].min(),data['Fitness'].min()]
    high = [data['Age'].max(),data['Gender'].max(),data['Fitness'].max()]

    experiment = Experiment(low,high,new_runner)
    experiment.run()