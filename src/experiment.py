import pandas as pd
import numpy as np
import runner

class Experiment(object):

    n = 10
    t = 2.262

    def __init__ (self,low,high,runner):
        self.low = low
        self.high = high
        self.runner = runner
        self.table_2k = None
        self.table_yates = None
        self.data_interval = {'means': [], 'stds': [], 'h': 0}

    def _running_repetitions (self,data_runner):
        all_runs = []
        for i in range(self.n):
            all_runs.append(self.runner.run_marathon([data_runner[0],data_runner[1],data_runner[2]]))
        return np.mean(all_runs), np.std(all_runs)

    def configuration_2k(self):
        m1,s1 = self._running_repetitions(
            [self.low[0],self.low[1],self.low[2]])
        m2,s2 = self._running_repetitions(
            [self.high[0],self.low[1],self.low[2]])
        m3,s3 = self._running_repetitions(
            [self.low[0],self.high[1],self.low[2]])
        m4,s4 = self._running_repetitions(
            [self.high[0],self.high[1],self.low[2]])
        m5,s5 = self._running_repetitions(
            [self.low[0],self.low[1],self.high[2]])
        m6,s6 = self._running_repetitions(
            [self.high[0],self.low[1],self.high[2]])
        m7,s7 = self._running_repetitions(
            [self.low[0],self.high[1],self.high[2]])
        m8,s8 = self._running_repetitions(
            [self.high[0],self.high[1],self.high[2]])
        self.data_interval['means'] = [m1,m2,m3,m4,m5,m6,m7,m8]
        self.data_interval['stds'] = [s1,s2,s3,s4,s5,s6,s7,s8]
        self.table_2k = pd.DataFrame()
        self.table_2k['Age'] = [self.low[0],self.high[0],self.low[0],self.high[0],self.low[0],self.high[0],self.low[0],self.high[0]]
        self.table_2k['Gender'] = [self.low[1],self.low[1],self.high[1],self.high[1],self.low[1],self.low[1],self.high[1],self.high[1]]
        self.table_2k['Fitness'] = [self.low[2],self.low[2],self.low[2],self.low[2],self.high[2],self.high[2],self.high[2],self.high[2]]
        self.table_2k['Results'] = list([m1,m2,m3,m4,m5,m6,m7,m8])

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

    def yates_algorithm (self):
        answers = self.data_interval['means']
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

    def _compute_h(self,std):
        return self.t * std/np.sqrt(self.n)

    def confidence_interval(self):
        df = pd.DataFrame()
        df['Mean'] = self.data_interval['means']
        df['S'] = self.data_interval['stds']
        df['h'] = [self._compute_h(s) for s in self.data_interval['stds']]
        df['Desired'] = 0.05*df['Mean']
        df['Low'] = df['Mean'] - df['h']
        df['High'] = df['Mean'] + df['h']
        self.data_interval['table'] = df

    def run(self):
        print('################### 2K CONFIGURATION ###################')
        answers = self.configuration_2k()
        print(self.table_2k.to_latex(),end='\n\n')

        print('################### YATES ###################')
        self.yates_algorithm()
        print(self.table_yates.to_latex(),end='\n\n')

        print('################### CONFIDENCE INTERVAL ###################')
        self.confidence_interval()
        print(self.data_interval['table'].to_latex(),end='\n\n')

if __name__ == '__main__':
    data = pd.read_csv('../data/final_marathon.csv')

    new_runner = runner.Runner(data)
    low = [data['Age'].min(),data['Gender'].min(),data['Fitness'].min()]
    high = [data['Age'].max(),data['Gender'].max(),data['Fitness'].max()]

    experiment = Experiment(low,high,new_runner)
    experiment.run()