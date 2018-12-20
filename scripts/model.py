import pandas as pd
import numpy as np
from scipy.stats import shapiro
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan


class Model(object):
    def __init__(self, coefs, intercept, residuals):
        self.coefs = coefs
        self.intercept = intercept
    
    def test_assumptions(self):
        print("Test Durbin-Watson")
        
        print("Test Breusch-Pagan")
        print("Test Shapiro")