# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:48:29 2020

@author: Santosh Sah
"""

import pandas as pd
from SupportVectorMachineUtils import readSupportVectorMachineModel, readSupportVectorMachineStandardScaler

def predict():
    
    supportVectorMachine = readSupportVectorMachineModel()
    supportVectorMachineStandardScaler = readSupportVectorMachineStandardScaler()
    
    inputValue = [[26, 1000]]
    inputValueDataframe = pd.DataFrame(supportVectorMachineStandardScaler.transform(inputValue))
    
    predictedValue = supportVectorMachine.predict(inputValueDataframe.values)
    
    print(predictedValue)

if __name__ == "__main__":
    predict()