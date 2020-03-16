# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:02:07 2020

@author: Santosh Sah
"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from SupportVectorMachineUtils import (saveSupportVectorMachineModel, readSupportVectorMachineXTrain, readSupportVectorMachineYTrain,
                                     saveSupportVectorMachineStandardScaler)

"""
Train SupportVectorMachine model 
"""
def trainSupportVectorMachineModel():
    
    supportVectorMachineStandardScalar = StandardScaler()
    
    X_train = readSupportVectorMachineXTrain()
    y_train = readSupportVectorMachineYTrain()
    
    supportVectorMachineStandardScalar.fit(X_train)
    saveSupportVectorMachineStandardScaler(supportVectorMachineStandardScalar)
    
    X_train = supportVectorMachineStandardScalar.transform(X_train)
    
    supportVectorMachine = SVC(kernel = "linear", random_state = 1234)
    supportVectorMachine.fit(X_train, y_train)
    
    saveSupportVectorMachineModel(supportVectorMachine)

if __name__ == "__main__":
    trainSupportVectorMachineModel()