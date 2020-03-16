# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:01:45 2020

@author: Santosh Sah
"""

from SupportVectorMachineUtils import (readSupportVectorMachineXTest, readSupportVectorMachineModel,
                                     saveSupportVectorMachineYPred, readSupportVectorMachineStandardScaler)

"""
test the model on testing dataset
"""
def testSupportVectorMachineModel():
    
    X_test = readSupportVectorMachineXTest()
    supportVectorMachineStandardScaler = readSupportVectorMachineStandardScaler()
    X_test = supportVectorMachineStandardScaler.transform(X_test)
    
    supportVectorMachineModel = readSupportVectorMachineModel()
    
    y_pred = supportVectorMachineModel.predict(X_test)
    saveSupportVectorMachineYPred(y_pred)
    
    print(y_pred)
    
if __name__ == "__main__":
    testSupportVectorMachineModel()