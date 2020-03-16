# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:46:53 2020

@author: Santosh Sah
"""

from SupportVectorMachineUtils import (importSupportVectorMachineDataset, saveTrainingAndTestingDataset)

def preprocess():
    
    X_train, X_test, y_train, y_test = importSupportVectorMachineDataset("Support_Vector_Machine_Social_Network_Ads.csv")
    
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    preprocess()