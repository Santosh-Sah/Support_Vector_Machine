# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:09:12 2020

@author: Santosh Sah
"""

"""
importing the libraries
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importSupportVectorMachineDataset(supportVectorMachineDatasetFileName):
    
    supportVectorMachineDataset = pd.read_csv(supportVectorMachineDatasetFileName)
    X = supportVectorMachineDataset.iloc[:, [2, 3]].values
    y = supportVectorMachineDataset.iloc[:, 4].values
    
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test

"""
Save standard scalar object as a pickel file. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def saveSupportVectorMachineStandardScaler(supportVectorMachineStandardScalar):
    
    #Write SupportVectorMachineStandardScaler in a picke file
    with open("SupportVectorMachineStandardScaler.pkl",'wb') as SupportVectorMachineStandardScaler_Pickle:
        pickle.dump(supportVectorMachineStandardScalar, SupportVectorMachineStandardScaler_Pickle, protocol = 2)

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)
    
    #Write y_test in a picke file
    with open("y_test.pkl",'wb') as y_test_Pickle:
        pickle.dump(y_test, y_test_Pickle, protocol = 2)

"""
Save SupportVectorMachineModel as a pickle file.
"""
def saveSupportVectorMachineModel(supportVectorMachineModel):
    
    #Write SupportVectorMachineModel as a picke file
    with open("SupportVectorMachineModel.pkl",'wb') as SupportVectorMachineModel_Pickle:
        pickle.dump(supportVectorMachineModel, SupportVectorMachineModel_Pickle, protocol = 2)

"""
read SupportVectorMachineStandardScalar from pickel file
"""
def readSupportVectorMachineStandardScaler():
    
    #load SupportVectorMachineStandardScaler object
    with open("SupportVectorMachineStandardScaler.pkl","rb") as SupportVectorMachineStandardScaler:
        supportVectorMachineStandardScalar = pickle.load(SupportVectorMachineStandardScaler)
    
    return supportVectorMachineStandardScalar

"""
read supportVectorMachineModel from pickle file
"""
def readSupportVectorMachineModel():
    
    #load SupportVectorMachineModel model
    with open("SupportVectorMachineModel.pkl","rb") as SupportVectorMachineModel:
        supportVectorMachineModel = pickle.load(SupportVectorMachineModel)
    
    return supportVectorMachineModel

"""
read X_train from pickle file
"""
def readSupportVectorMachineXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readSupportVectorMachineXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readSupportVectorMachineYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
read y_test from pickle file
"""
def readSupportVectorMachineYTest():
    
    #load y_test
    with open("y_test.pkl","rb") as y_test_pickle:
        y_test = pickle.load(y_test_pickle)
    
    return y_test

"""
save y_pred as a pickle file
"""

def saveSupportVectorMachineYPred(y_pred):
    
    #Write y_red in a picke file
    with open("y_pred.pkl",'wb') as y_pred_Pickle:
        pickle.dump(y_pred, y_pred_Pickle, protocol = 2)

"""
read y_predt from pickle file
"""
def readSupportVectorMachineYPred():
    
    #load y_test
    with open("y_pred.pkl","rb") as y_pred_pickle:
        y_pred = pickle.load(y_pred_pickle)
    
    return y_pred