# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:39:09 2020

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from SupportVectorMachineUtils import (readSupportVectorMachineYTest, readSupportVectorMachineYPred)

"""

calculating logistic regression confussion matrix

"""
def testSupportVectorMachineConfussionMatrix():
    
    y_test = readSupportVectorMachineYTest()
    y_pred = readSupportVectorMachineYPred()
    
    supportVectorMachineConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(supportVectorMachineConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[57  1]
    [ 6 16]]
    
    """
"""
calculating accuracy score

"""

def testSupportVectorMachineAccuracy():
    
    y_test = readSupportVectorMachineYTest()
    y_pred = readSupportVectorMachineYPred()
    
    supportVectorMachineConfussionAccuracy = accuracy_score(y_test, y_pred)
    
    print(supportVectorMachineConfussionAccuracy) #.9125%

"""
calculating classification report

"""

def testSupportVectorMachineClassificationReport():
    
    y_test = readSupportVectorMachineYTest()
    y_pred = readSupportVectorMachineYPred()
    
    supportVectorMachineConfussionClassificationReport = classification_report(y_test, y_pred)
    
    print(supportVectorMachineConfussionClassificationReport)
    
    """
               precision    recall  f1-score   support

          0       0.90      0.98      0.94        58
          1       0.94      0.73      0.82        22

avg / total       0.91      0.91      0.91        80
    """
    
if __name__ == "__main__":
    #testSupportVectorMachineConfussionMatrix()
    #testSupportVectorMachineAccuracy()
    testSupportVectorMachineClassificationReport()