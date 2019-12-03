# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:14:44 2018

@author: mayur.a
"""
               
import matplotlib.pyplot as plt
import csv
import random
import math

def Data_Normalization(dataset):
    mylist = []
    for  i in range(1,len(dataset[0])):
        for j in range(len(dataset)):
            mylist.append(float(dataset[j][i]))
#        plt.plot(mylist)
#        plt.show()
        minv = min(mylist)
        maxv = max(mylist)
        for j in range(len(dataset)):
            dataset[j][i] = float(mylist[j] - minv)/(maxv - minv)
        mylist.clear()
  
def loadcsvfile(filename):
    lines = csv.reader(open(filename, "r"))
    next(lines)
    dataset = list(lines)
    for i in range(len(dataset)):
        for j in range(1,len(dataset[0]) ):
            dataset[i][j] = float(dataset[i][j])
    Data_Normalization(dataset)
    return dataset
#    print(dataset)

def calculatemean(dataset):
    means = []
    for i in range(1,len(dataset[0])):
        means.append(dataset[0][i])
    for i in range(len(dataset)):
        for j in range(1,len(dataset[0])):
            means[j] += dataset[i][j]
    for j in range(1,len(dataset[0])):
        means[j] /= len(dataset)
    return means
        
def Calculatesd(dataset, mean):
    sd = []
    for i in range(len(dataset[0])):
        sd.append(pow(dataset[0][i] - mean[i], 2))
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            sd[j] += pow(dataset[i][j] - mean[j], 2)
    for j in range(len(dataset[0])):
        sd[j] /= len(dataset)
    return sd

def summary(dataset):
    mean = calculatemean(dataset)
    StandardDeviation = Calculatesd(dataset, mean)  #sigma square
    return mean, StandardDeviation

def findmeannVarianceforclass(classsummers):
    classdensity = {}
    for classvalue, instance in classsummers.items():
        classdensity[classvalue] = summary(instance)
#    print(classdensity)
    return classdensity

def findclassconditiondensity(dataset):
    classvalue = {}
    classprobability = {}
    for i in range(len(dataset)):
        vector = dataset[i][-1]
        if(vector not in classvalue):
            classvalue[vector] = []
        classvalue[vector].append(dataset[i])
    #      print(classvalue)
    for outclassvalue, instance in classvalue.items():
        classprobability[outclassvalue] = float(len(instance))/len(dataset)
    
    summeryclassdensity = findmeannVarianceforclass(classvalue)

    return classprobability, summeryclassdensity

def GaussianProb(x, mean, sigma):
    value = math.exp(-pow((x-mean), 2)/(2 * sigma))
    return value * math.sqrt(2*math.pi * sigma)

def getprobinclass(mean, sigma, testdata):
    factor = 1
    for i in range(len(testdata) - 1):
        factor *= GaussianProb(testdata[i], mean[i], sigma[i])
    return factor        

def getclasspredection(testdata, classprob, density):
    ccd = {}
    for classvalue, instance in density.items():
        ccd[classvalue] = getprobinclass(instance[0], instance[1], testdata) * classprob[classvalue]
    
    sortclass = sorted(ccd, key=ccd.__getitem__, reverse=True)
    return sortclass[0]

def main():
    dataset = loadcsvfile("Medical_data.csv")   
    traindatacount = len(dataset) * 0.7
    random.shuffle(dataset)
    count = 0
    traindata = []
    testdata = []
    for data in dataset:
        if count < traindatacount:
            traindata.append(data)
        else:
            count+=1
 #   print(testdata)
    classprobability, density = findclassconditiondensity(traindata)
            
#   print(classprobability)
#    print(density)

    output = []    
    for data in testdata:
        output.append(getclasspredection(data, classprobability, density))
        
    count = 0
    for i in range(len(testdata)):
        print(output[i], testdata[i][-1])
        if output[i] == testdata[i][-1]:
            count += 1
    
    print("Accuarcy:", count * 100/float(len(testdata)))
        
main()

