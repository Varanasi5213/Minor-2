import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

    
def data_description(data):
    def describe(data):
        print('\033[1m',"The data is distributed as:\n",'\033[0m',data.describe())
        print("\n\n")
    describe(data)
    def mean(data):
        print('\033[1m',"\n\n\nThe mean of the various attributes are:\n",'\033[0m',data.mean())
        sns.distplot(data.mean())
        
    mean(data)
    plt.title("Mean plot")
    plt.show()
    def var(data):
        print('\033[1m',"\n\n\nThe variance of various attributes are:\n",'\033[0m',data.var())
        sns.distplot(data.var())
        
    var(data)
    plt.title("Variance plot")
    plt.show()
    def corr(data):
        print('\033[1m',"\n\n\nThe correlation between various attributes:\n ",'\033[0m')
        plt.figure(figsize=(16,10))
        sns.heatmap(data.corr(), annot=True)
       
    corr(data)
    plt.title("Correlation plot")
    plt.show()
    def skew(data):
        print('\033[1m',"\n\n\nThe skewness in various attributes of data:\n",'\033[0m',data.skew())
        sns.distplot(data.skew())
       
    skew(data)
    plt.title("Skewness plot")
    plt.show()
    def kurt(data):
        print('\033[1m',"\n\n\nThe kurtosis in various attributes of data:\n",'\033[0m',data.kurtosis())
        sns.distplot(data.kurtosis())
       
    kurt(data)
    plt.title("Kurtosis plot")
    plt.show()
