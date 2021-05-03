#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def data_description():
    def describe():
        print('\033[1m',"The data is distributed as:\n",'\033[0m',data.describe())
        print("\n\n")
    describe()
    def mean():
        print('\033[1m',"\n\n\nThe mean of the various attributes are:\n",'\033[0m',data.mean())
        sns.distplot(data.mean())
        
    mean()
    plt.show()
    def var():
        print('\033[1m',"\n\n\nThe variance of various attributes are:\n",'\033[0m',data.var())
        sns.distplot(data.var())
        
    var()
    plt.show()
    def corr():
        print('\033[1m',"\n\n\nThe correlation between various attributes:\n ",'\033[0m')
        plt.figure(figsize=(16,10))
        sns.heatmap(data.corr(), annot=True)
       
    corr()
    plt.show()
    def skew():
        print('\033[1m',"\n\n\nThe skewness in various attributes of data:\n",'\033[0m',data.skew())
        sns.distplot(data.skew())
       
    skew()
    plt.show()
    def kurt():
        print('\033[1m',"\n\n\nThe kurtosis in various attributes of data:\n",'\033[0m',data.kurtosis())
        sns.distplot(data.kurtosis())
       
    kurt()
    plt.show()

