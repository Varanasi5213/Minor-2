import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def represent(parameters):
    for i in list(parameters):
        parameters[i]=parameters[i] - parameters[i].min()/(parameters[i].max()-parameters[i].min())
    
    for i in range(len(parameters)//2):
        plt.imshow( parameters["W"+str(i+1)].T,cmap='gray')
        plt.title( "W"+str(i+1) )
        plt.show()
        l=[]
        for j in parameters["W"+str(i+1)].T:
            l.append(abs(j).mean())
        per=l /(sum(l))*100
        df = {}
        per = np.array(per)
        df["Percentage Contribution"]=per
        df=pd.DataFrame(df)
        print(df)
    
    