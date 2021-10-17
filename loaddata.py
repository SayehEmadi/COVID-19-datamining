# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:03:44 2020
@author: karim

"""
import numpy as nu
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from scipy import stats
import seaborn as sns 
import matplotlib.pyplot as plt
sns.set(color_codes=True)
from sklearn import preprocessing as pre
import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.neural_network import MLPRegressor
class LocalDataMining():
    def RobustScaler(self,x):
        scaler = pre.RobustScaler()
        return scaler.fit_transform(x)
    def Standard(self,x):
        scaler = pre.StandardScaler()
        return scaler.fit_transform(x)
    def LoadData(self):
        filepath = 'data/ahmadfinal.xlsx'
        data   = pd.read_excel(filepath,sheet_name = "NewCase")
        newcase = nu.array(data)
        
        data = pd.read_excel(filepath,sheet_name = "Improved")
        improved = nu.array(data)
        
        data = pd.read_excel(filepath,sheet_name = "TotalDeath")
        totaldead = nu.array(data)
        
        data = pd.read_excel(filepath,sheet_name = "TotalCase")
        totalcase = nu.array(data)
    
        variable = pd.read_excel(filepath,sheet_name = "Variable")
        print (variable.columns)    
        return newcase , improved , totaldead , totalcase ,variable
    def ImputTimeSeries(self,data):
        imputeddata = []
        imp = IterativeImputer(max_iter=10, random_state=0)
        
        for index in range(0,len(data)):
            series = pd.Series(data[index,1:])
            series = series.fillna(series.rolling(4,min_periods=1).mean())
            
            try:
                imputed = series.interpolate(method='polynomial', order=2)
                imputed = series.interpolate(method='polynomial', order=3)
                #imputed = series.interpolate(method='polynomial', order=4)
                imputeddata.append(nu.array(imputed))
            except:
                print ("error")
                pass
        return imputeddata
    def KnnImputeData(self,data):
        imputeddata = []
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        imputeddata  =  imputer.fit_transform(data)
        return imputeddata
    def PrepareData(self):
        newcase , improved , totaldead , totalcase ,variable =self.LoadData()
        imputed =self.ImputTimeSeries(newcase)   
        newcase = self.KnnImputeData(imputed)    
        df = pd.DataFrame(newcase)
        df.to_csv('newcase.csv')
        
        imputed = self.ImputTimeSeries(improved)   
        improved = self.KnnImputeData(imputed)  
        df = pd.DataFrame(improved)
        df.to_csv('improved.csv')
        
        imputed = self.ImputTimeSeries(totaldead)   
        totaldead = self.KnnImputeData(imputed) 
        df = pd.DataFrame(totaldead)
        df.to_csv('totaldead.csv')
        
        imputed = self.ImputTimeSeries(totalcase)   
        totalcase = self.KnnImputeData(imputed) 
        df = pd.DataFrame(totalcase)
        df.to_csv('totalcase.csv')
        
        return newcase , improved , totaldead , totalcase ,variable
    
    def SideBySidePlot(self,dataframe,label,title):
        columns = list(dataframe.columns.values)
        sns.set(font_scale=0.6)
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20,10),dpi=300)
        colorsName= ['aqua','azure','brown','coral','crimson','cyan','darkblue','gold','grey','lavender','lime','lightblue','maroon','navy','olive','orange','pink','red','green','blue','magenta','white','yellowgreen' , 'tomato','silver']
        sns.regplot(ax=ax1,x=columns[0], y=columns[1], data=dataframe,scatter_kws={"s":20}, line_kws={"color": "red"})
        sns.jointplot(ax=ax2,x=columns[0], y=columns[1], data=dataframe,kind='reg' )
        plt.title(title,fontsize=16)
        plt.xlabel(columns[0],fontsize=14)
        plt.ylabel(columns[1],fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        for i, point in dataframe.iterrows():
            ax1.annotate(label[i], (point[columns[0]], point[columns[1]]))
            ax2.annotate(label[i], (point[columns[0]], point[columns[1]]))
            
        plt.tight_layout()
        plt.savefig('Regression.pdf')
    def RegressionPlot(self,dataframe,label,title):
        columns = list(dataframe.columns.values)
        sns.set(font_scale=0.6)
        plt.figure(figsize=(20,10),dpi=300)
        
        colorsName= ['aqua','azure','brown','coral','crimson','cyan','darkblue','gold','grey','lavender','lime','lightblue','maroon','navy','olive','orange','pink','red','green','blue','magenta','white','yellowgreen' , 'tomato','silver']
        ax = sns.lmplot(x=columns[0], y=columns[1], data=dataframe,legend=False,aspect=2,scatter_kws={"s":20}, line_kws={"color": "red"})
        #ax = sns.jointplot(x=columns[0], y=columns[1], data=dataframe,kind='reg')
        plt.title(title,fontsize=16)
        plt.xlabel(columns[0],fontsize=14)
        plt.ylabel(columns[1],fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        for i, point in dataframe.iterrows():
            plt.gca().annotate(label[i], (point[columns[0]], point[columns[1]]))
        plt.tight_layout()
        plt.savefig('reg{},{}.pdf'.format(columns[0],columns[1]))
    def JointPlot(self,dataframe,label,title):
        columns = list(dataframe.columns.values)
        sns.set(font_scale=0.6)
        plt.figure(figsize=(10,30),dpi=300)
        colorsName= ['aqua','azure','brown','coral','crimson','cyan','darkblue','gold','grey','lavender','lime','lightblue','maroon','navy','olive','orange','pink','red','green','blue','magenta','white','yellowgreen' , 'tomato','silver']
        ax = sns.jointplot(x=columns[0], y=columns[1], data=dataframe,kind='reg' , line_kws={"color": "red"})
        fig = ax.fig
        fig.suptitle(title)
        
        #plt.title(title,fontsize=16)
        plt.xlabel(columns[0],fontsize=14)
        plt.ylabel(columns[1],fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        for i, point in dataframe.iterrows():
            plt.gca().annotate(label[i], (point[columns[0]], point[columns[1]]))
        plt.tight_layout()
        plt.savefig('join{},{}.pdf'.format(columns[0],columns[1]))
    def LinearRegression(self,X,Y,label):
        X= nu.ndarray(X)
        Y = nu.ndarray(Y)
        X_train, X_test, y_train, y_test, trainindex , testindex = train_test_split(X, Y, test_size = 0.1, random_state=9)
        lin_reg_mod =linear_model.LinearRegression()
        lin_reg_mod.fit(X, Y)
        pred = lin_reg_mod.predict(X_test)
        test_set_rmse = (nu.sqrt(mean_squared_error(y_test, pred)))
        params = nu.append(lin_reg_mod.intercept_,lin_reg_mod.coef_)
        print (params)
        print ('Linear Regression:' , test_set_rmse)
        data = pd.DataFrame({'Real':Y,'Predict:':pred})
        p_values= self.PValues(X,Y)
        title = self.CalcCorrelation(Y,pred)
        self.JointPlot(data, label(testindex), title)
        print("Pvalues")
        print (p_values)
    def PValues(self,X,Y):
        import statsmodels.api as sm
        mod = sm.OLS(Y,X)
        fii = mod.fit()
        p_values = fii.summary2().tables[1]['P>|t|']
        print (p_values)
    def CalcCorrelation(self,X,Y):
        pear =  stats.pearsonr(X,Y)
        spearman = stats.spearmanr(X,Y)
        kendall = stats.kendalltau(X, Y)
        print (pear)
        print (spearman)
        print (kendall)
        title = 'Pearson :correlation:{} ,pvalue :{} \n Spearman :correlation:{} ,pvalue :{} \n Kendall :correlation:{} ,pvalue :{} \n'.format(round(pear[0],2),round(pear[1],8),                                                        round(spearman[0],2),round(spearman[1],8),                                                        round(kendall[0],2),round(kendall[1],8))
        return title
    def DecisionTreeRegressor(self,X,Y,label):
        X_train, X_test, y_train, y_test, trainindex , testindex = train_test_split(X, Y, test_size = 0.15, random_state=9)
        regr = tree.DecisionTreeRegressor(max_depth=5)
        regr.fit(X_train, y_train)
        pred = regr.predict(X_test)
        data = pd.DataFrame({'Real':y_test,'Predict By :':pred})
        title = self.CalcCorrelation(y_test,pred)
        self.JointPlot(data, label, title)
        dot_data = tree.export_graphviz(regr, out_file='tree.dot') 
        test_set_rmse = (nu.sqrt(mean_squared_error(y_test, pred)))
        print ('Decision Tree:' ,test_set_rmse)
        
    def MPLRegressor(self,X,Y):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state=9)
        regr = MLPRegressor(random_state=1, max_iter=500,hidden_layer_sizes=(5,3)).fit(X_train, y_train)
        pred = regr.predict(X_test)
        test_set_rmse = (nu.sqrt(mean_squared_error(Y, pred)))
        print (test_set_rmse)
        
datamining = LocalDataMining()
newcase , improved , totaldead , totalcase ,va = datamining.PrepareData()
X=[]
Y= []
incident = nu.array(va.loc[ : , ['Incident'] ])
for item in incident:
    for i in item:
        X.append(i)

population = nu.array(va.loc[ : , ['HDI_Q'] ])
for item in population:
    for i in item:
        Y.append(i)

title = datamining.CalcCorrelation(X, Y)
print (title)
#data = {Xlabel: X,YLabel:Y}
#dataframe = pd.DataFrame(data)
#datamining.JointPlot(dataframe,provincename,title)


#X = {'HDI':va.HDI_Q, 'Population':va.Population, 'Altitude':va.Altitude , 'Age 14-65': (va.Age_14_65/100)*va.Population}
#X= pd.DataFrame(X)

#scaler = pre.RobustScaler()
#scaler.fit_transform(X)
#scaler.fit_tran=datamining.RobustScaler(X)




