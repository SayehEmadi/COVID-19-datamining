# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:03:44 2020
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

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from itertools import chain, combinations
from sklearn.model_selection import KFold,train_test_split

class LocalDataMining():
    def Standard(self,x):
        scaler = pre.StandardScaler()
        return scaler.fit_transform(x)
    def LoadData(self,filepath):
        data   = pd.read_excel(filepath,sheet_name = "NewCase")
        newcase = nu.array(data)

        data = pd.read_excel(filepath,sheet_name = "TotalDeath")
        totaldead = nu.array(data)
        
        data = pd.read_excel(filepath,sheet_name = "TotalCase")
        totalcase = nu.array(data)
    
        variable = pd.read_excel(filepath,sheet_name = "Variable")
        print (variable.columns)

        return newcase ,  totaldead , totalcase ,variable
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
    def PrepareData(self,filepath):
        newcase ,  totaldead , totalcase ,variable =self.LoadData(filepath)
        imputed =self.ImputTimeSeries(newcase)   
        newcase = self.KnnImputeData(imputed)    
        df = pd.DataFrame(newcase)
        df.to_csv('newcase.csv')
        

        imputed = self.ImputTimeSeries(totaldead)   
        totaldead = self.KnnImputeData(imputed) 
        df = pd.DataFrame(totaldead)
        df.to_csv('totaldead.csv')
        
        imputed = self.ImputTimeSeries(totalcase)   
        totalcase = self.KnnImputeData(imputed) 
        df = pd.DataFrame(totalcase)
        df.to_csv('totalcase.csv')
        
        return newcase ,  totaldead , totalcase ,variable
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

    def RegressionPlotTrainTest(self, dttrain,dttest, labeltrain,labeltest, title):
        columnstrains = list(dttrain.columns.values)
        columnstest = list(dttest.columns.values)
        sns.set(font_scale=0.6)
        f,(ax1,ax2) =plt.subplots(1,2,figsize=(20, 10), dpi=300)

        ax1 = sns.lmplot(x=columnstrains[0], y=columnstrains[1], data=dttrain, legend=False, aspect=2, scatter_kws={"s": 20},  line_kws={"color": "red"})
        ax2 = sns.lmplot(x=columnstest[0], y=columnstest[1], data=dttest, legend=False, aspect=2, scatter_kws={"s": 20},  line_kws={"color": "blue"})


        plt.title(title, fontsize=16)
        plt.xlabel(columnstrains[0], fontsize=14)
        plt.ylabel(columnstrains[1], fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        for i, point in dttrain.iterrows():
            plt.gca().annotate(labeltrain[i], (point[columnstrains[0]], point[columnstrains[1]]))
        for i, point in dttest.iterrows():
            plt.gca().annotate(labeltest[i], (point[columnstest[0]], point[columnstest[1]]))

        plt.tight_layout()
        ax1.savefig('results/{}.pdf'.format(title))
        ax2.savefig('results/{}.pdf'.format(title))
        plt.show()
        ff = 1

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
        ax.savefig('results/{}.pdf'.format(title))
        plt.show()
        ff=1
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
        plt.savefig('results/join{},{}.pdf'.format(columns[0],columns[1]))
    def LinearRegression(self,X,Y,label):
        X= X.to_numpy()
        label = label.to_numpy()
        rs = ShuffleSplit(n_splits=1, test_size=.15, random_state=0)
        for train_index, test_index in rs.split(X):
            lin_reg_mod =linear_model.LinearRegression()
            lin_reg_mod.fit(X[train_index], Y[train_index])

            pred = lin_reg_mod.predict(X[test_index])
            test_set_rmse = (nu.sqrt(mean_squared_error(Y[test_index], pred)))
            params = nu.append(lin_reg_mod.intercept_,lin_reg_mod.coef_)

            print (params)
            print ('Linear Regression:' , test_set_rmse)
            data = pd.DataFrame({'Real':list(Y[test_index]),'Predict:':list(pred)})
            p_values= self.PValues(Y[test_index],pred)
            title = self.CalcCorrelation(Y[test_index],pred)
            self.RegressionPlot(data, label[test_index], 'Regression')
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
        X= X.to_numpy()
        label = label.to_numpy()
        rs = ShuffleSplit(n_splits=1, test_size=.2, random_state=2)
        for train_index, test_index in rs.split(X):
            regr = tree.DecisionTreeRegressor(max_depth=3)
            regr.fit(X[train_index], Y[train_index])
            pred = regr.predict(X[test_index])
            test_set_rmse = (nu.sqrt(mean_squared_error(Y[test_index], pred)))

            print ('Decision Tree:' , test_set_rmse)
            data = pd.DataFrame({'Real':list(Y[test_index]),'Predict:':list(pred)})

            title = self.CalcCorrelation(Y[test_index],pred)
            print (title)
            self.RegressionPlot(data, label[test_index], 'DecesionTree')
            dot_data = tree.export_graphviz(regr, out_file='results/tree.dot')
    def RobustScaler(self, x):
        scaler = pre.RobustScaler()
        return scaler.fit_transform(x)
    def MLPRegressorAll(self,XX,Y):
        scaler = pre.StandardScaler()
        maxrmse = 50000000
        maxr2=0
        bestsubset = 1
        power =list(self.powerset([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]))
        print ("Power Length", len(power))
        for i, item in enumerate(power):
            if (i==0):
                pass
            else:
                X= self.RobustScaler(XX[:,item])
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=9)
        
                regr = MLPRegressor(random_state=1, max_iter=500,hidden_layer_sizes=(8,2)).fit(X_train, y_train)
                pred = regr.predict(X_test)
                from sklearn.metrics import r2_score
                r2=r2_score(pred,y_test)
                rmse = (nu.sqrt(mean_squared_error(y_test, pred)))
                if (r2> maxr2 ):
                    print (i)
                    maxr2= r2;
                    bestrmse = rmse
                    bestsubset = item
                    print ("r2" , r2)
                    print ("rmse" ,rmse)
                    print ("subset" , bestsubset)
                    print ("--------------------------------")
        return bestsubset
    def MLPRegressorCrossFold(self,X,Y):
        X = self.RobustScaler(X)
        Data =X
        nfold = 5
        kf = KFold(n_splits=nfold, shuffle=True, random_state=random.randint(1, 100))
        totalr2 =[]
        totalmse =[]
        for train_index, test_index in kf.split(Data):
            mlpregressor = MLPRegressor(random_state=1, max_iter=100000, hidden_layer_sizes=(12, 2))
            mlpregressor.fit(Data[train_index], Y[train_index])
            y_pred = mlpregressor.predict(Data[test_index])
            y_test = Y[test_index]
            rr2 = r2_score(y_test, y_pred)
            totalr2.append(r2_score(y_test, y_pred))
            totalmse.append(nu.sqrt(mean_squared_error(y_test, y_pred)))
        averager2 = nu.mean(nu.array(totalr2))

        averagermse =nu.mean(nu.array(totalmse))

        return averager2,averagermse
    def MLPRegressor(self,X,Y,CityName):
        X= self.RobustScaler(X)
        indices = nu.arange(X.shape[0])
        X_train, X_test, y_train, y_test ,indices_train,indices_test = train_test_split(X, Y ,indices, test_size = 0.2,random_state=5)
        regr = MLPRegressor(random_state=1, max_iter=50000,hidden_layer_sizes=(20,2)).fit(X_train, y_train)
        predtrain = regr.predict(X_train)
        predtest = regr.predict(X_test)
        predall = regr.predict(X)
        print ("R2 : ")
        print ("      Train: ", r2_score(predtrain,y_train) )
        print ("      Test: ", r2_score(predtest,y_test) )
        print ("      All: ", r2_score(predall,Y) )

        print ("RMSE : ")
        rmse = (nu.sqrt(mean_squared_error(y_train, predtrain)))
        print ("      Train: ", rmse )
        rmse = (nu.sqrt(mean_squared_error(y_test, predtest)))
        print ("      Test: ", rmse )
        rmse = (nu.sqrt(mean_squared_error(predall, Y)))
        print ("      All: ", rmse )


        datatrain = pd.DataFrame({'Real': list(y_train), 'Predict:': list(predtrain)})
        datatest =  pd.DataFrame({'Real': list(y_test), 'Predict:': list(predtest)})
        dataall =  pd.DataFrame({'Real': list(Y), 'Predict:': list(predall)})

        labeltrain = CityName[indices_train]
        labeltest = CityName[indices_test]
        labelall = CityName

        #self.RegressionPlot(datatrain,labeltrain , 'MLPTrain' )
        #self.RegressionPlot(datatest,labeltest , 'MLPTest' )
        #self.RegressionPlot(dataall, labelall , 'MLPAll' )

#        title = self.CalcCorrelation(list(Y), list(pred))
#        print(title)
    def powerset(self,iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    def ConvertToMatrix(self,data,va,window):
        va = va.to_numpy()[:,1:]
        for i,item in enumerate(data):
            shape = item.shape[:-1] + (item.shape[-1] - window + 1, window)
            strides = item.strides + (item.strides[-1],)
            if (i==0):
                p = nu.lib.stride_tricks.as_strided(item, shape=shape, strides=strides)
                sizep = p.shape[0]
                vam = nu.matrix(va[i,:])
                vam = nu.repeat(vam,sizep,0)
                vam = nu.asmatrix(vam)
                alldata = nu.concatenate((vam,p ),axis =1)
            else:
                p= nu.lib.stride_tricks.as_strided(item, shape=shape, strides=strides)
                sizep = p.shape[0]
                vam = nu.matrix(va[i,:])
                vam = nu.repeat(vam,sizep,0)
                vam = nu.asmatrix(vam)
                t = nu.concatenate((vam,p ),axis =1)
                alldata = nu.concatenate((alldata,t))
        return alldata

datamining = LocalDataMining()
data = pd.read_excel("data/finaldata5.xlsx", sheet_name="data5")
header = data.columns
data = data.to_numpy()

totalr2 = []
indeices = [16,17, 18, 19, 20]
X = data[:, indeices]
Y = data[:, data.shape[1] - 1]
cityNames = data[:, 0]
r2,mse = datamining.MLPRegressorCrossFold(X, Y)
print (r2)
print (mse)
#totalr2.append(r2)
for i in range (1,16):
    print (header[i])
    indeices = [16,17, 18, 19, 20]
    indeices.append(i)
    X = data[:,indeices]
    Y = data[:,data.shape[1]-1]
    cityNames = data[:,0]
    #datamining.MLPRegressorCrossFold(X,Y)

    r2, rms = datamining.MLPRegressorCrossFold(X,Y)
    print (r2)
    print(rms)
    print ("-----------------------------")

#newcase , totaldead , totalcase , va = datamining.PrepareData('data/ahmadfinal.xlsx')
#data = datamining.ConvertToMatrix(newcase,va,5)
#nu.savetxt('data5.csv',data[:,1:],delimiter=',')
#f= open("names.csv","w")
#for item in data[:,0]:
#    f.write("{}\n".format(item))
#f.close()


#provincename = va.Province_Name
#newcase = nu.mean(newcase,axis =1)
#totalcase = nu.mean(totalcase,axis=1)
#totaldead = nu.mean(totaldead,axis=1)

#normalnewcase = newcase / va.Population

#Xlabel = 'NewCase'
#YLabel = 'HDI'
#X = normalnewcase
#Y =  va.HDI_Q
#title = datamining.CalcCorrelation(X, Y)

#print (title)
#data = {Xlabel: X,YLabel:Y}
#dataframe = pd.DataFrame(data)
#datamining.JointPlot(dataframe,provincename,title)

#X = {'HDI':va.HDI_Q, 'Population':va.Population, 'Altitude':va.Altitude}
#X= pd.DataFrame(X)
#scaler = pre.RobustScaler()
#scaler.fit_transform(X)
#scaler.fit_tran=datamining.RobustScaler(X)
#Y=normalnewcase
#datamining.LinearRegression(X,Y,provincename)
#datamining.DecisionTreeRegressor(X,Y,provincename)


