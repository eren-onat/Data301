import pandas as pd
import numpy as np

class BoostedHybrid:
    '''
    Time series analysis model
    Runs model_1 on data, then runs model_2 on residuals of model_1
    In practice model_1 detrends the data then model_2 uses a ML algorithm to forcast target values on detrended data
    
    The model has 2 methods:<br>
    fit:
    1. Fit a linear regressor (model_1) to a train set
    2. Calculate predictions (preds) on train set
    3. Calculate residuals (target-preds) to detrend the train set
    4. Fit catboost (model_2) on train set residuals 

    predict:
    1. model_1.predict + model_2.predict
    
    Problems:
    When training or predicting we have to have a list of n previous residuals. This means that the start of train and test sets will be missing these values. 
    This model drops those rows<br>
    
    Examples:
    from sklearn import linear_model
    from catboost import CatBoostRegressor
    hm=BoostedHybrid(model_1 = linear_model.LinearRegression(), model_2=CatBoostRegressor(silent=True, random_state=42), num_lags=2)

    #fit the model
    hm.fit(X_1,X_2,y_train)
    
    #predict
    X_1t=X_test.loc[:,['Time']]
    X_2t=X_test
    preds=hm.predict(X_1t,X_2t,y_test,False)
    '''
    def __init__(self, model_1, model_2, num_lags=0):
        self.model_1 = model_1
        self.model_2 = model_2
        self.num_lags=num_lags

    def fit(self, X_1,X_2, y):
        '''
        X_1 dataset for linear regression(uses column 'Time')
        X_2 dataset for GBT ('all other dataset columns')
        y  target ('NumVehicles')
        '''
        self.train_size=len(X_1)
        
        #fit the first model on train portion of dataset
        self.model_1.fit(X_1, y) # train model 1
        
        #generate residuals for entire dataset (only used to calculate lags)
        y_resid = y - self.model_1.predict(X_1)
        
        #make shallow copy, this ensures that the added lags do not stick to X_2
        X_2=X_2.copy(deep=False)
        
        #compute and add the lags to X_2
        X_2['y_resid']=y_resid
        for i in range(1,self.num_lags+1):
            X_2['Lag_'+str(i)]=X_2.y_resid.shift(i)
        
        #ignore lagged rows that have NaNs
        # display(X_2)
        # X_2.dropna(inplace=True)  #alas goodbye to some rows
        X_2=X_2.iloc[self.num_lags:,:]
        # display(X_2)
              
        #drop target from training data
        X_2.drop(columns=['y_resid'],inplace=True, axis=1)  #this gets rid of target value, so model only sees past lags
        # display(X_2)
        
        #fit the second model on residuals from first model, just the train data part
        self.model_2.fit(X_2, y_resid[-len(X_2):])
         
    def predict(self,X_1,X_2, y, dont_add_lags_to_X_2=True):        
        #make copy?
        if(dont_add_lags_to_X_2):
            #make shallow copy, this ensures that the lags are not added to X_2
            X_2=X_2.copy(deep=False)
        
        #to get a prediction add Model_1's prediction to Model_2's prediction
        #generate residuals for dataset
        X_2['preds'] = self.model_1.predict(X_1)
        X_2['y_resid']=y-X_2['preds'] 
        
        #compute and add the lags to X_2
        for i in range(1,self.num_lags+1):
            X_2['Lag_'+str(i)]=X_2.y_resid.shift(i)
        
        X_2.dropna(inplace=True)  #alas goodbye to some rows
        # display(X_2)
       
        #get the X_1 preds
        y_pred1 = X_2['preds']
        
        #drop columns that are not used for model_2
        X_2.drop(columns=['preds','y_resid'], inplace=True)
        y_pred2=self.model_2.predict(X_2)
        return(y_pred1+y_pred2)