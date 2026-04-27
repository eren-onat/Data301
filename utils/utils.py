'''
R squared - a way to qualify a models predictions

The following regressors use R squared as the default objective to optimize.  See <a href="https://www.youtube.com/watch?v=2AQKmw14mHM">Statquest: R-squared, Clearly Explained!!!</a> for a great explanation plus examples.

Usually 0<R squared<1  .  It ranges between these 2 values and is interpreted as how well the model fits the data. (In statistics this is called explained variance)

If R squared =0,the line fitted to data is no more accurate than taking the mean of the data.<br>
If R squared =1,the line fitted to the data is a perfect match<br>
If R squared is negative then the line fitted to the data is a worse fit than just taking the average value of the data.
'''
import numpy as np

#It was not clear what objective lightGBM optimizes
#so I implemented R squared below 
def rsquared(preds, y):
    '''
    preds: model predictions
    y: ground truth values
    returns: rsquared for above
    '''
    RSS=np.sum(np.square(preds-y))
    ymean=np.sum(y)/len(y)
    TSS=np.sum(np.square(y-ymean))
    return 1-RSS/TSS

def scoremodel(clf,X_test, y_test):
    '''
    clf: model
    X_test: test data for clf to predict on
    y_test: ground truth
    '''
    print("Score on test setusing models score function: {:.2f}".format(clf.score(X_test, y_test)))
    #run score using rsquared function above
    preds=clf.predict(X_test)
    rsq=rsquared(preds,y_test)
    print("Score on test set using rsquared: {:.2f}".format(rsq))
    
from sklearn import linear_model
def train_linreg_model(X,y):
    '''
    X: train on this, for this notebook its 1 column of data
    y: the target
    return: the trained model
    '''
    reg = linear_model.LinearRegression()
    reg.fit(X=X, y=y)
    return reg

def show_regression_formula(mod, target, independant_var, show_explanation=True):
    '''
    mod: the trained linear regression model
    target: string, the name of the dependant variable(s)
    independant_var: string, the name of the independant variable
    '''
    #what are the linear regression parameters for w1 and bias?
    if(show_explanation==True):
        print('Linear regression formula is:')
    
    s=f'{target}='
    for coef, ind_var in zip(mod.coef_,independant_var) :
        s=s+f'{coef:.3f}*{ind_var}'
    print(s +f' +{mod.intercept_:.3f}')
    
def show_thresholds(df, threshold=1):
    '''
    shows all columns with variance<=threshold
    df: a dataframe to inspect
    return: nothing
    '''
    #lets see what the low variance columns have in them
    for col in data.columns:
        if(data[f'{col}'].nunique() <= threshold):
            print(f'{col}, vals={data[f"{col}"].unique()} ')
    
NSAMPLES=300
RANDOM_STATE=999
MIN_SAMPLES = 6
from sklearn.neighbors import NearestNeighbors
def get_sorted_distances( X, min_samples=MIN_SAMPLES):
    '''
    X:data 
    distances: the sorted distances to MIN_SAMPLES points for every point in X
    plot: to plot the knee or not
    return: sorted distances in descending order
    '''
    nbrs = NearestNeighbors(n_neighbors=min_samples ).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distance_descending = sorted(distances[:,min_samples-1], reverse=True)
    return distance_descending


