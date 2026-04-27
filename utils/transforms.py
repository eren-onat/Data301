import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

def get_features(df,features, val1='object'):
    if(features is None):
        features=[df.dtypes.index[i] for i,val in enumerate(df.dtypes) if val==val1]
    return features
    
def impute_NaNs(df, strategy='most_frequent',verbose=True):
    '''
    use simple imputer to replace NaNs
    df: dataframe to operate on
    return: transformed df
    '''
    #are there any?
    nans=df.isnull().sum()
    tot=nans.sum()
    if tot==0:
        return df
    
    if verbose == True:
        print(f'Fixing {tot} NaNs using {strategy} strategy')
 
    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)  #works with strings
    nans=[nans.index[i] for i,val in enumerate(nans) if val>0]   
    for val in nans:        
        imp = imp.fit(df[[val]])  #determine replacement  
        df[[val]]=imp.transform(df[[val]])  #here is where the transform is applied 
    return df
  

def ps_lower_strip(df, features=None):
    '''
    preprocesses strings

    df: dataframe to operate on
    features: a list of columns to apply to or all object columns if None
    return: transformed df
    '''
    features=get_features(df,features)
        
    for feat in features:
        df[feat] = df[feat].map(str.lower).map(str.strip)
    return df


import re  #the regular expressions package
def ps_replace_punctuation(df,features,punc="[!\"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\`]", replace_with=''):
    '''
    preprocesses strings, replace punctuation, be careful not to run this 
    after you have generated a order_dict for cat _ordinal

    df: dataframe to operate on
    punc: punction to replace
    replace_with: replacement char
    features: a list of columns to apply transform to
    return: transformed df
    '''
    # need to have columns to work with
    if(not features):
        return df
    
    def psp_closure(x):
        return re.sub(punc,replace_with,x)
    
    for feat in features:
        df[feat] = df[feat].map(psp_closure)
    return df


def remove_duplicates(df,features=None, verbose=True):
    '''
    remove duplicate strings, duplicates determined based on columns in features
    
    df: dataframe to operate on
    features: a list of columns to consider for duplicates, if None then all considered
    returns: transformed df
    '''
    
    # need to have columns to work with
    if(not features):
        features=list(df.columns)
    
    #are there any?
    dups=df.duplicated(subset=features)
    ndups=dups.sum()
    if ndups==0:
        return df
    
    if verbose == True:
        print(f'Removing {ndups} duplicate rows')
        
    df.drop_duplicates(subset=features,inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def cat_ordinal(df, features, order):
    '''
    apply a numerical order on ordinal features

    df: dataframe to operate on
    features: a list of columns to apply to (likely 1)
    order: custom ordering dictionary of dictionaries, very likely hand generated
    return: transformed df
    
    ex
    features=['education','day_of_week']
    order={'education':{'illiterate':0,'unknown':1,'basic.4y':2, 'high.school':3},
         'day_of_week':{'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5}}
    df=cat_ordinal(df,features,order)
    '''
    for feat in features:
        df[feat] = df[feat].map(order[feat])
    return df


def cat_getdummies(df, features, dtype=float, drop_first=True):
    '''
    get dummy vars for each feature

    df: dataframe to operate on
    features: a list of columns to get dummy variables for
    dtype: type of the dummy variables
    return: transformed df
    '''
    for feat in features:
        df = pd.get_dummies(df, columns=[feat],dtype=dtype, drop_first=drop_first)
    return df


from sklearn.preprocessing import StandardScaler
def scale(df,features=None):
    '''
    in place scales numerical_features using the provided scaler
    min_max scales all features that only have 2 values
    standard scales all others

    df: dataframe to operate on
    features: a list of columns to apply to
    scaler: function that operates on df's features
    return: transformed df
    '''
    if(features is None):
        features=[df.dtypes.index[i] for i,val in enumerate(df.dtypes) if val != 'object']
        
    #get list of binary columns, these only have 2 values so will be min/max scaled (so they will be either 0 or 1
    bin_columns=[val for val in features if df[val].nunique()==2]

    #remove binary columns from feature columns
    features=[val for val in features if val not in bin_columns]

    #standard scale features columns
    df[features] = StandardScaler().fit_transform(df[features])
    
    def mm(x):
        '''
        min max scaler
        '''
        #check to see if its already scaled 0->1
        if ( x.min()==0 and x.max()==1):
            return x
        
        return (x-x.min())/(x.max()-x.min())
    df[bin_columns].apply(mm,axis=0)
    
    return df


#find extra correlated columns
def get_correlated_columns(df,correlation_threshold =.95):
    '''
    df: a dataframe
    correlation_threshold: select all rows and columns that have a correlation >= to this value
    return: list of tuples of form [ (col,row),...]
    '''
    #make sure we do correlations on non-object columns only
    df = df.loc[:, df.dtypes != 'object']
    
    # generate the correlation matrix (abs converts to absolute value, this way we only look for 1 color range)
    corr = df.corr().abs()
    # Generate mask for the upper triangle (see https://seaborn.pydata.org/examples/many_pairwise_correlations.html)
    # the matrix is symmetric, the diagonal (all 1's) and upper triangle are visual noise, use this to mask both out
    mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)    #k=-1 means get rid of the diagonal
    corr = corr.where(cond=mask)
    
    correlated=[]
    for col in corr.columns:
        for i,val in enumerate(corr.loc[col]):
            if( val>= correlation_threshold):
                correlated.append((col,corr.loc[col].index[i]))
    return correlated

def drop_correlated_columns(df,correlation_threshold = .95, verbose=True):
    '''
    Drops 1 of each 2 correlated columns
    CAREFUL WITH THIS ONE< YOU WANT TO DROP THE COLUMN WITH THE LEAST INFORMATION
    df: a dataframe
    return: df with 1 of each 2 correlated columns dropped
    '''
    correlated = get_correlated_columns(df, correlation_threshold)
    while correlated:
        if (verbose==True):
            print(f'dropping column {correlated[0][0]} which is correlated with {correlated[0][1]}')
            
        df.drop(columns=[correlated[0][0]], inplace=True)
        correlated = get_correlated_columns(df, correlation_threshold)
    return df

def drop_no_variance_columns(df,verbose=True):
    '''
    drops all columns that only have 1 value
    df: a dataframe to inspect
    return: df columns that only have 1 value dropped
    '''
    vals=df.nunique()
    
    #get list of columns that only have 1 value
    todrop=[df.dtypes.index[i] for i,val in enumerate(df.nunique()) if val ==1]
    
    #bail if no columns to drop
    if not todrop:
        return df
    
    if(verbose==True):
         print(f'dropping columns {todrop} since each only has 1 value')
    
    #drop em
    df.drop(columns=todrop, inplace=True)
    
    return df

def run_pipeline(df,dup_features, dummy_features, ordinal_features, ordering_dict):
    '''
    Convenience method: runs a pipeline of above transforms on a dataframe
    df:
    dup_features:a list of columns to consider for duplicates, if None then all considered
    dummy_features:a list of columns to get dummy variables for
    ordinal_features:a list of columns to apply to (likely 1)
    ordering_dict:custom ordering dictionary of dictionaries, very likely hand generated, see cat_ordinal for ex
    returns: transformed df
    '''
    return df.pipe(remove_duplicates,dup_features).pipe(cat_ordinal,ordinal_features,ordering_dict).pipe(drop_no_variance_columns).pipe(scale).pipe(cat_getdummies, dummy_features).pipe(drop_correlated_columns)
    
if __name__=='__main__':
    pass
    #if running this file as a script (ie python3 transforms1.py')
    #all code here will run
    #you can call unit tests from here