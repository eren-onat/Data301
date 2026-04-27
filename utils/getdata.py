# plt.style.use('ggplot')
import pandas as pd
import numpy as np
import random
from random import gauss
# from sklearn.preprocessing import StandardScaler

#to generate people names
import logging
INCLUDE_NAMES = True
try:
    import names
except ImportError as e:
    logging.error("Missing names package\nPlease install by typing '!pip install names' in a jupyter cell\nor by typing 'pip install names' in a terminal window")
    INCLUDE_NAMES = False
    # raise
    
    
PROCESSED_DATA = "./data.feather"

# make some duplicates just to show how to handle duplicates (delete them)
#lets find the oldest male and female
def fun1(df,numb=5):
    #generates numb rows from df
    return (df.iloc[0:numb,:])

def generate_tshirt_order1(number_each_size=100, sizes={'s':(75, 15),'m':(100, 15), 'l':(130, 20), 'xl':(165, 20), 'xxl':(200,25)},dups=0, percent_nans=0.0):
    '''
    generate a t-shirt order with the above mix of sizes, medians and std
    use the names module to generate random names associated with each order (see https://pypi.org/project/names/https://pypi.org/project/names/)
    add a color column with a random color
    add a name column with a random name
    add a gender column with a random gender appropriate to the name
    add an Age column with random ages between 8 and 18
    return: a Pandas DataFrame

    sizes: dict with the mean and std for each size
    dups: number duplicate rows appended to dataframe
    percent_nans: fraction of t_shirt_sizes to set to np.NaN
    returns: dataframe of t shirts        
    '''
    #generate a bunch of t-shirts with the following mean,std,numbershirts
    x = np.empty(0)
    s = np.empty(0,dtype=object)
    for size, (mean, std) in sizes.items():
        x = np.concatenate((x, np.random.normal(mean, std, number_each_size)))
        s = np.concatenate((s, np.full(number_each_size, size)))

    d = {'weight': x, 't_shirt_size': s}
    df = pd.DataFrame(data=d)

    ts_colors = ['green','blue','orange','red','black']

    df['t_shirt_color'] = np.random.choice(ts_colors, size=number_each_size*len(sizes))
    if(INCLUDE_NAMES):
        df['name'] = "Unknown"
        df.name = df.name.map(lambda x: names.get_full_name())

    #generate an age (integer)
    df['Age'] = np.random.randint(8, 18, len(df))
    
    #duplicates?
    if(dups>0):
        #generates numb rows from each group to be used as duplicates
        df_dups=df.groupby('t_shirt_size').apply(fun1)
        df=pd.concat([df,df_dups],ignore_index=True)
     
    #missing data?
    if(percent_nans>0.0):
        res = random.sample(range(0, len(df)), int(percent_nans * len(df)))
        #lose orig size
        df.loc[res,'t_shirt_size']=np.NaN
    return df

def generate_tshirt_order(numb_small=100, numb_medium=100,numb_large=100, dups=0, percent_nans=0.0):
    return generate_tshirt_order1(number_each_size=numb_small, sizes={'s':(75, 15),'m':(100, 15), 'l':(130, 20)}, dups=dups,percent_nans= percent_nans)

NUMB_SAMPLES = 100
RAND_MAX_VAL =10
RAND_MIN_VAL =0
MAX_RISE = 20
RANDOM_SEED=42

def gendata(ns, max_val= RAND_MAX_VAL, min_val=RAND_MIN_VAL, max_rise=MAX_RISE):
    '''
    generate dataset for linear regression
    :return: x,y dataset
    '''
    x = [val for val in range(ns)]
    y=[random.random()*(max_val-min_val)+min_val + val + max_rise + max_val*gauss(0,1) for val in range(ns)]
    # x=np.array(x).reshape(-1,1)
    # y=np.array(y).reshape(-1,1)
    return np.array(x),np.array(y)
