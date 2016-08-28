import pandas as pd
import numpy as np
import os
import logging
import random
import simplejson as json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def normalize(series):
    min_val = series.min()
    max_val = series.max()
    return (series-min_val)/(max_val-min_val)


def preprocess_date_numerical(df,date_field):    
    min_date = min(df[date_field])
    new_date_field = date_field+numerical_suffix
    df[new_date_field] = normalize(df[date_field].apply(lambda date: (date-min_date).days))


def preprocess_date_by_week(df,date_field):    
    fixed_date = pd.to_datetime('2000/1/1') 
    new_date_field = date_field+'_categorical'
    df[new_date_field] = df[date_field].apply(lambda date: str(((date-fixed_date).days) % 7))


def delete_existing_files():
    try:
        os.remove('data/train.json')
        os.remove('data/validation.json')
        os.remove('data/test.json')

    except:
        print 'failed to delete file'
        return


def preprocess(df,train,columns):
    if train:
        df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)

    #process date variables
    #df['date_ppl'] = pd.to_datetime(df.date_ppl)
    #df['date_act'] = pd.to_datetime(df.date_act)
    #preprocess_date_numerical(df,'date_ppl')
    #preprocess_date_numerical(df,'date_act')
    #preprocess_date_by_week(df,'date_act')

    #mark char_38 feature as numerical
    df['char_38'+numerical_suffix] = normalize(df['char_38'])

    #remove unnecessary columns
    ignored_columns = set(['activity_id','people_id','outcome','char_38'])
    columns.add('activity_id')
    columns.add('outcome')
    
    if train:
        delete_existing_files()
    f_train = open('data/train.json','a')
    f_validation = open('data/validation.json','a')
    f_test = open('data/test.json','a')

    for _,row in tqdm(df.iterrows()):
        tmp_dic = row.to_dict()
        dic_to_store = {}
        dic_to_store['activity_id'] = tmp_dic['activity_id']
        if train:
            dic_to_store['outcome'] = tmp_dic['outcome']
        for k,v in tmp_dic.items():
            #ignored column
            if k in ignored_columns:
                continue
            #numerical column - keep original
            if k.endswith(numerical_suffix):
                dic_to_store[k]=v
                columns.add(k)
            else:#categorical feature
                if type(v)!=type(True):
                    col = k+'_'+str(v)
                else:
                    col = k
                #if test category not in train - dont keep
                if (not train) and (col not in columns):
                    continue
                #if the value exist
                if not (v==0 or v==False or pd.isnull(v)):
                    dic_to_store[col]=1
                    if train:
                        #keep the train columns
                        columns.add(col)

        #dump the dictionary to file
        if train:
            validation = random.random() < 0.01
            if validation:
                f_validation.write(json.dumps(dic_to_store)+'\n')
            else:
                f_train.write(json.dumps(dic_to_store)+'\n')

        else:
            f_test.write(json.dumps(dic_to_store)+'\n')

    f_train.close()
    f_validation.close()
    f_test.close()


def remove_duplicates(df):
    print 'records:',len(df)
    columns_to_compare = set(df.columns)
    columns_to_compare.remove('people_id')
    columns_to_compare.remove('activity_id')
    df = df.drop_duplicates(columns_to_compare)
    print 'records without duplicates:',len(df)
    return df



numerical_suffix = '_numerical'

#read csv files
people_df = pd.read_csv('data/people.csv')
act_train_df = pd.read_csv('data/act_train.csv')
act_test_df = pd.read_csv('data/act_test.csv')

#process train df
df_train = pd.merge(act_train_df,people_df,on='people_id',how='left',suffixes=('_ppl','_act'))
df_train = remove_duplicates(df_train)

columns = set()
preprocess(df_train,True,columns)
del df_train
logging.info('done with train, now test...')
#process test df
df_test = pd.merge(act_test_df,people_df,on='people_id',how='left',suffixes=('_ppl','_act'))
preprocess(df_test,False,columns)

#store columns
f_columns = open('data/columns','w')
for c in columns:
    f_columns.write(c+'\n')
f_columns.close()




