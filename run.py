import pandas as pd
import simplejson as json
from itertools import islice
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.optimizers import Adam,SGD
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import logging
import glob


logging.basicConfig(level=logging.INFO)

_columns = [c.strip() for c in open('data/columns','r')]
ignore_columns = ['activity_id','outcome']
num_of_features = len(_columns)-len(ignore_columns)

#colum name to array index mapping
col_to_ind = {}
i = 0
for c in _columns:
    if c not in ignore_columns:
        col_to_ind[c] = i
        i += 1

train_file = 'data/train_duplicates.json'
validation_file = 'data/validation_duplicates.json'
test_file = 'data/test_duplicates.json'

def read_batch(batch_lines):
    X = np.zeros((len(batch_lines),num_of_features))
    y = []
    ids = []

    for i,line in enumerate(batch_lines):
        d = json.loads(line)
        for k,v in d.iteritems():
            if k not in ignore_columns:
                X[i][col_to_ind[k]]=v

        y.append(d.get('outcome',None))
        ids.append(d['activity_id'])

    return X,y,ids


def load_data_in_batches(json_path,batch_size=1024):
    with open(json_path) as f_train:
        while True:
            batch_lines = list(islice(f_train, batch_size))
            if not batch_lines:
                break
            yield read_batch(batch_lines)


def train_model(model,model_name,batch_size=8192,mini_batch_size=1024,epoches=5,epoches_per_batch=1,lr_decay=0.9):
    optimizer = Adam()
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #load mini validation
    X_mini_validation=None
    y_mini_validation=None
    for x,y,_ in load_data_in_batches(validation_file,5000):
        X_mini_validation=x
        y_mini_validation=y
        break

    hist_data = []
    for epoch in range(epoches):
        for i,(X_train,y_train,_) in enumerate(load_data_in_batches(train_file,batch_size)):
            print '{}/{}'.format(epoch,i)
            history = model.fit(X_train, y_train, nb_epoch=epoches_per_batch, batch_size=mini_batch_size)
            if i%20==0:
                val_loss,val_acc = model.evaluate(X_mini_validation,y_mini_validation)
                hist_data.append({'val_loss':val_loss,'val_acc':val_acc,'train_loss':history.history['loss'][-1],'train_acc':history.history['acc'][-1]})
                print
                print 'Validation loss/acc:',val_loss,val_acc
                print 'learning rate:',model.optimizer.lr.get_value()

        curr_lr = float(optimizer.lr.get_value())
        optimizer.lr.set_value(curr_lr*lr_decay)

    # save model
    model.save('models/{}.model'.format(model_name))
    df_loss = pd.DataFrame(hist_data)
    df_loss.to_csv('models/{}.hist'.format(model_name),index=False)
    evaluate_on_validation(model)


def evaluate_on_validation(model):
    # evaluate on all validation
    print 'evaluating on entire validation...'
    val_predictions = []
    val_labels = []
    for X_validation, y_validation, _ in tqdm(load_data_in_batches(validation_file)):
        val_predictions += predict(model,X_validation)
        val_labels += y_validation

    print
    print 'Validation AUC:', roc_auc_score(val_labels, val_predictions)
    print


def predict(model,X):
    predictions = model.predict(X,len(X))
    return [x[0] for x in predictions]


def make_submission(model_name):
    model = load_model('models/{}.model'.format(model_name))
    d = {'activity_id':[],'outcome':[]}
    for X_test,_,ids in tqdm(load_data_in_batches(test_file,5000)):
        d['outcome']+=predict(model,X_test)
        d['activity_id']+=ids

    pd.DataFrame(d).to_csv('submissions/{}.csv'.format(model_name),index=False)

def lr_model():
    model = Sequential()
    model.add(Dense(1, input_dim=num_of_features, activation='sigmoid'))
    return model

#create model
def mlp2_model(hidden1,hidden2,dropout=0.5):
    model = Sequential()
    model.add(Dense(hidden1, input_dim=num_of_features, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    return model

def mlp3_model(hidden1,hidden2,hidden3,dropout=0.5):
    model = Sequential()
    model.add(Dense(hidden1, input_dim=num_of_features, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden3, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    return model

def make_average_submission(model_name='average'):
    dfs = []
    for f in glob.glob('submissions/*'):
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs,axis=1)
    df_avg = pd.DataFrame()
    df_avg['outcome'] = df[['outcome']].mean(axis=1)
    df_avg['activity_id'] = dfs[0]['activity_id']
    pd.DataFrame(df_avg).to_csv('submissions/{}.csv'.format(model_name),index=False)


model = mlp3_model(1000,500,300,0.7)
model_name = 'mlp3_1000_500_300_dp07'
train_model(model,model_name)
make_submission(model_name)
