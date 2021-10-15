import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import pickle

data = pd.read_csv('IPL Player Stats - 2016 till 2019.csv')

ipl = pd.read_csv('ipl_dataset.csv')

ipl= ipl.drop(['Unnamed: 0','extras','match_id', 'runs_off_bat'],axis = 1)
new_ipl = pd.merge(ipl,data,left_on='striker',right_on='Player',how='left')
new_ipl.drop(['wicket_type', 'player_dismissed'],axis=1,inplace=True)

str_cols = new_ipl.columns[new_ipl.dtypes==object]
new_ipl[str_cols] = new_ipl[str_cols].fillna('.')

a1 = new_ipl['venue'].unique()
a2 = new_ipl['batting_team'].unique()
a3 = new_ipl['bowling_team'].unique()
a4 = new_ipl['striker'].unique()
a5 = new_ipl['bowler'].unique()

def labelEncoding(data):
    dataset = pd.DataFrame(new_ipl)
    feature_dict ={}
    for feature in dataset:
        if dataset[feature].dtype==object:
            le = preprocessing.LabelEncoder()
            fs = dataset[feature].unique()
            le.fit(fs)
            dataset[feature] = le.transform(dataset[feature])
            feature_dict[feature] = le
    return dataset
new_ipl = labelEncoding(new_ipl)

ip_dataset = new_ipl[['venue','innings', 'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler']]
b1 = ip_dataset['venue'].unique()
b2 = ip_dataset['batting_team'].unique()
b3 = ip_dataset['bowling_team'].unique()
b4 = ip_dataset['striker'].unique()
b5 = ip_dataset['bowler'].unique()

features={}
for i in range(len(a1)):
    features[a1[i]]=b1[i]
for i in range(len(a2)):
    features[a2[i]]=b2[i]
for i in range(len(a3)):
    features[a3[i]]=b3[i]
for i in range(len(a4)):
    features[a4[i]]=b4[i]
for i in range(len(a5)):
    features[a5[i]]=b5[i]

new_ipl.fillna(0,inplace=True)

X = new_ipl[['venue', 'innings','batting_team', 'bowling_team', 'striker','bowler']].values
y = new_ipl['y'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.callbacks import EarlyStopping

def my_model():    
    model = Sequential()

    model.add(Dense(43, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(22, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(11, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(x=X_train, y=y_train, epochs=400, validation_data=(X_test,y_test),callbacks=[early_stop] )
    return model

early_stop = EarlyStopping(monitor='val_loss',patience=25,verbose=1, mode='min')
model = my_model()
model.save('ipl_score_prediction.h5')

# pickle.dump(features, open("features.pkl", "wb"))


def prediction(d):
    lst = d
    plst = pd.DataFrame(d)
    labeled_data=[]
    for i in lst:
        if type(i)==str:
            if i in features.keys():
                labeled_data.append(features[i])
        else:
            labeled_data.append(i)
        
    bowls = labeled_data[5:]
    
    all_pred = []   
    preds = []
    l1 = labeled_data[:5]
    for i in range(len(bowls)):
        l1.append(bowls[i])
        all_pred.append(l1)

        l1=l1[:-1]
    
        
    # v = all_pred[0]
    # v = [15, 1, 7, 13, 259, 30]
    # v = np.array(v)
    # v = scaler.transform([v])

    new_model = tf.keras.models.load_model('ipl_score_prediction.h5')
    v = all_pred[0]
    v = scaler.transform([v])
    p = new_model.predict(v)
    return  (p[0])

# input_array = ['MA Chidambaram Stadium','Kolkata Knight Riders','Royal Challengers Bangalore','SC Ganguly', 
# 'BB McCullum','RT Ponting','P Kumar', 'TA Boult', 'K Rabada', 'PJ Cummins', 'Avesh Khan']

# prediction(input_array)

