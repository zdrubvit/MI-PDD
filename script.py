import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

np.random.seed(0)

# Load the users
usr_train = pd.read_csv('./data/train_users_2.csv')
usr_test = pd.read_csv('./data/test_users.csv')
labels = usr_train['country_destination'].values

# This column is not needed
usr_train = usr_train.drop(['country_destination'], axis=1)
usr_test_ids = usr_test['id']
usr_train_count = usr_train.shape[0]

# Creating a DataFrame with all the users, so that the following operations can be applied only once
usr_all = pd.concat((usr_train, usr_test), axis=0, ignore_index=True)
# Removing id and date_first_booking
usr_all = usr_all.drop(['id', 'date_first_booking'], axis=1)

# Now to the features...

# Split the 'date_account_created' and 'timestamp_first_active' columns into individual parts, then drop the columns
dac = np.vstack(usr_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
usr_all['dac_year'] = dac[:,0]
usr_all['dac_month'] = dac[:,1]
usr_all['dac_day'] = dac[:,2]
usr_all = usr_all.drop(['date_account_created'], axis=1)

tfa = np.vstack(usr_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
usr_all['tfa_year'] = tfa[:,0]
usr_all['tfa_month'] = tfa[:,1]
usr_all['tfa_day'] = tfa[:,2]
usr_all = usr_all.drop(['timestamp_first_active'], axis=1)

# Replace the age outliers
age_values = usr_all.age.values
usr_all['age'] = np.where(np.logical_or(age_values<18, age_values>120), np.nan, age_values)

# Replace the unknown gender info
usr_all.gender.replace('-unknown-', np.nan, inplace=True)

# Features encoding
features = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in features:
    usr_all_dummy = pd.get_dummies(usr_all[f], prefix=f)
    usr_all = usr_all.drop([f], axis=1)
    usr_all = pd.concat((usr_all, usr_all_dummy), axis=1)

# Splitting train and test again (according to the previously defined count)
X = usr_all.values[:usr_train_count]
X_test = usr_all.values[usr_train_count:]

le = LabelEncoder()
y = le.fit_transform(labels)

# Create the classifier instance
classifier = XGBClassifier(max_depth=5, learning_rate=0.6, n_estimators=40, objective='multi:softprob', subsample=0.6, colsample_bytree=0.5, seed=0)
# Train
classifier.fit(X, y)
# Predict values
predictions = classifier.predict_proba(X_test)

# Five most probable predictions are chosen for each user
ids = []
countries = []

for i in range(len(usr_test_ids)):
    idx = usr_test_ids[i]
    ids += [idx] * 5
    countries += le.inverse_transform(np.argsort(predictions[i])[::-1])[:5].tolist()

# Generate submission
submission = pd.DataFrame(np.column_stack((ids, countries)), columns=['id', 'country'])
submission.to_csv('submission.csv',index=False)
