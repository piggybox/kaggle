from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn import decomposition
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

import random
import numpy as np
import csv as csv
import pandas as pd


train = np.asarray(pd.read_csv("yelp_training_set/yelp_training_set_user.csv"))
test_user = np.asarray(pd.read_csv("yelp_test_set/yelp_test_set_user.csv"))
test_review = pd.read_csv("yelp_test_set/yelp_test_set_review.csv")

# delete name, type, user_id, cool, fun
train = np.delete(train, [1,3,4,5,6], 1)

# delete name, type, user_id
test_user = np.delete(test_user, [1,3,4], 1)

forest = RandomForestClassifier(n_estimators=10)
forest = forest.fit(train[0::,::-2], train[0::,-1])
output = forest.predict(test_user) # rating per user

test_user = pd.read_csv("yelp_test_set/yelp_test_set_user.csv")
test_user['vote'] = pd.Series(output) #, index=test_user.index)

with open('submission.csv', 'wb') as csvfile:
    result = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    result.writerow(["id","votes"])
    for i in range(len(test_review)):
    	review_id = test_review.review_id.loc[i]
    	if test_user[test_user.user_id == test_review.user_id.iloc[i]].empty :
    		result.writerow([review_id, 0])
    	else: 
    		result.writerow([review_id, test_user[test_user.user_id == test_review.user_id.iloc[i]].vote.iloc[0]])


