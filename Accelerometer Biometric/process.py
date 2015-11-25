import pandas as pd 
import csv
import numpy
import math

def interval(l):
    return numpy.mean([abs(y - x) for x, y in zip(l, l[1:])])

# train.groupby("Device")["T"].apply(interval).to_csv("intervals_train.csv")
# test.groupby("SequenceId")["T"].apply(interval).to_csv("intervals_test.csv")


# question = pd.read_csv("questions.csv")
train = pd.read_csv("intervals_train.csv")
test = pd.read_csv('intervals_test.csv')


reader = csv.reader( open( "questions.csv" ))
writer = csv.writer( open( "submission.csv", 'wb' ))

headers = reader.next()
writer.writerow( [ 'QuestionId', 'IsTrue' ] )

n = 0

for line in reader:
	q_id, q_sequence, q_device = line

	int_train = train[train["device_id"] == int(q_device)].iat[0,1] 
	int_test = test[test["squence_id"] == int(q_sequence)].iat[0,1] 
	
	diff = abs(int_train - int_test)
	prob = 1.0 / (diff + math.log(diff))
	writer.writerow( [ q_id, prob] )
	
	n += 1
	if n % 10000 == 0:
		print n