import csv
import numpy as np
import string
from sklearn.ensemble import RandomForestRegressor

def main():
  alchemy_category_set = {}
  #read train data
  train = []
  target = []
  with open("./data/train.tsv", 'rb') as csvfile:
    reader = csv.reader(csvfile, dialect='excel-tab')
    reader.next() #skip the header
    for row in reader:
      line = row[3:len(row)-1]
      train.append(line)
      target.append(row[len(row)-1])
      if row[3] not in alchemy_category_set:
        alchemy_category_set[row[3]] = len(alchemy_category_set)

  #read valid data
  valid = []
  valid_index = []
  with open("./data/test.tsv", 'rb') as csvfile:
    reader = csv.reader(csvfile, dialect='excel-tab')
    reader.next() #skip the header
    for row in reader:
      line = row[3:len(row)]
      valid.append(line)
      valid_index.append(row[1])
      if row[3] not in alchemy_category_set:
        alchemy_category_set[row[3]] = len(alchemy_category_set)
  
  #expand the alchemy_category
  for idx in range(len(train)):
    line = train[idx]
    alchemy_category = [0 for i in range(len(alchemy_category_set))]
    alchemy_category_idx = alchemy_category_set[line[0]]
    alchemy_category[alchemy_category_idx] = 1
    del line[0]
    line = [string.atof(x) if x != '?' else 0 for x in line]
    line = line + alchemy_category
    train[idx] = line

  for idx in range(len(valid)):
    line = valid[idx]
    alchemy_category = [0 for i in range(len(alchemy_category_set))]
    alchemy_category_idx = alchemy_category_set[line[0]]
    alchemy_category[alchemy_category_idx] = 1
    del line[0]
    line = [string.atof(x) if x != '?' else 0 for x in line]
    line = line + alchemy_category
    valid[idx] = line

  # run the model
  classifier = RandomForestRegressor(n_estimators=1000,verbose=2,n_jobs=20,min_samples_split=5,random_state=1034324)
  classifier.fit(train, target)
  predictions = classifier.predict(valid)

  # write
  writer = csv.writer(open("predictions", "w"), lineterminator="\n")
  rows = [x for x in zip(valid_index, predictions)]
  writer.writerow(("urlid","label"))
  writer.writerows(rows)


if __name__=="__main__":
  main()
