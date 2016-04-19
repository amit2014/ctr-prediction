import graphlab as gl
from graphlab.toolkits.feature_engineering import FeatureHasher, CountThresholder
import os
import math

filename = '../../data/train_data.txt' # replace with day_i.gz

outfile = 'criteo-sample'

if not os.path.exists(outfile):

    sf = gl.SFrame.read_csv(filename, delimiter='\t', header=False, skip_initial_space=False)

    sf.save(outfile)

else:

    sf = gl.SFrame(outfile)

# Quick in-browser visualization using Dato's Canvas
# sf.show()

# Create a random train/test split
train, valid = sf.random_split(.8)

# train.remove_column('X1')
# valid.remove_column('X1')

print train.head()

train.save('criteo_train')

valid.save('criteo_valid')

# Feature engineering
# This attempts to apply "Preprocessing-B" to the original data as described in
# http://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf

# Create an SFrame containing only the categorical variables
categ = ['X' + str(i) for i in range(2, 25)]
# numeric = ['X' + str(i) for i in range(15, 41)]

def int_trans(x):

    if x > 2:

        return math.floor(math.log(x) ** 2)

    return x

def transform_numeric_columns(sf):

    for n in numer:

        print(n)

        sf[n] = sf[n].apply(int_trans).astype(int)

    return sf

# train = transform_numeric_columns(train)
# valid = transform_numeric_columns(valid)
# print train.tail()
# print valid.tail()

for c in categ:

    print(c)

    chain = gl.feature_engineering.create(train,
        [CountThresholder(c, threshold=10),
         FeatureHasher(c)])

    train[c] = chain.transform(train)['hashed_features']

    valid[c] = chain.transform(valid)['hashed_features']

train.save('criteo_train_transformed')

valid.save('criteo_valid_transformed')

print(train)

