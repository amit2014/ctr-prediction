import ffm
import graphlab as gl
# from convert import read_libffm_file

# Output from examples/process-data.py
train = gl.SFrame('criteo_train_transformed')
valid = gl.SFrame('criteo_valid_transformed')

print(train['X2'].tail())

# Currently only dictionary columns are supported
features = [c for c in train.column_names() if train[c].dtype()==dict]

print(features)

# Train a model
m = ffm.FFM()
m.fit(train, valid, target='X1', features=features, nr_iters=20)

# Make predictions
yhat = m.predict(valid)
print(yhat)

