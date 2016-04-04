
# Following: https://dato.com/learn/gallery/notebooks/feature_engineering_with_graphlab_create.html

########################################################################################################################

# import package
import graphlab as gl

########################################################################################################################

# for training
data = gl.SFrame.read_csv('data/train_data.txt', delimiter='\t', verbose=False)
train_data, test_data = data.random_split(0.8, seed=1)

# get features
features = data.column_names()
# remove click label feature
features.remove('click')

# remove some features temporarily
features.remove('ip')
features.remove('url')
features.remove('domain')
features.remove('user_id')
features.remove('log_type')
features.remove('timestamp')
features.remove('user_tags')
features.remove('ad_slot_id')
features.remove('creative_id')
features.remove('key_page_url')
features.remove('advertiser_id')
features.remove('anonymous_url_id')

# for checking features
# print(features)

# for testing
# train_data = gl.SFrame.read_csv('data/train_data.txt', delimiter='\t', verbose=False)
# test_data = gl.SFrame.read_csv('data/test_data.txt', delimiter='\t', verbose=False)

########################################################################################################################

# baseline logistic regression model - uses all features
log_baseline = gl.logistic_classifier.create(train_data, target='click', features=features, validation_set=test_data,
                                             max_iterations=50)

# get logistic regression model predictions
log_predictions = log_baseline.predict(test_data, output_type='probability')

# open logistic regression model predictions file
with open('predictions/log_predictions.csv', mode='w') as log_prediction_file:
    # write headers to file
    log_prediction_file.write('Id,Prediction\n')
    # set logistic regression model prediction id to 1
    log_prediction_id = 1
    # for every logistic regression model prediction
    for log_prediction in log_predictions:
        # write logistic regression model prediction to file in requested format
        log_prediction_file.write('{},{:.5f}\n'.format(log_prediction_id, log_prediction))
        # increment logistic regression model prediction id
        log_prediction_id += 1

# close logistic regression model predictions file
log_prediction_file.close()

# calculate logistic regression model auc
log_auc = log_baseline.evaluate(test_data, metric='auc')

# print logistic regression model auc
print 'Logistic Regression Model AUC: {}\n'.format(log_auc)

########################################################################################################################

# baseline support vector machines model - uses all features
svm_baseline = gl.svm_classifier.create(train_data, target='click', features=features, validation_set=test_data,
                                        max_iterations=50)

# get support vector machines model predictions
svm_predictions = svm_baseline.predict(test_data, output_type='margin')

# open support vector machines model predictions file
with open('predictions/svm_predictions.csv', mode='w') as svm_prediction_file:
    # write headers to file
    svm_prediction_file.write('Id,Prediction\n')
    # set support vector machines model prediction id to 1
    svm_prediction_id = 1
    # for every support vector machines model prediction
    for svm_prediction in svm_predictions:
        # write support vector machines model prediction to file in requested format
        svm_prediction_file.write('{},{:.5f}\n'.format(svm_prediction_id, svm_prediction))
        # increment support vector machines model prediction id
        svm_prediction_id += 1

# close support vector machines model predictions file
svm_prediction_file.close()

# calculate support vector machines model auc
svm_auc = svm_baseline.evaluate(test_data, metric='f1_score')

# print support vector machines model auc
print 'Support Vector Machines Model AUC: {}'.format(svm_auc)

########################################################################################################################
