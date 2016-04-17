
# import package
import graphlab as gl

########################################################################################################################


# loads data
def load_data():

    # load train data
    train_set = gl.SFrame.read_csv('data/train_data.txt', delimiter='\t', verbose=False)
    # load test data
    test_set = gl.SFrame.read_csv('data/test_data.txt', delimiter='\t', verbose=False)
    # split train data to train set and validation set
    train_set, validation_set = train_set.random_split(0.8, seed=1)

    # get features
    features = train_set.column_names()
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

    # return train set, validation set and features
    return train_set, validation_set, test_set, features


########################################################################################################################


# runs logistic regression model
def log_reg(train_set, validation_set, test_set, features):

    # baseline logistic regression model - uses all features
    log_baseline = gl.logistic_classifier.create(train_set, target='click', features=features,
                                                 validation_set=validation_set, max_iterations=50)

    # calculate logistic regression model validation set auc
    log_auc = log_baseline.evaluate(validation_set, metric='auc')

    # print logistic regression model validation set auc
    print 'Logistic Regression Model - Validation Set - AUC: {}\n'.format(log_auc)

    # get logistic regression model predictions
    log_predictions = log_baseline.predict(test_set, output_type='probability')

    # open logistic regression model predictions file
    with open('log_reg/output/log_predictions.csv', mode='w') as log_prediction_file:
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


########################################################################################################################


# main function
def main():

    # load data
    train_set, validation_set, test_set, features = load_data()

    # run logistic regression model
    log_reg(train_set, validation_set, test_set, features)


########################################################################################################################


# runs main function
if __name__ == '__main__':
    main()


########################################################################################################################
