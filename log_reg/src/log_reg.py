
# import package
import graphlab as gl
import graphlab.toolkits.feature_engineering as fe

########################################################################################################################


# loads data
def load_data():

    # load train data
    train_set = gl.SFrame.read_csv('../../data/train_data.txt', delimiter='\t', verbose=False)

    # load test data
    test_set = gl.SFrame.read_csv('../../data/test_data.txt', delimiter='\t', verbose=False)

    # split train data to train set and validation set
    train_set, validation_set = train_set.random_split(0.9, seed=1337)

    train_set.remove_columns(['user_id', 'advertiser_id', 'log_type', 'key_page_url', 'anonymous_url_id'])
    validation_set.remove_columns(['user_id', 'advertiser_id', 'log_type', 'key_page_url', 'anonymous_url_id'])
    test_set.remove_columns(['user_id', 'advertiser_id', 'log_type', 'key_page_url', 'anonymous_url_id'])

    # return train set, validation set and features
    return train_set, validation_set, test_set


########################################################################################################################


# merges features
def feature_merger(train_set, validation_set):

    # if feature already in train set
    if 'ad_slot_width_height' in train_set.column_names():
        # return train set
        return train_set

    # merge features
    train_set['ad_slot_width_height'] = train_set['ad_slot_width'] + train_set['ad_slot_height']

    ####################################################################################################################

    # if feature already in validation set
    if 'ad_slot_width_height' in validation_set.column_names():
        # return validation set
        return validation_set

    # merge features
    validation_set['ad_slot_width_height'] = validation_set['ad_slot_width'] + validation_set['ad_slot_height']

    ####################################################################################################################

    # return train and validation set
    return train_set, validation_set


########################################################################################################################


# one hot encoder
def one_hot_encoder(train_set):

    # set features
    features = ['url',
                'domain']

    # create one hot encoder
    encoder = fe.create(train_set, fe.OneHotEncoder(features, max_categories=100))

    # return one hot encoder
    return encoder


########################################################################################################################


# encodes features
def one_hot_encode(train_set, validation_set):

    # create one hot encoder
    encoder = one_hot_encoder(train_set)

    # if train features are already encoded
    if 'encoded_features' in train_set.column_names():
        # return train set
        return train_set

    # encode train features
    encoded_train_features = encoder.transform(train_set)

    ####################################################################################################################

    # if validation features are already encoded
    if 'encoded_features' in validation_set.column_names():
        # return validation set
        return validation_set

    # encode validation features
    encoded_validation_features = encoder.transform(validation_set)

    ####################################################################################################################

    # return encoded train features
    return encoded_train_features, encoded_validation_features


########################################################################################################################


# feature hasher
def feature_hasher(train_set):

    # set features
    features = ['url',
                'domain']

    # create feature hasher
    hasher = fe.create(train_set, fe.FeatureHasher(features=features))

    # return feature hasher
    return hasher


########################################################################################################################


# hashes features
def feature_hash(train_set, validation_set):

    # create feature hasher
    hasher = feature_hasher(train_set)

    # if train features are already hashed
    if 'hashed_features' in train_set.column_names():
        # return train set
        return train_set

    # hash train features
    hashed_train_features = hasher.transform(train_set)

    ####################################################################################################################

    # if validation features are already hashed
    if 'hashed_features' in validation_set.column_names():
        # return validation set
        return validation_set

    # hash validation features
    hashed_validation_features = hasher.transform(validation_set)

    ####################################################################################################################

    # return hashed train and validation features
    return hashed_train_features, hashed_validation_features


########################################################################################################################


# runs logistic regression model
def log_reg(train_set, validation_set, test_set, features):

    # baseline logistic regression model - uses all features
    log_baseline = gl.logistic_classifier.create(train_set,
                                                 target='click',
                                                 features=features,
                                                 validation_set=validation_set,
                                                 max_iterations=20)

    ####################################################################################################################

    # calculate logistic regression model validation set auc
    log_auc = log_baseline.evaluate(validation_set, metric='auc')

    # calculate logistic regression model validation set loss
    log_loss = log_baseline.evaluate(validation_set, metric='log_loss')

    # calculate logistic regression model validation set accuracy
    log_accuracy = log_baseline.evaluate(validation_set, metric='accuracy')

    log_roc_curve = log_baseline.evaluate(validation_set, metric='roc_curve')

    # print logistic regression model validation set auc
    print 'Logistic Regression Model - Validation Set - AUC: {}\n'.format(log_auc)

    # print logistic regression model validation set loss
    print 'Logistic Regression Model - Validation Set - Log Loss: {}\n'.format(log_loss)

    # print logistic regression model validation set accuracy
    print 'Logistic Regression Model - Validation Set - Accuracy: {}\n'.format(log_accuracy)

    ####################################################################################################################

    # get logistic regression model predictions on validation set
    log_predictions_validation = log_baseline.predict(validation_set, output_type='probability')

    # get logistic regression model predictions on test set
    log_predictions_test = log_baseline.predict(test_set, output_type='probability')

    ####################################################################################################################

    # open logistic regression model predictions validation file
    with open('../output/log_predictions_validation.csv', mode='w') as log_prediction_validation_file:
        # write headers to file
        log_prediction_validation_file.write('Id,Prediction\n')
        # set logistic regression model prediction validation id to 1
        log_prediction_validation_id = 1
        # for every logistic regression model prediction validation
        for log_prediction_validation in log_predictions_validation:
            # write logistic regression model prediction validation to file in requested format
            log_prediction_validation_file.write('{},{:.5f}\n'.format(log_prediction_validation_id,
                                                                      log_prediction_validation))
            # increment logistic regression model prediction validation id
            log_prediction_validation_id += 1

    # close logistic regression model predictions validation file
    log_prediction_validation_file.close()

    ####################################################################################################################

    # open logistic regression model predictions test file
    with open('../output/log_predictions_test.csv', mode='w') as log_prediction_test_file:
        # write headers to file
        log_prediction_test_file.write('Id,Prediction\n')
        # set logistic regression model prediction test id to 1
        log_prediction_test_id = 1
        # for every logistic regression model prediction test
        for log_prediction_test in log_predictions_test:
            # write logistic regression model prediction test to file in requested format
            log_prediction_test_file.write('{},{:.5f}\n'.format(log_prediction_test_id, log_prediction_test))
            # increment logistic regression model prediction test id
            log_prediction_test_id += 1

    # close logistic regression model predictions test file
    log_prediction_test_file.close()


########################################################################################################################


# main function
def main():

    # load data
    train_set, validation_set, test_set = load_data()

    # train_set.print_rows(5)

    ####################################################################################################################

    # merge features
    # train_set, validation_set = feature_merger(train_set, validation_set)

    ####################################################################################################################

    # one hot encode train and validation set
    train_set_encoded, validation_set_encoded = one_hot_encode(train_set, validation_set)

    # train_set_encoded.print_rows(5)

    ####################################################################################################################

    # feature hash train and validation set
    # train_set_hashed, validation_set_hashed = feature_hash(train_set, validation_set)

    # train_set_hashed.print_rows(5)

    ####################################################################################################################

    encoded_features_train = train_set_encoded.select_column('encoded_features')
    encoded_features_validation = validation_set_encoded.select_column('encoded_features')
    train_set.add_column(encoded_features_train, 'encoded_features')
    validation_set.add_column(encoded_features_validation, 'encoded_features')

    # hashed_features_train = train_set_hashed.select_column('hashed_features')
    # hashed_features_validation = validation_set_hashed.select_column('hashed_features')
    # train_set.add_column(hashed_features_train, 'hashed_features')
    # validation_set.add_column(hashed_features_validation, 'hashed_features')

    ####################################################################################################################

    # train_set.print_rows(5)
    # validation_set.print_rows(5)

    '''
    click

    url                 - one hot encoded
    domain              - one hot encoded

    hour                - normal
    weekday             - normal
    user_agent          - normal
    ad_slot_width       - normal
    ad_slot_height      - normal
    ad_slot_format      - normal
    ad_slot_floor_price - normal

    ip                  - bad
    city                - bad
    region              - bad
    timestamp           - bad
    user_tags           - bad
    ad_slot_id          - bad
    ad_exchange         - bad
    creative_id         - bad
    ad_slot_visibility  - bad
    '''

    # set features
    features = ['hour',
                'weekday',

                'user_agent',

                'ad_slot_width',
                'ad_slot_height',
                'ad_slot_format',
                'ad_slot_floor_price',

                'encoded_features']

    # run logistic regression model
    log_reg(train_set, validation_set, test_set, features)


########################################################################################################################


# runs main function
if __name__ == '__main__':
    main()


########################################################################################################################
