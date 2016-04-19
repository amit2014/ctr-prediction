
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


# one hot encoder
def one_hot_encoder(train_set):

    # set features
    features = ['url',
                'domain',

                'user_agent',

                'ad_slot_id',
                'ad_exchange',
                'ad_slot_width',
                'ad_slot_height',
                'ad_slot_format',
                'ad_slot_visibility',
                'ad_slot_floor_price']

    # create one hot encoder
    encoder = fe.create(train_set, fe.OneHotEncoder(features, max_categories=100))

    # return one hot encoder
    return encoder


########################################################################################################################


# encodes features
def one_hot_encode(data_set, encoder):

    # if data set features are already encoded
    if 'encoded_features' in data_set.column_names():
        # return train set
        return data_set

    # encode data set features
    encoded_data_set_features = encoder.transform(data_set)

    # return encoded train features
    return encoded_data_set_features


########################################################################################################################


# runs logistic regression model
def log_reg(train_set, validation_set, test_set, features):

    # baseline logistic regression model - uses all features
    log_baseline = gl.logistic_classifier.create(train_set,
                                                 target='click',
                                                 features=features,
                                                 # class_weights='auto',
                                                 validation_set=validation_set,
                                                 max_iterations=20)

    ####################################################################################################################

    results = log_baseline.evaluate(validation_set)
    print results

    # calculate logistic regression model validation set auc
    log_auc = log_baseline.evaluate(validation_set, metric='auc')

    # calculate logistic regression model validation set loss
    log_loss = log_baseline.evaluate(validation_set, metric='log_loss')

    # calculate logistic regression model validation set accuracy
    log_accuracy = log_baseline.evaluate(validation_set, metric='accuracy')

    # calculate logistic regression model validation set roc curve
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
    # validation_set.print_rows(5)
    # test_set.print_rows(5)

    ####################################################################################################################

    # create encoder
    encoder = one_hot_encoder(train_set)

    # one hot encode train set
    train_set = one_hot_encode(train_set, encoder)
    # one hot validation set
    validation_set = one_hot_encode(validation_set, encoder)
    # one hot encode test set
    test_set = one_hot_encode(test_set, encoder)

    # train_set.print_rows(5)
    # validation_set.print_rows(5)
    # test_set.print_rows(5)

    ####################################################################################################################

    '''
    click

    hour                - normal
    weekday             - normal

    url                 - one hot encoded
    domain              - one hot encoded
    user_agent          - one hot encoded
    ad_slot_id          - one hot encoded
    ad_exchange         - one hot encoded
    ad_slot_width       - one hot encoded
    ad_slot_height      - one hot encoded
    ad_slot_format      - one hot encoded
    ad_slot_floor_price - one hot encoded
    ad_slot_visibility  - one hot encoded

    ip                  - bad
    city                - bad
    region              - bad
    timestamp           - bad
    user_tags           - bad
    creative_id         - bad
    '''

    # set features
    features = ['hour',
                'weekday',

                'encoded_features']

    # run logistic regression model
    log_reg(train_set, validation_set, test_set, features)


########################################################################################################################


# runs main function
if __name__ == '__main__':
    main()


########################################################################################################################
