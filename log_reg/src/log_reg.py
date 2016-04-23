
# import package
import graphlab as gl
import plotly.plotly as py
import plotly.graph_objs as go
from graphlab import model_parameter_search
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

    train_set.remove_columns(['ip',
                              'city',
                              'region',
                              'user_id',
                              'log_type',
                              'timestamp',
                              'creative_id',
                              'key_page_url',
                              'advertiser_id',
                              'anonymous_url_id'])

    validation_set.remove_columns(['ip',
                                   'city',
                                   'region',
                                   'user_id',
                                   'log_type',
                                   'timestamp',
                                   'creative_id',
                                   'key_page_url',
                                   'advertiser_id',
                                   'anonymous_url_id'])

    test_set.remove_columns(['ip',
                             'city',
                             'region',
                             'user_id',
                             'log_type',
                             'timestamp',
                             'creative_id',
                             'key_page_url',
                             'advertiser_id',
                             'anonymous_url_id'])

    # return train set, validation set and features
    return train_set, validation_set, test_set


########################################################################################################################


# splits user agent into os and browser
def split_user_agent(data, os_l, browser_l):
    data[os_l] = data.apply(lambda row: row['user_agent'].split('_')[0])
    data[browser_l] = data.apply(lambda row: row['user_agent'].split('_')[1])
    return data


########################################################################################################################


# turns ad slot floor price into price bucket
def bucket_price(col):
    price = int(col)
    if price > 100:
        return '101+'
    elif price > 50:
        return '51-100'
    elif price > 10:
        return '11-50'
    elif price > 0:
        return '1-10'
    else:
        return "0"


########################################################################################################################


# splits tags
def split_tags(tags_l):
    if len(tags_l) > 0:
        tags = tags_l.split(',')
        return dict(zip(tags, [1 for tag in tags]))
    else:
        return {}


########################################################################################################################


# merges features
def feature_merger(data_set):

    # merge features
    data_set['url_domain'] = data_set['url'] + data_set['domain']

    # merge features
    data_set['city_region'] = data_set['city'].split('_') + data_set['region']

    # merge features
    data_set['ad_slot_width_height'] = data_set['ad_slot_width'] + data_set['ad_slot_height']

    # return data set
    return data_set


########################################################################################################################


# feature hasher
def feature_hasher(train_set):

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

    # create hasher
    hasher = fe.create(train_set, fe.FeatureHasher(features, num_bits=22))

    # return hasher
    return hasher


########################################################################################################################


# one hot encoder
def one_hot_encoder(train_set):

    # set features
    features = ['url',
                'domain',

                'os',
                'browser',

                'tags',

                'ad_slot_id',
                'ad_exchange',
                'ad_slot_width',
                'ad_slot_height',
                'ad_slot_format',
                'ad_slot_visibility']

    # create one hot encoder
    encoder = fe.create(train_set, fe.OneHotEncoder(features, max_categories=120))

    # return one hot encoder
    return encoder


########################################################################################################################


# hashes features
def feature_hash(data_set, hasher):

    # hash data set features
    hashed_data_set_features = hasher.transform(data_set)

    # return hashed train features
    return hashed_data_set_features


########################################################################################################################


# encodes features
def one_hot_encode(data_set, encoder):

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
                                                 class_weights='auto',
                                                 validation_set=validation_set,
                                                 max_iterations=20)

    ####################################################################################################################

    # calculate logistic regression model validation set results
    results = log_baseline.evaluate(validation_set)

    # calculate logistic regression model validation set precision
    log_precision = log_baseline.evaluate(validation_set, metric='precision')

    # calculate logistic regression model validation set recall
    log_recall = log_baseline.evaluate(validation_set, metric='recall')

    # calculate logistic regression model validation set f1 score
    log_f1score = log_baseline.evaluate(validation_set, metric='f1_score')

    # calculate logistic regression model validation set loss
    log_loss = log_baseline.evaluate(validation_set, metric='log_loss')

    # calculate logistic regression model validation set accuracy
    log_accuracy = log_baseline.evaluate(validation_set, metric='accuracy')

    # calculate logistic regression model validation set auc
    log_auc = log_baseline.evaluate(validation_set, metric='auc')

    # print logistic regression model validation set results
    # print results

    # print logistic regression model validation set precision
    # print 'Logistic Regression Model - Validation Set - Precision: {}\n'.format(log_precision)

    # print logistic regression model validation set recall
    # print 'Logistic Regression Model - Validation Set - Recall: {}\n'.format(log_recall)

    # print logistic regression model validation set f1 score
    # print 'Logistic Regression Model - Validation Set - F1 Score: {}\n'.format(log_f1score)

    # print logistic regression model validation set loss
    # print 'Logistic Regression Model - Validation Set - Log Loss: {}\n'.format(log_loss)

    # print logistic regression model validation set accuracy
    # print 'Logistic Regression Model - Validation Set - Accuracy: {}\n'.format(log_accuracy)

    # print logistic regression model validation set auc
    print 'Logistic Regression Model - Validation Set - AUC: {}\n'.format(log_auc)

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

    # calculate logistic regression model validation set roc curve
    log_roc_curve = log_baseline.evaluate(validation_set, metric='roc_curve')

    # return logistic regression model validation set roc curve
    return log_roc_curve


########################################################################################################################


# plots log reg gbm roc curve figure
def plot_log_gbm_roc():

    # list to hold log false positive rates
    log_false_pos_rates = []
    # list to hold log true positive rates
    log_true_pos_rates = []
    # open file
    with open('../output/log_roc_curve.csv', mode='r') as log_roc_curve_file:
        next(log_roc_curve_file)
        # for every line in file
        for line in log_roc_curve_file:
            # split line and assign results to tokens
            tokens = line.strip('\n').split(",")
            # assign first token to log false positive rates
            log_false_pos_rates += [tokens[0]]
            # assign second token to log true positive rates
            log_true_pos_rates += [tokens[1]]
    # close file
    log_roc_curve_file.close()

    # list to hold gbm positive rates
    gbm_false_pos_rates = []
    # list to hold gbm positive rates
    gbm_true_pos_rates = []
    # open file
    with open('../../experiments/output/gbm_roc_curve.csv', mode='r') as gbm_roc_curve_file:
        next(gbm_roc_curve_file)
        # for every line in file
        for line in gbm_roc_curve_file:
            # split line and assign results to tokens
            tokens = line.strip('\n').split(",")
            # assign first token to gbm false positive rates
            gbm_false_pos_rates += [tokens[0]]
            # assign second token to gbm true positive rates
            gbm_true_pos_rates += [tokens[1]]
    # close file
    gbm_roc_curve_file.close()

    # setup trace 1
    trace1 = go.Scatter(
        x=log_false_pos_rates,
        y=log_true_pos_rates,
        name="Logistic Regression Model",
        mode="lines",
        line=dict(
            color='blue'
        )
    )

    # setup trace 2
    trace2 = go.Scatter(
        x=gbm_false_pos_rates,
        y=gbm_true_pos_rates,
        name="Gradient Boosted Machine",
        mode="lines",
        line=dict(
            color='red'
        )
    )

    # setup data
    data = [trace1, trace2]

    # setup layout
    layout = go.Layout(
        title='Logistic Regression vs Gradient Boosted Machine - ROC Curve',
        showlegend=True,
        xaxis=dict(
            title='False Positive Rate',
            showline=True,
            zeroline=False
        ),
        yaxis=dict(
            title='True Positive Rate',
            showline=True,
            zeroline=False
        ),
        legend=dict(
            x=0.62,
            y=0.1,
            borderwidth=1
        )
    )

    # setup figure
    fig = go.Figure(data=data, layout=layout)

    # save figure
    py.image.save_as(fig, filename='../output/log_gbm_roc_curve.png')


########################################################################################################################


# main function
def main():

    # load data
    train_set, validation_set, test_set = load_data()

    # train_set.print_rows(5)
    # validation_set.print_rows(5)
    # test_set.print_rows(5)

    ####################################################################################################################

    # price_cat_label = 'price_bucket'
    # train_set.add_column(train_set.select_column('ad_slot_floor_price').apply(lambda x: bucket_price(x)),
    #                      price_cat_label)

    # validation_set.add_column(validation_set.select_column('ad_slot_floor_price').apply(lambda x: bucket_price(x)),
    #                           price_cat_label)

    # test_set.add_column(test_set.select_column('ad_slot_floor_price').apply(lambda x: bucket_price(x)),
    #                     price_cat_label)

    # train_set.print_rows(5)
    # validation_set.print_rows(5)
    # test_set.print_rows(5)

    ####################################################################################################################

    tags_label = 'tags'
    train_set.add_column(train_set.select_column('user_tags').apply(lambda x: split_tags(x)), tags_label)
    validation_set.add_column(validation_set.select_column('user_tags').apply(lambda x: split_tags(x)), tags_label)
    test_set.add_column(test_set.select_column('user_tags').apply(lambda x: split_tags(x)), tags_label)

    # train_set.print_rows(5)
    # validation_set.print_rows(5)
    # test_set.print_rows(5)

    ####################################################################################################################

    os_label = 'os'
    browser_label = 'browser'
    train_set = split_user_agent(train_set, os_label, browser_label)
    validation_set = split_user_agent(validation_set, os_label, browser_label)
    test_set = split_user_agent(test_set, os_label, browser_label)

    # train_set.print_rows(5)
    # validation_set.print_rows(5)
    # test_set.print_rows(5)

    ####################################################################################################################

    # merge features
    # train_set = feature_merger(train_set)
    # validation_set = feature_merger(validation_set)
    # test_set = feature_merger(test_set)

    # train_set.print_rows(5)
    # validation_set.print_rows(5)
    # test_set.print_rows(5)

    ####################################################################################################################

    # create hasher
    # hasher = feature_hasher(train_set)

    # feature hash train set
    # train_set = one_hot_encode(train_set, hasher)
    # feature hash validation set
    # validation_set = one_hot_encode(validation_set, hasher)
    # feature hash test set
    # test_set = one_hot_encode(test_set, hasher)

    # train_set.print_rows(5)
    # validation_set.print_rows(5)
    # test_set.print_rows(5)

    ####################################################################################################################

    # create encoder
    encoder = one_hot_encoder(train_set)

    # one hot encode train set
    train_set = one_hot_encode(train_set, encoder)
    # one hot encode validation set
    validation_set = one_hot_encode(validation_set, encoder)
    # one hot encode test set
    test_set = one_hot_encode(test_set, encoder)

    # train_set.print_rows(5)
    # validation_set.print_rows(5)
    # test_set.print_rows(5)

    ####################################################################################################################

    # perform model parameter search
    # mps = model_parameter_search.create((train_set, validation_set), gl.logistic_classifier.create,
    #                                     {'target': 'click'})
    # mps_results = mps.get_results()
    # print mps_results

    ####################################################################################################################

    '''
    click

    hour                - normal
    weekday             - normal

    url                 - one hot encoded
    domain              - one hot encoded

    os                  - one hot encoded
    browser             - one hot encoded

    tags                - one hot encoded

    ad_slot_id          - one hot encoded
    ad_exchange         - one hot encoded
    ad_slot_width       - one hot encoded
    ad_slot_height      - one hot encoded
    ad_slot_format      - one hot encoded
    ad_slot_visibility  - one hot encoded

    ip                  - bad
    city                - bad
    region              - bad
    user_id             - bad
    timestamp           - bad
    user_tags           - bad
    creative_id         - bad
    key_page_url        - bad
    advertiser_id       - bad
    anonymous_url_id    - bad
    ad_slot_floor_price - bad
    '''

    # set features
    features = ['hour',
                'weekday',

                'encoded_features']

    # run logistic regression model
    log_roc_curve = log_reg(train_set, validation_set, test_set, features)

    # save logistic regression model validation set roc curve to file
    # log_roc_curve.get('roc_curve').select_columns(['fpr', 'tpr']).save('../output/log_roc_curve.csv')

    # plot log reg and gbm roc curve
    # plot_log_gbm_roc()


########################################################################################################################


# runs main function
if __name__ == '__main__':
    main()


########################################################################################################################
