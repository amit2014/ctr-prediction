
# import package
import graphlab as gl
import feature_engineering as cfe
import re

########################################################################################################################


# loads data
def load_data():

    # load train data
    train_set = gl.SFrame.read_csv('../../data/train_data.txt', delimiter='\t', verbose=False)
    # load test data
    test_set = gl.SFrame.read_csv('../../data/test_data.txt', delimiter='\t', verbose=False)
    # split train data to train set and validation set
    train_set, validation_set = train_set.random_split(0.8, seed=1)

    # get features
    features = train_set.column_names()
    # remove click label feature
    features.remove('click')

    # remove some features temporarily
    features.remove('ip') #very sparse
    features.remove('url')
    features.remove('domain')
    features.remove('user_id')
    features.remove('log_type') # just impressions=1
    features.remove('timestamp') # useless values
    features.remove('user_tags') # seems that a short No of instances actually use this feature
    features.remove('ad_slot_id')
    features.remove('creative_id')
    features.remove('key_page_url')
    features.remove('advertiser_id') # useless value
    features.remove('anonymous_url_id') # useless value

    # features = ['weekday', 'hour',
    #             # 'user_agent',
    #             'ad_exchange',
    #             #'domain',
    #             'url',
    #             'user_id',
    #             #'ad_slot_id',
    #             #'ad_slot_width', 'ad_slot_height',
    #             #'ad_slot_format', 'ad_slot_visibility',
    #             'ad_slot_floor_price',
    #             'creative_id',
    #             'key_page_url'#,
    #             #'region', 'city'
    #             ]

    features = ['weekday', 'hour',
                'ad_slot_id',
                'user_id',
                ]

    train_set, _ = cfe.site_feat(train_set)
    validation_set, _ = cfe.site_feat(validation_set)
    test_set, site_feat = cfe.site_feat(test_set)
    features.append(site_feat)

    train_set, _ = cfe.location_feat(train_set)
    validation_set, _ = cfe.location_feat(validation_set)
    test_set, location_feat = cfe.location_feat(test_set)
    features.append(location_feat)

    uagent_desktop_label = 'uagent_desktop'
    regexp = re.compile(r'windows|linux|mac')
    train_set = cfe.apply_separate_uagent(train_set, regexp, uagent_desktop_label)
    validation_set = cfe.apply_separate_uagent(validation_set, regexp, uagent_desktop_label)
    features.append(uagent_desktop_label)
    #
    uagent_mobile_label = 'uagent_mobile'
    regexp = re.compile(r'android|ios|other')
    train_set = cfe.apply_separate_uagent(train_set, regexp, uagent_mobile_label)
    validation_set = cfe.apply_separate_uagent(validation_set, regexp, uagent_mobile_label)
    features.append(uagent_mobile_label)

    # user_tags_label = 'user_tags_dict'
    # train_set = cfe.apply_dictionary(train_set, 'user_tags', user_tags_label)
    # validation_set = cfe.apply_dictionary(validation_set, 'user_tags', user_tags_label)
    # features.append(user_tags_label)

    interaction_columns = ['ad_slot_width', 'ad_slot_height', 'ad_slot_format', 'ad_slot_visibility', 'hour']
    interaction_columns2 = ['domain', 'region', 'city']

    # quad, quad_label = cfe.create_quad_features(train_set, interaction_columns)
    # train_set = cfe.apply_feature_eng(train_set, quad, quad_label)
    # validation_set = cfe.apply_feature_eng(validation_set, quad, quad_label)
    # features.append(quad_label)

    # onehot, onehot_label = cfe.create_onehot_features(train_set, interaction_columns2, 300)
    # train_set = cfe.apply_feature_eng(train_set, onehot, onehot_label)
    # validation_set = cfe.apply_feature_eng(validation_set, onehot, onehot_label)
    # features.append(onehot_label)

    # for checking features
    # print(features)

    # return train set, validation set, test set and features
    print features
    return train_set, validation_set, test_set, features

########################################################################################################################


# runs support vector machine model
def gbm(train_set, validation_set, test_set, features):

    # gbm_baseline = gl.boosted_trees_classifier.create(train_set, target='click', features=features,
    #                                                   validation_set=validation_set, max_depth=100, step_size=0.1,
    #                                                   max_iterations=50, min_child_weight=train_set.shape[0]/1000,
    #                                                   early_stopping_rounds=5)

    gbm_baseline = gl.logistic_classifier.create(train_set, target='click', features=features,
                                                 validation_set=validation_set, max_iterations=50)

    # targets = validation_set['click']
    # predictions = gbm_baseline.predict(validation_set, output_type='class')
    gbm_f1_score = gbm_baseline.evaluate(validation_set, metric='auc')
    accuracy = gbm_baseline.evaluate(validation_set, metric='accuracy')
    # gbm_f1_score = gl.evaluation.f1_score(targets, predictions)
    # log_loss = gl.evaluation.log_loss(targets, predictions)
    # precision = gl.evaluation.precision(targets, predictions)
    # recall = gl.evaluation.precision(targets, predictions)
    # roc_curve = gl.evaluation.roc_curve(targets, predictions)
    # auc = gl.evaluation.auc(targets, predictions)

    print 'Gradient Boosted Machine Model - Validation Set - F1 Score: {} - Precision: {} - Recall: {} Log-Loss: {}'\
        .format(gbm_f1_score, accuracy, 0, 0)
        # .format(gbm_f1_score, precision, recall, log_loss)

    results = gbm_baseline.evaluate(validation_set)
    print results

    gbm_predictions = gbm_baseline.predict(test_set, output_type='probability')

    with open('../output/gbm_predictions.csv', mode='w') as prediction_file:
        # write headers to file
        prediction_file.write('Id,Prediction\n')
        prediction_id = 1
        for prediction in gbm_predictions:
            prediction_file.write('{},{:.5f}\n'.format(prediction_id, prediction))
            prediction_id += 1

    prediction_file.close()


########################################################################################################################


# main function
def main():

    # load data
    train_set, validation_set, test_set, features = load_data()

    # run support vector machine model
    gbm(train_set, validation_set, test_set, features)


########################################################################################################################


# runs main function
if __name__ == '__main__':

    main()


########################################################################################################################
