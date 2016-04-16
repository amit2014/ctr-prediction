
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

    # return train set, validation set, test set and features
    return train_set, validation_set, test_set, features

########################################################################################################################


# runs support vector machine model
def svm(train_set, validation_set, test_set, features):

    # baseline support vector machines model - uses all features
    svm_baseline = gl.svm_classifier.create(train_set, target='click', features=features,
                                            validation_set=validation_set,
                                            max_iterations=50)

    # calculate support vector machines model validation set f1 score
    svm_f1_score = svm_baseline.evaluate(validation_set, metric='f1_score')

    # print support vector machines model validation set auc
    print 'Support Vector Machines Model - Validation Set - F1 Score: {}'.format(svm_f1_score)

    # get support vector machines model predictions
    svm_predictions = svm_baseline.predict(test_set, output_type='margin')

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


########################################################################################################################


# main function
def main():

    # load data
    train_set, validation_set, test_set, features = load_data()

    # run support vector machine model
    svm(train_set, validation_set, test_set, features)


########################################################################################################################


# runs main function
if __name__ == '__main__':
    main()


########################################################################################################################
