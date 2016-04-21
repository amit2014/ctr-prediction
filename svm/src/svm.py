
# import package
import graphlab as gl
import math

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
                                            max_iterations=10)

    # calculate support vector machines model validation set f1 score
    svm_f1_score = svm_baseline.evaluate(validation_set, metric='f1_score')

    # print support vector machines model validation set auc
    print 'Support Vector Machines Model - Validation Set - F1 Score: {}'.format(svm_f1_score)

    # get support vector machines model predictions
    svm_predictions = svm_baseline.predict(test_set, output_type='class')
    svm_values = svm_baseline.predict(test_set, output_type='margin')

    n_1 = svm_predictions.filter(lambda x: x == 1).size()
    n_0 = svm_predictions.filter(lambda x: x == 0).size()

    _a, _b = platt_scaling(svm_values, svm_predictions, n_1, n_0)
    svm_predictions = svm_predictions.apply(lambda x: apply_platt(x, _a, _b))

    # open support vector machines model predictions file
    with open('../output/svm_predictions.csv', mode='w') as svm_prediction_file:
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


def platt_scaling(svm_output, svm_prediction, n_1, n_0):
    _a = 0.
    _b = math.log((n_0+1) / (n_1+1))
    hi_target = (n_1+1) / (n_1+2)
    lo_target = 1 / (n_0+2)
    lambda_v = 1e-3
    old_err = 1e300
    pp = gl.SArray(data=[((n_1+1)/(n_0+n_1+2)) for _ in xrange(svm_output.size())], dtype=float)
    count = 0
    for it in xrange(100):
        a = b = c = d = e = 0.
        # compute the Hessian & gradient error function w.r.t. A & B
        for i in xrange(pp.size()):
            t = hi_target if svm_prediction[i] else lo_target
            d1 = pp[i] - t
            d2 = pp[i] * (1 - pp[i])
            a += svm_output[i] * svm_output[i] * d2
            b += d2
            c += svm_output[i] * d2
            d += svm_output[i] * d1
            e += d1

        # if gradient is really tiny, then stop
        if abs(d) < 1e-9 and abs(e) < 1e-9:
            break
        old_a = _a
        old_b = _b
        err = 0.
        while True:
            det = (a + lambda_v) * (b + lambda_v) - c*c
            if det == 0.: # if determinant of Hessian is zero
                # increases stabilizer
                lambda_v *= 10
                continue
            _a = old_a + ((b + lambda_v) * d - c*e) / det
            _b = old_b + ((a + lambda_v) * e - c*d) / det

            # now perform the goodness of fit
            err = 0.
            for j in xrange(pp.size()):
                p = 1 / (1 + math.exp(svm_output[j]*_a + _b))
                pp[j] = p
                ## At this step, make sure log(0) returns -200
                if p <= 1.383897e-87:
                    err -= t * (-200) + (1 - t) * math.log(1 - p)
                elif p == 1:
                    err -= t * math.log(p) + (1 - t) * (-200)
                else:
                    err -= t*math.log(p) + (1-t)*math.log(1-p)

                if err == -float("inf"):
                    print '==Something is wrong=='

            if err < old_err*(1 + 1e-7):
                lambda_v *= 0.1
                break

            # error did not decrease: increase stabilizer by factor of 10 & try again
            lambda_v *= 10
            if lambda_v >= 1e6: # something is broken: give up
                print '==Somethig is broken... giving up=='
                break

        diff = err - old_err
        scale = 0.5 * (err + old_err + 1)
        if diff > -1e-3*scale and diff < 1e-7*scale:
            count += 1
        else:
            count = 0
        print count
        old_err = err
        if count == 3:
            break

    return _a, _b


########################################################################################################################

def apply_platt(x, _a, _b):
    return 1 / (1 + math.exp(x*_a + _b))


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
