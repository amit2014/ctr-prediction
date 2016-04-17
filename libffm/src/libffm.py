
# import package
import graphlab as gl
import subprocess, sys, os, time

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


# runs libffm
def libffm():

    pass




########################################################################################################################


# main function
def main():

    start = time.time()

    train_set, validation_set, test_set, features = load_data()

    print('time taken to run the model: {0:.0f}'.format(time.time()-start))


########################################################################################################################


# runs main function
if __name__ == '__main__':
    main()


########################################################################################################################
