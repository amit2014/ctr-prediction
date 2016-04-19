import graphlab as gl
import ffm
from convert import read_libffm_file

# loads data
def load_data():

    # load train data
    train_set = gl.SFrame.read_csv('../../../data/train_data.txt', delimiter='\t', verbose=False)
    # load test data
    test_set = gl.SFrame.read_csv('../../../data/test_data.txt', delimiter='\t', verbose=False)
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


# main function
def main():

    # # load data
    # train_set, validation_set, test_set, features = load_data()
    #
    # train_set['click'] = train_set['click'].astype(int)

    # print(train_set['click'])

    # # Train a model
    # m = ffm.FFM()
    # m.fit(train_set, features, target='click', features=features, nr_iters=25)
    # yhat = m.predict(features)
    # print yhat



########################################################################################################################


    trainfile = 'lib/bigdata.tr.txt'
    validfile = 'lib/bigdata.te.txt'
    train = read_libffm_file(trainfile)
    valid = read_libffm_file(validfile)

    print(train)

    train['y'] = train['y'].astype(int)
    del train['features.0']
    valid = valid[train.column_names()]
    train.save('examples/small.tr.sframe')
    valid.save('examples/small.te.sframe')

    features = [c for c in train.column_names() if c != 'y']

    # Train a model
    m = ffm.FFM()
    m.fit(train, valid, target='y', features=features, nr_iters=15)
    yhat = m.predict(valid)
    print yhat


########################################################################################################################


# runs main function
if __name__ == '__main__':
    main()


########################################################################################################################
