import graphlab as gl
import ffm
from convert import read_libffm_file

# loads data
def load_data():

    # load train data
    train_set = gl.SFrame.read_csv('../../data/train_data.txt', delimiter='\t', verbose=False, column_type_hints=str)
    # load test data
    test_set = gl.SFrame.read_csv('../../data/test_data.txt', delimiter='\t', verbose=False, column_type_hints=str)
    # split train data to train set and validation set
    train_set, validation_set = train_set.random_split(0.8, seed=1)

    # get features
    features = train_set.column_names()
    remove click label feature
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

    # return train set, validation set and features
    return train_set, validation_set, test_set, features


########################################################################################################################

# def make_dict(z):
#
#         d = {}
#
#         for (f, k, v) in z:
#             if f not in d:
#                 d[f] = {}
#             d[f][str(k)] = str(v)
#         return d
#
#
#     x = gl.SFrame.read_csv(filename, header=False)
#     x['s'] = x['X1'].apply(lambda x: x.split(','))
#     x['y'] = x['s'].apply(lambda x: x[0])
#     x['y'] = x['y'].astype(int)
#     x['features'] = x['s'].apply(lambda x: x[1:])
#     x['features'] = x['features'].apply(lambda x: [z.split(':') for z in x])
#     x['features'] = x['features'].apply(lambda x: make_dict(x))
#     sf = x[['y', 'features']]
#     return sf.unpack('features')


########################################################################################################################


# main function
def main():

    # load data
    train_set, validation_set, test_set, features = load_data()

    print(train_set)

    # train_set.save('no-header.csv', format='csv')

    # train_set = gl.SFrame.read_csv('no-header.csv', delimiter=',', verbose=False, column_type_hints=str)
    #

    train_set = train_set.add_row_number()

    def transform_row(row):

        return [':'.join([str(row['id']), str(k), v]) for k, v in row.items() if k != 'id']

    train_set['formatted_data'] = train_set.apply(lambda row: ' '.join(sorted(transform_row(row))))

    print(train_set['formatted_data'])

    # train_set['answer'].save('output')

    # train_set['answer'].save('training_set.csv', format='csv')

    # train_set['answer'].export_csv('output.csv', delimiter=' ', header=False, line_terminator='\n')

    # trainfile = 'no-header.csv'

    print ("converting")

    train = read_libffm_file(trainfile)

    print(train)


    # Train a model
    m = ffm.FFM()
    m.fit(train, features, target='click', features=features, nr_iters=10)
    yhat = m.predict(features)
    print yhat



########################################################################################################################


    # trainfile = 'lib/bigdata.tr.txt'
    # validfile = 'lib/bigdata.te.txt'
    # train = read_libffm_file(trainfile)
    # valid = read_libffm_file(validfile)
    #
    # print(train)
    #
    # train['y'] = train['y'].astype(int)
    # del train['features.0']
    # valid = valid[train.column_names()]
    # train.save('examples/small.tr.sframe')
    # valid.save('examples/small.te.sframe')
    #
    # features = [c for c in train.column_names() if c != 'y']
    #
    # # Train a model
    # m = ffm.FFM()
    # m.fit(train, valid, target='y', features=features, nr_iters=15)
    # yhat = m.predict(valid)
    # print yhat


########################################################################################################################


# runs main function
if __name__ == '__main__':
    main()


########################################################################################################################
