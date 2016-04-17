
# import packages
import argparse
import graphlab as gl

########################################################################################################################


# loads data
def load_data():

    # load data
    train_set = gl.SFrame.read_csv('../../data/train_data.txt', delimiter='\t', verbose=False)

    # return train set
    return train_set


########################################################################################################################


# gets click count
def click_count(train_set):

    # get click count
    click = train_set.groupby('click',
                              {'count': gl.aggregate.COUNT('click')}).sort('count', ascending=False)

    # save click count to file
    click.save('../output/click.csv')


########################################################################################################################


# gets user agent ctr and count
def user_agent_ctr(train_set):

    # get user agent ctr and count
    user_agent = train_set.groupby('user_agent',
                                   {'ctr': gl.aggregate.MEAN('click')},
                                   {'count': gl.aggregate.COUNT('user_agent')}).sort('ctr', ascending=False)

    # save user agent ctr and count to file
    user_agent.save('../output/user_agent.csv')


########################################################################################################################


# gets hour ctr and count
def hour_ctr(train_set):

    # get hour ctr and count
    hour = train_set.groupby('hour',
                             {'ctr': gl.aggregate.MEAN('click')},
                             {'count': gl.aggregate.COUNT('hour')}).sort('ctr', ascending=False)

    # save hour ctr and count to file
    hour.save('../output/hour.csv')


########################################################################################################################


# gets weekday ctr and count
def weekday_ctr(train_set):

    # get weekday ctr and count
    weekday = train_set.groupby('weekday',
                                {'ctr': gl.aggregate.MEAN('click')},
                                {'count': gl.aggregate.COUNT('weekday')}).sort('ctr', ascending=False)

    # save weekday ctr and count to file
    weekday.save('../output/weekday.csv')


########################################################################################################################


# gets timestamp ctr and count
def timestamp_ctr(train_set):

    # get timestamp ctr and count
    timestamp = train_set.groupby('timestamp',
                                  {'ctr': gl.aggregate.MEAN('click')},
                                  {'count': gl.aggregate.COUNT('timestamp')}).sort('ctr', ascending=False)

    # save timestamp ctr and count to file
    timestamp.save('../output/timestamp.csv')


########################################################################################################################


# gets city ctr and count
def city_ctr(train_set):

    # get city ctr and count
    city = train_set.groupby('city',
                             {'ctr': gl.aggregate.MEAN('click')},
                             {'count': gl.aggregate.COUNT('city')}).sort('ctr', ascending=False)

    # save city ctr and count to file
    city.save('../output/city.csv')


########################################################################################################################


# gets region ctr and count
def region_ctr(train_set):

    # get region ctr and count
    region = train_set.groupby('region',
                               {'ctr': gl.aggregate.MEAN('click')},
                               {'count': gl.aggregate.COUNT('region')}).sort('ctr', ascending=False)

    # save region ctr and count to file
    region.save('../output/region.csv')


########################################################################################################################


# gets ad exchange ctr and count
def ad_exchange_ctr(train_set):

    # get ad exchange ctr and count
    ad_exchange = train_set.groupby('ad_exchange',
                                    {'ctr': gl.aggregate.MEAN('click')},
                                    {'count': gl.aggregate.COUNT('ad_exchange')}).sort('ctr', ascending=False)

    # save ad exchange ctr and count to file
    ad_exchange.save('../output/ad_exchange.csv')


########################################################################################################################


# gets ad slot floor price ctr and count
def ad_slot_floor_price_ctr(train_set):

    # get ad slot floor price ctr and count
    ad_slot_floor_price = train_set.groupby('ad_slot_floor_price',
                                            {'ctr': gl.aggregate.MEAN('click')},
                                            {'count': gl.aggregate.COUNT('ad_slot_floor_price')}
                                            ).sort('ctr', ascending=False)

    # save ad slot floor price ctr and count to file
    ad_slot_floor_price.save('../output/ad_slot_floor_price.csv')


########################################################################################################################


# gets ad slot visibility ctr and count
def ad_slot_visibility_ctr(train_set):

    # get ad slot visibility ctr and count
    ad_slot_visibility = train_set.groupby('ad_slot_visibility',
                                           {'ctr': gl.aggregate.MEAN('click')},
                                           {'count': gl.aggregate.COUNT('ad_slot_visibility')}
                                           ).sort('ctr', ascending=False)

    # save ad slot visibility ctr and count to file
    ad_slot_visibility.save('../output/ad_slot_visibility.csv')


########################################################################################################################


# gets ad slot format ctr and count
def ad_slot_format_ctr(train_set):

    # get ad slot format ctr and count
    ad_slot_format = train_set.groupby('ad_slot_format',
                                       {'ctr': gl.aggregate.MEAN('click')},
                                       {'count': gl.aggregate.COUNT('ad_slot_format')}).sort('ctr', ascending=False)

    # save ad slot format ctr and count to file
    ad_slot_format.save('../output/ad_slot_format.csv')


########################################################################################################################


# gets ad slot width ctr and count
def ad_slot_width_ctr(train_set):

    # get ad slot width ctr and count
    ad_slot_width = train_set.groupby('ad_slot_width',
                                      {'ctr': gl.aggregate.MEAN('click')},
                                      {'count': gl.aggregate.COUNT('ad_slot_width')}).sort('ctr', ascending=False)

    # save ad slot width ctr and count to file
    ad_slot_width.save('../output/ad_slot_width.csv')


########################################################################################################################


# gets ad slot height ctr and count
def ad_slot_height_ctr(train_set):

    # get ad slot height ctr and count
    ad_slot_height = train_set.groupby('ad_slot_height',
                                       {'ctr': gl.aggregate.MEAN('click')},
                                       {'count': gl.aggregate.COUNT('ad_slot_height')}).sort('ctr', ascending=False)

    # save ad slot height ctr and count to file
    ad_slot_height.save('../output/ad_slot_height.csv')


########################################################################################################################


# gets ad slot width height ctr and count
def ad_slot_width_height_ctr(train_set):

    # get ad slot width height ctr and count
    ad_slot_width_height = train_set.groupby('ad_slot_width',
                                             {'ad_slot_height': gl.aggregate.SELECT_ONE('ad_slot_height')},
                                             {'ctr': gl.aggregate.MEAN('click')},
                                             {'width_count': gl.aggregate.COUNT('ad_slot_width')},
                                             {'height_count': gl.aggregate.COUNT('ad_slot_height')}
                                             ).sort('ctr', ascending=False)

    # save ad slot width height ctr and count to file
    ad_slot_width_height.save('../output/ad_slot_width_height.csv')


########################################################################################################################


# gets advertiser id ctr and count
def advertiser_id_ctr(train_set):

    # get advertiser id ctr and count
    advertiser_id = train_set.groupby('advertiser_id',
                                      {'ctr': gl.aggregate.MEAN('click')},
                                      {'count': gl.aggregate.COUNT('advertiser_id')}).sort('ctr', ascending=False)

    # save advertiser id ctr and count to file
    advertiser_id.save('../output/advertiser_id.csv')


########################################################################################################################


# gets anonymous url id ctr and count
def anonymous_url_id_ctr(train_set):

    # get anonymous url id ctr and count
    anonymous_url_id = train_set.groupby('anonymous_url_id',
                                         {'ctr': gl.aggregate.MEAN('click')},
                                         {'count': gl.aggregate.COUNT('anonymous_url_id')}).sort('ctr', ascending=False)

    # save anonymous url id ctr and count to file
    anonymous_url_id.save('../output/anonymous_url_id.csv')


########################################################################################################################


# gets log type ctr and count
def log_type_ctr(train_set):

    # get log type ctr and count
    log_type = train_set.groupby('log_type',
                                 {'ctr': gl.aggregate.MEAN('click')},
                                 {'count': gl.aggregate.COUNT('log_type')}).sort('ctr', ascending=False)

    # save log type ctr and count to file
    log_type.save('../output/log_type.csv')


########################################################################################################################


# gets user id ctr and count
def user_id_ctr(train_set):

    # get user id ctr and count
    user_id = train_set.groupby('user_id',
                                {'ctr': gl.aggregate.MEAN('click')},
                                {'count': gl.aggregate.COUNT('user_id')}).sort('ctr', ascending=False)

    # save user id ctr and count to file
    user_id.save('../output/user_id.csv')


########################################################################################################################


# gets ip ctr and count
def ip_ctr(train_set):

    # get ip ctr and count
    ip = train_set.groupby('ip',
                           {'ctr': gl.aggregate.MEAN('click')},
                           {'count': gl.aggregate.COUNT('ip')}).sort('ctr', ascending=False)

    # save ip ctr and count to file
    ip.save('../output/ip.csv')


########################################################################################################################


# gets domain ctr and count
def domain_ctr(train_set):

    # get domain ctr and count
    domain = train_set.groupby('domain',
                               {'ctr': gl.aggregate.MEAN('click')},
                               {'count': gl.aggregate.COUNT('domain')}).sort('ctr', ascending=False)

    # save domain ctr and count to file
    domain.save('../output/domain.csv')


########################################################################################################################


# gets url ctr and count
def url_ctr(train_set):

    # get url ctr and count
    url = train_set.groupby('url',
                            {'ctr': gl.aggregate.MEAN('click')},
                            {'count': gl.aggregate.COUNT('url')}).sort('ctr', ascending=False)

    # save url ctr and count to file
    url.save('../output/url.csv')


########################################################################################################################


# gets ad slot id ctr and count
def ad_slot_id_ctr(train_set):

    # get ad slot id ctr and count
    ad_slot_id = train_set.groupby('ad_slot_id',
                                   {'ctr': gl.aggregate.MEAN('click')},
                                   {'count': gl.aggregate.COUNT('ad_slot_id')}).sort('ctr', ascending=False)

    # save ad slot id ctr and count to file
    ad_slot_id.save('../output/ad_slot_id.csv')


########################################################################################################################


# gets creative id ctr and count
def creative_id_ctr(train_set):

    # get creative id ctr and count
    creative_id = train_set.groupby('creative_id',
                                    {'ctr': gl.aggregate.MEAN('click')},
                                    {'count': gl.aggregate.COUNT('creative_id')}).sort('ctr', ascending=False)

    # save creative id ctr and count to file
    creative_id.save('../output/creative_id.csv')


########################################################################################################################


# gets creative id ctr and count
def key_page_url_ctr(train_set):

    # get key page url ctr and count
    key_page_url = train_set.groupby('key_page_url',
                                     {'ctr': gl.aggregate.MEAN('click')},
                                     {'count': gl.aggregate.COUNT('key_page_url')}).sort('ctr', ascending=False)

    # save key page url ctr and count to file
    key_page_url.save('../output/key_page_url.csv')


########################################################################################################################


# gets user tags ctr and count
def user_tags_ctr(train_set):

    # get user tags ctr and count
    user_tags = train_set.groupby('user_tags',
                                  {'ctr': gl.aggregate.MEAN('click')},
                                  {'count': gl.aggregate.COUNT('user_tags')}).sort('ctr', ascending=False)

    # save user tags ctr and count to file
    user_tags.save('../output/user_tags.csv')


########################################################################################################################


# main function
def main(analysis):

    # load data
    train_set = load_data()

    if analysis == 'click':
        # get click count
        click_count(train_set)
    elif analysis == 'user_agent':
        # get user agent ctr and count
        user_agent_ctr(train_set)
    elif analysis == 'hour':
        # get hour ctr and count
        hour_ctr(train_set)
    elif analysis == 'weekday':
        # get weekday ctr and count
        weekday_ctr(train_set)
    elif analysis == 'timestamp':
        # get timestamp ctr and count
        timestamp_ctr(train_set)
    elif analysis == 'city':
        # get city ctr and count
        city_ctr(train_set)
    elif analysis == 'region':
        # get region ctr and count
        region_ctr(train_set)
    elif analysis == 'ad_exchange':
        # get ad exchange ctr and count
        ad_exchange_ctr(train_set)
    elif analysis == 'ad_slot_floor_price':
        # get ad slot floor price ctr and count
        ad_slot_floor_price_ctr(train_set)
    elif analysis == 'ad_slot_visibility':
        # get ad slot visibility ctr and count
        ad_slot_visibility_ctr(train_set)
    elif analysis == 'ad_slot_format':
        # get ad slot format ctr and count
        ad_slot_format_ctr(train_set)
    elif analysis == 'ad_slot_width':
        # get ad slot width ctr and count
        ad_slot_width_ctr(train_set)
    elif analysis == 'ad_slot_height':
        # get ad slot height ctr and count
        ad_slot_height_ctr(train_set)
    elif analysis == 'ad_slot_width_height':
        # get ad slot width height ctr and count
        ad_slot_width_height_ctr(train_set)
    elif analysis == 'advertiser_id':
        # get advertiser id ctr and count
        advertiser_id_ctr(train_set)
    elif analysis == 'anonymous_url_id':
        # get anonymous url id ctr and count
        anonymous_url_id_ctr(train_set)
    elif analysis == 'log_type':
        # get log type ctr and count
        log_type_ctr(train_set)
    elif analysis == 'user_id':
        # get user id ctr and count
        user_id_ctr(train_set)
    elif analysis == 'ip':
        # get ip ctr and count
        ip_ctr(train_set)
    elif analysis == 'domain':
        # get domain ctr and count
        domain_ctr(train_set)
    elif analysis == 'url':
        # get url ctr and count
        url_ctr(train_set)
    elif analysis == 'ad_slot_id':
        # get ad slot id ctr and count
        ad_slot_id_ctr(train_set)
    elif analysis == 'creative_id':
        # get creative id ctr and count
        creative_id_ctr(train_set)
    elif analysis == 'key_page_url':
        # get key page url ctr and count
        key_page_url_ctr(train_set)
    elif analysis == 'user_tags':
        # get user tags ctr and count
        user_tags_ctr(train_set)


########################################################################################################################


# runs main function
if __name__ == '__main__':

    # parse script argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--analysis', type=str, dest='analysis', metavar='analysis type',
                        help='analysis type', required=True)
    args = parser.parse_args()

    # call main function with the script argument as parameter
    main(args.analysis)


########################################################################################################################
