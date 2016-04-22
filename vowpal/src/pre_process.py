
########################################################################################################################


# loads data
def load_data():

    # list to hold strings
    strings = []
    # open file
    with open('../../data/train_data.txt', mode='r') as roc_curve_file:
        next(roc_curve_file)
        # for every line in file
        for line in roc_curve_file:
            # split line and assign results to tokens
            tokens = line.strip('\n').split('\t')

            click = tokens[0]
            weekday = tokens[1]
            hour = tokens[2]
            user_agent = tokens[6]
            region = tokens[8]
            city = tokens[9]
            domain = tokens[11]
            url = tokens[12]
            ad_slot_id = tokens[14]
            ad_slot_width = tokens[15]
            ad_slot_height = tokens[16]
            ad_slot_visibility = tokens[17]
            ad_slot_format = tokens[18]
            ad_slot_floor_price = tokens[19]

            string = '{} |CategoricalFeatures' \
                     ' weekday:{}'              \
                     ' hour:{}'                 \
                     ' region:{}'               \
                     ' city:{}'                 \
                     ' ad_slot_width:{}'        \
                     ' ad_slot_height:{}'       \
                     ' ad_slot_visibility:{}'   \
                     ' ad_slot_format:{}'       \
                     ' ad_slot_floor_price:{}'.format(click,
                                                      weekday,
                                                      hour,
                                                      region,
                                                      city,
                                                      ad_slot_width,
                                                      ad_slot_height,
                                                      ad_slot_visibility,
                                                      ad_slot_format,
                                                      ad_slot_floor_price)

            strings += [string]
    # close roc curve file
    roc_curve_file.close()
    # return strings
    return strings


########################################################################################################################


# main function
def main():

    # load data
    strings = load_data()

    # open train data file
    with open('../output/train_data.txt', mode='w') as train_data_file:
        # for every string
        for string in strings:
            # write strings to file
            train_data_file.write('{}\n'.format(string))

    # close train data file
    train_data_file.close()


########################################################################################################################


# runs main function
if __name__ == '__main__':
    main()


########################################################################################################################
