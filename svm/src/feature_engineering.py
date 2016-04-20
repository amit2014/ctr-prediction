from graphlab import feature_engineering as fe


def site_feat(data):
    label = 'site_map'
    if label in data.column_names():
        return data

    data[label] = data['domain'] + data['user_agent']
    return data, label


def slot_feat(data):
    label = 'slot_features'
    if label in data.column_names():
        return data

    data[label] = data['ad_slot_width'] + data['ad_slot_height']
    return data, label


def location_feat(data):
    label = 'location_feat'
    if label in data.column_names():
        return data

    data[label] = data['region'] + data['city']
    return data, label


def create_quad_features(train, interaction_columns, label='quadratic_features'):
    return fe.create(train, fe.QuadraticFeatures(features=interaction_columns, output_column_name=label)), label


def apply_quadratic(data, quad, label):

    if label in data.column_names():
        # operation already performed, do nothing
        return data
    # for feature in interaction_columns:
    #     dataset[feature] = data[feature].astype(int)
    return quad.transform(data)


def create_onehot_features(train, interaction_columns, categories=300, label='encoded_features'):
    return fe.create(train, fe.OneHotEncoder(features=interaction_columns,
                                             max_categories=categories, output_column_name=label)), label


def apply_feature_eng(data, impl, label):
    if label in data.column_names():
        return data
    return impl.transform(data)


def apply_dictionary(data, feature, label):
    tags_to_dict = lambda tags: dict(zip(tags, [1 for tag in tags]))
    data[label] = data.apply(lambda row: tags_to_dict(row[feature].split(',')))
    print data[label].head(5)
    return data


def tags_to_dict(row):
    tags = row.split(',')
    return dict(zip(tags, [1 for _ in tags]))


def apply_separate_uagent(data, regexp, label):
    data[label] = data.apply(lambda row: filter_uagent(row['user_agent'], regexp))


def filter_uagent(value, regexp):
    return 1 if regexp.search(value) is not None else 0
