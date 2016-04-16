
# Following: http://blog.dato.com/beginners-guide-to-click-through-rate-prediction-with-logistic-regression

########################################################################################################################

# import package
import graphlab as gl

########################################################################################################################

# read data
data = gl.SFrame.read_csv('train_data.txt', delimiter='\t', verbose=False)

# print(data.head(1))

# print 'Overall CTR: {}'.format(data['click'].mean())

'''
print(data.groupby('hour',
                   {'CTR': gl.aggregate.MEAN('click')}
                   ).sort('CTR', ascending=False).print_rows(197, 2))
'''

'''
print(data.groupby('click',
                   {'FREQ': gl.aggregate.FREQ_COUNT('click')}
                   ).print_rows(2, 2))
'''

print(data.groupby('key_page_url',
                   {'CTR': gl.aggregate.MEAN('click')},
                   {'FREQ': gl.aggregate.FREQ_COUNT('key_page_url')}
                   ).sort('CTR', ascending=False).print_rows(2, 3))

'''
print(data.groupby('ad_slot_width',
                       {'ad_slot_height': gl.aggregate.SELECT_ONE('ad_slot_height')},
                   {'CTR': gl.aggregate.MEAN('click')},
                   {'WIDTH FREQ': gl.aggregate.FREQ_COUNT('ad_slot_width')},
                   {'HEIGHT FREQ': gl.aggregate.FREQ_COUNT('ad_slot_height')}
                   ).sort('CTR', ascending=False).print_rows(11, 5))
'''
