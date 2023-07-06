# General imports
import numpy as np
import pandas
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

from math import ceil

from sklearn.preprocessing import LabelEncoder

from FE.feature_engineering import sizeof_fmt, merge_by_concat, reduce_mem_usage

train_df = pd.read_csv('../data/sales_train_validation.csv')
prices_df = pd.read_csv('../data/sell_prices.csv')
calendar_df = pd.read_csv('../data/calendar.csv')
TARGET = 'sales'  # Our main target
END_TRAIN = 1913  # Last day in train set
MAIN_INDEX = ['id', 'd']  # We can identify item by these columns


def data_precess_1():
    global train_df
    ########################### Vars
    #################################################################################
    ########################### Load Data
    #################################################################################
    print('Load Main Data')
    # Here are reafing all our data
    # without any limitations and dtype modification

    ########################### Make Grid
    #################################################################################
    print('Create Grid')
    # We can tranform horizontal representation
    # to vertical "view"
    # Our "index" will be 'id','item_id','dept_id','cat_id','store_id','state_id'
    # and labels are 'd_' coulmns

    # 在实际处理数据的过程中，会出现按照宽表格式表征的时间序列数据，宽表格式适合于人类观察但不适合进行数据处理。因此需要利用pd.melt函数将宽表变为长表。其中，id为不会旋转的列，var_name表示变量列名，value_name为值得列名
    index_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    grid_df = pd.melt(train_df,
                      id_vars=index_columns,
                      var_name='d',
                      value_name=TARGET)

    # If we look on train_df we se that
    # we don't have a lot of traning rows
    # but each day can provide more train data
    print('Train rows:', len(train_df), len(grid_df))

    # To be able to make predictions
    # we need to add "test set" to our grid
    # 这里每一天都创建一组数据，每组数据的天数标签不同，用来保存未来28天的结果，作为测试集。
    add_grid = pd.DataFrame()
    for i in range(1, 29):
        temp_df = train_df[index_columns]
        temp_df = temp_df.drop_duplicates()
        temp_df['d'] = 'd_' + str(END_TRAIN + i)
        temp_df[TARGET] = np.nan
        add_grid = pd.concat([add_grid, temp_df])

    grid_df = pd.concat([grid_df, add_grid])
    grid_df = grid_df.reset_index(drop=True)

    # Remove some temoprary DFs
    del temp_df, add_grid

    # We will not need original train_df
    # anymore and can remove it
    del train_df

    # You don't have to use df = df construction
    # you can use inplace=True instead.
    # like this
    # grid_df.reset_index(drop=True, inplace=True)

    # Let's check our memory usage
    print("{:>20}: {:>8}".format('Original grid_df', sizeof_fmt(grid_df.memory_usage(index=True).sum())))

    # We can free some memory
    # by converting "strings" to categorical
    # it will not affect merging and
    # we will not lose any valuable data
    for col in index_columns:
        grid_df[col] = grid_df[col].astype('category')

    # Let's check again memory usage
    print("{:>20}: {:>8}".format('Reduced grid_df', sizeof_fmt(grid_df.memory_usage(index=True).sum())))

    ########################### Product Release date
    #################################################################################
    print('Release week')

    # It seems that leadings zero values
    # in each train_df item row
    # are not real 0 sales but mean
    # absence for the item in the store
    # we can safe some memory by removing
    # such zeros

    # Prices are set by week
    # so it we will have not very accurate release week
    # 这一步相当于是在
    release_df = prices_df.groupby(['store_id', 'item_id'])['wm_yr_wk'].agg(['min']).reset_index()
    release_df.columns = ['store_id', 'item_id', 'release']

    # Now we can merge release_df
    # 按照物品属性作为合并点，合并两个表。就是给每个物品填上这一物品第一次出现的时间属性
    grid_df = merge_by_concat(grid_df, release_df, ['store_id', 'item_id'])
    del release_df

    # We want to remove some "zeros" rows
    # from grid_df
    # to do it we need wm_yr_wk column
    # let's merge partly calendar_df to have it
    # 按照d列对两个表进行合并，就是给出每个天所对应的星期
    grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk', 'd']], ['d'])

    # Now we can cutoff some rows
    # and safe memory
    # 删除无用的行
    grid_df = grid_df[grid_df['wm_yr_wk'] >= grid_df['release']]
    grid_df = grid_df.reset_index(drop=True)

    # Let's check our memory usage
    print("{:>20}: {:>8}".format('Original grid_df', sizeof_fmt(grid_df.memory_usage(index=True).sum())))

    # Should we keep release week
    # as one of the features?
    # Only good CV can give the answer.
    # Let's minify the release values.
    # Min transformation will not help here
    # as int16 -> Integer (-32768 to 32767)
    # and our grid_df['release'].max() serves for int16
    # but we have have an idea how to transform
    # other columns in case we will need it
    # 重新初始化时间
    grid_df['release'] = grid_df['release'] - grid_df['release'].min()
    grid_df['release'] = grid_df['release'].astype(np.int16)

    # Let's check again memory usage
    print("{:>20}: {:>8}".format('Reduced grid_df', sizeof_fmt(grid_df.memory_usage(index=True).sum())))

    ########################### Save part 1
    #################################################################################
    print('Save Part 1')
    grid_df.to_pickle('../data/processed/grid_part_1.pkl')




def data_precess_2():
    print('Prices')
    # 对时间序列数据的静态特征进行提取，非常适合用于时间序列数据的处理
    # We can do some basic aggregations

    global prices_df
    prices_df['price_max'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
    prices_df['price_min'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('min')
    prices_df['price_std'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('std')
    prices_df['price_mean'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')

    # and do price normalization (min/max scaling)
    prices_df['price_norm'] = prices_df['sell_price'] / prices_df['price_max']

    # Some items are can be inflation dependent
    # and some items are very "stable"
    #
    prices_df['price_nunique'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('nunique')
    prices_df['item_nunique'] = prices_df.groupby(['store_id', 'sell_price'])['item_id'].transform('nunique')

    # I would like some "rolling" aggregations
    # but would like months and years as "window"
    calendar_prices = calendar_df[['wm_yr_wk', 'month', 'year']]
    calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
    prices_df = prices_df.merge(calendar_prices[['wm_yr_wk', 'month', 'year']], on=['wm_yr_wk'], how='left')
    del calendar_prices

    # Now we can add price "momentum" (some sort of)
    # Shifted by week
    # by month mean
    # by year mean
    # 通过shift函数对数据做了多种偏移，并利用偏移计算出了每种商品价格得变化趋势
    prices_df['price_momentum'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id'])[
        'sell_price'].transform(lambda x: x.shift(1))
    prices_df['price_momentum_m'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'month'])[
        'sell_price'].transform('mean')
    prices_df['price_momentum_y'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'year'])[
        'sell_price'].transform('mean')

    del prices_df['month'], prices_df['year']

    ########################### Merge prices and save part 2
    #################################################################################
    print('Merge prices and save part 2')

    # Merge Prices
    grid_df = pd.read_pickle('../data/processed/grid_part_1.pkl')
    original_columns = list(grid_df)
    grid_df = grid_df.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    keep_columns = [col for col in list(grid_df) if col not in original_columns]
    grid_df = grid_df[MAIN_INDEX + keep_columns]
    grid_df = reduce_mem_usage(grid_df)

    # Safe part 2
    grid_df.to_pickle('../data/processed/grid_part_2.pkl')
    print('Size:', grid_df.shape)

    # We don't need prices_df anymore
    del prices_df


def data_precess_3():
    grid_df = pd.read_pickle('../data/processed/grid_part_1.pkl')
    ########################### Merge calendar
    #################################################################################
    grid_df = grid_df[MAIN_INDEX]
    global calendar_df
    # Merge calendar partly
    icols = ['date',
             'd',
             'event_name_1',
             'event_type_1',
             'event_name_2',
             'event_type_2',
             'snap_CA',
             'snap_TX',
             'snap_WI']

    grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')

    # Minify data
    # 'snap_' columns we can convert to bool or int8
    icols = ['event_name_1',
             'event_type_1',
             'event_name_2',
             'event_type_2',
             'snap_CA',
             'snap_TX',
             'snap_WI']
    for col in icols:
        grid_df[col] = grid_df[col].astype('category')

    # Convert to DateTime
    grid_df['date'] = pd.to_datetime(grid_df['date'])

    # Make some features from date
    grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
    grid_df['tm_w'] = grid_df['date'].dt.week.astype(np.int8)
    grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
    grid_df['tm_y'] = grid_df['date'].dt.year
    grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
    grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x / 7)).astype(np.int8)

    grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)
    grid_df['tm_w_end'] = (grid_df['tm_dw'] >= 5).astype(np.int8)

    # Remove date
    del grid_df['date']
    ########################### Save part 3 (Dates)
    #################################################################################
    print('Save part 3')

    # Safe part 3
    grid_df.to_pickle('../data/processed/grid_part_3.pkl')
    print('Size:', grid_df.shape)

    # We don't need calendar_df anymore
    del calendar_df
    del grid_df


def data_precess_4():

    ## Part 1
    # Convert 'd' to int
    # grid_df = pd.read_pickle('../data/processed/grid_part_1.pkl')
    # grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)
    #
    # # Remove 'wm_yr_wk'
    # # as test values are not in train set
    # del grid_df['wm_yr_wk']
    # grid_df.to_pickle('../data/processed/grid_part_1.pkl')

    # print('Size:', grid_df.shape)


    # Now we have 3 sets of features
    grid_df = pd.concat([pd.read_pickle('../data/processed/grid_part_1.pkl'),
                         pd.read_pickle('../data/processed/grid_part_2.pkl').iloc[:, 2:],
                         pd.read_pickle('../data/processed/grid_part_3.pkl').iloc[:, 2:]],
                        axis=1)

    # Let's check again memory usage
    print("{:>20}: {:>8}".format('Full Grid', sizeof_fmt(grid_df.memory_usage(index=True).sum())))
    print('Size:', grid_df.shape)

    # 2.5GiB + is is still too big to train our model
    # (on kaggle with its memory limits)
    # and we don't have lag features yet
    # But what if we can train by state_id or shop_id?
    state_id = 'CA'
    grid_df = grid_df[grid_df['state_id'] == state_id]
    print("{:>20}: {:>8}".format('Full Grid', sizeof_fmt(grid_df.memory_usage(index=True).sum())))
    #           Full Grid:   1.2GiB

    store_id = 'CA_1'
    grid_df = grid_df[grid_df['store_id'] == store_id]
    print("{:>20}: {:>8}".format('Full Grid', sizeof_fmt(grid_df.memory_usage(index=True).sum())))
    #           Full Grid: 321.2MiB

    # Seems its good enough now
    # In other kernel we will talk about LAGS features
    # Thank you.
    grid_df.to_pickle('../data/processed/grid_total.pkl')


if __name__ == '__main__':
    data_precess_4()