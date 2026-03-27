#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os,argparse,math
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit

def stat_data(path):
    # get stat feature
    veh_df = pd.read_csv(os.path.join(path,'vehicle3.csv'))
    cols = ['VIN','车主性质','车型','family_name']
    veh_df = veh_df[cols]
    veh_df.columns = ['VIN','owner_type','car_mode','car_level']

    # get membership feature
    member_info = pd.read_csv(os.path.join(path, 'member_info.csv'))
    member_info = member_info[['VIN','会员等级']].drop_duplicates()
    member_info = member_info[~member_info['会员等级'].isna()].reset_index(drop=True)
    member_info.columns = ['VIN','member_level']

    df = veh_df.merge(member_info, on=['VIN'], how='outer').drop_duplicates().reset_index(drop=True)

    # fillna directly for some col based on knowing on the business logic
    df['member_level'] = df['member_level'].fillna('无')
    df['owner_type'] = df['owner_type'].fillna('个人')

    # filter data after filling
    all_customer_df = df.dropna()

    # get purchase date feature which is important to describe the car status, if customer has purchase date, then here delete directly here, 
    # in reality, it could be fill up by statistics
    repair_df = pd.read_csv(os.path.join(path, 'repare_maintain_info1.csv'))
    buy_df = repair_df[['VIN','purchase_date']].drop_duplicates().reset_index(drop=True)
    buy_df = buy_df.dropna()
    all_customer_df = all_customer_df.merge(buy_df, on=['VIN'],how='inner')
    
    return all_customer_df

def clean_main_template(path, all_customer_df):
    repair_df = pd.read_csv(os.path.join(path, 'repare_maintain_info1.csv'))
    # here, we just choose some important feature for simple test, actually other feature such as maintanience payment we put in the model before.
    cols = ['VIN','修理日期','公里数','修理类型']
    repair_df = repair_df[cols]
    repair_df = repair_df[repair_df['VIN'].isin(all_customer_df['VIN'].values)]
    new_cols = ['VIN','date','mile','repair_type']
    repair_df.columns = new_cols
    repair_df = repair_df[new_cols].sort_values(['VIN','date'], ascending=True)
    repair_df = repair_df.dropna()
    # print(repair_df.shape)
    # print(repair_df.head())
    
    repair_df['date'] = pd.to_datetime(repair_df['date']).dt.date

    # clean mile data further for every maintance, such as two samples for one day maintance, but the mile record is different
    repair_df_1 = repair_df.groupby(['VIN','date']).agg(mile = ('mile','mean')).reset_index()

    # clean repair type, which will be helpful for filtering data, also could be as one of features to describe different purpose for in-store maintance
    repair_df_2 = repair_df[['VIN','date','repair_type']].drop_duplicates().groupby(['VIN','date'])['repair_type'].agg(lambda x:','.join(x)).reset_index()

    # now, the sample df is unique about (user_id + maintance date)
    new_repair_df = repair_df_1.merge(repair_df_2,on=['VIN','date'],how='outer') 
    print(new_repair_df.shape)
    print(new_repair_df[['VIN','date']].drop_duplicates().shape)
    
    new_repair_df['date'] = pd.to_datetime(new_repair_df['date'])

    # start to split data by date, we keep data as train_x before three months and keep data as get train_y for other three months
    data_split_date = new_repair_df['date'].max()+pd.DateOffset(months=-3)
    
    train_x = new_repair_df[new_repair_df['date']<=data_split_date] # use to get features
    train_y = new_repair_df[new_repair_df['date']>data_split_date] # use to get train_y label

    return train_x, train_y, data_split_date

def get_features(train_x, train_y):
    main_df = train_x.sort_values(['VIN','date'],ascending=False).reset_index(drop=True)

    # get current maintain data
    feat_1 = main_df.sort_values(['VIN','date'],ascending=False).groupby('VIN').first().reset_index()
    feat_1['last_till_now_days'] = feat_1['date'].apply(lambda x:(pd.to_datetime(today)-pd.to_datetime(x)).days)
    feat_1 = feat_1.rename(columns={'date':'last_date','mile':'last_mile','repair_type':'last_repair_type'})

    # get statistic feature of maintain frequency on day_diff, mile_diff and day speed
    main_df['relative_last_date'] = main_df.groupby('VIN')['date'].shift(-1)
    main_df['relative_last_mile'] = main_df.groupby('VIN')['mile'].shift(-1)
    main_df = main_df.merge(buy_df[['VIN','purchase_date']].drop_duplicates(), on = 'VIN', how='left')
    main_df['purchase_date'] = pd.to_datetime(main_df['purchase_date'])
    main_df.loc[main_df['relative_last_date'].isna(),'relative_last_date'] = main_df['purchase_date']
    main_df.loc[main_df['relative_last_mile'].isna(),'relative_last_mile'] = 0
    del main_df['purchase_date']
    
    main_df['day_diff'] = main_df[['date','relative_last_date']].apply(lambda row:(pd.to_datetime(row[0])-pd.to_datetime(row[1])).days, axis=1, raw=True)
    main_df['mile_diff'] = main_df['mile'] - main_df['relative_last_mile']
    main_df['day_speed'] = main_df['mile_diff']/main_df['day_diff']

    # when day_diff<0, it is usually cased by the wrong record of purhase date and some are date for the record of second time buying
    del_vin = main_df[main_df['day_diff']<0]

    # here I filter data directly for simiplity, in reality, the purchase date could be filled up
    main_df = main_df[~main_df['VIN'].isin(del_vin['VIN'].values)]
    main_df['rk'] = main_df.sort_values(by=['VIN','date'],ascending=False).groupby('VIN')['date'].rank(method='first')

    # get the feature for user's first and second maintain behaviour features
    feat_2 = main_df[main_df['rk']==1][['VIN','day_diff','mile_diff']].drop_duplicates().rename(columns={'day_diff':'first_to_purchase_day_diff',
                                                                                                         'mile_diff':'first_to_purchase_mile_diff'})
    feat_3 = main_df[main_df['rk']==2][['VIN','day_diff','mile_diff']].drop_duplicates().rename(columns={'day_diff':'second_to_first_day_diff',
                                                                                                         'mile_diff':'second_to_first_mile_diff'})
    # get user's historical maintaince behaviour features
    feat_4 = main_df.groupby('VIN').agg(day_diff_median = ('day_diff','median'), 
                                        day_diff_std = ('day_diff','std'), day_diff_mean = ('day_diff','mean'),
                                        mile_diff_median = ('mile_diff','median'), 
                                        mile_diff_std = ('mile_diff','std'), mile_diff_mean = ('mile_diff','mean'),
                                        day_speed_median = ('day_speed','median'), 
                                        day_speed_std = ('day_speed','std'), day_speed_mean = ('day_speed','mean')
                                       ).reset_index()
    feat_4.loc[feat_4['day_diff_std'].isna(), 'day_cv'] = 0  # for those who only in-store maintaince once
    feat_4.loc[~feat_4['day_diff_std'].isna(), 'day_cv'] = feat_4['day_diff_std']/feat_4['day_diff_mean']
    feat_4.loc[feat_4['mile_diff_std'].isna(), 'mile_cv'] = 0
    feat_4.loc[~feat_4['mile_diff_std'].isna(), 'mile_cv'] = feat_4['mile_diff_std']/feat_4['mile_diff_mean']
    feat_4.loc[feat_4['day_speed_std'].isna(), 'day_speed_cv'] = 0
    feat_4.loc[~feat_4['day_speed_std'].isna(), 'day_speed_cv'] = feat_4['day_speed_std']/feat_4['day_speed_mean']
    
    # smooth for outlier
    feat_4.loc[feat_4['day_cv']>2, 'day_cv'] = 2
    feat_4.loc[feat_4['mile_cv']>2, 'mile_cv'] = 2
    
    # get user's historical instore maintain times
    feat_5 = main_df.groupby('VIN').agg(all_times = ('date','count')).reset_index()

    # get user's historical instore maintain types
    feat_6 = main_df.groupby('VIN')['repair_type'].agg(','.join).rename('all_repair_types').reset_index()

    feat_df = feat_1.merge(feat_2,on='VIN',how='left').merge(feat_3,on='VIN',how='left').merge(feat_4,on='VIN',how='left').merge(feat_5,on='VIN',how='left').merge(feat_6,on='VIN',how='left')
    # print(feat_df.shape)
    # print(feat_df['VIN'].nunique())
    
    return feat_df

def add_label(feat_df, all_customer_df, train_y, data_split_date):
    feat_df = feat_df.merge(all_customer_df, on=['VIN'], how='inner')
    feat_df['car_age'] = feat_df['purchase_date'].apply(lambda x:(pd.to_datetime(today)-pd.to_datetime(x)).days)
    feat_df['car_age'] = feat_df['car_age'].apply(lambda x:math.ceil(x/365))
    
    # use user's historical info to estimate the next in-store date
    feat_df['relative_next_instore_date'] = (
        pd.to_datetime(feat_df['last_date']) + 
        pd.to_timedelta(feat_df['day_diff_median'], unit='D')
    )
    
    # add more three months for the estimation
    feat_df['max_relative_next_instore_date'] = pd.to_datetime(feat_df['relative_next_instore_date']) + pd.DateOffset(months=3)
    
    train_y_df = feat_df[['VIN','last_date','day_diff_median','relative_next_instore_date','max_relative_next_instore_date']].drop_duplicates().reset_index(drop=True)
    
    # if user do not come to maintain in max estimation, then mark churn_label = 1 means we have lost the user
    train_y_df['churn_label'] = 0
    train_y_df.loc[train_y_df['max_relative_next_instore_date'] <= data_split_date, 'churn_label'] = 1
    
    feat_df = feat_df.merge(train_y_df[['VIN','churn_label']].drop_duplicates(),on=['VIN'],how='left')
    return feat_df

def train_valid_split(feat_df):
    # here we use stratifiedshufflesplit in order to split data according to its balance based on the binary classification
    sss =  StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    
    for train_index, val_index in sss.split(feat_df, feat_df['churn_label']):
        train_df = feat_df.iloc[train_index].reset_index(drop=True)
        train_df['dataset'] = 'train'
        val_df = feat_df.iloc[val_index].reset_index(drop=True)
        val_df['dataset'] = 'valid'
    
    all_feat_df = pd.concat([train_df, val_df],axis=0)
    all_feat_df.to_csv(os.path.join(save_path,'cleaned_data.csv'),index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing pipeline')
    parser.add_argument('--data_path', type=str, default='../clean_data',
                        help='original data saved path')
    parser.add_argument('--save_path', type=str, default='./',
                        help='cleaned data save path')
    args = parser.parse_args()
    
    all_customer_df = stat_data(args.data_path)
    train_x, train_y, data_split_date = clean_main_template(args.data_path, all_customer_df)
    feat_df = get_main_template(train_x)
    data = add_label(feat_df, all_customer_df, train_y, data_split_date)
    train_valid_split(data, args.save_path) # save file name is fixed as cleaned_data.csv





