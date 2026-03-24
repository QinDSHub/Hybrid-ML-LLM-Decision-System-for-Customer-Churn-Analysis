#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os, dotenv, math
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import hstack
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 0-2 years: Like-new car
# 3-5 years: Young car
# 6-8 years: Prime car
# 9-15 years: Middle-aged car
# 15+ years: Old car

def car_age_bin(x):
    if x<=2:
        return '车龄：0-2年'
    elif x>2 and x<=5:
        return '车龄：3-5年'
    elif x>5 and x<=8:
        return '车龄：6-8年'    
    elif x>8 and x<=15:
        return '车龄：9-15年' 
    else:
        return '车龄：15年以上'

def car_level_bin(x):
    if x=='family_1':
        return '高档车'
    elif x == 'family_2':
        return '中档车'
    else:
        return '低档车'

def all_times_bin(x):
    if x<=1:
        return "历史进店总次数：1次"
    elif x>1 and x<=5:
        return "历史进店总次数：2-5次"
    elif x>5 and x<=10:
        return "历史进店总次数：6-10次"
    elif x>10 and x<=20:
        return "历史进店总次数：11-20次"
    elif x>20 and x<=50:
        return "历史进店总次数：21-50次"
    else:
        return "历史进店总次数：大于50次"

def cv_bin(x, name):
    if x==0:
        return f"{name}0"
    elif x<=0.2:
        return f"{name}0-0.2"
    elif x>0.2 and x<=0.5:
        return f"{name}0.2-0.5"
    elif x>0.5 and x<=1:
        return f"{name}0.5-1"
    elif x>1 and x<=1.5:
        return f"{name}1-1.5"
    else:
        return f"{name}1.5-2"

def day_diff_medium_bin(x, name):
    if math.isnan(x): # value值，无法使用x.isna()或x.isnull()
        return f'{name}仅进店一次，无法推测其平均进店一次的时间间隔'
    elif x>0 and x<=90:
        return f'{name}0-3个月'
    elif x>90 and x<=180:
        return f'{name}3-6个月'
    elif x>180 and x<=360:
        return f'{name}6-12个月'
    elif x>360 and x<=720:
        return f'{name}1-2年'
    else:
        return f'{name}2年以上'

def mile_diff_medium_bin(x, name):
    if math.isnan(x): 
        return f'{name}仅进店一次，无法推测其平均进店一次的行驶里程间隔'
    elif x>0 and x<=3000:
        return f'{name}0-3千公里'
    elif x>3000 and x<=6000:
        return f'{name}3-6千公里'
    elif x>6000 and x<=12000:
        return f'{name}6-12千公里'
    elif x>12000 and x<=24000:
        return f'{name}12-24千公里'
    else:
        return f'{name}24千公里以上'

def day_speed_medium_bin(x, name):
    if math.isnan(x):
        return f'{name}仅进店一次，无法推测其平均日均行驶速度'
    elif x>0 and x<=20:
        return f'{name}0-20公里/天'
    elif x>20 and x<=40:
        return f'{name}20-40公里/天'
    elif x>40 and x<=60:
        return f'{name}40-60公里/天'
    elif x>60 and x<=80:
        return f'{name}60-80公里/天'
    else:
        return f'{name}80公里/天以上'

def last_mile_bin(x, name):
    if x<=10000:
        return f'{name}0-10千公里'
    elif x>10000 and x<=20000:
        return f'{name}10-20千公里'
    elif x>20000 and x<=40000:
        return f'{name}20-40千公里'
    elif x>40000 and x<=80000:
        return f'{name}40-80千公里'
    elif x>80000 and x<=140000:
        return f'{name}80-140千公里'
    elif x>140000 and x<=200000:
        return f'{name}140-200千公里'
    else:
        return f'{name}200千公里以上'

def last_till_now_days_bin(x, name):
    if x<=180:
        return f'{name}6个月以内'
    elif x>180 and x<=360:
        return f'{name}6-12个月'
    elif x>360 and x<=720:
        return f'{name}12-24个月'
    else:
        return f'{name}24个月以上'
        
def second_to_first_day_diff_bin(x, name):
    if math.isnan(x):
        return f'{name}只进店一次'
    elif x<=180:
        return f'{name}6个月以内再次进店'
    elif x>180 and x<=360:
        return f'{name}6-12个月再次进店'
    elif x>360 and x<=720:
        return f'{name}12-24个月再次进店'
    else:
        return f'{name}24个月以上再次进店'

def first_to_purchase_day_diff_bin(x, name):
    if math.isnan(x):
        return f'{name}没有进店'
    elif x<=180:
        return f'{name}6个月以内再次进店'
    elif x>180 and x<=360:
        return f'{name}6-12个月再次进店'
    elif x>360 and x<=720:
        return f'{name}12-24个月再次进店'
    else:
        return f'{name}24个月以上再次进店'

def second_to_first_mile_diff_bin(x, name):
    if math.isnan(x):
        return f'{name}没有进店'
    elif x<=5000:
        return f'{name}0-5千公里'
    elif x>5000 and x<=10000:
        return f'{name}5-10千公里'
    elif x>10000 and x<=15000:
        return f'{name}10-15千公里'
    elif x>15000 and x<=20000:
        return f'{name}15-20千公里'
    else:
        return f'{name}20千公里以上'

def first_to_purchase_mile_diff_bin(x, name):
    if math.isnan(x):
        return f'{name}没有进店'
    elif x<=5000:
        return f'{name}0-5千公里'
    elif x>5000 and x<=10000:
        return f'{name}5-10千公里'
    elif x>10000 and x<=20000:
        return f'{name}10-20千公里'
    elif x>20000 and x<=40000:
        return f'{name}20-40千公里'
    else:
        return f'{name}40千公里以上'

def llm_data_preprocess(save_path):
    # choose important features for simplity, now the data sample is unique for user
    dff = pd.read_csv(os.path.join(save_path, 'cleaned_data.csv'))
    need_cols = ['VIN', 'last_mile', 'last_repair_type', 'last_till_now_days',
           'first_to_purchase_day_diff', 'first_to_purchase_mile_diff',
           'second_to_first_day_diff', 'second_to_first_mile_diff',
           'day_diff_median', 'mile_diff_median', 'day_speed_median', 'day_cv',
           'mile_cv', 'day_speed_cv', 'all_times', 'all_repair_types',
           'owner_type', 'car_mode', 'car_level', 'member_level', 'car_age',
           'churn_label', 'dataset']
    dff = dff[need_cols]
    print(df.shape)
    print(dff['VIN'].nunique())
    
    dff['car_age'] = dff['car_age'].apply(lambda x:car_age_bin(x))
    dff['car_level'] = dff['car_level'].apply(lambda x:car_level_bin(x))
    dff['all_times'] = dff['all_times'].apply(lambda x:all_times_bin(x))
    dff['day_cv'] = dff['day_cv'].apply(lambda x:cv_bin(x, '进店时间间隔波动系数：'))
    dff['mile_cv'] = dff['mile_cv'].apply(lambda x:cv_bin(x, '行驶里程波动系数：'))
    dff['day_speed_cv'] = dff['day_speed_cv'].apply(lambda x:cv_bin(x, '日均行驶速度波动系数：'))
    
    dff['day_diff_median'] = dff['day_diff_median'].apply(lambda x:day_diff_medium_bin(x, '平均进店间隔：')) 
    dff['mile_diff_median'] = dff['mile_diff_median'].apply(lambda x:mile_diff_medium_bin(x, '平均行驶里程间隔：'))
    dff['day_speed_median'] = dff['day_speed_median'].apply(lambda x:day_speed_medium_bin(x, '平均日均行驶速度间隔：'))
    dff['last_mile'] = dff['last_mile'].apply(lambda x:last_mile_bin(x, '最近一次进店里程数：'))
    
    dff['last_till_now_days'] = dff['last_till_now_days'].apply(lambda x:last_till_now_days_bin(x, '最后一次进店距离今天：'))
    dff['second_to_first_day_diff'] = dff['second_to_first_day_diff'].apply(lambda x:second_to_first_day_diff_bin(x, '第二次进店距离第一次进店的时间间隔：'))
    dff['first_to_purchase_day_diff'] = dff['first_to_purchase_day_diff'].apply(lambda x:first_to_purchase_day_diff_bin(x, '购车后到第一次进店的时间间隔：'))
    dff['second_to_first_mile_diff'] = dff['second_to_first_mile_diff'].apply(lambda x:second_to_first_mile_diff_bin(x, '第二次进店距离第一次进店的里程间隔：'))
    dff['first_to_purchase_mile_diff'] = dff['first_to_purchase_mile_diff'].apply(lambda x:first_to_purchase_mile_diff_bin(x, '购车后到第一次进店的里程间隔：'))
    return dff

def get_most_feat_lgb(data):

    num_classes = data['churn_label'].nunique()
    
    del_columns = ['VIN' ,'churn_label','dataset',
                   'last_repair_type',
                   'all_repair_types', ]
    
    all_hot_columns = [x for x in data.columns if x not in del_columns]
    y = 'churn_label'
    
    train_mask = data['dataset'] == 'train'
    valid_mask = data['dataset'] == 'valid'
    
    train_features = [] 
    valid_features = []
    all_feature_names = []

    for col in all_hot_columns:
    
        le = LabelEncoder()
    
        all_values = pd.concat([data.loc[train_mask, col], data.loc[valid_mask, col]])
        le.fit(all_values)
        
        train_encoded = le.transform(data.loc[train_mask, col])
        valid_encoded = le.transform(data.loc[valid_mask, col])
        train_encoded = train_encoded.reshape(-1, 1)
        valid_encoded = valid_encoded.reshape(-1, 1)
        
        ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        ohe.fit(train_encoded)  
        original_feature_names = ohe.get_feature_names_out(['x0'])
        custom_feature_names = [name.replace('x0', col) for name in original_feature_names]
    
        train_ohe = ohe.transform(train_encoded)
        valid_ohe = ohe.transform(valid_encoded)
        
        train_features.append(train_ohe)
        valid_features.append(valid_ohe)
        all_feature_names.extend(custom_feature_names)

    if train_features:
        train_data_x = hstack(train_features)  # 使用hstack合并稀疏矩阵
        valid_data_x = hstack(valid_features)
    
    train_data_y = data.loc[train_mask, y].values
    valid_data_y = data.loc[valid_mask, y].values
    
    lgb_train = lgb.Dataset(train_data_x, train_data_y)
    lgb_valid = lgb.Dataset(valid_data_x, valid_data_y, reference=lgb_train)

    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'learning_rate': 0.01,
        'random_state': 2026,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'max_bin': 63,
        'verbose': -1
    }
    
    model = lgb.train(params, lgb_train, valid_sets=[lgb_valid])

    # get top-k most important features according to lgb
    importance = model.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    topk=30
    
    rank_feat = []
    for col in feature_importance_df['feature'].values[:topk]:
        del_col = col.split('_')[-1]
        new_col = col.replace(del_col, '')
        if new_col not in rank_feat:
            rank_feat.append(new_col)

    new_rank_feat = []
    for col in rank_feat:
        new_col = col.strip('_')
        new_rank_feat.append(new_col)
    # print(new_rank_feat)
    return new_rank_feat

def rag_train_data(knowledges):
    docs = [Document(page_content=s) for s in knowledges]
    print(f"sample number: {len(docs)}")
    
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings_model,
        collection_name="openai_embed",
        persist_directory="./chroma_db"
    )
    print('Have saved all embedding into chroma_db!')

def llm_reason(dff, new_rank_feat):
    dff['idx'] = dff.index+1
    dff['idx'] = dff['idx'].apply(int)
    dff['idx'] = dff['idx'].apply(lambda x: '用户：'+str(x))
    
    dff['member_level'] = dff['member_level'].apply(lambda x:'会员卡：'+str(x))
    
    label_map = {1:'标签：流失', 0:'标签：未流失'}
    dff['label_txt'] = dff['churn_label'].map(label_map)
    
    # transform data into type which llm could regonize
    train_dt = dff[dff['dataset']=='train']
    valid_dt = dff[dff['dataset']=='valid']
    
    train_dt = train_dt[['idx'] + new_rank_feat + ['label_txt']]
    valid_dt = valid_dt[['idx'] + new_rank_feat + ['label_txt']]
    
    alls = train_dt.values
    knowledges = []
    
    # mark, need to add churn label into data which helps llm to reason
    for i in range(len(alls)):
        ss = alls[i]
        k = ss[0] + '，' + ss[1] + '，' + ss[2] + '，' + ss[3] + '，' + ss[4] + '，' + ss[5] + '，' + ss[6] + '，' + ss[7] + '，' + ss[8] + '，' + ss[9] + '，' + ss[10] + '，' + ss[11]
        knowledges.append(k)
    print(len(alls))
    print(len(knowledges))
    
    rag_train_data(knowledges)
    
    valid_alls = valid_dt.values
    valid_knowledges = []
    valid_y = []
    for i in range(len(valid_alls)):
        ss = valid_alls[i]
        k = ss[0] + '，' + ss[1] + '，' + ss[2] + '，' + ss[3] + '，' + ss[4] + '，' + ss[5] + '，' + ss[6] + '，' + ss[7] + '，' + ss[8] + '，' + ss[9] + '，' + ss[10]
        valid_knowledges.append(k)
        valid_y.append(ss[0] + '，' + ss[11])
    print(len(valid_alls))
    print(len(valid_knowledges))
    print(len(valid_y))
    
    
    valid_df = []
    
    for query in valid_knowledges:
        tt = [query, valid_y[valid_knowledges.index(query)]]
        result = vectorstore.similarity_search(query, k=10)
    
        gpt = ChatOpenAI(
            model="gpt-5-nano",
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY
        )
    
        prompt = [
            SystemMessage(
                content=(
                    "你是一个汽车售后服务与客户流失分析专家。"
                    "请根据用户行为与历史相似用户数据，判断该用户标签。"
                )
            ),
            HumanMessage(
                content=f"""
        用户进店行为数据：
        {query}
    
        历史相似用户数据：
        {"\n\n".join(doc.page_content for doc in result)}
    
        判断该用户标签【流失 / 未流失】，不需要给出解释和建议。
        """
            )
        ]

        response = gpt.invoke(prompt)
        tt.append(response.content)
        valid_df.append(tt)
        return valid_df

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'llm reason')
    parser.add_argument('--save_path', type=str, default='./', help='save path for cleaned data or outputs')
    args = parser.parse_args()
    
    dff = llm_data_preprocess(args.save_path)
    new_rank_feat = get_most_feat_lgb(dff)
    valid_df = llm_reason(dff, new_rank_feat)
    
    valid_df_df = pd.DataFrame(valid_df, columns=['query', 'true_label', 'predicted_label'])
    valid_df_df['real_label'] = valid_df_df['true_label'].apply(lambda x: x.split('：')[-1])
    # valid_df_df['correct'] = valid_df_df.apply(lambda row: 1 if row['real_label'] == row['predicted_label'] else 0, axis=1)
    valid_df_df.to_csv(os.path.join(args.save_path, 'llm_predictions.csv'), index=False, encoding='utf-8-sig')


# In[ ]:




