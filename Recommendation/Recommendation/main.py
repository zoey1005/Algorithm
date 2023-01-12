from zoneinfo._common import load_data

import pandas as pd
users=["User1", "User2", "User3","User4", "User5"]
items=["Item A", "Item B", "Item C", "Item D", "Item E"]

data2=[
    [5,3,4,4,None],
    [3,1,2,3,3],
    [4,3,4,3,5],
    [3,3,1,5,4],
    [1,5,5,2,1]
]
df2 = pd.DataFrame(data2, columns=items, index=users)
user_similar = round(df2.T.corr(),4)

def predict(uid, iid, rating_matrix, user_similar):

    """
    :param uid: 用户id
    :param iid: 物品id
    :param rating_matrix: 用户—物品评分矩阵
    :param user_similar:  用户-用户相似矩阵
    :return: 预测的分数
    """

    print('开始预测用户<%d>对电影<%d>的评分...'%(uid, iid))
    # 1.找出uid用户的相似用户
    # 去掉用户自身
    similar_users = user_similar[uid].drop([uid]).dropna()
    # 相似用户筛选规则：正相关用户
    similar_users = similar_users.where(similar_users > 0).dropna()
    if similar_users.empty is True:
        raise Exception('用户<%d>没有相似的用户' % uid)

    # 2. 从uid用户的近邻用户中筛选出对iid物品有评分记录的近邻用户
    ids = set(rating_matrix[iid].dropna().index) & set(similar_users.index)
    final_similar_users = similar_users.loc[list(ids)]

    # 3. 结合uid用户与其临近相似用户的相似度预测uid用户对iid物品的评分
    sum_numerator = 0
    sum_denominator = 0
    for sum_uid, similarity in final_similar_users.iteritems():
        # 邻近用户的评分数据
        sum_user_rated_movies = rating_matrix.loc[sum_uid].dropna()
        # 邻近用户对iid物品的评分
        sim_user_rating_for_item = sum_user_rated_movies[iid]
        # 计算分子值
        sum_numerator += sim_user_rating_for_item
        # 计算分母值
        sum_denominator += similarity

    # 计算预测值并返回
    predict_rating = sum_numerator/sum_denominator
    print('预测用户<%d>对电影<%d>的评分：%0.2f' % (uid, iid, predict_rating))
    return round(predict_rating, 2)



if __name__ == '__main__':
    rating_matrix = load_data(DATA_PATH)
    user_similar = compute_pearson_similarity(rating_matrix, based = 'user')
    predict(1, 1, rating_matrix, user_similar)
    predict(1, 2, rating_matrix, user_similar)


def predict_all(uid, ratings_matrix, user_similar):
    """
    :param uid: 用户id
    :param ratings_matrix: 用户-物品打分矩阵
    :param user_similar: 用户-用户之间相似度
    :return: 返回预测评分
    """
    # 需要准备预测的物品id列表
    item_ids = ratings_matrix.columns
    # 逐个预测
    for iid in item_ids:
        try:
            rating = predict(uid,iid, rating_matrix,user_similar)
        except Exception as e:
            print(e)
        else:
            yield uid, iid, rating

if __name__'__main__':
    rating_matrix = load_data(DATA_PATH)
    user_similar = compute_pearson_similarity(rating_matrix, based = 'user')
    for i in predict_all(1, ratings_matrix, user_similar):
        pass

# 添加过滤规则
def predict_all_rules(uid, item_ids, ratings_matrix, user_similar):
    """
    :param uid: 用户id
    :param item_ids: 要预测的物品id列表
    :param ratings_matrix: 用户-物品打分矩阵
    :param user_similar: 用户-用户相似度
    :return: 返回预测评分
    """
    # 逐个预测
    for iid in item_ids:
        try:
            rating = predict(uid, iid, rating_matrix, user_similar)



