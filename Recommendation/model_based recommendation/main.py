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
rating_matrix = pd.DataFrame(data2, columns=items, index=users)
user_similar = round(rating_matrix.T.corr(),4)

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


def predict_all_score(uid, ratings_matrix, user_similar):
    """
    预测全部评分
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
    for i in predict_all_score(uid, rating_matrix, user_similar):
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
        except Exception as e:
            print(e)
        else:
            yield uid, iid, rating

def predict_all(uid, ratings_matrix, user_similar, filter_rule = None):
    """
    预测全部评分，并可根据条件进行前置过滤
    :param uid: 用户id
    :param ratings_matrix: 用户-物品打分矩阵
    :param user_similar: 用户两两相似度
    :param filter_rule: 过滤规则，只能是四选一，否则抛出异常：'unhot'，'rated'
    :return: 生成器，逐个返回预测评分
    """
    if not filter_rule:
        item_ids = ratings_matrix.columns
    elif isinstance(filter_rule, str) and filter_rule == 'unhot':
        # 过滤非热门电影
        count = ratings_matrix.count()
        # 过滤出评分数高于10的电影，作为热门电影
        item_ids = count.where(count>10).dropna().index
    elif isinstance(filter_rule, str) and filter_rule == 'rated':
        # 过滤用户评分过的电影
        # 获取用户的评分记录
        user_ratings = ratings_matrix.loc[uid]
        # 评分范围是1-5，小于6都是评分过的，除此之外都是没有评分的
        rule = user_ratings<6
        item_ids = rule.where(rule == False).dropna().index
    elif isinstance(filter_rule, list) and set(filter_rule) == set(['unhot', 'rated']):
        # 过滤非热门和用户已经评分过的电影
        count = rating_matrix.count()
        ids1 = count.where(count>10).dropna().index
        user_ratings = ratings_matrix.loc[uid]
        rule = user_ratings<6
        ids2 = rule.where(rule ==False).dropna().index
        # 取二者交集
        item_ids = set(ids1)&set(ids2)
    else:
        raise Exception("无效的过滤参数")
    yield from predict_all_rules(uid, item_ids, ratings_matrix, user_similar)

if __name__ == '__main__':
    rating_matrix = load_data(DATA_PATH)
    user_similar = compute_pearson_similarity(rating_matrix, based='user')
    for result in predict_all(1, rating_matrix, user_similar, filter_rule=['unhot','rated']):
        print(result)


# 根据预测评分给用户进行topN推荐
def top_k_rs_result(k):
    ratings_matrix = load_data(DATA_PATH)
    user_similar = compute_pearson_similarity(ratings_matrix, based='user')
    results = predict_all(1, ratings_matrix, user_similar, filter_rule=['unhot','rated'])
    return sorted(results, key = lambda x:x[2], reverse=True)[k]
if __name__=='__main__':
    from pprint import pprint
    result = top_k_rs_result(20)
    pprint(result)

