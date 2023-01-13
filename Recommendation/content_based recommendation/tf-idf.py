import collections
from pprint import pprint
import pandas as pd
import numpy as np

"""
-利用tag.csv中每部电影的标签作为电影的候选关键词
-利用tf-idf值，选取topN关键词作为电影画像标签
-并将电影的分类词作为每部电影的画像标签
"""


def get_movie_dateset(path, path2):
    # 加载基于所有电影的标签
    tags = pd.read_csv(path, usecols=range(1, 3)).dropna().groupby('movieId').agg(list)

    # 加载列表数据集
    movies = pd.read_csv(path2, index_col='movieId')
    # 将类别词分开
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
    # 为每部电影匹配对应的标签数据，如果没有用nan填充
    movie_index = set(movies.index) & set(tags.index)
    new_tags = tags.loc[list(movie_index)]
    ret = movies.join(new_tags)

    # 构建数据集，包含电影id、电影名称、类别、标签四个字段
    # 如果电影没有标签数据，替换为nan
    movie_dataset = pd.DataFrame(map(lambda x: (x[0], x[1], x[2], x[2] + x[3]) if x[3] is not np.nan else (
        x[0], x[1], x[2], []), ret.itertuples()), columns=['movieId', 'title', 'genres', 'tags'])
    movie_dataset.set_index('movieId', inplace=True)
    return movie_dataset


def create_movie_profile(movie_dataset):
    dataset = movie_dataset['tags'].values
    from gensim.corpora import Dictionary
    # 根据数据集建立关键词袋，并统计词频，将所有词放入一个词典，使用索引进行获取
    dct = Dictionary(dataset)
    # 根据每条数据，返回对应的词索引和词频
    corpus = [dct.doc2bow(line) for line in dataset]
    # 训练TF-IDF模型，计算每个词的值
    from gensim.models import TfidfModel
    model = TfidfModel(corpus)

    movie_profile = {}
    for i, mid in enumerate(movie_dataset.index):
        # 根据每条数据返回，向量
        vector = model[corpus[i]]
        # 按照TF-IDF值得到topN关键词
        movie_tags = sorted(vector, key=lambda x: x[1], reverse=True)[:30]
        # 根据关键词提取对应的名称
        movie_profile[mid] = dict(lambda x: (dict[x[0]], x[1]), movie_tags)
    return movie_profile


movie_dataset = get_movie_dateset()
pprint(create_movie_profile(movie_dataset))


# 通过标签找到对应的电影,产生推荐结果
def create_inverted_table(movie_profile):
    inverted_table = []
    for mid, weights in movie_profile['weights'].iteritems():
        for tag, weight in weights.items():
            movie_found = inverted_table.get(tag, [])
            movie_found.append((mid, weight))
            inverted_table.setdefault(tag, movie_found)
    return inverted_table


# inverted_table = create_inverted_table(movie_profile)


# 完善用户画像

from functools import reduce

"""
- 提取用户观看列表
- 根据观看列表和物品画像为用户匹配关键词，并统计词频
- 排序，作为用户标签
"""


def create_user_profile():
    watch_record = pd.read_csv(path, usecols=range(2), dtype={'userId': np.int32, 'movieId': np.int32})
    watch_record = watch_record.groupby('userId').agg(list)

    movie_dataset = get_movie_dateset()
    movie_profile = create_movie_profile(movie_dataset)

    user_profile = {}
    for uid, mids in watch_record.itertuples():
        record_movie_profile = movie_profile.loc[list(mids)]
        counter = collections.Counter(reduce(lambda x, y: list(x) + list(y), record_movie_profile['profile'].values))
        interest_word = counter.most_common(50)
        maxcount = interest_word[0][1]
        interest_word = [(w, round(c / maxcount, 4)) for w, c in interest_word]
        user_profile[uid] = interest_word
    return user_profile


user_profile = create_user_profile()
pprint(user_profile)

# 产生推荐结果
inverted_table = create_inverted_table(movie_profile)
def user_result(user_profile):
    rs_dict ={}
    for uid, interest_words in user_profile.items():
        result_table={}
        for interest_word, interest_weight in interest_words:
            related_movie = inverted_table[interest_word]
            for mid, related_weight in related_movie:
                movie_found=result_table.get(mid, []) # 电影id和评分list
                movie_found.append(interest_weight) # 只考虑用户的兴趣程度
                # movie_found.append(related_weight) 只考虑兴趣词与电影的关联程度
                # movie_found.append(interest_weight*related_weight) 都考虑
                result_table.setdefault(mid, movie_found)
        rs_result = map(lambda x:(x[0], sum(x[1])), result_table.items())
        rs_result = sorted(rs_result, key = lambda x: x[1], reverse=True)[:10]
        rs_dict[uid] = rs_dict
    return rs_dict

