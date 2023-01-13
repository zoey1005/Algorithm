from gensim.models import TfidfModel

import pandas as pd
import numpy as np

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

    movie_profile = []
    for i, data in enumerate(movie_dataset.itertuples()):
        mid = data[0]
        title = data[1]
        genres = data[2]
        vector = model[corpus[i]]
        movie_tags = sorted(vector, key = lambda x:x[1], reverse=True)[:30]
        topN_tags_weights = dict(map(lambda x:(dict[x[0]], x[1]), movie_tags))
        # 添加类别，权重设为1
        for g in genres:
            topN_tags_weights[g] = 1.0
        topN_tags = [i[0] for i in topN_tags_weights.items()]
        movie_profile.append(mid, title, topN_tags, topN_tags_weights)

    movie_profile_df = pd.DataFrame(movie_profile, columns = ['movieId', 'title', 'profile', 'weights'])
    movie_profile_df.set_index('movieId', inplace = True)
    return movie_profile_df

movie_dataset = get_movie_dateset()
movie_profile_df= create_movie_profile(movie_dataset)


import gensim, logging

logging.basicConfig(format = '%(asctime)s :%(levelname)s: %(message)s', level=logging.INFO)

sentences = list(movie_profile_df['profile'].values)

model = gensim.models.Word2Vec(sentences, window=3, min_count=1, iter=20) # windows =3是三个词一起比较

while True：
    words = input('words: ') # eg. action
    ret = model.wv.most_similar(postive = [words], topn =10)
    print(ret)

