import pandas as pd
import numpy as np


def data_split(data_path, x=0.8, random=False):
    """
    切分数据集，这里为了保证用户数量不变，将每个用户的评分数据按比例进行拆分
    :param data_path: 数据集路径
    :param x: 训练集的比例，如x是0.8,测试集是0.2
    :param random: 是否随即切分，默认是False
    :return: 用户-物品评分矩阵
    """
    print('开始切分的数据集...')
    # 设置要加载的数据字段的类型
    dtype = {'userId': np.int32, 'movieId': np.int32, 'rating': np.float}
    # 加载数据，我们只使用前三列数据，分别是用户id，电影id，用户对电影的评分
    ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))

    testset_index = []
    # 为了保证每个用户在测试集和训练集都有数据，因此按userId聚合
    for uid in ratings.groupby('userId').any().index:
        user_rating_data = ratings.where(ratings['userId'] == uid).dropna()
        if random:
            # 因为不可变类型不能被shuffle方法作用，所以需要强行转换为列表
            index = list(user_rating_data.index)
            np.random.shuffle(index)  # 打乱列表
            index_num = round(len(user_rating_data) * x)
            testset_index += list(index[index_num:])
        else:
            # 将每个用户的x比例数据作为训练集，神谕的作为测试集
            index = round(len(user_rating_data) * x)
            testset_index += list(user_rating_data.index.values[index:])
    testset = ratings.loc[testset_index]
    trainset = ratings.drop(testset_index)
    print('完成数据集切分...')
    return trainset, testset


def accuray(predict_results, method='all'):
    """
    准确性指标计算方法
    :param predict_results: 预测结果，类型为容器，每个元素是一个包含 uid, iid, real_rating, pred_rating的序列
    :param method: 指标方法，类型为字符串，rmse或mae，否则返回两者rmse和mae
    :return:
    """

    def rmse(predict_results):
        """
        rmse评估指标
        :param predict_results:
        :return: rmse
        """
        length = 0
        rmse_sum = 0
        for uid, iid, real_rating, predict_rating in predict_results:
            length += 1
            rmse_sum += (predict_rating - real_rating) ** 2
        return round(np.sqrt(rmse_sum / length), 4)

    def mae(predict_results):
        """
        mae评估指标
        :param predict_results:
        :return: mae
        """
        length = 0
        mae_sum = 0
        for uid, iid, real_rating, predict_rating in predict_results:
            length += 1
            mae_sum += abs(predict_rating - real_rating)
        return round(mae_sum / length, 4)
