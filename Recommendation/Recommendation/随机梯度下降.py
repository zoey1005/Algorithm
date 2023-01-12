import pandas as pd
import numpy as np

# Stochastic Gradient Descent
class BaselineCFBySGD(object):
    def __init__(self, number_epochs, alpha, reg, columns = ['uid', 'iid', 'rating']):
        # 梯度下降最高迭代次数
        self.number_epochs = number_epochs
        # 学习率
        self.alpha = alpha
        # 正则参数
        self.reg = reg
        # 数据集中user-item-rating字段名称
        self.columns =columns

    def fit(self, dataset):
        """
        :param dataset: uid, iid, rating
        :return:
        """
        self.dataset =dataset # '__main__'下载
        # 用户评分数据
        self.users_ratings = dataset.groupby(self.columns[0].agg([list]))[[self.columns[1], self.columns[2]]]
        # agg([list])表示聚合为一个list，可以修改为.agg('max')
        # 物品评分数据
        self.items_ratings = dataset.groupby(self.columns[1].agg([list]))[[self.columns[0], self.columns[2]]]
        # 调用sgd方法训练模型参数
        self.bu, self.bi = self.sgd()

    def sgd(self):
        """利用随即梯度下降，优化bu，bi值
        :return: bu,bi值
        """
        bu = dict(zip(self.users_ratings.index, np.zero(len(self.users_ratings)))) # np.zero生成一个全零的数组
        bi = dict(zip(self.users_ratings.index, np.zero(len(self.users_ratings))))

        for i in range(self.number_epochs):
            print('iter%d' % i)
            for uid, iid, real_rating in self.dataset.itertuples(index = False):
                error = real_rating - (self.global_mean + bu[uid] +bi[uid])
                bu[uid] += self.alpha*(error - self.reg * bu[uid])
                bi[uid] += self.alpha*(error - self.reg * bu[iid])

        return bu, bi

    def predict(self, uid, iid):
        # 评分预测
        if iid not in self.items_ratings.index:
            raise Exception('无法预测用户<{uid}>对电影<{iid}>的评分，因为训练集中缺失<{iid}>的数据'.format(uid = uid, iid = iid))
        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating

    def test(self, testset):
        # 预测测试集的数据
        for uid, iid, real_rating in testset.itertuples(index = False):
            try:
                pred_rating = self.predict(uid, iid)
            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating

if __name__ = '__main__':
    dtype = [('userId', np.int32), ('movieId', np.int32), ('rating', np.float32)]
    dataset = pd.read_csv('dataset/ml-latest-small/rating.csv', usecols = range(3), dtype= dict(type))

    # 创建对象
    bcf = BaselineCFBySGD(20, 0.1, 0.1, ['uid', 'iid', 'rating'])
    bcf.fit(dataset)

    while True:
        uid = int(input('uid: '))
        iid = int(input('iid: '))
        print(bcf.predict(uid, iid))


