import pandas as pd
import numpy as np

"""
LFM model1
"""

# 评分预测1-5

class LFM(object):
    def __init__(self, alpha, reg_p, reg_q,
                 number_LatentFactors = 10, number_epochs =10, columns =['uid', 'iid', 'rating']):
        # 学习率
        self.alpha = alpha
        # p矩阵正则化
        self.reg_p = reg_p
        # q矩阵正则化
        self.reg_q = reg_q
        # 隐式类别数量
        self.number_LatentFactors = number_LatentFactors
        # 最大迭代次数
        self.number_epochs = number_epochs
        self.columns =columns

    def fit(self, dataset):
        """
        fit dataset
        :param dataset: uid, iid, rating
        :return:
        """
        self.dataset = pd.DataFrame(dataset)
        self.user_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.item_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]

        self.global_mean= self.dataset[self.columns[2]].mean()
        self.p, self.q = self.sgd()

    def init_matrix(self):
        """
        初始化P和Q矩阵，同时设置为0， 1 之间的随机值作为初始值
        :return:
        """
        # User-LF
        p = dict(zip(self.user_ratings.index,
                     np.random.rand(len(self.item_ratings), self.number_LatentFactors).astype(np.float32)))
        # Item_LF
        q = dict(zip(self.user_ratings.index,
                     np.random.rand(len(self.item_ratings), self.number_LatentFactors).astype(np.float32)))

        return p, q

    def sgd(self):
        """
        使用随机梯度下降，优化结果
        :return:
        """
        p, q = self.init_matrix()
        # 随机梯度下降的损失函数
        for i in range(self.number_epochs):
            print('iter%d' % i)
            error_list=[]
            # 遍历用户、物品的评分数据，通过用户id到用户矩阵中获取用户向量、物品向量
            for uid, iid, r_ui in self.dataset.itertuples(index = False):
                # 用户向量
                v_pu = p[uid]
                # 物品向量
                v_qi = q[uid]
                # 真实值减去向量点乘的预测值
                error = np.float32(r_ui - np.dot(v_pu, v_qi))

                v_pu += self.alpha * (error * v_qi - self.reg_p * v_pu)
                v_qi += self.alpha * (error * v_pu - self.reg_q * v_qi)

                p[uid] = v_pu
                q[uid] = v_qi

                error_list.append(error ** 2)
            print(np.sqrt(np.mean(error_list)))
        return p, q

    def predic(self, uid, iid):
        # 如果uid, iid不在，我们使用全局平均分作为预测结果返回
        if uid not in self.users_ratings.index or iid not in self.item_ratings.index:
            return self.global_mean
        p_u = self.p[uid]
        q_i = self.q[iid]
        return np.dot(p_u, q_i)
