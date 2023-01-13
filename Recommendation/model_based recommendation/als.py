import numpy as np
import pandas as pd

# 除了这个方法，其余与随机梯度下降完全类似
def als(self): # 随机梯度下降 sgc()
    """
    利用梯度下降，优化bu，bi
    :param self:
    :return:
    """
    bu = dict(zip(self.users_ratings.index, np.zero(len(self.users_ratings))))  # np.zero生成一个全零的数组
    bi = dict(zip(self.users_ratings.index, np.zero(len(self.users_ratings))))

    for i in range(self.number_epochs):
        print('iter%d' % i)
        for uids, iid, ratings in self.dataset.itertuples(index=True):
            sum_num =0
            for uid, rating in zip(uids, ratings):
                sum_num += rating -self.global_mean-bu[uid]
            bi[iid] = sum_num/ (self.reg_bi + len(iids))
    return bu, bi

