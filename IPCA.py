# -*- coding: utf-8 -*-
# author @wwj

from sklearn.linear_model import LinearRegression

import numpy as np

import pandas as pd


def get_charport_ret(data):
    n_sec = data.shape[0]
    return_vec = data.iloc[:, 2:3].values/n_sec
    char_ma = data.iloc[:, 3:].values
    a = np.dot(char_ma.T, return_vec)
    res = pd.DataFrame(a.T, columns=data.axes[1][3:])
    res_date = pd.DataFrame({'date': [data['date'].iloc[0]]})
    res = pd.concat([res_date, res],1)
    return res


def get_interaction(data, k=3):
    data_z = data.iloc[:, 3:-k].values
    data_f = data.iloc[0:1, -k:].values
    data_res = np.kron(data_z, data_f)
    res = pd.DataFrame(data_res)
    res_date = pd.DataFrame({'date': list(data['date'])})
    res = pd.concat([res_date, res], 1)
    return res


def get_interaction_full(data, k=3,lag=2):
    data_z = data.iloc[:, 3:-(k*lag+k)].values
    data_f_full = data.iloc[0:1, -(k*lag+k):].values
    data_res = np.kron(data_z, data_f_full[:,0:k])
    for i in range(lag):
        temp = np.kron(data_z, data_f_full[:, (i+1)*k:(i+2)*k])
        data_res = np.concatenate((data_res, temp), axis=1)
    res = pd.DataFrame(data_res)
    res_date = pd.DataFrame({'date': list(data['date'])})
    res = pd.concat([res_date, res], 1)
    return res


def est_f(data, coef_ma, k):
    data_z = data.iloc[:, 3:-k].values
    data_x = np.dot(data_z, coef_ma)
    data_y = data['return_rate'].values
    model_for_f = LinearRegression(fit_intercept=False).fit(data_x, data_y)
    temp = pd.DataFrame(model_for_f.coef_.reshape((1, k)))
    temp_date = pd.DataFrame({'date': [data['date'].iloc[0]]})
    res = pd.concat([temp_date, temp], 1)
    return res


def est_f_ini(data, coef_ma):
    data_z = data.iloc[:, 3:].values
    data_x = np.dot(data_z, coef_ma)
    data_y = data['return_rate'].values
    model_for_f = LinearRegression(fit_intercept=False).fit(data_x, data_y)
    temp = pd.DataFrame(model_for_f.coef_.reshape((1, coef_ma.shape[1])))
    temp_date = pd.DataFrame({'date': [data['date'].iloc[0]]})
    res = pd.concat([temp_date, temp], 1)
    return res


def est_f_full(data, coef_ma, k, lag):
    data_z = data.iloc[:, 3:-(lag*k+k)].values
    data_f_lag = data.iloc[0:1, -(lag*k):].values
    data_x_de = np.dot(data_z, coef_ma[:, -(lag*k):])
    data_x_de = np.dot(data_x_de, data_f_lag.T).reshape((data.shape[0],))
    data_residual = data['return_rate'].values-data_x_de
    data_x = np.dot(data_z, coef_ma[:, 0:k])
    model_for_f = LinearRegression(fit_intercept=False).fit(data_x, data_residual)
    temp = pd.DataFrame(model_for_f.coef_.reshape((1, k)))
    temp_date = pd.DataFrame({'date': [data['date'].iloc[0]]})
    res = pd.concat([temp_date, temp], 1)
    return res


def ipca(data,k=3,lag=2,n_loop=100):
    n_z = data.shape[1]-3
    n_t = data['date'].drop_duplicates.count
    data_return_charport = data.groupby('date', sort=True, as_index=False).apply(get_charport_ret)
    data_return_charport_arr = data_return_charport.iloc[:, 1:].values
    cov = np.cov(data_return_charport_arr, rowvar=False)
    eig, eig_vec = np.linalg.eigh(cov)
    eig_vec_used = eig_vec[:, -k:][:, ::-1]
    factor_return = np.dot(data_return_charport_arr, eig_vec_used)
    factor_return_pd = pd.DataFrame(factor_return)
    factor_return_pd['date'] = list(data_return_charport['date'])
    data = pd.merge(data, factor_return_pd, how='left', on=['date'])
    coef_ma = None
    for i in range(n_loop):
        data_x = data.groupby('date', sort=True, as_index=False).apply(lambda x: get_interaction(x, k))
        data_x = data_x.iloc[:, 1:].values
        data_y = data['return_rate'].values
        model_temp = LinearRegression(fit_intercept=False).fit(data_x, data_y)
        coef_vec = model_temp.coef_
        coef_ma = coef_vec.reshape((n_z, k))

        factor_return_pd = data.groupby('date', sort=True, as_index=False).apply(lambda x: est_f(x, coef_ma, k))
        data = data.iloc[:, 0:-k]
        data = pd.merge(data, factor_return_pd, how='left', on='date')

    return coef_ma, data


def get_model(data,k=3,lag=2,n_loop=100):
    """
    IPCA的lag版本
    :param data: dataFrame,columns=['date','sec_code','return_rate',z1,..zP]
    :param k:
    :param lag:
    :param n_loop:
    :return:
    """
    n_z = data.shape[1] - 3
    n_t = data['date'].drop_duplicates().shape[0]
    data_return_charport = data.groupby('date', sort=True, as_index=False).apply(get_charport_ret)
    data_return_charport_arr = data_return_charport.iloc[:, 1:].values
    cov = np.cov(data_return_charport_arr, rowvar=False)
    eig, eig_vec = np.linalg.eigh(cov)
    eig_vec_used = eig_vec[:, -k:][:, ::-1]

    factor_return_pd = data.groupby('date', sort=True, as_index=False).apply(lambda x: est_f_ini(x,eig_vec_used))
    factor_return = factor_return_pd.iloc[:, 1:].values

    #factor_return = np.dot(data_return_charport_arr, eig_vec_used)
    factor_return_full_lag_pd = pd.DataFrame(factor_return)
    for ll in range(lag):
        temp = pd.DataFrame(factor_return).shift(ll+1).fillna(0.0)
        factor_return_full_lag_pd = pd.concat([factor_return_full_lag_pd, temp], axis=1)
    factor_return_full_lag_pd['date'] = list(data_return_charport['date'])
    data = pd.merge(data, factor_return_full_lag_pd, how='left', on=['date'])
    coef_ma = np.zeros((n_z, k*(lag+1)))
    opt_info = np.zeros((n_loop, 2))
    opt_info[:, 0] = np.arange(n_loop)
    for i in range(n_loop):
        coef_ma_old = coef_ma
        factor_return_old = factor_return
        data_x = data.groupby('date', sort=True, as_index=False).apply(lambda x: get_interaction_full(x, k, lag))
        data_x = data_x.iloc[:, 1:].values
        data_y = data['return_rate'].values
        model_temp = LinearRegression(fit_intercept=False).fit(data_x, data_y)
        coef_vec = model_temp.coef_
        coef_temp = coef_vec.reshape((lag+1, -1))
        coef_ma = coef_temp[0].reshape((n_z, k))
        for j in range(lag):
            temp = coef_temp[j+1].reshape((n_z, k))
            coef_ma = np.concatenate((coef_ma, temp), axis=1)

        factor_return_pd = data.groupby('date', sort=True, as_index=False).apply(lambda x: est_f_full(x, coef_ma, k, lag))

        factor_return = factor_return_pd.iloc[:, 1:].values
        factor_return_full_lag_pd = pd.DataFrame(factor_return)
        for ll in range(lag):
            temp = pd.DataFrame(factor_return).shift(ll + 1).fillna(0.0)
            factor_return_full_lag_pd = pd.concat([factor_return_full_lag_pd, temp], axis=1)
        factor_return_full_lag_pd['date'] = list(data_return_charport['date'])
        data = data.iloc[:, 0:(3+n_z)]
        data = pd.merge(data, factor_return_full_lag_pd, how='left', on=['date'])
        diff_coef = np.amax(np.abs(coef_ma - coef_ma_old))
        diff_f = np.amax(np.abs(factor_return - factor_return_old))
        diff_com = max(diff_coef, diff_f)
        opt_info[i, 1] = diff_com
        print('current iteration num :{}, diff:{}'.format(i, diff_com))
        if diff_com < 1e-8:
            print('Convergency is good')
            break
        elif i == (n_loop - 1):
            print('Convergency is not good,but max iteration num is reached')
    pd.DataFrame(opt_info).to_csv('opt_info.csv')
    final_model = coef_ma, factor_return[-lag:][::-1].reshape((1, -1)).T, k, lag, data
    return final_model


def predict_er(final_model,z):
    """预测

    :param model: keras,Model 训练好的收益模型
    :param x: ndarray, 2维， 第一维是观测，第二维是自变量
    :return: ndarray, 2维， 第一维是观测，第二维是因变量
    """
    coef_ma = final_model[0]
    factor_return = final_model[1]
    k = final_model[2]
    lag = final_model[3]
    pred = np.dot(z, coef_ma[:, k:])
    pred = np.dot(pred, factor_return)
    return pred


def model_update(final_model, x,z):
    """

    :param final_model:
    :param x: n*1,t+1的return_rate
    :param z: n*p,t的信息
    :return:
    """
    coef_ma = final_model[0]
    factor_return = final_model[1]
    k = final_model[2]
    lag = final_model[3]
    pred = np.dot(z, coef_ma[:, k:])
    pred = np.dot(pred, factor_return)
    residual_return = x-pred
    data_x = np.dot(z, coef_ma[:, 0:k])
    f_temp = LinearRegression(fit_intercept=False).fit(data_x, residual_return.reshape((z.shape[0],))).coef_.reshape((k, 1))
    factor_return = np.concatenate((f_temp, factor_return[0:-k]))

    final_model = coef_ma, factor_return, k, lag
    return final_model


def gen_data(n_t=120,n_sec=1000,n_z=10,k=3,lag=2,sigma=0.0,coef_l0=None, coef_l1=None, coef_l2=None):
    factor_return = np.random.randn(n_t+lag, k)
    factor_return = factor_return * np.array([1.0,1.0,1.0])
    #factor_return[0:lag*2]=np.zeros((lag*2,k))
    data_z = np.random.randn(n_t*n_sec, n_z)
    data_r = np.zeros((n_t,n_sec))
    data_r_pr_full = np.zeros((n_t, n_sec))
    data_r_pr = np.zeros((n_t, n_sec))
    coef_ma = np.concatenate((coef_l0,coef_l1,coef_l2),axis=1)
    coef_ma_pr = np.concatenate((coef_l1, coef_l2), axis=1)
    for t in range(n_t):
        temp_z = data_z[t*n_sec:(t+1)*n_sec]
        temp_f = factor_return[t:t+lag+1][::-1].reshape((1, -1)).T
        temp_r = np.dot(np.dot(temp_z, coef_ma), temp_f)+np.random.randn(n_sec, 1)*sigma
        data_r[t:t+1] = temp_r.T
        temp_r_pr_full = np.dot(np.dot(temp_z, coef_ma), temp_f)
        data_r_pr_full[t:t + 1] = temp_r_pr_full.T


        temp_f_pr = factor_return[t:t + lag][::-1].reshape((1, -1)).T
        temp_r_pr = np.dot(np.dot(temp_z, coef_ma_pr), temp_f_pr)
        data_r_pr[t:t + 1] = temp_r_pr.T


    data_r_s = data_r.reshape((n_t*n_sec,))
    temp_index = pd.MultiIndex.from_product([np.arange(n_t), np.arange(n_sec)], names=['date', 'stockcode'])
    data_temp = pd.DataFrame(index=temp_index).reset_index()
    data_temp['return_rate'] = data_r_s
    data_res = pd.concat([data_temp.loc[:,['date','stockcode','return_rate']], pd.DataFrame(data_z)], axis=1)
    return data_res, data_r_pr_full, data_r_pr

coef0=np.random.randn(10,3)

coef1=np.random.randn(10,3)*1.0

coef2=np.random.randn(10,3)*1.0

n_t=200
n_z=10
n_sec=500
n_test=20
k=3
lag=2

data, data_r_pr_full, data_r_pr = gen_data(n_t=n_t+n_test,n_sec=n_sec,sigma=0.0,coef_l0=coef0, coef_l1=coef1, coef_l2=coef2)
data_train = data.iloc[0:n_t*n_sec]
data_test = data.iloc[n_t*n_sec:]
data_r_pr_full = data_r_pr_full[0:n_t]
data_r_pr_test = data_r_pr[n_t:]
data_r_pr = data_r_pr[0:n_t]

#pred = np.zeros((n_test,n_sec))
#true_pred = data_r_s_pr[-n_test*n_sec:].reshape((n_test,n_sec))

data_r_pr_full_fit = np.zeros(data_r_pr_full.shape)
data_r_pr_fit = np.zeros(data_r_pr.shape)
data_r_pr_oos = np.zeros(data_r_pr_test.shape)
"""
model = ipca(data_train,k=3,n_loop=10)
for t in range(data_r1.shape[0]):
    z_temp = model[1].iloc[t*n_sec:(t+1)*n_sec, 3:-k].values
    f_return = model[1].iloc[t*n_sec:t*n_sec+1, -k:].values
    coef = model[0]
    a = np.dot(np.dot(z_temp, coef),f_return.T)
    data_r1_fit[t:t+1] = a.T
"""

model = get_model(data_train,k=3,lag=2,n_loop=1000)
for t in range(n_t):
    z_temp = model[4].iloc[t * n_sec:(t + 1) * n_sec, 3:3+n_z].values
    f_return = model[4].iloc[t * n_sec:t * n_sec + 1, -(lag+1)*k:].values
    coef = model[0]
    a = np.dot(np.dot(z_temp, coef), f_return.T)
    data_r_pr_full_fit[t:t + 1] = a.T

    f_return = model[4].iloc[t * n_sec:t * n_sec + 1, -lag * k:].values
    coef = model[0][:, -(lag*k):]
    a = np.dot(np.dot(z_temp, coef), f_return.T)
    data_r_pr_fit[t:t + 1] = a.T


for i in range(n_test):
    z = data_test.iloc[i*n_sec:(i+1)*n_sec, 3:3+n_z].values
    r = data_test['return_rate'].iloc[i*n_sec:(i+1)*n_sec].values.reshape((n_sec, 1))
    pred = predict_er(model, z)
    data_r_pr_oos[i:i+1] = pred.T
    model = model_update(model, r, z)


kkk=3