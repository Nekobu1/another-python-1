import numpy as np
from statistics import mean
import pandas as pd
from pandas import read_csv

data = read_csv("data_3.1.csv", sep=';', encoding="Windows-1251")
x1 = data['x1']
x2 = data['x2']
y = data['y']


def f(x1, x2):
    return 1, 1 / x1, x1, x2 ** 2, x2, x1 * x2


def func_u(x1, x2, theta):
    return theta[0] * f(x1, x2)[0] + theta[1] * f(x1, x2)[1] + theta[2] * f(x1, x2)[2] + theta[3] * f(x1, x2)[3] + \
           theta[4] * f(x1, x2)[4] + theta[5] * f(x1, x2)[5]


def simulation(u, n, p):
    omega = np.dot((u - mean(u)), (u - mean(u)).T) / (n - 1)
    # omega = 6.42282
    dispersion = np.math.sqrt(p * omega)
    # dispersion = 0.80143
    e = np.random.normal(0, dispersion, n)
    y = u + e
    # y = data['y']
    return dispersion, y, omega


def theta_estimate(x1, x2, y, n):
    x = np.array([np.ones(n), f(x1, x2)[1], f(x1, x2)[2], f(x1, x2)[3], f(x1, x2)[4], f(x1, x2)[5]])
    estimate = np.linalg.inv(x.dot(x.T)).dot(x).dot(y)
    return estimate


def sigma_estimate(x1, x2, y, n, m):
    t_est = theta_estimate(x1, x2, y, n)
    u_est = func_u(x1, x2, t_est)
    e = y - u_est
    estimate = np.dot(e, e.T) / (n - m)
    return estimate, u_est, e


# Построить доверительные интервалы для каждого параметра модели регрессии.
def sigma_teta(x1, x2, y, n, m):
    x = np.array([np.ones(n), f(x1, x2)[1], f(x1, x2)[2], f(x1, x2)[3], f(x1, x2)[4], f(x1, x2)[5]])
    d = np.linalg.inv(x.dot(x.T))
    sigma = sigma_estimate(x1, x2, y, n, m)[0]
    d_jj = np.diagonal(d)
    s_t = np.sqrt(sigma * d_jj)
    return s_t, sigma, d_jj


def confidence_interval(x1, x2, y, n, m, theta):
    global lower_bound, upper_bound
    t = 1.984
    theta_hat = theta_estimate(x1, x2, y, n)
    sigma = sigma_teta(x1, x2, y, n, m)[0]
    print('\n', 'Доверительное оценивание для отдельного параметра:')
    print('j', '\t', 'teta_j', '\t\t', 'teta_hat_j', '\t\t\t\t', 'a', '\t\t\t\t\t\t', 'b')
    for j in range(len(theta)):
        lower_bound = theta_hat - t * sigma
        upper_bound = theta_hat + t * sigma
        print(j + 1, '\t', '%.2f' % theta[j], '\t\t\t', '%.10f' % theta_hat[j], '\t\t\t', '%.10f' % lower_bound[j],
              '\t\t\t', '%.10f' % upper_bound[j])
    return lower_bound, upper_bound, theta_hat


# Проверка гипотезы о незначимости для каждого параметра
def hypothesis_testing(x1, x2, y, n, m, theta):
    theta_hat = confidence_interval(x1, x2, y, n, m, theta)[2]
    sigma = sigma_teta(x1, x2, y, n, m)[1]
    d_jj = sigma_teta(x1, x2, y, n, m)[2]
    f = theta_hat ** 2 / ((sigma ** 2) * d_jj)
    f_critical = 3.94
    print('\n', 'Проверка гипотезы о незначимости:')
    print('j', '\t', 'F', '\t\t\t\t\t', 'Гипотеза о незначимости')
    for j in range(len(d_jj)):
        if f[j] < f_critical:
            print(j + 1, '\t', '%.10f' % f[j], '\t\t', 'Не отвергается')
        else:
            print(j + 1, '\t', '%.10f' % f[j], '\t\t', 'Отвергается')


# Проверка гипотезы о незначимости регрессии
def regression_hypothesis_testing(x1, x2, n, m, theta):
    x = np.array([np.ones(n), f(x1, x2)[1], f(x1, x2)[2], f(x1, x2)[3], f(x1, x2)[4], f(x1, x2)[5]])
    theta_hat = confidence_interval(x1, x2, y, n, m, theta)[2]
    rss_h = np.sum((y - np.average(y)) ** 2)
    rss = np.sum((np.squeeze(np.asarray(y - np.dot(theta_hat, x)))) ** 2)
    fr = ((rss_h - rss) / 5) / (rss / (n - m))
    f_critical = 2.3
    print('\n', 'Проверка гипотезы о незначимости регрессии:')
    if fr < f_critical:
        print('F=', '%.10f' % fr, '=> гипотеза о незначимости регрессии не отвергается')
    else:
        print('F=', '%.10f' % fr, '=> гипотеза о незначимости регрессии отвергается')


# Прогнозирование математического ожидания функции отклика
def eta(x1, x2, y, n):
    global f_x
    etas = np.array([])
    theta_hat = theta_estimate(x1, x2, y, n)
    x11 = [1, 1, 1, 1, 1]
    x22 = [1, 1.5, 2, 2.5, 3]
    for j in range(len(x11)):
        f_x = np.array([1, 1 / x11[j], x11[j], x22[j] ** 2, x22[j], x11[j] * x22[j]])
        eta = np.dot(f_x, theta_hat)
        etas=np.append(eta)
        print("x1 = %d, x2 = %d, f(x) = " % (x11[j], x22[j]), f_x[j], ", f(x)T*theta = ", eta)
        print("theta_hat = ", theta_hat)
    return f_x, etas


# Построение доверительных интервалов для математического ожидания функции отклика
def sigma_eta(x1, x2, y, n, m):
    x=[]
    x_1 = np.array([1, 1, 1, 1, 1])
    x_2 = np.array([1, 1.5, 2, 2.5, 3])
    onesss=np.ones(len(x_1))
    f_x = eta(x1, x2, y, n)[0]
    eta_hat = eta(x1, x2, y, n)[1]
    for j in range(len(x_1)):
        xx= np.array([onesss[j], f(x_1[j], x_2[j])[1], f(x_1[j], x_2[j])[2], f(x_1[j], x_2[j])[3], f(x_1[j], x_2[j])[4], f(x_1[j], x_2[j])[5]])
        x.append(xx)
    d = np.linalg.inv(x.dot(x.T))
    sigma = sigma_estimate(x1, x2, y, n, m)[0]
    d_jj = np.diagonal(d)
    s_t = np.sqrt(f_x * d_jj)
    return s_t, sigma, d_jj


def main():
    a = 1
    b = 3
    p = 0.1
    n = 100
    m = 6
    theta = (1, 1, 0.01, 1, 0.01, 0)
    # part 1
    u = func_u(x1, x2, theta)
    disp, y, omega = simulation(u, n, p)
    print('omega =', omega)
    print('dispersion =', disp)
    # part 2
    sigma_est = sigma_estimate(x1, x2, y, n, m)[0]
    print('theta_estimate =', theta_estimate(x1, x2, y, n))
    print('sigma_estimate =', sigma_estimate(x1, x2, y, n, m)[0])
    result1 = pd.DataFrame({'x1': x1, 'x2': x2, 'u': u, 'y': y,
                            'тета*х': sigma_estimate(x1, x2, y, n, m)[1],
                            'e': sigma_estimate(x1, x2, y, n, m)[2]})
    # result1.to_csv("data3.csv", sep=';', encoding="Windows-1251")
    confidence_interval(x1, x2, y, n, m, theta)
    # hypothesis_testing(x1, x2, y, n, m, theta)
    # regression_hypothesis_testing(x1, x2, n, m, theta)
    eta(x1, x2, y, n)
    sigma_eta(x1, x2, y, n, m)

if __name__ == '__main__':
    main()
