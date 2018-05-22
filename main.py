import numpy as np
from sklearn import svm
from scipy import stats
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt

def extend(a, b, r=0.01):
    return a * (1 + r) - b * r, -a * r + b * (1 + r)

def startjob():
    np.set_printoptions(suppress=True)
    np.random.seed(0)
    N = 200
    x = np.empty((4 * N, 2))
    means = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
    #np.eye，生成对角单位矩阵；np.diag，根据给定的值生成对角矩阵
    sigmas = [np.eye(2), 2 * np.eye(2), np.diag((1, 2)), np.array(((3, 2), (2, 3)))]
    # print(x)
    print(sigmas)
    #为x赋值
    for i in range(4):
        #多元正态分布，将均值与方差喂给此函数，即可得到模型
        mn = stats.multivariate_normal(means[i], sigmas[i] * 0.1)
        x[i * N: (i + 1) * N, :] = mn.rvs(N)
    # print(x)
    #得到分类向量y(四分类)
    a = np.array((0, 1, 2, 3)).reshape((-1, 1))
    y = np.tile(a, N).flatten()
    # clf = svm.SVC(C = 1, kernel='rbf', gamma=1, decision_function_shape='ovr')
    clf = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovo')
    clf.fit(x, y)
    y_hat = clf.predict(x)
    acc = accuracy_score(y, y_hat)
    print('预测正确的样本个数: %d, 正确率: %.2f%%' % (round(acc * 4 * N), 100 * acc))
    print(clf.decision_function(x))
    print(y_hat)

    x1_min, x2_min = np.min(x, axis=0)
    x1_max, x2_max = np.max(x, axis=0)
    x1_min, x1_max = extend(x1_min, x1_max)
    x2_min, x2_max = extend(x2_min, x2_max)

    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    x_test = np.stack((x1.flat, x2.flat), axis=1)
    # print(x_test)
    y_test = clf.predict(x_test)
    y_test = y_test.reshape(x1.shape)
    print(y_test)
    cm_light = mpl.colors.ListedColormap(['#FF8080', '#80FF80', '#8080FF', '#F0F080'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g', 'b', 'y'])
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    #将x1, x2两个网格矩阵和对应的预测结果y_test绘制在图片上，输出为颜色区块
    #区块数量与y的类别一致
    plt.pcolormesh(x1, x2, y_test, cmap=cm_light)
    #绘制等高线
    plt.contour(x1, x2, y_test, levels=(0,1,2), colors='k', linestyles='--')
    plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=cm_dark, edgecolors='k', alpha=0.7)
    plt.xlabel('$X_1$', fontsize=15)
    plt.ylabel('$X_2$', fontsize=15)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(b=True)
    plt.tight_layout(pad=1.5)
    plt.title('SVM多分类方法: One/One or One/Other', fontsize=18)
    plt.show()

if __name__ == '__main__':
    startjob()