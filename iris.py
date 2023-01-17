from sklearn.datasets import load_iris                  # 获取数据集
from sklearn.model_selection import train_test_split    # 划分数据集
from sklearn.preprocessing import StandardScaler        # 标准化
from sklearn.neighbors import KNeighborsClassifier      # KNN算法分类

def knn():
    """
    KNN算法预测鸢尾花分类
    :return:
    """
    # 1、获取数据
    iris = load_iris()
    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)
    # 3、特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)  # 训练集标准化
    x_test = transfer.transform(x_test)        # 测试集标准化
    # 4、KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)   # 模型训练
    # 5、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接必读真实值和预测值：\n", y_test == y_predict)  # 直接比对
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)  # 测试集的特征值，测试集的目标值
    print("准确率：\n", score)

    return None


if __name__ == "__main__":
    knn()          # 调用knn()函数
