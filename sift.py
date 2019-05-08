# coding=utf-8
"""
face recognition by sift features

"""
import pickle
import os

# from scipy.misc import toimage
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


# utils function


def show_img(fea):
    # toimage(fea.reshape(64, 64)).show()
    # or
    plt.imshow(fea.reshape(64, 64), cmap='gray')


# def save_img(fea, file_name):
#     toimage(fea.reshape(64, 64)).save(file_name)


def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


def gen_sift_features(img):
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints

    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc


def show_sift_features(gray_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, gray_img.copy()))


def load_data(shuffle_=True):

    file_ids = ['05', '07', '09', '27', '29']
    train_feas, train_labels = [], []
    test_feas, test_labels = [], []
    for file_id in file_ids:
        path = 'PIE_dataset/Pose{}_64x64.mat'.format(file_id)
        data = loadmat(path)

        train_indices = (data['isTest'] == 0).reshape(-1)
        test_indices = (data['isTest'] == 1).reshape(-1)

        train_feas.append(data['fea'][train_indices, :])
        test_feas.append(data['fea'][test_indices, :])

        train_labels.append(data['gnd'][train_indices, :])
        test_labels.append(data['gnd'][test_indices, :])

    train_feas = np.concatenate(train_feas)
    train_feas = train_feas / 255.0
    train_labels = np.concatenate(train_labels)

    test_feas = np.concatenate(test_feas)
    test_feas = test_feas / 255.0
    test_labels = np.concatenate(test_labels)

    if shuffle_:
        train_feas, train_labels = shuffle(train_feas, train_labels)
        test_feas, test_labels = shuffle(test_feas, test_labels)
        train_labels = train_labels.reshape(-1)
        test_labels = test_labels.reshape(-1)

    return train_feas, train_labels, test_feas, test_labels


def build_histogram(descs, cluster_model):

    histogram = np.zeros(len(cluster_model.cluster_centers_))
    if descs is None:
        return histogram

    cluster_result = cluster_model.predict(descs)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram


def build_sift_features(feas, cluster_model):
    histograms = []
    for fea in tqdm(feas):
        img = (fea*255).reshape(64, 64).astype(np.uint8)
        key_points, descs = gen_sift_features(img)

        histogram = build_histogram(descs, cluster_model)
        histograms.append(histogram)

    histograms = np.stack(histograms, axis=0)

    return histograms


def transform_to_sift_feas(train_feas, test_feas, k):
    kmeans_model_path = 'kmeans_model_{}.pkl'.format(k)

    if not os.path.isfile(kmeans_model_path):
        train_descs = None
        print("抽取训练集上的sift特征...")
        for fea in tqdm(train_feas):
            img = (fea*255).reshape(64, 64).astype(np.uint8)
            key_points, desc = gen_sift_features(img)
            if train_descs is None:
                train_descs = desc
            else:
                if desc is not None:
                    train_descs = np.append(train_descs, desc, axis=0)

        # 速度极慢....
        print("使用k-means对提取出来的特征进行聚类...")
        kmeans_model = KMeans(n_clusters=k)
        kmeans_model.fit(train_descs)

        # 换成mini_batch k-means 可以加快速度
        # kmeans_model = MiniBatchKMeans(
        #     n_clusters=100, batch_size=16, random_state=0)
        # kmeans_model.fit(train_descs)

        # 写入 下回可以直接从文件中加载
        with open(kmeans_model_path, 'wb') as w:
            pickle.dump(kmeans_model, w)
    else:
        print("加载k-means模型...")
        with open(kmeans_model_path, 'rb') as f:
            kmeans_model = pickle.load(f)

    # 提取特征
    print("构建词袋模型...")
    train_sift_feas = build_sift_features(train_feas, kmeans_model)
    test_sift_feas = build_sift_features(test_feas, kmeans_model)

    # 标准化
    scaler = StandardScaler()
    train_sift_feas = scaler.fit_transform(train_sift_feas)
    test_sift_feas = scaler.transform(test_sift_feas)

    return train_sift_feas, test_sift_feas


def transform_to_pca_feas(train_feas, test_feas, dim):
    pca = PCA(n_components=dim, copy=False)
    train_pca_feas = pca.fit_transform(train_feas)
    test_pca_feas = pca.transform(test_feas)

    return train_pca_feas, test_pca_feas


def train(train_feas, train_labels, test_feas, test_labels):
    # 使用LR
    lr_clf = LogisticRegression(
        solver="lbfgs", max_iter=300, multi_class='multinomial')
    lr_clf.fit(train_feas, train_labels)
    predicted = lr_clf.predict(test_feas)
    acc = np.mean(predicted == test_labels)
    print("Accuracy of LogisticRegression: {:.2f}%".format(acc * 100))

    # 朴素贝叶斯(效果不好)
    # nb_clf = MultinomialNB()
    # nb_clf.fit(train_feas, train_labels)
    # predicted = nb_clf.predict(test_feas)
    # acc = np.mean(predicted == test_labels)
    # print("Accuracy of Naive Bayes: {:.2f}%".format(acc * 100))

    # SVM
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
    sgd_clf.fit(train_feas, train_labels)
    predicted = sgd_clf.predict(test_feas)
    acc = np.mean(predicted == test_labels)
    print("Accuracy of SVM: {:.2f}%".format(acc * 100))

    # 随机森林
    rf_clf = RandomForestClassifier(n_estimators=20)
    rf_clf.fit(train_feas, train_labels)
    predicted = rf_clf.predict(test_feas)
    acc = np.mean(predicted == test_labels)
    print("Accuracy of RandomForest: {:.2f}%".format(acc * 100))

    # K-means:(效果不好)
    # km_clf = KMeans(n_clusters=68).fit(train_feas)
    # predicted = km_clf.predict(test_feas)
    # acc = np.mean(predicted == np.array(test_labels))
    # print("Accuracy of K means: {:.2f}%".format(acc * 100))

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train_feas, train_labels)
    predicted = neigh.predict(test_feas)
    acc = np.mean(predicted == np.array(test_labels))
    print("Accuracy of KNN: {:.2f}%".format(acc * 100))


if __name__ == "__main__":
    train_feas, train_labels, test_feas, test_labels = load_data()

    # 基于原始图像的方法
    train(train_feas, train_labels, test_feas, test_labels)

    # # PCA 降维之后再分类
    DIM = 200
    print("use PCA, DIM = {}..".format(DIM))
    train_pca_feas, test_pca_feas = transform_to_pca_feas(
        train_feas, test_feas, DIM
    )
    train(train_pca_feas, train_labels, test_pca_feas, test_labels)

    # 基于SIFT特征的方法 sample 提取特征
    sample_img = (train_feas[0]*255).reshape(64, 64).astype(np.uint8)
    show_img(sample_img)
    key_points, desc = gen_sift_features(sample_img)
    show_sift_features(sample_img, key_points)

    K = 300
    train_sift_feas, test_sift_feas = transform_to_sift_feas(
        train_feas, test_feas, K
    )
    train(train_sift_feas, train_labels, test_sift_feas, test_labels)

    K = 600
    train_sift_feas, test_sift_feas = transform_to_sift_feas(
        train_feas, test_feas, K
    )
    train(train_sift_feas, train_labels, test_sift_feas, test_labels)


"""
准确率:

使用像素点作为特征:
Accuracy of LogisticRegression: 98.14%
Accuracy of SVM: 95.67%
Accuracy of RandomForest: 93.81%
Accuracy of KNN: 88.16%


PCA降维之后(200维):
use PCA, DIM = 200..
Accuracy of LogisticRegression: 98.14%
Accuracy of SVM: 91.33%
Accuracy of RandomForest: 90.94%
Accuracy of KNN: 86.30%


使用SIFT特征(k-means 300):
效果一般般
Accuracy of LogisticRegression: 59.52%
Accuracy of SVM: 60.29%
Accuracy of RandomForest: 41.02%
Accuracy of KNN: 23.92%

使用SIFT特征(k-means 600):
Accuracy of LogisticRegression: 77.79%
Accuracy of SVM: 76.86%
Accuracy of RandomForest: 50.46%
Accuracy of KNN: 14.32%

"""
