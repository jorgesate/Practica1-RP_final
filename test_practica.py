# Author: Jose' Miguel Buenaposada (josemiguel.buenaposada@urjc.es)

import numpy as np
import pylab as plt
import matplotlib
import math
from matplotlib.colors import ListedColormap
from sklearn import metrics, covariance
from GMMBayes import *
from GaussianBayes import *
from KNNClassifier import *
# from ParzenClassifier import *
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, range(cm.shape[0]))
    plt.yticks(tick_marks, range(cm.shape[0]))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ax = plt.gca()
    width = cm.shape[1]
    height = cm.shape[0]

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[y, x]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')


def compute_estimated_labels_map(clf, data_range):
    h = .5  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max, y_min, y_max = data_range

    # x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    # y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    z = z.reshape(xx.shape)

    return xx, yy, z


def plot_classification_results(clf, data_range, X, y=None, title=None):
    xx, yy, z = compute_estimated_labels_map(clf, data_range)
    x_min, x_max, y_min, y_max = data_range

    if y is None:
        y = clf.predict(X)

    cm = plt.cm.get_cmap('jet')
    plt.contourf(xx, yy, z, cmap=cm, alpha=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X')
    plt.ylabel('Y')
    if title is not None:
        plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm)


def sample_from_gmm(n_samples, gmm):
    # n_samples is a list with the number of data to sample per class.
    #
    # gmm is a list of dictionaries, each class is a dictionary
    # with three fields:
    #   - 'covs', a numpy array (n_features x n_features x n_gaussians)
    #   - 'means', a numpy array (n_gaussians x n_features)
    #   - 'weights', a numpy array (n_gaussians x 1)
    #   - 'prior', a float with class prior.
    #
    # Returns (X, y) where X[c] is a numpy array n_samples[c] x d for class c
    #                and y[c] ia a numpy array with labels for data from class c

    X = []
    y = []
    n_classes = len(gmm)
    for c in range(n_classes):
        class_c = gmm[c]
        weights_ = class_c['weights']
        n_gaussians = len(weights_)
        means_ = class_c['means']
        covs_ = class_c['covs']
        prior_ = class_c['prior']
        X_c = []
        for g in range(n_gaussians):
            if n_gaussians > 1:
                mean_ = means_[g, :]
                cov_ = covs_[:, :, g]
            else:
                mean_ = means_
                cov_ = covs_
            X_c.append(np.random.multivariate_normal(mean_, cov_, int(n_samples[c] * weights_[g])))

        X_c = np.concatenate(X_c, axis=0)
        X.append(X_c)
        y.append(c * np.ones((X_c.shape[0], 1)))

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y)

    return X, y


def rotMatrix(angle, radians=False):
    if not radians:
        angle = np.deg2rad(angle)

    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])

    return R


def gmm_classes_1(D):
    # Classes is a list of dictionaries, each class is a dictionary
    # with three fields:
    #   - 'covs', a numpy array (n_features x n_features x n_gaussians)
    #   - 'means', a numpy array (n_gaussians x n_features)
    #   - 'weights', a numpy array (n_gaussians x 1)
    #   - 'prior', a float with class prior.
    VAR = D
    classes = []
    R = rotMatrix(45)

    # Class number 0
    class_dict = {}
    class_dict['covs'] = np.zeros((2, 2, 1))
    cov = np.array([[VAR, 0],
                    [0, VAR / 8]])
    class_dict['covs'] = np.dot(R.T, np.dot(cov, R))
    class_dict['means'] = np.array([D / 2, D / 2])
    class_dict['weights'] = np.array([1.])
    class_dict['prior'] = 1. / 4.
    classes.append(class_dict)

    # Class number 1
    class_dict = {}
    class_dict['covs'] = np.zeros((2, 2, 1))
    cov = np.array([[VAR / 2, 0],
                    [0, VAR / 8]])
    class_dict['covs'] = np.dot(R.T, np.dot(cov, R))
    class_dict['means'] = np.array([2 * D / 6, 2 * D / 6])
    class_dict['weights'] = np.array([1.])
    class_dict['prior'] = 1. / 4.
    classes.append(class_dict)

    # Class number 2
    class_dict = {}
    class_dict['covs'] = np.zeros((2, 2, 1))
    cov = np.array([[VAR / 2, 0],
                    [0, VAR / 8]])
    class_dict['covs'] = np.dot(R.T, np.dot(cov, R))
    class_dict['means'] = np.array([4 * D / 6, 4 * D / 6])
    class_dict['weights'] = np.array([1.])
    class_dict['prior'] = 1. / 4.
    classes.append(class_dict)

    # Class number 3 (it is a mixture of  gaussians around a circle).
    class_dict = {}
    num_gaussians = 8
    radius = 2.5 * D / 6.
    angle = 0.
    angle_inc = 2 * math.pi / num_gaussians
    cov_angle = 0.
    covs_ = np.zeros((2, 2, num_gaussians))
    means_ = np.zeros((num_gaussians, 2))
    weights_ = np.zeros((num_gaussians, 1))

    for j in range(num_gaussians):
        mu_x = D / 2. + radius * np.cos(angle)
        mu_y = D / 2. + radius * np.sin(angle)
        means_[j, :] = np.array([mu_y, mu_x])

        cov = np.array([[VAR, 0],
                        [0, VAR / 2.]])
        R = rotMatrix(np.rad2deg(cov_angle))
        covs_[:, :, j] = np.dot(R.T, np.dot(cov, R))

        angle += angle_inc
        cov_angle += angle_inc
        weights_[j] = 1. / num_gaussians

    class_dict['covs'] = covs_
    class_dict['means'] = means_
    class_dict['weights'] = weights_
    class_dict['prior'] = 1. / 4.
    classes.append(class_dict)

    return classes


def gmm_classes_2(D):
    # Classes is a list of dictionaries, each class is a dictionary
    # with three fields:
    #   - 'covs', a numpy array (n_features x n_features x n_gaussians)
    #   - 'means', a numpy array (n_gaussians x n_features)
    #   - 'weights', a numpy array (n_gaussians x 1)
    #   - 'prior', a float with class prior.
    classes = []

    R = rotMatrix(45)

    # Class number 0 (it is a mixture of  gaussians around a circle).
    VAR = D / 10
    class_dict = {}
    num_gaussians = 8
    radius = D / 5.
    angle = 0.
    angle_inc = 2 * math.pi / num_gaussians
    cov_angle = 0.
    covs_ = np.zeros((2, 2, num_gaussians))
    means_ = np.zeros((num_gaussians, 2))
    weights_ = np.zeros((num_gaussians, 1))

    for j in range(num_gaussians):
        mu_x = D / 2. + radius * np.cos(angle)
        mu_y = D / 2. + radius * np.sin(angle)
        means_[j, :] = np.array([mu_y, mu_x])

        cov = np.array([[3 * VAR, 0],
                        [0, VAR]])
        R = rotMatrix(np.rad2deg(cov_angle))
        covs_[:, :, j] = np.dot(R.T, np.dot(cov, R))

        angle += angle_inc
        cov_angle += angle_inc
        weights_[j] = 1. / num_gaussians

    class_dict['covs'] = covs_
    class_dict['means'] = means_
    class_dict['weights'] = weights_
    class_dict['prior'] = 1. / 3.
    classes.append(class_dict)

    # Class number 2 (it is a mixture of  gaussians around a circle).
    VAR = D / 10
    class_dict = {}
    num_gaussians = 8
    radius = 2.5 * D / 6.
    angle = 0.
    angle_inc = 2 * math.pi / num_gaussians
    cov_angle = 0.
    covs_ = np.zeros((2, 2, num_gaussians))
    means_ = np.zeros((num_gaussians, 2))
    weights_ = np.zeros((num_gaussians, 1))

    for j in range(num_gaussians):
        mu_x = D / 2. + radius * np.cos(angle)
        mu_y = D / 2. + radius * np.sin(angle)
        means_[j, :] = np.array([mu_y, mu_x])

        cov = np.array([[4 * VAR, 0],
                        [0, VAR]])
        R = rotMatrix(np.rad2deg(cov_angle))
        covs_[:, :, j] = np.dot(R.T, np.dot(cov, R))

        angle += angle_inc
        cov_angle += angle_inc
        weights_[j] = 1. / num_gaussians

    class_dict['covs'] = covs_
    class_dict['means'] = means_
    class_dict['weights'] = weights_
    class_dict['prior'] = 1. / 3.
    classes.append(class_dict)

    # Class number 3 (it is a mixture of 3 gaussians).
    VAR = D / 10
    class_dict = {}
    num_gaussians = 3
    covs_ = np.zeros((2, 2, num_gaussians))
    means_ = np.zeros((num_gaussians, 2))
    weights_ = np.zeros((num_gaussians, 1))
    cov = np.array([[VAR, 0],
                    [0, VAR]])
    covs_[:, :, 0] = cov
    means_[0, :] = np.array([D / 2, D / 2])
    covs_[:, :, 1] = cov
    means_[1, :] = np.array([7 * (D / 8.), 7 * (D / 8.)])
    covs_[:, :, 2] = cov
    means_[2, :] = np.array([D / 8., D / 8.])
    class_dict['covs'] = covs_
    class_dict['means'] = means_
    class_dict['weights'] = np.array([1. / 3., 1. / 3., 1. / 3.])
    class_dict['prior'] = 1. / 3.
    classes.append(class_dict)

    return classes


def gmm_classes_3(D):
    # Classes is a list of dictionaries, each class is a dictionary
    # with three fields:
    #   - 'covs', a numpy array (n_features x n_features x n_gaussians)
    #   - 'means', a numpy array (n_gaussians x n_features)
    #   - 'weights', a numpy array (n_gaussians x 1)
    #   - 'prior', a float with class prior.
    classes = []

    R = rotMatrix(45)

    # Class number 0 (it is a mixture of  gaussians around a circle).
    VAR = D / 10
    class_dict = {}
    num_gaussians = 8
    radius = D / 5.
    angle = 0
    #    angle_inc = 2 * math.pi / num_gaussians
    angle_inc = 1.2 * math.pi / num_gaussians
    cov_angle = 0.
    covs_ = np.zeros((2, 2, num_gaussians))
    means_ = np.zeros((num_gaussians, 2))
    weights_ = np.zeros((num_gaussians, 1))

    for j in range(num_gaussians):
        mu_x = (D / 10.) + (D / 2.) + radius * np.cos(angle)
        mu_y = (D / 10.) + (D / 2.) + radius * np.sin(angle)
        means_[j, :] = np.array([mu_y, mu_x])

        cov = np.array([[3 * VAR, 0],
                        [0, VAR]])
        R = rotMatrix(np.rad2deg(cov_angle))
        covs_[:, :, j] = np.dot(R.T, np.dot(cov, R))

        angle += angle_inc
        cov_angle += angle_inc
        weights_[j] = 1. / num_gaussians

    class_dict['covs'] = covs_
    class_dict['means'] = means_
    class_dict['weights'] = weights_
    class_dict['prior'] = 1. / 2.
    classes.append(class_dict)

    ## Class number 1 (it is a mixture of  gaussians around a circle).
    VAR = D / 10
    class_dict = {}
    num_gaussians = 8
    radius = D / 5.
    angle = 0.8 * math.pi
    #    angle_inc = 2 * math.pi / num_gaussians
    angle_inc = 1.2 * math.pi / num_gaussians
    cov_angle = 0.
    covs_ = np.zeros((2, 2, num_gaussians))
    means_ = np.zeros((num_gaussians, 2))
    weights_ = np.zeros((num_gaussians, 1))

    for j in range(num_gaussians):
        mu_x = (D / 2.) - (D / 10.) + radius * np.cos(angle)
        mu_y = (D / 2.) - (D / 10.) + radius * np.sin(angle)
        means_[j, :] = np.array([mu_y, mu_x])

        cov = np.array([[3 * VAR, 0],
                        [0, VAR]])
        R = rotMatrix(np.rad2deg(cov_angle))
        covs_[:, :, j] = np.dot(R.T, np.dot(cov, R))

        angle += angle_inc
        cov_angle += angle_inc
        weights_[j] = 1. / num_gaussians

    class_dict['covs'] = covs_
    class_dict['means'] = means_
    class_dict['weights'] = weights_
    class_dict['prior'] = 1. / 2.
    classes.append(class_dict)

    ## Class number 1 (it is a mixture of  gaussians around a circle).
    VAR = D / 10
    class_dict = {}
    num_gaussians = 8
    radius = 2. * D / 5.
    angle = 0.8 * math.pi
    #    angle_inc = 2 * math.pi / num_gaussians
    angle_inc = 1.2 * math.pi / num_gaussians
    cov_angle = 0.
    covs_ = np.zeros((2, 2, num_gaussians))
    means_ = np.zeros((num_gaussians, 2))
    weights_ = np.zeros((num_gaussians, 1))

    for j in range(num_gaussians):
        mu_x = (D / 2.) - (D / 10.) + radius * np.cos(angle)
        mu_y = (D / 2.) - (D / 10.) + radius * np.sin(angle)
        means_[j, :] = np.array([mu_y, mu_x])

        cov = np.array([[3 * VAR, 0],
                        [0, VAR]])
        R = rotMatrix(np.rad2deg(cov_angle))
        covs_[:, :, j] = np.dot(R.T, np.dot(cov, R))

        angle += angle_inc
        cov_angle += angle_inc
        weights_[j] = 1. / num_gaussians

    class_dict['covs'] = covs_
    class_dict['means'] = means_
    class_dict['weights'] = weights_
    class_dict['prior'] = 1. / 2.
    classes.append(class_dict)

    return classes


def main(GAUSSIANS):
    # Set an square size on the feature space in order to
    # generate ground thruth distribution of classes on it.
    square_size = 50.

    # Choose the ground thruth distribution of the classes
    if GAUSSIANS == 1:  # 4 classess
        real_classes = gmm_classes_1(square_size)  # , var_data)
        n_samples_train = np.array([200, 200, 200, 200])
        n_samples_test = [1000, 1000, 1000, 1000]
        exp_name = 'Gaussians 1'
    elif GAUSSIANS == 2:  # 3 clasess
        real_classes = gmm_classes_2(square_size)  # , var_data)
        n_samples_train = np.array([200, 200, 200])
        n_samples_test = [1000, 1000, 1000]
        exp_name = 'Gaussians 2'
    elif GAUSSIANS == 3:  # 3 clasess
        real_classes = gmm_classes_3(square_size)  # , var_data)
        n_samples_train = np.array([200, 200, 200])
        n_samples_test = [1000, 1000, 1000]
        exp_name = 'Gaussians 3 - Balanced'
    elif GAUSSIANS == 4:  # 3 clases
        real_classes = gmm_classes_3(square_size)  # , var_data)
        n_samples_train = np.array([100, 200, 700]) #np.array([100, 200, 700])
        n_samples_test = [1000, 1000, 1000]
        exp_name = 'Gaussians 3 - Unbalanced'

    # Create a classifier that uses the real classes distribution without training
    clf = GMMBayes(real_classes)

    # Sample data from the real classes distributions to train and test..
    X_train, y_train = sample_from_gmm(n_samples_train, real_classes)
    X_test, y_test = sample_from_gmm(n_samples_test, real_classes)

    # x_min, x_max, y_min, y_max and the sampled points
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    data_range = (x_min, x_max, y_min, y_max)
    xx, yy, z = compute_estimated_labels_map(clf, data_range)

    # Plot the classification ground truth Bayes min error boundaries.
    fig0 = plt.figure(0)
    mytitle = exp_name + ' real class boundaries + Training samples'
    fig0.canvas.set_window_title(mytitle)
    plt.subplot(1, 2, 1)
    plot_classification_results(clf, data_range, X_train, y=y_train, title=mytitle)
    plt.subplot(1, 2, 2)
    mytitle = exp_name + ' real class boundaries + Testing samples'
    plot_classification_results(clf, data_range, X_test, y=y_test, title=mytitle)
    fig0.show()

    # ----------------------------------------------------------------
    # Gaussians based Bayes classifier training and evaluation
    gaussBayes = GaussianBayes()
    gaussBayes.fit(X_train, y_train)
    predicted_bayes = gaussBayes.predict(X_test)

    conf_matrix_gauss = metrics.confusion_matrix(y_test, predicted_bayes)

    fig1 = plt.figure(1)
    mytitle = exp_name + ': Gaussian Bayes classifier (Test)'
    fig1.canvas.set_window_title(mytitle)
    plt.subplot(1,2,1)
    plot_classification_results(gaussBayes, data_range, X_test, y=y_test, title=mytitle)
    plt.subplot(1,2,2)
    plot_confusion_matrix(conf_matrix_gauss, cmap=plt.cm.get_cmap('jet'))
    fig1.show()
    print ('Gaussian Bayes (Test) Done. Score: {:.2f}'.format(gaussBayes.score(X_test, y_test)))

    # ---------------------------------------------------------------------
    # KNN Classifier

    KNN = KNNClassifier(80)
    KNN.fit(X_train, y_train.ravel())
    predicted_KNN = KNN.predict(X_test)

    conf_matrix_KNN = metrics.confusion_matrix(y_test, predicted_KNN)

    fig2 = plt.figure(2)
    mytitle = exp_name + ': KNN classifier'
    fig2.canvas.set_window_title(mytitle)
    plt.subplot(1,2,1)
    plot_classification_results(KNN, data_range, X_test, y=y_test, title=mytitle)
    plt.subplot(1,2,2)
    plot_confusion_matrix(conf_matrix_KNN, cmap=plt.cm.get_cmap('jet'))
    fig2.show()

    print ('KNN Done. Score: {:.2f}  K = {:d}'.format(KNN.score(X_test, y_test), KNN.k_best))

#-----------------------------------------------------------------------
    # PARTE A TERMINAR EN LA PRACTICA
    #-----------------------------------------------------------------------

    # GMM ==> Falta terminar de implementar GMMBayes.fit y evaluar
    #                  (BUSCANDO PARAMETROS POR VAL. CRUZADA 5-fold!  - en este caso nu'mero
    #                   componentes Gaussianas de la mezcla -)

    # Parzen ==> Implentar y evaluar (BUSCANDO PARAMETROS POR VAL. CRUZADA 5-fold)
    
    # K-NN ==> Implentar y evaluar (BUSCANDO PARAMETROS POR VAL. CRUZADA 5-f/old)

    plt.figure()
    plt.show()

if __name__ == "__main__":

    # Cambiar este valor de 1 a 4 dependiendo del experimento en el que
    # queramos probar nuestros clasificadores.
    GAUSSIANS = 1

    main(GAUSSIANS)
