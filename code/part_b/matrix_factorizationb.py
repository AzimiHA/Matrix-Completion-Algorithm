from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

import numpy as np

#from utils import _load_student_meta


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss_m(data, u, z, mean, b_s, b_q, lambd):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0

    for i, q in enumerate(data["question_id"]):
        n = data["user_id"][i]
        loss += (data["is_correct"][i] - mean - b_s[n] - b_q[q] - np.sum(
            u[data["user_id"][i]] * z[q])) ** 2 + lambd * (
                            b_s[n] ** 2 + b_q[q] ** 2)
    return 0.5 * loss


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    u[n] = (u[n] + lr * (c - u[n].T.dot(z[q])) * (z[q]))
    z[q] = (z[q] + lr * (c - u[n].T.dot(z[q])) * (u[n]))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def update_u_z_m(train_data, lr, u, z, lambd, b_s, b_q, mean):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    # new formula
    b_s[n] = b_s[n] + lr * (
                c - mean - b_s[n] - b_q[q] - u[n].T.dot(z[q]) - lambd * b_s[n])
    b_q[q] = b_q[q] + lr * (
                c - mean - b_s[n] - b_q[q] - u[n].T.dot(z[q]) - lambd * b_q[q])
    u[n] = u[n] + lr * (
                (c - mean - b_s[n] - b_q[q] - u[n].T.dot(z[q])) * (z[q]))
    z[q] = z[q] + lr * (
                (c - mean - b_s[n] - b_q[q] - u[n].T.dot(z[q])) * (u[n]))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z, b_s, b_q


def als(train_data, k, lr, num_iteration, val_data):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    mat = None
    loss_train = np.zeros((int(num_iteration / 5000), 1))
    # loss_val = np.zeros((int(num_iteration / 5000), 1))
    for j in range(0, num_iteration):
        update_u_z(train_data, lr, u, z)
        if j % 5000 == 0:
            # print(j)
            loss_train[int(j / 5000)] = squared_error_loss(train_data, u, z)
            # loss_val[int(j / 5000)] = squared_error_loss(val_data, u, z)
    mat = u.dot(z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, loss_train


def alsmodified(train_data, k, lr, num_iteration, val_data, lambd,
                train_matrix):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    mat = None
    b_s = np.zeros((len(u)))
    b_q = np.zeros((len(z)))
    mean = np.nanmean(train_matrix)
    loss_train = np.zeros((int(num_iteration / 5000), 1))
    # loss_val = np.zeros((int(num_iteration / 5000), 1))
    for j in range(0, num_iteration):
        update_u_z_m(train_data, lr, u, z, lambd, b_s, b_q, mean)
        if j % 5000 == 0:
            # print(j)
            loss_train[int(j / 5000)] = squared_error_loss_m(train_data, u, z,
                                                             mean, b_s, b_q,
                                                             lambd)
            # loss_val[int(j / 5000)] = squared_error_loss_m(val_data, u, z,
            # mean,b_s,b_q,lambd)

    b_s = b_s.reshape(len(u), 1)
    b_q = b_q.reshape(1, len(z))
    mat = u.dot(z.T) + mean + b_s + b_q
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, loss_train


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    lr = 0.01  # 0.01 seems a good starting point to tune
    lambd = 0.02  # 0.02 seems a good starting point to tune
    num_iterations = 500000
    # k-value of 30 resulted in highest validation accuracy
    kvalues = [2, 4, 7, 10, 20, 30, 60]
    kvalues = [30]
    max = 0
    optimal_k = 0


    # This is to calculate the losses for the new model vs the old
    for k in kvalues:
        mat, loss = alsmodified(train_data, k, lr, num_iterations, val_data,
                                lambd, train_matrix)
        mat_o, loss_o = als(train_data, k, lr, num_iterations, val_data)

        #This is to determine the best k-value hyperparameter for the modified model
        '''
        print("ALS WITH Bias K Value: ", k, " Training Accuracy: ",
              sparse_matrix_evaluate(train_data, mat), " Validation Accuracy: ",
              sparse_matrix_evaluate(val_data, mat))
        '''

        x_axis = np.arange(0, num_iterations / 5000, 1)
        plt.plot(x_axis, loss, label="New Model Loss")
        plt.plot(x_axis, loss_o, label="Original Model Loss", color='orange')
        plt.xlabel("500000 Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss for ALS Model")
        plt.legend(loc="upper right")
        plt.show()



if __name__ == "__main__":
    main()
