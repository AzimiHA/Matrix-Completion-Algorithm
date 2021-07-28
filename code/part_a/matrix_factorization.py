from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

import numpy as np


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
    loss_val = np.zeros((int(num_iteration / 5000), 1))
    for j in range(0,num_iteration):
        update_u_z(train_data,lr,u,z)
        if j%5000==0:
            #print(j)
            loss_train[int(j / 5000)] = squared_error_loss(train_data, u, z)
            loss_val[int(j / 5000)] = squared_error_loss(val_data, u, z)
    mat = u.dot(z.T)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat,loss_val,loss_train


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    '''
    k_values = [2,5,10,30,50,100]
    for k in k_values:
        train_r = svd_reconstruct(train_matrix, k)
        temp=sparse_matrix_evaluate(val_data,train_r)
        print("K Value: ",k, " Accuracy: ", temp)
        #Best k value is a k-value of 5
    train_r = svd_reconstruct(train_matrix, 5)
    val_score = sparse_matrix_evaluate(val_data,train_r)
    test_score = sparse_matrix_evaluate(test_data,train_r)
    print("Validation Score: ", val_score)
    print("Test Score: ", test_score)
    '''

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    lr = 0.01 #0.01 seems a good starting point to tune
    num_iterations = 500000
    #k-value of 30 resulted in highest validation accuracy
    kvalues = [2,4,7,10,20,30,60]
    kvalues = [30]
    #This is for comparing the Losses for the two models for question 3(i)
    for k in kvalues:
        mat,loss_val,loss_train = als(train_data,k,lr,num_iterations,val_data)
        val_accuracy = sparse_matrix_evaluate(val_data, mat)
        train_accuracy = sparse_matrix_evaluate(train_data, mat)
        test_accuracy = sparse_matrix_evaluate(test_data, mat)
        print("K Value: ", k, "Validation Accuracy: ", val_accuracy, " Training Accuracy: ", train_accuracy, "Test Accuracy: ", test_accuracy)
        x_axis = np.arange(0, num_iterations / 5000, 1)
        plt.plot(x_axis, loss_val, label="Validation Loss")
        plt.plot(x_axis, loss_train, label="Training Loss", color='orange')
        plt.xlabel("5000 Iterations")
        plt.ylabel("Loss")
        plt.title("Loss for ALS Model")
        plt.legend(loc="upper right")
        plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
