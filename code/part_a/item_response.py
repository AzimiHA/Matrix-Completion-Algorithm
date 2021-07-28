from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    l = sigmoid(np.array(theta[data['user_id']]) -
                np.array(beta[data['question_id']]))

    ll = np.array(data['is_correct']) * np.log(l) + (1-np.array(data['is_correct']) * np.log(1-l))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -ll.sum()


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    student_entries = data['user_id']
    question_entries = data['question_id']
    # Assign params
    theta_entries = np.array(theta[student_entries])
    beta_entries = np.array(beta[question_entries])
    ll = sigmoid(theta_entries - beta_entries)
    theta = theta + lr * np.bincount(data['user_id'], np.array(data['is_correct']) - ll)

    theta_entries = np.array(theta[student_entries])
    ll = sigmoid(theta_entries - beta_entries)
    beta = beta + lr * np.bincount(data['question_id'], ll - np.array(data['is_correct']))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.ones(len(np.unique(data['user_id'])))
    beta = np.ones(len(np.unique(data['question_id'])))

    val_acc_lst = []
    train_ll_lst = []
    val_ll_lst = []
    iter = []
    for i in range(iterations):
        train_ll_lst.append(neg_log_likelihood(data, theta=theta, beta=beta))
        val_ll_lst.append(neg_log_likelihood(val_data, theta=theta, beta=beta))
        score = evaluate(val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        iter.append(i)
        theta, beta = update_theta_beta(data, lr, theta, beta)

    """plt.figure(0)
    plt.plot(iter, train_ll_lst)
    plt.plot(iter, val_ll_lst)
    plt.show()"""
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.47)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("../data")
    # using dictionaries for better runtime
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    print("start")
    # Params
    lr = 0.01
    iterations = 400
    theta, beta, val_acc_lst = irt(train_data, val_data, lr, iterations)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print("end")
    print("the final validation accuracy is \n", val_acc)
    print("the final test accuracy is \n", test_acc)
    """plt.figure(1)
    for i in range(5):
        prob = []
        thetas = []
        for student in theta:
            prob.append(sigmoid(student - beta[i]))
            thetas.append(student)
        thetas.sort()
        prob.sort()
        plt.plot(thetas, prob)
    plt.show()"""

    p = len(np.unique(train_data['user_id']))
    q = len(np.unique(train_data['question_id']))
    mat = np.zeros((p, q))
    for i in range(p):
        for j in range(q):
            mat[i, j] = sigmoid((theta[i] - beta[j]).sum())

    """private_test = load_private_test_csv("../data")
    is_c = sparse_matrix_predictions(private_test, mat)
    private_test["is_correct"] = is_c
    save_private_test_csv(private_test,
                          "/Users/wu/Desktop/private_test_result.csv")"""

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
