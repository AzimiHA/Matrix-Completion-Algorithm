# TODO: complete this file.
from part_a.knn import *
from part_a.matrix_factorization import *
from part_a.item_response import *
import matplotlib.pyplot as plt


def als_helper(train_data, k, lr, num_iteration, val_data, p, q):
    """ Performs ALS algorithm. Return reconstructed matrix."""
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(p, k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(q, k))
    for j in range(0, num_iteration):
        update_u_z(train_data, lr, u, z)
    mat = u.dot(z.T)
    return mat


def irt_helper(data, lr, iterations, p, q):
    """ Train IRT model."""
    theta = np.ones(p)
    beta = np.ones(q)
    for i in range(iterations):
        theta, beta = update_theta_beta(data, lr, theta, beta)
    return theta, beta


def get_matrix_knn_user(train_sparse, num_bootstrap: int, k: int, val_data):
    """train user-based knn model by bootstrapping"""
    n = train_sparse.shape[1]
    matrix_list = []
    val_acc_list = []
    for rep in range(num_bootstrap):
        rdm_index_n = np.random.choice(n, size=n, replace=True)
        rdm_sample = train_sparse[:, rdm_index_n]
        nbrs = KNNImputer(n_neighbors=k)
        mat = nbrs.fit_transform(rdm_sample)
        matrix_list.append(mat)
        print("knn_user rep", rep + 1, "done")
        acc_temp = sparse_matrix_evaluate(val_data, sum(matrix_list) / (rep+1))
        print("val_acc:", acc_temp)
        val_acc_list.append(acc_temp)
    matrix = sum(matrix_list) / num_bootstrap
    return matrix


def get_matrix_knn_item(train_sparse, num_bootstrap: int, k: int, val_data):
    """train item-based knn model by bootstrapping"""
    n = train_sparse.shape[1]
    matrix_list = []
    val_acc_list = []
    for rep in range(num_bootstrap):
        rdm_index_n = np.random.choice(n, size=n, replace=True)
        rdm_sample = train_sparse[:, rdm_index_n]
        nbrs = KNNImputer(n_neighbors=k)
        mat = nbrs.fit_transform(rdm_sample.T).T
        matrix_list.append(mat)
        print("knn_item rep", rep + 1, "done")
        acc_temp = sparse_matrix_evaluate(val_data,
                                          sum(matrix_list) / (rep + 1))
        print("val_acc:", acc_temp)
        val_acc_list.append(acc_temp)
    matrix = sum(matrix_list) / num_bootstrap
    return matrix, val_acc_list


def get_matrix_als(train_data, k, lr, num_iteration: int,
                   num_bootstrap: int, val_data):
    """train ALS by bootstrapping"""
    p = len(np.unique(train_data['user_id']))
    q = len(np.unique(train_data['question_id']))
    n = len(train_data["user_id"])
    matrix_list = []
    val_acc_list = []
    for rep in range(num_bootstrap):
        rdm_index = np.random.choice(n, size=n, replace=True)
        rdm_sample = {"user_id":
                          np.array(train_data["user_id"])[rdm_index],
                      "question_id":
                          np.array(train_data["question_id"])[rdm_index],
                      "is_correct":
                          np.array(train_data["is_correct"])[rdm_index]}
        mat = als_helper(rdm_sample, k, lr, num_iteration, val_data, p, q)
        matrix_list.append(mat)
        print("als rep", rep + 1, "done")
        acc_temp = sparse_matrix_evaluate(val_data,
                                          sum(matrix_list) / (rep + 1))
        print("val_acc:", acc_temp)
        val_acc_list.append(acc_temp)
    matrix = sum(matrix_list) / num_bootstrap
    return matrix, val_acc_list


def get_matrix_svd(matrix, k, num_bootstrap: int, val_data):
    """train svd model"""
    n = matrix.shape[0]
    matrix_list = []
    val_acc_list = []
    for rep in range(num_bootstrap):
        rdm_index = np.random.choice(n, size=n, replace=True)
        rdm_sample = matrix[rdm_index, :]
        mat = svd_reconstruct(rdm_sample, k)
        matrix_list.append(mat)
        print("svd rep", rep + 1, "done")
        acc_temp = sparse_matrix_evaluate(val_data,
                                          sum(matrix_list) / (rep + 1))
        print("val_acc:", acc_temp)
        val_acc_list.append(acc_temp)
    matrix = sum(matrix_list) / num_bootstrap
    return matrix, val_acc_list


def get_matrix_irt(data, lr, iterations, num_bootstrap: int, val_data):
    """train IRT model"""
    p = len(np.unique(data['user_id']))
    q = len(np.unique(data['question_id']))
    n = len(data["user_id"])
    val_acc_list = []
    matrix_list = []
    for rep in range(num_bootstrap):
        rdm_index = np.random.choice(n, size=n, replace=True)
        rdm_sample = {"user_id":
                          np.array(data["user_id"])[rdm_index],
                      "question_id":
                          np.array(data["question_id"])[rdm_index],
                      "is_correct":
                          np.array(data["is_correct"])[rdm_index]}
        theta, beta = irt_helper(rdm_sample, lr, iterations, p, q)
        mat = np.zeros((p, q))
        for i in range(p):
            for j in range(q):
                mat[i, j] = sigmoid((theta[i] - beta[j]).sum())
        matrix_list.append(mat)
        print("irt rep", rep + 1, "done")
        acc_temp = sparse_matrix_evaluate(val_data,
                                          sum(matrix_list) / (rep + 1))
        print("val_acc:", acc_temp)
        val_acc_list.append(acc_temp)
    matrix = sum(matrix_list) / num_bootstrap
    return matrix, val_acc_list


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    train_data = load_train_csv("../data")

    # hyperparameters
    # todo: tune these hyperparameters
    n = 50

    k_knn= [7, 11, 15]

    als_lr = [0.01, 0.015, 0.02, 0.03, 0.05]
    als_num_iteration = [10000, 100000, 300000, 500000]
    als_k = [2, 5, 10, 20, 30]

    svd_k = 5

    irt_lr = [0.001, 0.002, 0.01, 0.015, 0.02]
    irt_iteration = [50, 100, 1000, 10000]

    # emsemble
    # todo: select 3 models

    m1, acc_m1 = get_matrix_irt(train_data, 0.01, 300,
                                20, val_data)
    m2, acc_m2 = get_matrix_irt(train_data, 0.01, 300,
                                20, val_data)
    m3, acc_m3 = get_matrix_irt(train_data, 0.01, 300,
                                20, val_data)

    model_avg = (m1 + m2 + m3) / 3
    acc = sparse_matrix_evaluate(val_data, model_avg)
    print("\nAccuracy on validation set: ")
    print(acc)
    test_acc = sparse_matrix_evaluate(test_data, model_avg)
    print("\nAccuracy on test set: ")
    print(test_acc)
    plt.plot(acc_m1)
    plt.plot(acc_m2)
    plt.plot(acc_m3)
    plt.legend(['m1', 'm2', 'm3'], loc='upper left')
    plt.show()


    """private_test = load_private_test_csv("../data")
    is_c = sparse_matrix_predictions(private_test, model_avg)
    private_test["is_correct"] = is_c
    save_private_test_csv(private_test,
                          "/Users/wu/Desktop/private_test_result.csv")"""



if __name__ == "__main__":
    main()



