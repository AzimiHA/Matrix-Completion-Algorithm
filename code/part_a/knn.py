from sklearn.impute import KNNImputer
from utils import *
#import chart_studio.plotly as ply
#import plotly.graph_objects as go


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    """print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)"""

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    # Q1(a)

    print("\nUser based knn:\n")
    k_list = [k for k in range(1, 27, 5)]
    train_accuracy = {}
    for k in k_list:
        print("\nk =", k, ":")
        train_accuracy[k] = knn_impute_by_user(sparse_matrix, val_data, k)

    """train_accuracy_plot = go.Figure(go.Scatter(x=k_list,
                                               y=list(train_accuracy.values()),
                                                    mode="lines+markers",
                                                    name="User based knn"))
    train_accuracy_plot.update_layout(title="KNN", width=800,
                                      height=600, xaxis=dict(
                                        tickmode="array",
                                        tickvals=k_list))
    train_accuracy_plot.update_xaxes(title_text="k")
    train_accuracy_plot.update_yaxes(title_text="accuracy")
    train_accuracy_plot.show()"""

    # Q1(b)

    best_k = max(train_accuracy, key=train_accuracy.get)
    print("\nthe best k is", best_k)
    test_accuracy = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print("\nthe test accuracy for k =", best_k)
    print(test_accuracy)

    # Q1(c)

    print("\nItem based knn:\n")
    train_accuracy_item = {}
    for k in k_list:
        print("\nk =", k, ":")
        train_accuracy_item[k] = knn_impute_by_item(sparse_matrix, val_data, k)

    """train_accuracy_plot.add_trace(go.Scatter(x=k_list,
                                        y=list(train_accuracy_item.values()),
                                               mode="lines+markers",
                                             name="Item based knn"))
    train_accuracy_plot.show()"""

    # Q1(d)

    best_k_item = max(train_accuracy_item, key=train_accuracy_item.get)
    print("\nthe best k is", best_k_item)
    test_accuracy_item = knn_impute_by_item(sparse_matrix, test_data,
                                            best_k_item)
    print("\nthe test accuracy for k =", best_k_item)
    print(test_accuracy_item)


    # Q1(e)


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
