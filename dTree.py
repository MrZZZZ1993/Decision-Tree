from process_data import *
from calculate_score import *
import pandas as pd
from decision_tree import *


def final(load_path, full_sample, depth, gain_ratio):

    load_data = load_dataset(load_path)
    title_type, all_data = get_title_data(load_data)

    np.random.shuffle(all_data)

    titles = [t[0] for t in title_type]
    train_data_set, valid_data_set = stratified_cross_validation(all_data, 5)
    length = 1 if full_sample == 1 else 5

    accuracies = []
    sizes = []
    depths = []
    first_features = []

    print('training...')
    for i in range(length):
        thresholds = find_all_thresholds(train_data_set[i], title_type, 'thresholds.csv')
        train_data = continuous_2_nominal(train_data_set[i], thresholds)
        valid_data = continuous_2_nominal(valid_data_set[i], thresholds)

        np.random.shuffle(train_data)
        np.random.shuffle(valid_data)

        t = creat_decision_tree(train_data, title_type.copy(), gain_ratio, depth, 1)

        num_correct = 0
        num_sum = len(valid_data)
        for d in valid_data:
            res = classify(t, titles, d[:-1])
            if res == d[-1]:
                num_correct += 1

        accuracies.append(num_correct/num_sum)
        sizes.append(get_node_num(t))
        depths.append(get_depth(t))
        first_features.append(list(t.keys())[0])

    return sum(accuracies)/len(accuracies), sizes, depths, first_features


def show(path, is_full_sample, maximum_depth, is_gain_ratio):
    print('resources: ', path)
    if is_full_sample == 0:
        print('training set: CV')
    else:
        print('training set: full samples')
    print('maximum depth: ', maximum_depth)
    if is_gain_ratio == 0:
        print('split criterion: information gain')
    else:
        print('split criterion: gain ration')

    accuracy, size, depth, first_feature = final(path, is_full_sample, maximum_depth, is_gain_ratio)

    print()
    print()
    print('Accuracy: ', accuracy)
    print('Size: ')
    for i in range(len(size)):
        print('    Tree ', i + 1, ': ', size[i])
    print('Maximum depth: ')
    for i in range(len(size)):
        print('    Tree ', i + 1, ': ', depth[i])
    print('First Feature: ')
    for i in range(len(first_feature)):
        print('    Tree ', i + 1, ': ', first_feature[i])

    print()
    print()


if __name__ == '__main__':

    # path of data
    path = '/440data/voting'
    # 0: cross validation, 1: full samples
    is_full_sample = 1
    # set the maximum depth of the tree
    maximum_depth = 32
    # 0: information gain, 1: gain ratio
    is_gain_ratio = 0

    show(path, is_full_sample, maximum_depth, is_gain_ratio)


