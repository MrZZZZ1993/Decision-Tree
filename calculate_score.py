import math
import numpy as np
import pandas as pd


def entropy(dataset, col):
    classes = {}

    for data in dataset:
        label = data[col]
        classes[label] = 1 if label not in classes else classes[label]+1

    #print(classes)

    sum_num = len(dataset)

    m_entropy = 0.0
    for label in classes:
        probability = float(classes[label])/sum_num
        m_entropy -= probability * math.log2(probability)

    return m_entropy


def gini(dataset, col):
    classes = {}

    for data in dataset:
        label = data[col]
        classes[label] = 1 if label not in classes else classes[label]+1

    sum_num = len(dataset)

    m_entropy = 1.0
    for label in classes:
        probability = float(classes[label])/sum_num
        m_entropy -= probability * probability

    return m_entropy


def information_gain(dataset, col):
    h_y = entropy(dataset, -1)

    classes = {}
    for data in dataset:
        x = data[col]
        if x not in classes:
            classes[x] = [data]
        else:
            classes[x].append(data)

    #print(classes)

    h_y_x = 0.0
    sum_num = len(dataset)

    for x in classes:
        probability = float(len(classes[x])) / sum_num
        h_y_x += probability * entropy(classes[x], -1)

    return h_y - h_y_x


def gain_ratio(dataset, col):
    e = entropy(dataset, col)
    if e == 0:
        return 0
    return information_gain(dataset, col)/entropy(dataset, col)


def find_threshold(np_dataset, col):
    # np_dataset = np.array(dataset)
    np_col_y = np_dataset[:, [col, -1]]
    np_col_y = np_col_y[np.argsort(np_col_y[:, 0])]

    thresholds = set()

    for i in range(1, len(np_col_y)):
        if np_col_y[i][-1] != np_col_y[i - 1][-1]:
            thresholds.add((np_col_y[i][0] + np_col_y[i - 1][0]) / 2)

    thresholds = list(thresholds)

    temp = np_col_y.copy()
    max_info_gain = float('-inf')
    max_threshold = 0.0
    for threshold in thresholds:
        # print(threshold)
        temp[:, 0][temp[:, 0] > threshold] = 1
        temp[:, 0][temp[:, 0] != 1] = 0

        ig = information_gain(temp, 0)
        #ig = gini(temp, 0)
        # print(ig)

        if ig > max_info_gain:
            max_info_gain = ig
            max_threshold = threshold
        temp = np_col_y.copy()

    return max_threshold


def find_all_thresholds(dataset, title, name):

    dataset = np.array(dataset)

    thresholds = [float('inf')] * len(dataset[0])
    for col in range(len(title)):
        if title[col][1] == 'CONTINUOUS':
            threshold = find_threshold(dataset, col)
            thresholds[col] = threshold

    pd_thresholds = pd.DataFrame(thresholds)
    pd_thresholds.to_csv(name, index=False)

    return thresholds


def continuous_2_nominal(dataset, thresholds):
    dataset = np.array(dataset)

    for col in range(len(thresholds)):
        if thresholds[col] != float('inf'):
            threshold = thresholds[col]
            dataset[:, col][dataset[:, col] > threshold] = 1
            dataset[:, col][dataset[:, col] != 1] = 0
            thresholds[col] = threshold

    return dataset.tolist()


def find_attribute_gini(dataset):
    len_attr = len(dataset[0])-1
    max_info_gain = float('-inf')
    max_col = 0
    for col in range(len_attr):
        info_gain = gini(dataset, col)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_col = col

    return max_col


def find_attribute_info_gain(dataset):
    len_attr = len(dataset[0])-1
    max_info_gain = float('-inf')
    max_col = 0
    for col in range(len_attr):
        info_gain = information_gain(dataset, col)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_col = col

    return max_col


def find_attribute_gain_ratio(dataset):
    len_attr = len(dataset[0]) - 1
    max_gain_ratio = float('-inf')
    max_col = 0
    for col in range(len_attr):
        info_gain = gain_ratio(dataset, col)
        if info_gain > max_gain_ratio:
            max_gain_ratio = info_gain
            max_col = col

    return max_col





