from process_data import *
from calculate_score import *


def creat_decision_tree(dataset, title, score_type, max_depth, curr):

    y = [data[-1] for data in dataset]
    if len(y) == y.count(y[0]):
        return y[0]

    if len(dataset[0]) == 1 or curr > max_depth != 0:
        return count_majority(dataset)

    best_attr = find_attribute_info_gain(dataset) if score_type == 0 else find_attribute_gain_ratio(dataset)
    #best_attr = find_attribute_gini(dataset)

    best_title = title[best_attr][0]

    del(title[best_attr])

    m_tree = {best_title: {}}

    unique_attributes = get_unique_class(dataset, best_attr)

    for attr in unique_attributes:
        m_tree[best_title][attr] = creat_decision_tree(split_dataset(dataset, best_attr, attr), title.copy(),
                                                       score_type, max_depth, curr+1)

    return m_tree


def classify(decision_tree, title, test_item):
    first_title = list(decision_tree.keys())[0]
    sub_tree = decision_tree[first_title]
    first_title_idx = title.index(first_title)

    res = ''
    for k in sub_tree.keys():
        if test_item[first_title_idx] == k:
            if type(sub_tree[k]).__name__ == 'dict':
                res = classify(sub_tree[k], title, test_item)
            else:
                res = sub_tree[k]

    return res


def get_node_num(decision_tree):
    num = 1

    first_title = list(decision_tree.keys())[0]
    sub_tree = decision_tree[first_title]

    for k in sub_tree.keys():
        if type(sub_tree[k]).__name__ == 'dict':
            num += get_node_num(sub_tree[k]) + 1
        else:
            num += 1

    return num


def get_depth(decision_tree):
    num = 1

    first_title = list(decision_tree.keys())[0]
    sub_tree = decision_tree[first_title]

    for k in sub_tree.keys():
        if type(sub_tree[k]).__name__ == 'dict':
            num_t = get_depth(sub_tree[k]) + 1
        else:
            num_t = 1

        num = max(num, num_t)

    return num

