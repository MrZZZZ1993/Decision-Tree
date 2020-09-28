from mldata import *
import numpy as np
import operator


def load_dataset(path):
    paths = path.split('/')
    root = '/'.join(paths[:-1])
    if path[0] == '/' or root == '':
        root = '.'+root
    print('root dir: ', root)
    return parse_c45(paths[-1], root)


# def to_numpy(dataset):
#     data = []
#     for example in dataset.examples:
#         data.append(example)
#     data = np.array(data, dtype=np.float32)
#     return data


def get_title_data(dataset):
    title = [(schema.name, schema.type) for schema in dataset.schema]
    # data = to_numpy(dataset)
    data = np.array(dataset.to_float())
    data = data[:, 1:]
    return title[1:], data.tolist()


def get_subset(dataset, col, val):
    subset = []
    for data in dataset:
        if data[col] == val:
            subset.append(data)
    return subset


def split_dataset(dataset, col, val):
    subset = np.array(get_subset(dataset, col, val))
    subset = np.delete(subset, col, axis=1)
    return subset.tolist()


def get_unique_class(dataset, col):
    uni_class = [data[col] for data in dataset]
    return list(set(uni_class))


def stratified_cross_validation(dataset, n_folds):

    unique_label = get_unique_class(dataset, -1)

    class_data = [get_subset(dataset, -1, label) for label in unique_label]
    #print(len(class_data[0]), len(class_data[1]))

    train_data = [[] for _ in range(n_folds)]
    valid_data = [[] for _ in range(n_folds)]

    for each in class_data:
        num_per_fold = len(each)//n_folds
        for i in range(n_folds):
            valid = each[i * num_per_fold: (i + 1) * num_per_fold]
            train = each[0: i*num_per_fold] + each[(i+1)*num_per_fold:]
            train_data[i].extend(train)
            valid_data[i].extend(valid)

    return train_data, valid_data


def count_majority(dataset):
    classes = {}
    for data in dataset:
        x = data[-1]
        if x not in classes:
            classes[x] = 0
        else:
            classes[x] += 1

    classes = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)
    return classes[0][0]




