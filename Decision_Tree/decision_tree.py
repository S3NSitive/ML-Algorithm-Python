import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from graphviz import Digraph

data = {
    'name': ["Kang", "Kim", "Choi", "Park", "Yoon"],
    '짱절미': [True, False, False, False, False],
    '셀스타그램': [False, False, True, False, False],
    '우산': [False, False, False, False, False],
    'follower': [0, 0, 100, 210, 0],
    'like': [True, False, True, True, False]
}

data = pd.DataFrame(data)
data = data.set_index("name")
label_name = 'like'

min_sample_split = 2
max_depth = None

print(data, "\n")


def display_node(dot, key, node):
    if node["leaf"]:
        probability = node['probability']
        probability = round(probability, 4)
        probability = str(probability)

        dot.node(key, probability)
    else:
        description = node['description']
        dot.node(key, description)

        if "left" in node:
            left_key = key + "L"
            display_node(dot, left_key, node['left'])
            dot.edge(key, left_key)

        if "right" in node:
            right_key = key + "R"
            display_node(dot, right_key, node['right'])
            dot.edge(key, right_key)


def display_tree(tree):
    dot = Digraph(comment='Decision Tree')

    display_node(dot, "Root", tree)

    return graphviz.Source(dot.source)


def predict(data, node):
    if node['leaf'] == True:
        probability = node["probability"]
        result = dict(zip(data.index, len(data) * [probability]))
    else:
        condition = node['condition']

        left_data = data[condition(data)]
        left_result = predict(left_data, node['left'])

        right_data = data[~condition(data)]
        right_result = predict(right_data, node['right'])

        return {**left_result, **right_result}

    return result


def binary_condition(data, feature_name, value):
    return data[feature_name] == value


def make_condition(method, feature_name, value):
    def call_condition(data):
        return method(data, feature_name, value)

    return call_condition


def evaluate_gini_impurity(data, label_name):
    true_probability = data[label_name].mean()
    false_probability = 1 - true_probability

    true_gini_impurity = true_probability * (1 - true_probability)
    false_gini_impurity = false_probability * (1 - false_probability)

    gini_impurity = true_gini_impurity + false_gini_impurity

    return gini_impurity


def make_node(data, condition_list, current_depth):
    condition = (len(condition_list) != 0)
    sample = (len(data) >= min_sample_split)
    depth = (max_depth is None) or (current_depth < max_depth)

    if condition and sample and depth:
        description = list(condition_list.keys())[0]
        condition = condition_list[description]

        node = {"leaf": False, "description": description, "condition": condition}

        left_data = data[condition(data)]
        right_data = data[~condition(data)]

        if len(left_data) != 0 and len(right_data) != 0:
            del condition_list[description]

            node['left'] = make_node(left_data, condition_list.copy(), current_depth + 1)
            node['right'] = make_node(right_data, condition_list.copy(), current_depth + 1)

            return node

    probability = data[label_name].mean()
    node = {"leaf": True, "probability": probability}

    return node


def make_tree(data):
    tree = {}
    feature_name = '짱절미'

    condition1 = make_condition(binary_condition, '짱절미', True)
    condition2 = make_condition(binary_condition, '셀스타그램', True)
    condition3 = make_condition(binary_condition, '우산', True)
    condition_list = {
        '짱절미 == True': condition1,
        '셀스타그램 == True': condition2,
        '우산 == True': condition3
    }

    tree = make_node(data, condition_list, current_depth=0)

    return tree


tree = make_tree(data)
print(tree)
