import pandas as pd
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

print(data.shape)
print(data.head())

label_name = "like"
feature_names = data.columns.difference([label_name])

# Visualize Tree
def display_node(dot, key, node):
    if node["leaf"] == True:
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


# Predict
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


def continuous_condition(data, feature_name, value):
    return data[feature_name] < value


def make_condition(method, feature_name, value):
    def call_condition(data):
        return method(data, feature_name, value)

    return call_condition


def make_condition_list(data, feature_names):
    condition_list = {}

    for feature_name in feature_names:
        if data[feature_name].dtype == "bool":
            description = f"{feature_name} == True"
            condition = make_condition(binary_condition, feature_name, True)

            condition_list[description] = condition
        else:
            values = data[feature_name].unique()
            values = values[1:-1]

            for value in values:
                description = f"{feature_name} < {value}"
                condition = make_condition(continuous_condition, feature_name, value)

                condition_list[description] = condition

    return condition_list


def evaluate_gini_impurity(data, label_name):
    if len(data) == 0:
        return 0

    true_probability = data[label_name].mean()
    false_probability = 1 - true_probability

    true_gini_impurity = true_probability * (1 - true_probability)
    false_gini_impurity = false_probability * (1 - false_probability)

    gini_impurity = true_gini_impurity + false_gini_impurity

    return gini_impurity


def evaluate_average_gini_impurity(data, condition, label_name):
    true_data = data[condition(data)]
    false_data = data[~condition(data)]

    true_impurity = evaluate_gini_impurity(true_data, label_name)
    false_impurity = evaluate_gini_impurity(false_data, label_name)

    gini_impurity = (len(true_data) * true_impurity + len(false_data) * false_impurity)
    gini_impurity = gini_impurity / len(data)

    return gini_impurity


def find_best_condition(data, condition_list, label_name):
    best_gini_impurity = 0.51
    best_condition = None
    best_description = None

    for description, condition in condition_list.items():
        gini_impurity = evaluate_average_gini_impurity(data, condition, label_name)

        if gini_impurity < best_gini_impurity:
            best_gini_impurity = gini_impurity
            best_condition = condition
            best_description = description

    return best_condition, best_description, best_gini_impurity


condition1 = make_condition(binary_condition, "짱절미", True)
condition2 = make_condition(binary_condition, "셀스타그램", True)
condition3 = make_condition(binary_condition, "우산", True)

condition_list = {
    "짱절미 == True": condition1,
    "셀스타그램 == True": condition2,
    "우산 == True": condition3,
}

print(find_best_condition(data, condition_list, label_name))

max_depth = None
min_sample_split = 2


def make_node(data, condition_list, current_gini, current_depth):
    condition = (len(condition_list) != 0)
    sample = (len(data) >= min_sample_split)
    depth = (max_depth == None) or (current_depth < max_depth)

    if condition and sample and depth:
        condition, description, gini = \
            find_best_condition(data, condition_list, label_name)

        left_data = data[condition(data)]
        right_data = data[~condition(data)]

        if gini < current_gini and len(left_data) != 0 and len(right_data) != 0:
            node = {'leaf': False, 'description': description, 'condition': condition}

            del condition_list[description]

            node['left'] = make_node(left_data, condition_list.copy(), gini, current_depth + 1)
            node['right'] = make_node(right_data, condition_list.copy(), gini, current_depth + 1)

            return node

    probability = data[label_name].mean()
    node = {'leaf': True, 'probability': probability}

    return node


def make_tree(data, feature_names):
    condition_list = make_condition_list(data, feature_names)

    tree = make_node(data, condition_list, current_gini=0.51, current_depth=0)

    return tree


tree = make_tree(data, feature_names)
display_tree(tree)

predictions = predict(data, tree)
predictions = pd.Series(predictions)

result = data.copy()
result["like(predict)"] = predictions

print(result)
