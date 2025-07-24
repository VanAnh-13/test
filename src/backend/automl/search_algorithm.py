import  itertools
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def grid_search(param_grid, model_func, data, targets, cv = 5, scoring = None, metric_sort='accuracy'):
    best_all_scores = None

    if scoring is None:
        scoring = {
            'accuracy': accuracy_score,
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro', zero_division=0)
        }

    if metric_sort not in scoring:
        raise ValueError(f"metric_sort '{metric_sort}' not in scoring metrics")

    best_score = float('-inf')
    best_params = None

    keys = param_grid.keys()
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))

    for combination in combinations:
        params = dict(zip(keys, combination))

        metric_scores = {metric: [] for metric in scoring.keys()}
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(data):
            train_data, test_data = data[train_index], data[test_index]
            train_targets, test_targets = targets[train_index], targets[test_index]

            model = model_func(**params)
            model.fit(train_data, train_targets)

            predictions = model.predict(test_data)

            for metric_name, metric_func in scoring.items():
                score = metric_func(test_targets, predictions)
                metric_scores[metric_name].append(score)

        average_score = {metric: np.mean(scores) for metric, scores in metric_scores.items()}

        current_score = average_score[metric_sort]
        if current_score > best_score:
            best_score = current_score
            best_params = params
            best_all_scores = average_score

    return best_params, best_score, best_all_scores