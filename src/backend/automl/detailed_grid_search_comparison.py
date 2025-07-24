"""
Bảng thống kê chi tiết kết quả của tất cả parameter combinations
Detailed statistics table for all parameter combinations
"""

import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from search_algorithm import grid_search
import itertools


def get_all_results_custom(X, y, param_grid, cv=5):
    """Get results for all parameter combinations using custom grid search"""
    results = []
    
    keys = param_grid.keys()
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))
    
    scoring = {
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    for combination in combinations:
        params = dict(zip(keys, combination))
        
        metric_scores = {metric: [] for metric in scoring.keys()}
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        for train_index, test_index in kf.split(X):
            train_data, test_data = X[train_index], X[test_index]
            train_targets, test_targets = y[train_index], y[test_index]
            
            model = SVC(**params)
            model.fit(train_data, train_targets)
            
            predictions = model.predict(test_data)
            
            for metric_name, metric_func in scoring.items():
                score = metric_func(test_targets, predictions)
                metric_scores[metric_name].append(score)
        
        average_scores = {metric: np.mean(scores) for metric, scores in metric_scores.items()}
        
        result = {'params': params}
        result.update(average_scores)
        results.append(result)
    
    return results


def get_all_results_sklearn(X, y, param_grid, cv=5):
    """Get results for all parameter combinations using sklearn grid search"""
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    grid_search = GridSearchCV(
        SVC(), 
        param_grid, 
        cv=cv, 
        scoring=scoring,
        refit='accuracy',
        n_jobs=1
    )
    
    grid_search.fit(X, y)
    
    results = []
    for i in range(len(grid_search.cv_results_['params'])):
        result = {
            'params': grid_search.cv_results_['params'][i],
            'accuracy': grid_search.cv_results_['mean_test_accuracy'][i],
            'precision': grid_search.cv_results_['mean_test_precision_macro'][i],
            'recall': grid_search.cv_results_['mean_test_recall_macro'][i],
            'f1': grid_search.cv_results_['mean_test_f1_macro'][i]
        }
        results.append(result)
    
    return results


def create_comparison_table():
    """Create detailed comparison table for all parameter combinations"""
    
    # Load Iris dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    print("\nParameter Grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    
    # Get results from both methods
    print("\nRunning Custom Grid Search...")
    custom_start = time.time()
    custom_results = get_all_results_custom(X, y, param_grid, cv=5)
    custom_time = time.time() - custom_start
    
    print("Running sklearn Grid Search...")
    sklearn_start = time.time()
    sklearn_results = get_all_results_sklearn(X, y, param_grid, cv=5)
    sklearn_time = time.time() - sklearn_start
    
    # Create comparison dataframes
    print("\n" + "="*120)
    print("BẢNG THỐNG KÊ CHI TIẾT KẾT QUẢ CỦA TẤT CẢ PARAMETER COMBINATIONS")
    print("DETAILED STATISTICS TABLE FOR ALL PARAMETER COMBINATIONS")
    print("="*120)
    
    # Custom Grid Search Results
    print("\n1. CUSTOM GRID SEARCH RESULTS:")
    print("-"*100)
    custom_df = pd.DataFrame(custom_results)
    custom_df['params_str'] = custom_df['params'].apply(str)
    custom_display = custom_df[['params_str', 'accuracy', 'precision', 'recall', 'f1']].round(4)
    custom_display.columns = ['Parameters', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    print(custom_display.to_string(index=False))
    
    # sklearn Grid Search Results
    print("\n2. SKLEARN GRID SEARCH RESULTS:")
    print("-"*100)
    sklearn_df = pd.DataFrame(sklearn_results)
    sklearn_df['params_str'] = sklearn_df['params'].apply(str)
    sklearn_display = sklearn_df[['params_str', 'accuracy', 'precision', 'recall', 'f1']].round(4)
    sklearn_display.columns = ['Parameters', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    print(sklearn_display.to_string(index=False))
    
    # Side-by-side comparison
    print("\n" + "="*120)
    print("SO SÁNH TRỰC TIẾP / DIRECT COMPARISON")
    print("="*120)
    
    comparison_data = []
    for i in range(len(custom_results)):
        params = custom_results[i]['params']
        
        # Find matching sklearn result
        sklearn_match = None
        for sk_result in sklearn_results:
            if sk_result['params'] == params:
                sklearn_match = sk_result
                break
        
        if sklearn_match:
            comparison_data.append([
                str(params),
                f"{custom_results[i]['accuracy']:.4f}",
                f"{sklearn_match['accuracy']:.4f}",
                f"{abs(custom_results[i]['accuracy'] - sklearn_match['accuracy']):.4f}",
                f"{custom_results[i]['f1']:.4f}",
                f"{sklearn_match['f1']:.4f}",
                f"{abs(custom_results[i]['f1'] - sklearn_match['f1']):.4f}"
            ])
    
    comparison_df = pd.DataFrame(comparison_data, 
                               columns=['Parameters', 
                                      'Custom Accuracy', 'sklearn Accuracy', 'Acc Diff',
                                      'Custom F1', 'sklearn F1', 'F1 Diff'])
    print(comparison_df.to_string(index=False))
    
    # Export to CSV
    csv_filename = 'grid_search_comparison_results.csv'
    comparison_df.to_csv(csv_filename, index=False)
    print(f"\n✓ Kết quả đã được xuất ra file: {csv_filename}")
    
    # Summary statistics
    print("\n" + "="*120)
    print("THỐNG KÊ TÓM TẮT / SUMMARY STATISTICS")
    print("="*120)
    
    # Best parameters
    best_custom_idx = custom_df['accuracy'].idxmax()
    best_sklearn_idx = sklearn_df['accuracy'].idxmax()
    
    print(f"\nBest Parameters:")
    print(f"  Custom Grid Search: {custom_df.iloc[best_custom_idx]['params_str']}")
    print(f"    - Accuracy: {custom_df.iloc[best_custom_idx]['accuracy']:.4f}")
    print(f"    - F1-Score: {custom_df.iloc[best_custom_idx]['f1']:.4f}")
    
    print(f"\n  sklearn Grid Search: {sklearn_df.iloc[best_sklearn_idx]['params_str']}")
    print(f"    - Accuracy: {sklearn_df.iloc[best_sklearn_idx]['accuracy']:.4f}")
    print(f"    - F1-Score: {sklearn_df.iloc[best_sklearn_idx]['f1']:.4f}")
    
    # Average differences
    acc_diffs = [abs(custom_results[i]['accuracy'] - sklearn_results[i]['accuracy']) 
                 for i in range(len(custom_results))]
    f1_diffs = [abs(custom_results[i]['f1'] - sklearn_results[i]['f1']) 
                for i in range(len(custom_results))]
    
    print(f"\nAverage Differences:")
    print(f"  Mean Accuracy Difference: {np.mean(acc_diffs):.6f}")
    print(f"  Max Accuracy Difference: {np.max(acc_diffs):.6f}")
    print(f"  Mean F1-Score Difference: {np.mean(f1_diffs):.6f}")
    print(f"  Max F1-Score Difference: {np.max(f1_diffs):.6f}")
    
    print(f"\nExecution Time:")
    print(f"  Custom Grid Search: {custom_time:.2f} seconds")
    print(f"  sklearn Grid Search: {sklearn_time:.2f} seconds")
    print(f"  Time Ratio (Custom/sklearn): {custom_time/sklearn_time:.2f}x")
    
    # Ranking comparison
    print("\n" + "="*120)
    print("XẾP HẠNG PARAMETERS / PARAMETER RANKING")
    print("="*120)
    
    # Sort by accuracy
    custom_sorted = custom_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    sklearn_sorted = sklearn_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    
    ranking_data = []
    for rank in range(min(5, len(custom_sorted))):  # Top 5
        ranking_data.append([
            rank + 1,
            custom_sorted.iloc[rank]['params_str'],
            f"{custom_sorted.iloc[rank]['accuracy']:.4f}",
            sklearn_sorted.iloc[rank]['params_str'],
            f"{sklearn_sorted.iloc[rank]['accuracy']:.4f}"
        ])
    
    ranking_df = pd.DataFrame(ranking_data,
                            columns=['Rank', 'Custom Params', 'Custom Acc', 'sklearn Params', 'sklearn Acc'])
    print("\nTop 5 Parameters by Accuracy:")
    print(ranking_df.to_string(index=False))


if __name__ == "__main__":
    create_comparison_table()
