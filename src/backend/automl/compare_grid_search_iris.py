"""
So sánh Custom Grid Search với sklearn Grid Search trên bộ dữ liệu Iris
Comparison between Custom Grid Search and sklearn Grid Search using the Iris dataset
"""

import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import itertools
from typing import Dict, List, Any, Tuple


from search_algorithm import grid_search


def compare_grid_search_methods():
    """
    So sánh hiệu suất giữa Custom Grid Search và sklearn Grid Search
    Compare performance between Custom Grid Search and sklearn Grid Search
    """
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    results = {}
    
    # 1. sklearn GridSearchCV
    sklearn_start = time.time()
    sklearn_grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=1)
    sklearn_grid.fit(X, y)
    sklearn_time = time.time() - sklearn_start
    
    results['sklearn'] = {
        'method': 'sklearn GridSearchCV',
        'best_params': sklearn_grid.best_params_,
        'best_cv_score': sklearn_grid.best_score_,
        'time': sklearn_time
    }
    
    # 2. Custom GridSearchCV from search_algorithm
    custom_start = time.time()
    best_params, best_score, all_scores, *_ = grid_search(param_grid, SVC, X, y, cv=5, metric_sort='accuracy')
    custom_time = time.time() - custom_start

    results['custom'] = {
        'method': 'Custom GridSearchCV',
        'best_params': best_params,
        'best_cv_score': best_score,
        'all_scores': all_scores,
        'time': custom_time
    }
    
    # Create comparison table
    comparison_data = []
    for key, value in results.items():
        comparison_data.append([
            value['method'],
            str(value['best_params']),
            f"{value['best_cv_score']:.4f}",
            f"{value['time']:.2f}s"
        ])
    
    df = pd.DataFrame(comparison_data, 
                     columns=['Phương pháp/Method', 'Best Parameters', 'CV Score', 'Thời gian/Time'])
    
    print("\n" + "="*80)
    print("BẢNG SO SÁNH KẾT QUẢ / RESULTS COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    # Print all metrics from custom grid search
    print("\n" + "="*80)
    print("TẤT CẢ METRICS TỪ CUSTOM GRID SEARCH / ALL METRICS FROM CUSTOM GRID SEARCH")
    print("="*80)
    if 'all_scores' in results['custom']:
        print(f"\nBest parameters: {results['custom']['best_params']}")
        print("\nAll metrics for best parameters:")
        for metric, score in results['custom']['all_scores'].items():
            print(f"   - {metric}: {score:.4f}")
    
    # Show detailed comparison
    print("\n" + "="*80)
    print("CHI TIẾT SO SÁNH / DETAILED COMPARISON")
    print("="*80)
    
    print(f"\n1. Độ chính xác CV (CV Accuracy):")
    print(f"   - sklearn GridSearchCV: {results['sklearn']['best_cv_score']:.4f}")
    print(f"   - Custom Grid Search (search_algorithm.py): {results['custom']['best_cv_score']:.4f}")
    print(f"   - Chênh lệch (Difference): {abs(results['sklearn']['best_cv_score'] - results['custom']['best_cv_score']):.4f}")
    
    print(f"\n2. Thời gian thực thi (Execution Time):")
    print(f"   - sklearn GridSearchCV: {results['sklearn']['time']:.2f} seconds")
    print(f"   - Custom Grid Search (search_algorithm.py): {results['custom']['time']:.2f} seconds")
    print(f"   - Tỷ lệ (Ratio): {results['custom']['time']/results['sklearn']['time']:.2f}x")
    
    print(f"\n3. Tham số tối ưu (Best Parameters):")
    print(f"   - sklearn: {results['sklearn']['best_params']}")
    print(f"   - Custom: {results['custom']['best_params']}")
    
    # Feature comparison table
    print("\n" + "="*80)
    print("SO SÁNH TÍNH NĂNG / FEATURE COMPARISON")
    print("="*80)
    
    features = [
        ['Phương pháp tìm kiếm/Search method', 'Grid Search', 'Grid Search'],
        ['Parallel processing', 'Có (Yes)', 'Không (No)'],
        ['Multiple scoring metrics', 'Có (Yes)', 'Có (Yes) - accuracy, precision, recall, f1'],
        ['Detailed CV results', 'Có (Yes)', 'Có (Yes) - all metrics'],
        ['Memory efficiency', 'Tốt (Good)', 'Trung bình (Average)'],
        ['Customization flexibility', 'Trung bình (Average)', 'Cao (High)'],
        ['Built-in preprocessing', 'Có (Yes)', 'Không (No)'],
        ['Error handling', 'Tốt (Good)', 'Cơ bản (Basic)'],
        ['Documentation', 'Xuất sắc (Excellent)', 'Code comment'],
        ['Return format', 'Object với nhiều attributes', 'Tuple (best_params, best_score, all_scores)']
    ]
    
    feature_df = pd.DataFrame(features, 
                            columns=['Tính năng/Feature', 'sklearn GridSearchCV', 'Custom Grid Search (search_algorithm.py)'])
    print(feature_df.to_string(index=False))


if __name__ == "__main__":
    print("="*80)
    print("SO SÁNH CUSTOM GRID SEARCH VỚI SKLEARN GRID SEARCH")
    print("COMPARISON BETWEEN CUSTOM GRID SEARCH AND SKLEARN GRID SEARCH")
    print("="*80)
    compare_grid_search_methods()
    
    print("\nLựa chọn tùy thuộc vào nhu cầu cụ thể của dự án!")

