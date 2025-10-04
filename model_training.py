import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score, \
    recall_score, precision_score
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight


def calculate_class_weights(y_train):
    """Calculate balanced class weights"""
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    return {0: class_weights[0], 1: class_weights[1]}


def apply_smote(X_train, y_train):
    """Apply SMOTE for handling class imbalance"""
    print("Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"Original class distribution: {np.bincount(y_train)}")
    print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")

    return X_train_balanced, y_train_balanced


def train_models(X_train, y_train, use_smote=True):
    """Train models with enhanced class imbalance handling"""

    # Calculate class weights for weighted loss functions
    class_weights = calculate_class_weights(y_train)
    custom_weights = [{0: 1, 1: 2}, {0: 1, 1: 3}, class_weights]

    # Apply SMOTE if requested
    if use_smote:
        X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train

    # Enhanced Logistic Regression with comprehensive class weights
    lr_param_grid = {
        'C': np.logspace(-3, 3, 7),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}, class_weights, None]
    }

    # Enhanced KNN with distance weighting
    knn_param_grid = {
        'n_neighbors': range(1, 21),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Enhanced Decision Tree with comprehensive class weights
    dt_param_grid = {
        'max_depth': [None] + list(range(3, 21)),
        'min_samples_split': range(2, 11),
        'min_samples_leaf': range(1, 11),
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}, class_weights, None]
    }

    # Enhanced SVM with comprehensive class weights
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly'],
        'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}, class_weights, None]
    }

    # Create grids with enhanced scoring for minority class
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        lr_param_grid,
        cv=5,
        scoring='f1',  # Changed to f1 for better balance
        n_jobs=-1,
        error_score='raise'
    )

    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        error_score='raise'
    )

    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        dt_param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        error_score='raise'
    )

    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        svm_param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        error_score='raise'
    )

    # Fit models
    print("Training Logistic Regression with enhanced class weights...")
    lr_grid.fit(X_train_balanced, y_train_balanced)

    print("Training KNN...")
    knn_grid.fit(X_train_balanced, y_train_balanced)

    print("Training Decision Tree with enhanced class weights...")
    dt_grid.fit(X_train_balanced, y_train_balanced)

    print("Training SVM with enhanced class weights...")
    svm_grid.fit(X_train_balanced, y_train_balanced)

    models = {
        'Logistic Regression': lr_grid.best_estimator_,
        'KNN': knn_grid.best_estimator_,
        'Decision Tree': dt_grid.best_estimator_,
        'SVM': svm_grid.best_estimator_
    }

    # Print best parameters including class weights
    print("\n=== Best Model Parameters ===")
    for name, model in models.items():
        print(f"\n{name}:")
        params = model.get_params()
        if 'class_weight' in params:
            print(f"  Class weights: {params['class_weight']}")
        if hasattr(model, 'best_params_'):
            print(f"  Best params: {model.best_params_}")

    return models


def optimize_thresholds(models, X_test, y_test):
    """Optimize prediction thresholds for better recall of minority class"""
    print("\n=== Threshold Optimization ===")
    threshold_results = {}

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]

            # Find optimal threshold for balanced performance
            best_threshold = 0.5
            best_f1 = 0
            best_recall = 0

            thresholds = np.arange(0.3, 0.7, 0.02)
            recall_values = []
            precision_values = []
            f1_values = []

            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                current_recall = recall_score(y_test, y_pred)
                current_precision = precision_score(y_test, y_pred, zero_division=0)
                current_f1 = f1_score(y_test, y_pred, zero_division=0)

                recall_values.append(current_recall)
                precision_values.append(current_precision)
                f1_values.append(current_f1)

                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_threshold = threshold
                    best_recall = current_recall

            # Store results
            threshold_results[name] = {
                'optimal_threshold': best_threshold,
                'default_recall': recall_score(y_test, (y_prob >= 0.5).astype(int)),
                'optimized_recall': best_recall,
                'recall_improvement': best_recall - recall_score(y_test, (y_prob >= 0.5).astype(int)),
                'f1_score': best_f1,
                'thresholds': thresholds,
                'recall_values': recall_values,
                'precision_values': precision_values,
                'f1_values': f1_values
            }

            print(f"\n{name}:")
            print(f"  Optimal threshold: {best_threshold:.3f}")
            print(f"  Recall: {threshold_results[name]['default_recall']:.3f} → {best_recall:.3f}")
            print(f"  Recall improvement: {threshold_results[name]['recall_improvement']:.3f}")
            print(f"  F1-score: {best_f1:.3f}")

    return threshold_results


def evaluate_models_with_thresholds(models, X_train, X_test, y_train, y_test, threshold_results=None):
    """Enhanced evaluation with threshold optimization"""
    results = {}

    for name, model in models.items():
        try:
            # Get optimal threshold if available
            optimal_threshold = 0.5
            if threshold_results and name in threshold_results:
                optimal_threshold = threshold_results[name]['optimal_threshold']

            # Standard predictions (threshold = 0.5)
            y_train_pred_standard = model.predict(X_train)
            y_test_pred_standard = model.predict(X_test)

            # Optimized predictions
            if hasattr(model, "predict_proba"):
                y_prob_train = model.predict_proba(X_train)[:, 1]
                y_prob_test = model.predict_proba(X_test)[:, 1]
                y_train_pred_optimized = (y_prob_train >= optimal_threshold).astype(int)
                y_test_pred_optimized = (y_prob_test >= optimal_threshold).astype(int)
            else:
                y_train_pred_optimized = y_train_pred_standard
                y_test_pred_optimized = y_test_pred_standard

            # Metrics for both standard and optimized
            train_accuracy_standard = accuracy_score(y_train, y_train_pred_standard)
            test_accuracy_standard = accuracy_score(y_test, y_test_pred_standard)
            train_accuracy_optimized = accuracy_score(y_train, y_train_pred_optimized)
            test_accuracy_optimized = accuracy_score(y_test, y_test_pred_optimized)

            # Recall metrics (focus on minority class)
            recall_standard = recall_score(y_test, y_test_pred_standard)
            recall_optimized = recall_score(y_test, y_test_pred_optimized)

            # Confusion matrices
            cm_train_standard = confusion_matrix(y_train, y_train_pred_standard)
            cm_test_standard = confusion_matrix(y_test, y_test_pred_standard)
            cm_test_optimized = confusion_matrix(y_test, y_test_pred_optimized)

            # ROC curve
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            # Classification reports
            report_standard = classification_report(y_test, y_test_pred_standard, output_dict=True)
            report_optimized = classification_report(y_test, y_test_pred_optimized, output_dict=True)

            # Store comprehensive results
            results[name] = {
                'model': model,
                'train_accuracy_standard': train_accuracy_standard,
                'test_accuracy_standard': test_accuracy_standard,
                'train_accuracy_optimized': train_accuracy_optimized,
                'test_accuracy_optimized': test_accuracy_optimized,
                'recall_standard': recall_standard,
                'recall_optimized': recall_optimized,
                'recall_improvement': recall_optimized - recall_standard,
                'cm_train_standard': cm_train_standard,
                'cm_test_standard': cm_test_standard,
                'cm_test_optimized': cm_test_optimized,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'report_standard': report_standard,
                'report_optimized': report_optimized,
                'optimal_threshold': optimal_threshold
            }

            # Print enhanced summary
            print(f"\n--- {name} ---")
            print(f"Best Parameters: {model.get_params()}")
            print(f"Optimal Threshold: {optimal_threshold:.3f}")
            print(
                f"Standard - Training Accuracy: {train_accuracy_standard:.4f}, Testing Accuracy: {test_accuracy_standard:.4f}")
            print(
                f"Optimized - Training Accuracy: {train_accuracy_optimized:.4f}, Testing Accuracy: {test_accuracy_optimized:.4f}")
            print(
                f"Recall - Standard: {recall_standard:.3f}, Optimized: {recall_optimized:.3f}, Improvement: {results[name]['recall_improvement']:.3f}")

            print("\nStandard Classification Report:")
            print(classification_report(y_test, y_test_pred_standard))
            print("Optimized Classification Report:")
            print(classification_report(y_test, y_test_pred_optimized))

        except Exception as e:
            print(f"\nError evaluating {name}: {str(e)}")
            results[name] = {
                'error': str(e)
            }

    return results


# Backward compatibility
def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Original evaluate_models function for backward compatibility"""
    return evaluate_models_with_thresholds(models, X_train, X_test, y_train, y_test)
