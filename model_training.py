import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


from sklearn.svm import SVC

def train_models(X_train, y_train):
    # Logistic Regression
    lr_param_grid = {
        'C': np.logspace(-3, 3, 7),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', None]
    }

    # KNN
    knn_param_grid = {
        'n_neighbors': range(1, 21),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Decision Tree
    dt_param_grid = {
        'max_depth': [None] + list(range(3, 21)),
        'min_samples_split': range(2, 11),
        'min_samples_leaf': range(1, 11),
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None]
    }

    # SVM
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly'],
        'class_weight': ['balanced', None]
    }

    # Create grids
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        lr_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        error_score='raise'
    )

    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        error_score='raise'
    )

    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        dt_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        error_score='raise'
    )

    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        svm_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        error_score='raise'
    )

    # Fit models
    print("Training Logistic Regression...")
    lr_grid.fit(X_train, y_train)

    print("Training KNN...")
    knn_grid.fit(X_train, y_train)

    print("Training Decision Tree...")
    dt_grid.fit(X_train, y_train)

    print("Training SVM...")
    svm_grid.fit(X_train, y_train)

    return {
        'Logistic Regression': lr_grid.best_estimator_,
        'KNN': knn_grid.best_estimator_,
        'Decision Tree': dt_grid.best_estimator_,
        'SVM': svm_grid.best_estimator_
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}

    for name, model in models.items():
        try:
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Probabilities for ROC curve
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:  # For models without predict_proba
                y_prob = model.decision_function(X_test)
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())  # Scale to [0,1]

            # Metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Confusion matrices
            cm_train = confusion_matrix(y_train, y_train_pred)
            cm_test = confusion_matrix(y_test, y_test_pred)

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            # Classification report
            report = classification_report(y_test, y_test_pred, output_dict=True)

            # Store results
            results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cm_train': cm_train,
                'cm_test': cm_test,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'report': report
            }

            # Print summary
            print(f"\n--- {name} ---")
            print(f"Best Parameters: {model.get_params()}")
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Testing Accuracy: {test_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_test_pred))
            print("Confusion Matrix (Test):")
            print(cm_test)

        except Exception as e:
            print(f"\nError evaluating {name}: {str(e)}")
            results[name] = {
                'error': str(e)
            }

    return results