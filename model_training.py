# model_training.py (updated)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


def train_models(X_train, y_train):
    # Logistic Regression
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['liblinear'],
        'class_weight': ['balanced', None]
    }

    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=500),
        lr_param_grid,
        cv=5,
        scoring='accuracy'
    )
    lr_grid.fit(X_train, y_train)

    # KNN
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_param_grid,
        cv=5,
        scoring='accuracy'
    )
    knn_grid.fit(X_train, y_train)

    return {
        'Logistic Regression': lr_grid.best_estimator_,
        'KNN': knn_grid.best_estimator_
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}

    for name, model in models.items():
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Probabilities for ROC curve
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:  # For KNN with some configurations
            y_prob = model.decision_function(X_test)

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
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred))
        print("Confusion Matrix (Test):")
        print(cm_test)

    return results