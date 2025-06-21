import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve


def plot_pairplot(data, features):
    sns.pairplot(data[features], hue='Outcome', height=2.5)
    plt.suptitle('Pairplot of Selected Features', y=1.02)
    plt.show()


def plot_correlation_matrix(data, features):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[features].corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix of Selected Features', fontsize=16)
    plt.show()


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (name, result) in zip(axes, results.items()):
        sns.heatmap(result['cm_test'], annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'],
                    ax=ax)
        ax.set_title(f'{name} - Test Data')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    plt.suptitle('Confusion Matrices Comparison', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_roc_curves(results):
    plt.figure(figsize=(8, 6))
    for name, result in results.items():
        plt.plot(result['fpr'], result['tpr'],
                 label=f'{name} (AUC = {result["roc_auc"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.show()


def plot_model_comparison(results):
    metrics = ['train_accuracy', 'test_accuracy']
    models = list(results.keys())

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, [results[m]['train_accuracy'] for m in models],
                    width, label='Training')
    rects2 = ax.bar(x + width / 2, [results[m]['test_accuracy'] for m in models],
                    width, label='Testing')

    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    fig.tight_layout()
    plt.show()


def plot_learning_curves(models, X_train, y_train):
    """
    Plot learning curves for multiple models to visualize training vs validation performance
    """
    plt.figure(figsize=(12, 8))

    for model_name, model in models.items():
        # Calculate learning curve data
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy')

        # Calculate mean and standard deviation
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Plot learning curve
        plt.plot(train_sizes, train_scores_mean, 'o-', color='blue',
                 label=f'{model_name} Training')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='red',
                 label=f'{model_name} Cross-validation')

        # Plot std deviation bands
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color='blue')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color='red')

    plt.title('Learning Curves', fontsize=16)
    plt.xlabel('Training Examples', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()