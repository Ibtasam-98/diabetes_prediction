import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve
from tabulate import tabulate

# Set Seaborn theme for blue palette
sns.set_palette("Blues")


def print_metrics_table(results):
    """Print all metrics in a formatted table in terminal"""
    table_data = []
    headers = ["Model", "Train Accuracy", "Test Accuracy", "F1 Score", "ROC AUC"]

    for name, result in results.items():
        if 'error' not in result:
            row = [
                name,
                f"{result['train_accuracy']:.4f}",
                f"{result['test_accuracy']:.4f}",
                f"{result.get('f1_score', 'N/A'):.4f}",
                f"{result['roc_auc']:.4f}"
            ]
            table_data.append(row)

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE SUMMARY".center(80))
    print("=" * 80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("=" * 80 + "\n")


def plot_pairplot(data, features):
    """Pairplot visualization in blue tones"""
    print("\nGenerating Pairplot...")
    sns.pairplot(data[features], hue='Outcome', height=2.5,
                 palette=sns.color_palette("Blues", n_colors=2))
    plt.suptitle('Pairplot of Selected Features', y=1.02)
    plt.show()
    print("Pairplot completed showing feature relationships by Outcome class")


def plot_correlation_matrix(data, features):
    """Correlation matrix in blue shades"""
    print("\nCalculating Feature Correlations...")
    corr_matrix = data[features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='Blues', square=True)
    plt.title('Correlation Matrix of Selected Features', fontsize=16)
    plt.show()

    print("\nTop Feature Correlations:")
    print(corr_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(10))


def plot_confusion_matrices(results):
    """Confusion matrices using blue shades"""
    print("\nGenerating Confusion Matrices...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for ax, (name, result) in zip(axes, results.items()):
        if 'error' not in result:
            sns.heatmap(result['cm_test'], annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['No Diabetes', 'Diabetes'],
                        yticklabels=['No Diabetes', 'Diabetes'],
                        ax=ax)
            ax.set_title(f'{name} (Acc: {result["test_accuracy"]:.2f})')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')

            print(f"\n{name} Confusion Matrix:")
            print(result['cm_test'])

    plt.suptitle('Confusion Matrices Comparison', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_roc_curves(results):
    """ROC curves using shades of blue"""
    print("\nGenerating ROC Curves...")
    plt.figure(figsize=(10, 8))
    blue_shades = ['#1f77b4', '#005f99', '#007acc', '#3399ff']

    for (name, result), color in zip(results.items(), blue_shades):
        if 'error' not in result:
            plt.plot(result['fpr'], result['tpr'], color=color,
                     label=f'{name} (AUC = {result["roc_auc"]:.2f})')
            print(f"{name} AUC: {result['roc_auc']:.4f}")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_model_comparison(results):
    """Bar chart using blue tones"""
    print("\nGenerating Model Comparison Chart...")
    metrics = ['train_accuracy', 'test_accuracy']
    models = list(results.keys())

    print("\nModel Accuracy Comparison:")
    for name in models:
        if 'error' not in results[name]:
            print(f"{name}: Train={results[name]['train_accuracy']:.4f}, Test={results[name]['test_accuracy']:.4f}")

    x = np.arange(len(models))
    width = 0.35

    train_accuracies = [results[m]['train_accuracy'] for m in models if 'error' not in results[m]]
    test_accuracies = [results[m]['test_accuracy'] for m in models if 'error' not in results[m]]

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, train_accuracies, width, label='Training', color='#4c72b0')
    rects2 = ax.bar(x + width / 2, test_accuracies, width, label='Testing', color='#6baed6')

    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m for m in models if 'error' not in results[m]], rotation=45, ha='right')
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    plt.tight_layout()
    plt.show()


def plot_learning_curves(models, X_train, y_train):
    """Learning curves with blue tones"""
    print("\nGenerating Learning Curves...")
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#005f99', '#007acc', '#3399ff']

    for (model_name, model), color in zip(models.items(), colors):
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy')

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            plt.plot(train_sizes, train_mean, 'o-', color=color,
                     label=f'{model_name} Training')
            plt.plot(train_sizes, test_mean, 'o--', color=color,
                     label=f'{model_name} CV')

            plt.fill_between(train_sizes, train_mean - train_std,
                             train_mean + train_std, alpha=0.1,
                             color=color)
            plt.fill_between(train_sizes, test_mean - test_std,
                             test_mean + test_std, alpha=0.1,
                             color=color)

            print(f"\n{model_name} Learning Curve:")
            for size, t_mean, v_mean in zip(train_sizes, train_mean, test_mean):
                print(f"Train Size: {size:.0f} | Train Acc: {t_mean:.3f} | Val Acc: {v_mean:.3f}")

        except Exception as e:
            print(f"\nError generating learning curve for {model_name}: {str(e)}")

    plt.title('Learning Curves', fontsize=16)
    plt.xlabel('Training Examples', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()
