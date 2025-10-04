import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve
from tabulate import tabulate
from sklearn.metrics import precision_score, recall_score, f1_score

# Set Seaborn theme for blue palette
sns.set_palette("Blues")


def print_metrics_table(results):
    """Print all metrics in a formatted table in terminal"""
    table_data = []
    headers = ["Model", "Train Acc", "Test Acc", "Test Acc (Opt)", "Recall", "Recall (Opt)", "Recall Imp", "F1 Score",
               "ROC AUC", "Opt Threshold"]

    for name, result in results.items():
        if 'error' not in result:
            # Handle both old and new result formats
            train_acc = result.get('train_accuracy_standard', result.get('train_accuracy', 'N/A'))
            test_acc_std = result.get('test_accuracy_standard', result.get('test_accuracy', 'N/A'))
            test_acc_opt = result.get('test_accuracy_optimized', test_acc_std)
            recall_std = result.get('recall_standard', result.get('report', {}).get('1', {}).get('recall', 'N/A'))
            recall_opt = result.get('recall_optimized', recall_std)
            recall_imp = result.get('recall_improvement', 0)
            f1 = result.get('report_standard', result.get('report', {})).get('1', {}).get('f1-score', 'N/A')
            roc_auc = result.get('roc_auc', 'N/A')
            opt_threshold = result.get('optimal_threshold', 0.5)

            row = [
                name,
                f"{train_acc:.4f}" if isinstance(train_acc, (int, float)) else train_acc,
                f"{test_acc_std:.4f}" if isinstance(test_acc_std, (int, float)) else test_acc_std,
                f"{test_acc_opt:.4f}" if isinstance(test_acc_opt, (int, float)) else test_acc_opt,
                f"{recall_std:.4f}" if isinstance(recall_std, (int, float)) else recall_std,
                f"{recall_opt:.4f}" if isinstance(recall_opt, (int, float)) else recall_opt,
                f"{recall_imp:.4f}" if isinstance(recall_imp, (int, float)) else recall_imp,
                f"{f1:.4f}" if isinstance(f1, (int, float)) else f1,
                f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else roc_auc,
                f"{opt_threshold:.3f}" if isinstance(opt_threshold, (int, float)) else opt_threshold
            ]
            table_data.append(row)

    print("\n" + "=" * 100)
    print("ENHANCED MODEL PERFORMANCE SUMMARY (WITH THRESHOLD OPTIMIZATION)".center(100))
    print("=" * 100)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("=" * 100 + "\n")


def plot_pairplot(data, features):
    """Pairplot visualization in blue tones"""
    print("\nGenerating Pairplot...")
    sns.pairplot(data[features], hue='Outcome', height=2.5,
                 palette=sns.color_palette("Blues", n_colors=2),
                 plot_kws={'alpha': 0.7})
    plt.suptitle('Pairplot of Selected Features', y=1.02)
    plt.show()
    print("Pairplot completed showing feature relationships by Outcome class")


def plot_correlation_matrix(data, features):
    """Correlation matrix in blue shades"""
    print("\nCalculating Feature Correlations...")
    corr_matrix = data[features].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='Blues', square=True, mask=mask)
    plt.title('Correlation Matrix of Selected Features', fontsize=16)
    plt.tight_layout()
    plt.show()

    print("\nTop Feature Correlations with Outcome:")
    outcome_correlations = corr_matrix['Outcome'].drop('Outcome').sort_values(ascending=False)
    for feature, corr in outcome_correlations.items():
        print(f"  {feature}: {corr:.3f}")


def plot_confusion_matrices(results):
    """Enhanced confusion matrices with both standard and optimized results"""
    print("\nGenerating Enhanced Confusion Matrices...")

    # Create subplots for standard and optimized
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, (name, result) in enumerate(results.items()):
        if 'error' not in result:
            # Standard confusion matrix
            ax_std = axes[0, idx]
            cm_std = result.get('cm_test_standard', result.get('cm_test'))
            sns.heatmap(cm_std, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['No Diabetes', 'Diabetes'],
                        yticklabels=['No Diabetes', 'Diabetes'],
                        ax=ax_std)
            acc_std = result.get('test_accuracy_standard', result.get('test_accuracy', 0))
            ax_std.set_title(f'{name}\nStandard (Acc: {acc_std:.3f})')
            ax_std.set_xlabel('Predicted')
            ax_std.set_ylabel('Actual')

            # Optimized confusion matrix
            ax_opt = axes[1, idx]
            cm_opt = result.get('cm_test_optimized', cm_std)
            sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', cbar=False,
                        xticklabels=['No Diabetes', 'Diabetes'],
                        yticklabels=['No Diabetes', 'Diabetes'],
                        ax=ax_opt)
            acc_opt = result.get('test_accuracy_optimized', acc_std)
            ax_opt.set_title(f'{name}\nOptimized (Acc: {acc_opt:.3f})')
            ax_opt.set_xlabel('Predicted')
            ax_opt.set_ylabel('Actual')

    # Remove empty subplots if needed
    for idx in range(len(results), 4):
        axes[0, idx].set_visible(False)
        axes[1, idx].set_visible(False)

    plt.suptitle('Confusion Matrices: Standard vs Optimized Thresholds', y=0.98, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_roc_curves(results):
    """Enhanced ROC curves with AUC values"""
    print("\nGenerating Enhanced ROC Curves...")
    plt.figure(figsize=(12, 8))
    blue_shades = ['#1f77b4', '#005f99', '#007acc', '#3399ff']

    for (name, result), color in zip(results.items(), blue_shades):
        if 'error' not in result:
            plt.plot(result['fpr'], result['tpr'], color=color, linewidth=2.5,
                     label=f'{name} (AUC = {result["roc_auc"]:.3f})')
            print(f"{name} AUC: {result['roc_auc']:.4f}")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - Model Discriminative Performance', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results):
    """Enhanced bar chart with both standard and optimized performance"""
    print("\nGenerating Enhanced Model Comparison Chart...")

    models = [m for m in results.keys() if 'error' not in results[m]]

    # Prepare data for comparison
    standard_acc = []
    optimized_acc = []
    standard_recall = []
    optimized_recall = []

    for name in models:
        result = results[name]
        standard_acc.append(result.get('test_accuracy_standard', result.get('test_accuracy', 0)))
        optimized_acc.append(result.get('test_accuracy_optimized', standard_acc[-1]))
        standard_recall.append(result.get('recall_standard', result.get('report', {}).get('1', {}).get('recall', 0)))
        optimized_recall.append(result.get('recall_optimized', standard_recall[-1]))

    x = np.arange(len(models))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Accuracy comparison
    ax1.bar(x - width, standard_acc, width, label='Standard Accuracy', color='#4c72b0', alpha=0.7)
    ax1.bar(x, optimized_acc, width, label='Optimized Accuracy', color='#2e59a8')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy: Standard vs Optimized Thresholds')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (std, opt) in enumerate(zip(standard_acc, optimized_acc)):
        ax1.text(i - width, std + 0.01, f'{std:.3f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i, opt + 0.01, f'{opt:.3f}', ha='center', va='bottom', fontsize=9)

    # Recall comparison
    ax2.bar(x - width, standard_recall, width, label='Standard Recall', color='#6baed6', alpha=0.7)
    ax2.bar(x, optimized_recall, width, label='Optimized Recall', color='#2171b5')
    ax2.set_ylabel('Recall (Diabetes Class)')
    ax2.set_title('Model Recall: Standard vs Optimized Thresholds')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (std, opt) in enumerate(zip(standard_recall, optimized_recall)):
        ax2.text(i - width, std + 0.01, f'{std:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i, opt + 0.01, f'{opt:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Print comparison summary
    print("\nModel Performance Comparison Summary:")
    for i, name in enumerate(models):
        recall_imp = optimized_recall[i] - standard_recall[i]
        print(f"{name}:")
        print(f"  Accuracy: {standard_acc[i]:.3f} → {optimized_acc[i]:.3f}")
        print(f"  Recall: {standard_recall[i]:.3f} → {optimized_recall[i]:.3f} (Improvement: {recall_imp:+.3f})")


def plot_learning_curves(models, X_train, y_train):
    """Enhanced learning curves with better visualization"""
    print("\nGenerating Enhanced Learning Curves...")
    plt.figure(figsize=(14, 8))
    colors = ['#1f77b4', '#005f99', '#007acc', '#3399ff']

    for (model_name, model), color in zip(models.items(), colors):
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 15),
                scoring='accuracy', random_state=42)

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            # Plot training scores
            plt.plot(train_sizes, train_mean, 'o-', color=color, linewidth=2,
                     markersize=6, label=f'{model_name} - Training')
            plt.fill_between(train_sizes, train_mean - train_std,
                             train_mean + train_std, alpha=0.15, color=color)

            # Plot cross-validation scores
            plt.plot(train_sizes, test_mean, 's--', color=color, linewidth=2,
                     markersize=4, label=f'{model_name} - Cross-validation')
            plt.fill_between(train_sizes, test_mean - test_std,
                             test_mean + test_std, alpha=0.15, color=color)

            # Print learning curve summary
            final_gap = train_mean[-1] - test_mean[-1]
            print(f"\n{model_name} Learning Curve Summary:")
            print(f"  Final Training Accuracy: {train_mean[-1]:.3f}")
            print(f"  Final CV Accuracy: {test_mean[-1]:.3f}")
            print(f"  Generalization Gap: {final_gap:.3f}")

        except Exception as e:
            print(f"\nError generating learning curve for {model_name}: {str(e)}")

    plt.title('Learning Curves - Model Training Dynamics', fontsize=16)
    plt.xlabel('Training Examples', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_threshold_analysis(threshold_results):
    """Comprehensive threshold optimization analysis"""
    if not threshold_results:
        print("No threshold results available for analysis")
        return

    print("\nGenerating Threshold Optimization Analysis...")

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    colors = ['#1f77b4', '#005f99', '#007acc', '#3399ff']

    # Plot 1: Threshold vs F1-Score
    for idx, (model_name, result) in enumerate(threshold_results.items()):
        ax1.plot(result['thresholds'], result['f1_values'],
                 color=colors[idx], linewidth=2.5, label=model_name)
        # Mark optimal threshold
        optimal_idx = np.argmax(result['f1_values'])
        ax1.scatter(result['thresholds'][optimal_idx], result['f1_values'][optimal_idx],
                    color=colors[idx], s=100, zorder=5, edgecolor='black')

    ax1.set_xlabel('Classification Threshold')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('F1-Score vs Classification Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Threshold vs Recall
    for idx, (model_name, result) in enumerate(threshold_results.items()):
        ax2.plot(result['thresholds'], result['recall_values'],
                 color=colors[idx], linewidth=2.5, label=model_name)
        optimal_idx = np.argmax(result['f1_values'])
        ax2.scatter(result['thresholds'][optimal_idx], result['recall_values'][optimal_idx],
                    color=colors[idx], s=100, zorder=5, edgecolor='black')

    ax2.set_xlabel('Classification Threshold')
    ax2.set_ylabel('Recall')
    ax2.set_title('Recall vs Classification Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Threshold vs Precision
    for idx, (model_name, result) in enumerate(threshold_results.items()):
        ax3.plot(result['thresholds'], result['precision_values'],
                 color=colors[idx], linewidth=2.5, label=model_name)
        optimal_idx = np.argmax(result['f1_values'])
        ax3.scatter(result['thresholds'][optimal_idx], result['precision_values'][optimal_idx],
                    color=colors[idx], s=100, zorder=5, edgecolor='black')

    ax3.set_xlabel('Classification Threshold')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision vs Classification Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Recall Improvement
    models = list(threshold_results.keys())
    recall_improvements = [result['recall_improvement'] for result in threshold_results.values()]

    bars = ax4.bar(models, recall_improvements, color=colors[:len(models)], alpha=0.7)
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Recall Improvement')
    ax4.set_title('Recall Improvement with Optimized Thresholds')
    ax4.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, improvement in zip(bars, recall_improvements):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f'{improvement:.3f}', ha='center', va='bottom', fontweight='bold')

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Comprehensive Threshold Optimization Analysis', y=1.02, fontsize=16)
    plt.show()

    # Print threshold optimization summary
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION SUMMARY".center(80))
    print("=" * 80)
    for model_name, result in threshold_results.items():
        print(f"\n{model_name}:")
        print(f"  Optimal Threshold: {result['optimal_threshold']:.3f}")
        print(f"  Default Recall: {result['default_recall']:.3f}")
        print(f"  Optimized Recall: {result['optimized_recall']:.3f}")
        print(f"  Recall Improvement: {result['recall_improvement']:+.3f}")
        print(f"  Best F1-Score: {result['f1_score']:.3f}")
    print("=" * 80)
