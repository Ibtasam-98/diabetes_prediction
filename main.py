from data_loading import load_data
from data_visualization import (plot_pairplot, plot_correlation_matrix,
                                plot_confusion_matrices, plot_roc_curves,
                                plot_model_comparison, plot_learning_curves,
                                print_metrics_table, plot_threshold_analysis)
from data_preprocessing import preprocess_data
from model_training import train_models, evaluate_models_with_thresholds, optimize_thresholds
from prediction import predict_new_data
import joblib
import os
import numpy as np


def main():
    # Configuration
    filepath = "dataset/diabetes.csv"
    selected_features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age', 'Outcome']
    target = 'Outcome'

    print("=== SmartGluco Diabetes Prediction System ===")
    print("Enhanced with Class Imbalance Handling and Theoretical Framework Integration")

    # Load data
    data = load_data(filepath)

    # Visualize data
    plot_pairplot(data, selected_features)
    plot_correlation_matrix(data, selected_features)

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, poly = preprocess_data(
        data, selected_features, target)

    # Train models with enhanced class imbalance handling
    print("\n=== Training Models with Enhanced Class Imbalance Handling ===")
    models = train_models(X_train, y_train, use_smote=True)

    # Optimize thresholds for better recall
    print("\n=== Optimizing Prediction Thresholds ===")
    threshold_results = optimize_thresholds(models, X_test, y_test)

    # Evaluate models with optimized thresholds
    print("\n=== Evaluating Models with Optimized Thresholds ===")
    results = evaluate_models_with_thresholds(models, X_train, X_test, y_train, y_test, threshold_results)

    # Print comprehensive metrics table
    print_metrics_table(results)

    # Plot evaluation metrics
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    plot_model_comparison(results)
    plot_learning_curves(models, X_train, y_train)
    plot_threshold_analysis(threshold_results)

    # Prepare models for saving with enhanced information
    models_to_save = {}
    for name, result in results.items():
        if 'error' not in result:
            models_to_save[name] = {
                'model': result['model'],
                'test_accuracy_standard': result['test_accuracy_standard'],
                'test_accuracy_optimized': result['test_accuracy_optimized'],
                'recall_standard': result['recall_standard'],
                'recall_optimized': result['recall_optimized'],
                'optimal_threshold': result['optimal_threshold'],
                'roc_auc': result['roc_auc']
            }

    # Create directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)

    # Save models and preprocessing objects with enhanced metadata
    joblib.dump(models_to_save, 'saved_models/models.pkl')
    joblib.dump(scaler, 'saved_models/scaler.pkl')
    joblib.dump(poly, 'saved_models/poly.pkl')

    # Save threshold results for analysis
    joblib.dump(threshold_results, 'saved_models/threshold_results.pkl')

    print("\n=== Model Performance Summary ===")
    for name, result in results.items():
        if 'error' not in result:
            print(f"\n{name}:")
            print(
                f"  Standard - Accuracy: {result['test_accuracy_standard']:.3f}, Recall: {result['recall_standard']:.3f}")
            print(
                f"  Optimized - Accuracy: {result['test_accuracy_optimized']:.3f}, Recall: {result['recall_optimized']:.3f}")
            print(f"  Recall Improvement: {result['recall_improvement']:.3f}")
            print(f"  Optimal Threshold: {result['optimal_threshold']:.3f}")

    print("\nModels and preprocessing objects saved successfully in 'saved_models' directory.")

    # Make predictions on new data (using best model based on optimized performance)
    best_model_name = max(results, key=lambda x: results[x]['test_accuracy_optimized'])
    best_model = results[best_model_name]['model']
    print(f"\nUsing {best_model_name} (optimized) for predictions...")
    predict_new_data(best_model, scaler, poly)

    print("\n=== Theoretical Framework Integration ===")
    print("SmartGluco incorporates established psychological theories:")
    print("• Health Belief Model: Enhances perceived susceptibility through risk assessment")
    print("• Cognitive-Behavioral Principles: Real-time feedback reinforces health behaviors")
    print("• Technology Acceptance Model: User-friendly interface for clinical adoption")
    print("• Self-Determination Theory: Supports autonomy, competence, and relatedness")

    print("\nModel evaluation and visualization completed.")


if __name__ == "__main__":
    main()
