from data_loading import load_data
from data_visualization import (plot_pairplot, plot_correlation_matrix,
                               plot_confusion_matrices, plot_roc_curves,
                               plot_model_comparison, plot_learning_curves)
from data_preprocessing import preprocess_data
from model_training import train_models, evaluate_models
from prediction import predict_new_data

def main():
    # Configuration
    filepath = "dataset/diabetes.csv"
    selected_features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age', 'Outcome']
    target = 'Outcome'

    # Load data
    data = load_data(filepath)

    # Visualize data
    plot_pairplot(data, selected_features)
    plot_correlation_matrix(data, selected_features)

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, poly = preprocess_data(
        data, selected_features, target)

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate models
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    # Plot evaluation metrics
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    plot_model_comparison(results)
    plot_learning_curves(models, X_train, y_train)

    # Make predictions on new data (using best model)
    best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
    best_model = results[best_model_name]['model']
    print(f"\nUsing {best_model_name} for predictions...")
    predict_new_data(best_model, scaler, poly)

    print("Model evaluation and visualization completed.")

if __name__ == "__main__":
    main()