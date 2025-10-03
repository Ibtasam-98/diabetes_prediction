
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split


def preprocess_data(data, features, target, test_size=0.2, random_state=42):
    # Select features
    data = data[features]

    # Split features and target
    X = data.drop(target, axis=1)
    y = data[target]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Add polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test, scaler, poly

