import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from collections import defaultdict

# File paths
initial_dataroot = '../Data/sales_with_weather_tx.xlsx'
training_dataroot = '../Data/sales_with_weather_tx_train_detailed.csv'
testing_dataroot = '../Data/sales_with_weather_tx_test_detailed.csv'
output_path = '../Data/product_count_predictions_detailed.csv'

print("=== Advanced Product Breakdown Analysis ===\n")


def preprocess_dataframe_detailed(df):
    """
    Preprocess data to predict unique product counts PER CATEGORY.
    Creates multi-output target: [coffee, tea, drinking_chocolate, bakery]
    """
    df = df.copy()

    # Convert datetime and handle decimals
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    for col in ['temperature_C', 'rain_mm', 'snow_cm', 'wind_speed_kmh']:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    # Extract hour features
    df['hour'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_morning'] = df['hour'].isin([7, 8, 9, 10, 11]).astype(int)
    df['is_afternoon'] = df['hour'].isin([12, 13, 14, 15, 16, 17]).astype(int)
    df['is_evening'] = df['hour'].isin([18, 19, 20, 21, 22]).astype(int)

    # Group by hour and category, then count unique products per category
    hourly_category = df.groupby([
        df['datetime'].dt.date, 'hour', 'product_category'
    ]).agg({
        'product_detail': 'nunique',
        'temperature_C': 'mean',
        'rain_mm': 'mean',
        'snow_cm': 'mean',
        'cloud_cover_pct': 'mean',
        'wind_speed_kmh': 'mean'
    }).reset_index()

    # Pivot to create one column per category
    hourly_pivot = hourly_category.pivot_table(
        index=['datetime', 'hour', 'temperature_C', 'rain_mm', 'snow_cm',
               'cloud_cover_pct', 'wind_speed_kmh'],
        columns='product_category',
        values='product_detail',
        fill_value=0
    ).reset_index()

    # Flatten column names
    hourly_pivot.columns.name = None
    category_columns = ['Coffee', 'Tea', 'Drinking Chocolate', 'Bakery']
    for col in category_columns:
        if col not in hourly_pivot.columns:
            hourly_pivot[col] = 0

    # Add time features
    hourly_pivot['hour_sin'] = np.sin(2 * np.pi * hourly_pivot['hour'] / 24)
    hourly_pivot['hour_cos'] = np.cos(2 * np.pi * hourly_pivot['hour'] / 24)
    hourly_pivot['is_morning'] = hourly_pivot['hour'].isin([7, 8, 9, 10, 11]).astype(int)
    hourly_pivot['is_afternoon'] = hourly_pivot['hour'].isin([12, 13, 14, 15, 16, 17]).astype(int)
    hourly_pivot['is_evening'] = hourly_pivot['hour'].isin([18, 19, 20, 21, 22]).astype(int)

    # Create total count column
    hourly_pivot['total_unique_products'] = hourly_pivot[category_columns].sum(axis=1)

    return hourly_pivot, category_columns


# Check if data needs to be processed
regenerate = False
if os.path.exists(training_dataroot) and os.path.exists(testing_dataroot):
    try:
        df_test = pd.read_csv(testing_dataroot, nrows=5)
        required = ['Coffee', 'Tea', 'Drinking Chocolate', 'Bakery', 'total_unique_products']
        if not all(col in df_test.columns for col in required):
            regenerate = True
            print("Existing files lack category breakdown. Regenerating...")
    except:
        regenerate = True
else:
    regenerate = True

if regenerate:
    print("Creating detailed category-level dataset...")
    df_raw = pd.read_excel(initial_dataroot, sheet_name=0)
    df_processed, category_columns = preprocess_dataframe_detailed(df_raw)

    print(f"Processed data shape: {df_processed.shape}")
    print(f"Category columns: {category_columns}")
    print(f"Sample data:\n{df_processed[['datetime', 'hour'] + category_columns + ['total_unique_products']].head()}\n")

    # Split dataset
    train_df, test_df = train_test_split(
        df_processed,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_df.to_csv(training_dataroot, index=False)
    test_df.to_csv(testing_dataroot, index=False)
    print(f"Created detailed training data: {train_df.shape}")
    print(f"Created detailed testing data: {test_df.shape}\n")

# Load datasets
df_train = pd.read_csv(training_dataroot)
df_test = pd.read_csv(testing_dataroot)

# Define features and targets
numeric_features = ['temperature_C', 'rain_mm', 'snow_cm', 'cloud_cover_pct', 'wind_speed_kmh']
cyclical_features = ['hour_sin', 'hour_cos']
time_category_features = ['is_morning', 'is_afternoon', 'is_evening']
feature_columns = numeric_features + cyclical_features + time_category_features

target_columns = ['Coffee', 'Tea', 'Drinking Chocolate', 'Bakery']
total_column = 'total_unique_products'

print("=== Multi-Output Model Training ===\n")
print(f"Features: {feature_columns}")
print(f"Targets: {target_columns}\n")

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True))
        ]), numeric_features),
        ('pass', 'passthrough', cyclical_features + time_category_features)
    ]
)

# Create multi-output model
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0, random_state=42))
])

# Prepare data
X_train = df_train[feature_columns]
y_train = df_train[target_columns]
X_test = df_test[feature_columns]
y_test = df_test[target_columns]

# Train model
print("Training multi-output model...")
model_pipeline.fit(X_train, y_train)

# Predict all categories
y_pred = model_pipeline.predict(X_test)
pred_df = pd.DataFrame(y_pred, columns=[f'pred_{col}' for col in target_columns])

# Calculate total predicted
df_test['predicted_total'] = pred_df.sum(axis=1)
df_test['actual_total'] = df_test[target_columns].sum(axis=1)

# Calculate MAPE for total
total_mape = mean_absolute_percentage_error(df_test['actual_total'], df_test['predicted_total'])
print(f"\nTotal Unique Products MAPE: {total_mape:.2%}")

# Add category predictions to test dataframe
for i, col in enumerate(target_columns):
    df_test[f'pred_{col}'] = np.round(y_pred[:, i], 1)

# Save detailed predictions
df_test.to_csv(output_path, index=False)
print(f"\nDetailed predictions saved to: {output_path}")

# Feature importance analysis
print("\n=== Weather Impact Analysis by Product Category ===")
print("Analyzing how weather affects each product category...\n")


def get_feature_importance(category_name, category_index):
    """Calculate feature importance for a specific category"""
    # Get the preprocessor's feature names
    numeric_transformer = model_pipeline.named_steps['preprocessor'].named_transformers_['num']
    poly_features = numeric_transformer.named_steps['poly']

    # Get polynomial feature names
    numeric_feature_names = numeric_features
    poly_feature_names = poly_features.get_feature_names_out(numeric_feature_names)

    # Combine all feature names
    all_feature_names = list(poly_feature_names) + cyclical_features + time_category_features

    # Get coefficients for this category
    coeffs = model_pipeline.named_steps['regressor'].coef_[category_index]

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'coefficient': coeffs
    })

    # Calculate absolute importance
    importance_df['importance'] = np.abs(importance_df['coefficient'])
    importance_df = importance_df.sort_values('importance', ascending=False)

    return importance_df.head(5)


# Show top weather influences for each category
for i, category in enumerate(target_columns):
    print(f"\nTop Weather Influences for {category}:")
    importance = get_feature_importance(category, i)
    print(importance.to_string(index=False))


def predict_with_breakdown(weather_params, hour=12):
    """
    Predict the breakdown of unique products per category for given weather.

    Parameters:
    -----------
    weather_params : dict
        Dictionary with weather parameters
    hour : int
        Hour of day (0-23)

    Returns:
    --------
    dict: Breakdown of predictions per category
    """
    # Create cyclical hour features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # Create time-of-day categories
    is_morning = 1 if hour in [7, 8, 9, 10, 11] else 0
    is_afternoon = 1 if hour in [12, 13, 14, 15, 16, 17] else 0
    is_evening = 1 if hour in [18, 19, 20, 21, 22] else 0

    # Create input dataframe
    input_data = pd.DataFrame({
        'temperature_C': [weather_params['temperature_C']],
        'rain_mm': [weather_params['rain_mm']],
        'snow_cm': [weather_params['snow_cm']],
        'cloud_cover_pct': [weather_params['cloud_cover_pct']],
        'wind_speed_kmh': [weather_params['wind_speed_kmh']],
        'hour_sin': [hour_sin],
        'hour_cos': [hour_cos],
        'is_morning': [is_morning],
        'is_afternoon': [is_afternoon],
        'is_evening': [is_evening]
    })

    # Make predictions for all categories
    predictions = model_pipeline.predict(input_data)[0]

    # Create breakdown dictionary
    breakdown = {}
    for i, category in enumerate(target_columns):
        breakdown[category] = max(0, round(predictions[i], 1))

    breakdown['total_unique_products'] = sum(breakdown.values())

    return breakdown


def analyze_product_popularity(df):
    """
    Analyze which specific products are most popular in different conditions.
    """
    print("\n=== Historical Product Popularity Analysis ===\n")

    # Popular products by category
    for category in ['Coffee', 'Tea', 'Drinking Chocolate', 'Bakery']:
        print(f"Top 5 Most Popular {category} Products:")
        popular = df[df['product_category'] == category]['product_detail'].value_counts().head(5)
        for idx, (product, count) in enumerate(popular.items(), 1):
            print(f"  {idx}. {product} ({count} sales)")
        print()

    # Weather correlations
    print("Weather impact on categories:")
    weather_corr = df.groupby('product_category')[['temperature_C', 'rain_mm', 'cloud_cover_pct']].corr()
    print(weather_corr)


def main():
    """Test model with detailed breakdown"""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Detailed Product Breakdown Prediction")
    print("=" * 70)

    # Historical analysis
    df_raw = pd.read_excel(initial_dataroot, sheet_name=0)
    analyze_product_popularity(df_raw)

    # Test scenarios
    scenarios = [
        {
            'name': 'Cold Winter Morning (8 AM)',
            'weather': {
                'temperature_C': -2.5,
                'rain_mm': 0.0,
                'snow_cm': 3.2,
                'cloud_cover_pct': 85,
                'wind_speed_kmh': 25.0
            },
            'hour': 8
        },
        {
            'name': 'Warm Summer Afternoon (2 PM)',
            'weather': {
                'temperature_C': 28.0,
                'rain_mm': 0.0,
                'snow_cm': 0.0,
                'cloud_cover_pct': 15,
                'wind_speed_kmh': 5.5
            },
            'hour': 14
        },
        {
            'name': 'Rainy Evening (7 PM)',
            'weather': {
                'temperature_C': 12.0,
                'rain_mm': 8.5,
                'snow_cm': 0.0,
                'cloud_cover_pct': 95,
                'wind_speed_kmh': 18.0
            },
            'hour': 19
        }
    ]

    for scenario in scenarios:
        print(f"\n{'=' * 70}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'=' * 70}")

        result = predict_with_breakdown(scenario['weather'], scenario['hour'])

        print(f"\nWeather Conditions:")
        for key, value in scenario['weather'].items():
            print(f"  {key}: {value}")

        print(f"\nPredicted Unique Product Breakdown:")
        print(f"  {'Category':<25} {'Count':>8}")
        print(f"  {'-' * 25} {'-' * 8}")
        for category in target_columns:
            print(f"  {category:<25} {result[category]:>8.1f}")
        print(f"  {'-' * 25} {'-' * 8}")
        print(f"  {'TOTAL':<25} {result['total_unique_products']:>8.1f}")

        # Insights
        print(f"\nKey Insights:")
        dominant = max(target_columns, key=lambda x: result[x])
        print(f"  • Highest selling category: {dominant} ({result[dominant]:.1f} unique products)")
        print(f"  • Best time for hot drinks: {'Morning' if scenario['hour'] < 12 else 'Afternoon/Evening'}")
        if scenario['weather']['temperature_C'] < 5:
            print(f"  • Cold weather boosts hot drink sales")
        if scenario['weather']['rain_mm'] > 5:
            print(f"  • Rainy weather increases comfort food demand")

    # Interactive mode
    print("\n" + "=" * 70)
    print("Interactive Mode - Enter Your Own Weather")
    print("=" * 70)

    try:
        print("\nEnter weather parameters:")
        temp = float(input("Temperature (°C): "))
        rain = float(input("Rain (mm): "))
        snow = float(input("Snow (cm): "))
        cloud = float(input("Cloud cover (%): "))
        wind = float(input("Wind speed (km/h): "))
        hour_input = int(input("Hour (0-23): "))

        custom_weather = {
            'temperature_C': temp,
            'rain_mm': rain,
            'snow_cm': snow,
            'cloud_cover_pct': cloud,
            'wind_speed_kmh': wind
        }

        custom_result = predict_with_breakdown(custom_weather, hour_input)

        print(f"\n{'=' * 50}")
        print(f"YOUR PREDICTION RESULTS")
        print(f"{'=' * 50}")
        print(f"\nUnique Product Breakdown:")
        for category in target_columns:
            print(f"  {category}: {custom_result[category]:.1f}")
        print(f"  {'Total':<20} {custom_result['total_unique_products']:>8.1f}")

    except (ValueError, KeyboardInterrupt):
        print("\nInteractive test completed or cancelled.")


if __name__ == "__main__":
    main()