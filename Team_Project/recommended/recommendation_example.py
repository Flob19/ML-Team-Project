import pandas as pd
import numpy as np

data = pd.read_csv("./weather_based_predictions.csv")



# Filter for Coffee category
coffee_data = data[data['product_category'] == 'Coffee']

# Get unique predictions for each hour to avoid repetition
# Assuming 'Hour' and 'Predicted_Product' are the relevant columns
unique_predictions = coffee_data.groupby('timestamp')['predicted_product_detail'].unique()

# Print the results
for hour, products in unique_predictions.items():
    print(f"Hour {hour}: {', '.join(products)}")