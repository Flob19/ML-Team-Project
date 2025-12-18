# Coffee Shop Helper

## Overview


## Installation
 - Create a conda environment:
   ```bash
   conda create -n coffee_shop_helper
   conda activate coffee_shop_helper
   ```
    - Install required packages:
    ```bash
    conda install --file requirements.txt
    ```
## Usage
    - Run the dashboard:
    ```bash
    streamlit run Team_Project/dashboard.py
    ```
## Features
 - In the order forecast tab, you can view demand prediction. With the sidebar, you can switch between Random Forest, MLP & Linear Regression models.
- In the recommendations tab, you can view product recommendations generated with a Decision Tree model.