# House Price Predictor(Using Train-Test-Split)

## Overview
This project is a simple **House Price Predictor** using **Linear Regression**. It takes input features such as square footage, number of bedrooms, and number of bathrooms to predict house prices. The dataset is generated within the script, and the model is trained using **scikit-learn**.

## Features
- Uses a **Linear Regression** model.
- Splits data into training (80%) and testing (20%) sets.
- Evaluates the model using **Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE)**.
- Visualizes the **Actual vs. Predicted** house prices using Matplotlib.

## Installation
Ensure you have **Python 3.x** installed along with the required libraries:
```sh
pip install pandas numpy matplotlib scikit-learn
```

## Usage
Run the Python script:
```sh
python house_price_predictor.py
```

## Code Explanation
1. **Dataset Creation:**
   - A sample dataset is created using pandas.
2. **Data Preprocessing:**
   - Splitting into independent (`Square_Feet`, `Bedrooms`, `Bathrooms`) and dependent (`Price`) variables.
   - Splitting into **training (80%) and testing (20%)** datasets.
3. **Model Training:**
   - Using **Linear Regression** to fit the training data.
4. **Prediction & Evaluation:**
   - Predicts house prices on the test set.
   - Evaluates the model using **MAE, MSE, and RMSE**.
5. **Visualization:**
   - A scatter plot comparing **actual vs. predicted** prices.

## Sample Output
```
Trained Data: (8,3)
Tested Data: (3,3)
The Mean Absolute Error is: 0.0
The Mean Squared Error is: 0.0
The Root Mean Squared Error: 0.0
```
(A perfect prediction may appear in this example due to the simplicity of the dataset.)

## Graph Output
The script generates a **scatter plot** comparing the actual and predicted house prices with an ideal fit line (`y = x`).

## License
This project is open-source and free to use.

---
Developed using **Python, scikit-learn, pandas, numpy, and Matplotlib**.

