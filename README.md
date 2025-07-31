# IR Drop Model Performance Dashboard

I generated a synthetic dataset that simulates IR drop in a VLSI chip using engineering-inspired features like:

Power density

Current density

Resistance

Via density

Layer type (which affects conductivity)

IR Drop (voltage drop) was calculated using a weighted linear formula. Noise was added to make it realistic.

✅ Purpose Achieved:

Created a realistic dataset for machine learning.

Embedded physical insight into the IR drop formula.

Then Loaded the above dataset.

Encoded categorical variable (layer_type).

Split it into training and testing.

Applied multiple regression models:

Linear Regression

Ridge

Lasso

Decision Tree

Random Forest

Gradient Boosting

Stacking Regressor (meta-model)

Performed:

Hyperparameter tuning

K-Fold cross-validation

Calculated performance metrics (R², MAE, RMSE)

Plotted actual vs predicted graphs.

Created a GUI with:

Results table

Export buttons (CSV, dataset)

Graph visualizations

## Features

- Synthetic dataset generator with realistic electrical parameters
- Multiple regression models:
  - Linear, Ridge, Lasso
  - Decision Tree, Random Forest, Gradient Boosting
  - Stacking Regressor (Meta Model)
- Model evaluation with RMSE, MAE, MSE, R²
- Tkinter GUI dashboard with performance viewer, graph plotting, and export buttons

I simulated a realistic chip design problem (IR drop) and solved it using end-to-end machine learning:

Data generation with physical realism

Model training and optimization

Model comparison

GUI for presentation


