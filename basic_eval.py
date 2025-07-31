import pandas as pd

df = pd.read_csv('synthetic_ir_drop_dataset.csv')

df.isnull().sum()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df[['power_density', 'current_density','resistance']] = scaler.fit_transform(
    df[['power_density', 'current_density','resistance']]
)

df = pd.get_dummies(df, columns=['layer_type'])

print("Dataset preview: ", df.head())

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

X = df.drop(columns=['ir_drop'])
y = df['ir_drop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random forest regression

Random_model = RandomForestRegressor()
Random_model.fit(X_train, y_train)
Random_preds = Random_model.predict(X_test)
Random_mse = mean_squared_error(y_test, Random_preds)
Random_mae = mean_absolute_error(y_test, Random_preds)
Random_rmse = np.sqrt(Random_mse)
Random_r2 = r2_score(y_test, Random_preds)

print("Random Forest: \n")
print(f"MSE:  {Random_mse:.4f}")
print(f"MAE:  {Random_mae:.4f}")
print(f"RMSE:  {Random_rmse:.4f}")
print(f"r2:  {Random_r2:.4f}\n")

#Linear regression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_preds)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, lr_preds)

print("Linear Regression: \n")
print(f"MSE: {lr_mse:.4f}")
print(f"MAE:  {lr_mae:.4f}")
print(f"RMSE:  {lr_rmse:.4f}")
print(f"r2:  {lr_r2:.4f}\n")

#Ridge regression

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_preds)
ridge_mae = mean_absolute_error(y_test, ridge_preds)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test, ridge_preds)

print("Ridge Regression: \n")
print(f"MSE:  {ridge_mse:.4f}")
print(f"MAE:  {ridge_mae:.4f}")
print(f"RMSE:  {ridge_rmse:.4f}")
print(f"r2:  {ridge_r2:.4f}\n")

# Lasso Regression after optimization

lasso = Lasso(max_iter=10000)
lasso_params = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
grid_lasso = GridSearchCV(lasso, lasso_params, cv=5)
grid_lasso.fit(X_train, y_train)
preds_lasso = grid_lasso.predict(X_test)
lasso_mse = mean_squared_error(y_test, preds_lasso)
lasso_mae = mean_absolute_error(y_test, preds_lasso)
lasso_rmse = np.sqrt(lasso_mse)
lasso_r2 = r2_score(y_test, preds_lasso)

print("Lasso regression after hyperparamter tuning: \n")
print("Best Lasso alpha: ", grid_lasso.best_params_)
print(f"MSE: {lasso_mse:.4f}")
print(f"MAE:  {lasso_mae:.4f}")
print(f"RMSE:  {lasso_rmse:.4f}\n")
print(f"r2:  {lasso_r2:.4f}\n")

#Decision Tree regression after optimization

tree_opt = DecisionTreeRegressor(random_state=42)
tree_params = {
    'max_depth' : [2, 5, 10, 20],
    'min_samples_split' : [2, 5, 10]
}
grid_tree = GridSearchCV(tree_opt, tree_params, cv=5)
grid_tree.fit(X_train, y_train)
preds_tree = grid_tree.predict(X_test)
tree_mse = mean_squared_error(y_test, preds_tree)
tree_mae = mean_absolute_error(y_test, preds_tree)
tree_rmse = np.sqrt(tree_mse)
tree_r2 = r2_score(y_test, preds_tree)

print("Decision Tree after hyperparamter tuning: \n")
print("Best Decision Tree params: ", grid_tree.best_params_)
print(f"MSE: {tree_mse:.4f}")
print(f"MAE:  {tree_mae:.4f}")
print(f"RMSE:  {tree_rmse:.4f}\n")
print(f"r2:  {tree_r2:.4f}\n")

#Gradient Boosting Regression

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)
gbr_preds = gbr.predict(X_test)
gbr_mse = mean_squared_error(y_test, gbr_preds)
gbr_mae = mean_absolute_error(y_test, gbr_preds)
gbr_rmse = np.sqrt(gbr_mse)
gbr_r2 = r2_score(y_test, gbr_preds)

print("Gradient Boosting Regression: \n")
print(f"MSE: {gbr_mse:.4f}")
print(f"MAE:  {gbr_mae:.4f}")
print(f"RMSE:  {gbr_rmse:.4f}")
print(f"r2: {gbr_r2:.4f}\n")

# K-Fold cross-validation

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest" : Random_model,
    "Linear Regression" : lr_model,
    "Ridge Regression" : ridge,
    "Lasso Regression" : lasso,
    "Decision Tree" : tree_opt,
    "Gradient Boosting" : gbr
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=kfold)
    rmse_scores = -scores
    print(f"{name} - RMSE per fold: {rmse_scores}\n")
    print(f"{name} - Mean RMSE: {rmse_scores.mean():.4f}, Std: {rmse_scores.std():.4f}\n")    
    

# Combing all models into one meta model

base_models = [
    ('rf', Random_model),
    ('lr', lr_model),
    ('ridge', ridge),
    ('lasso', lasso),
    ('tree', tree_opt),
    ('gbr', gbr)
]

meta_model = LinearRegression()

stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=True
)

stacking_model.fit(X_train, y_train)
stacking_preds = stacking_model.predict(X_test)
stack_mse = mean_squared_error(y_test, stacking_preds)
stack_mae = mean_absolute_error(y_test, stacking_preds)
stack_rmse = np.sqrt(mean_squared_error(y_test, stacking_preds))
stack_r2 = r2_score(y_test, stacking_preds)

print("Stacking Regressor: \n")
print(f"MSE: {stack_mse:.4f}")
print(f"MAE:  {stack_mae:.4f}")
print(f"RMSE:  {stack_rmse:.4f}")
print(f"r2: {stack_r2:.4f}\n")

stack_scores = cross_val_score(stacking_model, X, y, scoring='neg_root_mean_squared_error', cv=5)
stack_rmse_scores = -scores

print(f"Stacked Model - CV RMSE per fold: {stack_rmse_scores}\n")
print(f"Mean RMSE: {stack_rmse_scores.mean():.4f}, Std: {stack_rmse_scores.std():.4f}\n")


# # Plotting comparison graphs

# performance_data = {
#     'Model' : [
#         'Linear Reg', 'Ridge Reg', 'Lasso Reg',
#         'Random Forest', 'Decision Tree', 'Gradient Boost',
#         'Stack Reg'
#     ],
#     'R2 Score' : [0.9736, 0.9736, 0.9733, 0.9621, 0.9293, 0.9694, 0.9736],
#     'MSE' : [0.0025, 0.0025, 0.0025, 0.0035, 0.0066, 0.0028, 0.0025],
#     'MAE' : [0.0396, 0.0396, 0.0399, 0.0468, 0.0646, 0.0425, 0.0396],
#     'RMSE' : [0.0495, 0.0495, 0.0498, 0.0593, 0.0811, 0.0534, 0.0495],
    
# }

# performace_df = pd.DataFrame(performance_data)

# fig, axes = plt.subplots(2, 2, figsize=(14,10))
# metrics = ['R2 Score', 'MSE', 'MAE', 'RMSE']
# colors = ['skyblue', 'lightgreen', 'salmon', 'orange']

# for i, metric in enumerate(metrics):
#     ax = axes[i // 2, i %2]
#     performace_df.sort_values(by=metric, ascending=(metric != 'R2 Score')).plot(
#         x='Model', y=metric, kind='bar', ax=ax, color=colors[i], legend=False
#     )
#     ax.set_title(f'{metric} Comparison')
#     ax.set_ylabel(metric)
#     ax.set_xlabel('Model')
#     ax.grid(axis='y')
    
# plt.tight_layout()
# plt.show()

#GUI to show results
model_results = [
    {"Model": "Linear Reg", "R2": 0.9736, "MSE": 0.0025, "MAE": 0.0396, "RMSE": 0.0495},
    {"Model": "Ridge Reg", "R2": 0.9736, "MSE": 0.0025, "MAE": 0.0396, "RMSE": 0.0495},
    {"Model": "Lasso Reg", "R2": 0.9733, "MSE": 0.0025, "MAE": 0.0399, "RMSE": 0.0498},
    {"Model": "Decision Tree", "R2": 0.9621, "MSE": 0.0035, "MAE": 0.0468, "RMSE": 0.0593},
    {"Model": "Random Forest", "R2": 0.9293, "MSE": 0.0066, "MAE": 0.0646, "RMSE": 0.0811},
    {"Model": "Gradient Boosting", "R2": 0.9694, "MSE": 0.0028, "MAE": 0.0425, "RMSE": 0.0534},
    {"Model": "Stack Reg", "R2": 0.9736, "MSE": 0.0025, "MAE": 0.0396, "RMSE": 0.0495},

]

dataset = pd.read_csv('synthetic_ir_drop_dataset.csv')

root = tk.Tk()
root.title("Model Performance Dashboard")
root.geometry("1000x700")

tree_view = ttk.Treeview(root, columns=("Model", "R2", "MSE", "MAE", "RMSE"), show='headings')
for col in ("Model", "R2", "MSE", "MAE", "RMSE"):
    tree_view.heading(col, text=col)
    tree_view.column(col, width=150)

for result in model_results:
    tree_view.insert("", "end", values=(result["Model"], result["R2"], result["MSE"], result["MAE"], result["RMSE"]))

tree_view.pack(pady=10)

#Button: Export results to CSV
def export_csv():
    model_df = pd.DataFrame(model_results)
    model_df.to_csv("model_performance.csv", index=False)
    messagebox.showinfo("Success", "Model performance exported as model_performance.csv")

btn_export = tk.Button(root, text="Export Results to CSV", command=export_csv)
btn_export.pack(pady=5)

# Button: Download dataset
def download_dataset():
    dataset.to_csv("dataset.csv", index=False)
    messagebox.showinfo("Success", "Dataset saved as dataset.csv")

btn_dataset = tk.Button(root, text="Download Dataset", command=download_dataset)
btn_dataset.pack(pady=5)

# Plot performance metrics
def show_graph_window():
    new_window = tk.Toplevel(root)
    new_window.title("Performance Graphs")
    new_window.geometry("1000x700")

    df = pd.DataFrame(model_results)
    metrics = ["R2", "MSE", "MAE", "RMSE"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        axs[i].bar(df["Model"], df[metric], color="skyblue")
        axs[i].set_title(metric)
        axs[i].tick_params(axis='x', labelrotation=45)

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)

btn_plot = tk.Button(root, text="Show Performance Graphs", command=show_graph_window)
btn_plot.pack(pady=5)

# --- START ---
root.mainloop()
