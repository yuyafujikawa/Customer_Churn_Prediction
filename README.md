# Customer_Churn_Prediction
Customer Churn Prediction based on several numerical features. Classic classification problem. 

Data Source: https://www.kaggle.com/code/kmalit/bank-customer-churn-prediction/data

#### Future Improvement 1 - Create the ROC curve (False Positive vs. True Positive) to visualize the performance of classification - Greater the area under the curve, better the performance. - Done 19/09/2022

Reference:  https://www.statology.org/plot-multiple-roc-curves-python/

##### Example below:
Model instance definition and fitting has been done in previous step. Refer to JupyterNotebook for more details.

```
from sklearn.metrics import roc_curve, roc_auc_score

plt.figure(figsize=(10,10))

y_pred = knnc.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred)
auc = round(roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="KNeighborsClassifier AUC="+str(auc))
```

#### Future Improvement 2 - Run a more refined GridsearchCV with more parameters and respecitve values/range for the MLPClassifier for fine-tuning. Used only alpha in the first version, but could *incorporate other parameters such as hidden_layer_sizes, activation, solver and learning rate*. 

Reference: https://datascience.stackexchange.com/questions/36049/how-to-adjust-the-hyperparameters-of-mlp-classifier-to-get-more-perfect-performa

#### Set probability param to True before fitting certain models so that the ROC Curve can be plotted.(e.g. for SVC). 

#### Example of parameter grid for GridsearchCV
```
param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
```
