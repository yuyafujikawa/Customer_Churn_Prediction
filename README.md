# Customer_Churn_Prediction
Customer Churn Prediction based on several numerical features. Classic classification problem. 

Data Source: https://www.kaggle.com/code/kmalit/bank-customer-churn-prediction/data

#### Future Improvement - Create the ROC curve (False Positive vs. True Positive) to visualize the performance of classification - Greater the area under the curve, better the performance. 

Refer to: https://www.projectpro.io/recipes/plot-roc-curve-in-python

#### Set probability param to True before fitting models (e.g. for SVC). 

```
from sklearn.metrics import roc_curve, roc_auc_score
    
y_score1 = knnc.predict_proba(X_test)[:,1]
y_score2 = svc.predict_proba(X_test)[:,1]

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, y_score2)
print('roc_auc_score for KNN: ', roc_auc_score(y_test, y_score1))
print('roc_auc_score for SVC: ', roc_auc_score(y_test, y_score2))

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - KNN')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - SVC')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
```
