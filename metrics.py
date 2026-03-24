#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def metrics(save_path):
    pred = pd.read_csv(os.path.join(save_path, 'llm_predictions.csv'))
    
    label_map = {'流失':1, '未流失':0}
    pred['y_true'] = pred['real_label'].map(label_map)
    pred['y_pred'] = pred['predicted_label'].map(label_map)
    pred.loc[pred['predicted_label'].str.contains('未流失'), 'y_pred'] = 0
    pred.loc[pred['y_pred'].isna(), 'y_pred'] = 1
    
    for col in ['y_true','y_pred']:
        pred[col] = pred[col].map(int)

    y_true = pred['y_true'].values
    y_pred = pred['y_pred'].values
    auc_score = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"AUC Score: {auc_score:.3f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    fpr, tpr, thresholds = roc_curve(pred['y_true'].values, pred['y_pred'].values)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'metric for the model')
    parser.add_argument('--save_path', type=str, default='./', help='save cleaned data or outputs')
    args = parser.parser_args()

    metrics(args.save_path)

