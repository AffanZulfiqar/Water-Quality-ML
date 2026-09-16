import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

def main():
    features = ['ph', 'Solids', 'Sulfate', 'Conductivity', 'Turbidity']
    
    print("Loading data and model...")
    df_wqp = pd.read_csv('data/wqp_clean.csv')
    X_wqp = df_wqp[features]
    
    with open('results/wqp_5feat_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('results/wqp_5feat_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    X_wqp_s = scaler.transform(X_wqp)
    
    print("Running predictions...")
    preds_wqp = model.predict_proba(X_wqp_s)[:, 1]
    np.save('results/wqp_5feat_preds.npy', preds_wqp)
    
    preds_kaggle = np.load('results/kaggle_5feat_test_preds.npy')
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(preds_kaggle, label='Kaggle Test Set (Original)', fill=True, alpha=0.5)
    sns.kdeplot(preds_wqp, label='USGS WQP Set (External)', fill=True, alpha=0.5)
    plt.title('Domain Shift Analysis: Prediction Probability Distributions')
    plt.xlabel('Predicted Probability of Potability')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/figures/domain_shift_distribution.png', dpi=300)
    plt.close()
    
    print("Running SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_wqp_s)
    
    if isinstance(shap_values, list):
        shap_values_pos = np.array(shap_values[1])
    elif len(shap_values.shape) == 3:
        shap_values_pos = np.array(shap_values[:, :, 1])
    else:
        shap_values_pos = np.array(shap_values)
        
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_pos, X_wqp, show=False)
    plt.title('SHAP Feature Importance (External USGS WQP Data)')
    plt.tight_layout()
    plt.savefig('results/figures/wqp_shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    mean_shap = np.abs(shap_values_pos).mean(axis=0).flatten()
    shap_df = pd.DataFrame({'Feature': features, 'Mean_SHAP_WQP': mean_shap})
    shap_df = shap_df.sort_values(by='Mean_SHAP_WQP', ascending=False)
    shap_df.to_csv('results/wqp_shap_importance.csv', index=False)
    
    print("Analysis complete. Saved distribution and SHAP plots to results/figures/.")

if __name__ == '__main__':
    main()
