import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def main():
    df = pd.read_csv('data/water_potability.csv')
    features = ['ph', 'Solids', 'Sulfate', 'Conductivity', 'Turbidity']
    
    # Fill missing values using training medians (mimicking original pipeline)
    # Actually, the original pipeline drops exact duplicates and then imputes.
    df = df.drop_duplicates()
    X = df[features]
    y = df['Potability']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Impute medians from train
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train_s, y_train)
    
    with open('results/wqp_5feat_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('results/wqp_5feat_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Trained Kaggle 5-feature model (n={len(X_train)} train, {len(X_test)} test).")
    
    np.save('results/kaggle_5feat_test_preds.npy', model.predict_proba(X_test_s)[:, 1])
    pd.DataFrame(X_train_s, columns=features).to_csv('results/kaggle_5feat_Xtrain_s.csv', index=False)

if __name__ == '__main__':
    main()
