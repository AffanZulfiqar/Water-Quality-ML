"""
run_pipeline.py  — Complete end-to-end pipeline
Run from project root:
    python run_pipeline.py
"""

import json, logging, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).parent
DATA_CSV     = ROOT / "data" / "water_potability.csv"
METRICS      = ROOT / "results" / "metrics"
MODELS_DIR   = ROOT / "results" / "models"
FIGS_DIR     = ROOT / "results" / "figures"
TABLES_DIR   = ROOT / "results" / "tables"
SHAP_DIR     = ROOT / "results" / "shap"
ABLATION_DIR = ROOT / "results" / "ablation"
ROBUST_DIR   = ROOT / "results" / "robustness"

for d in [METRICS, MODELS_DIR, FIGS_DIR, TABLES_DIR, SHAP_DIR, ABLATION_DIR, ROBUST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, cross_validate)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, roc_auc_score, average_precision_score,
                              confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

def savefig(path, dpi=150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {Path(path).name}")

def evaluate(model, X, y, name, split):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1] if hasattr(model,"predict_proba") else y_pred.astype(float)
    return {
        "model": name, "split": split, "n": len(y),
        "accuracy":          round(accuracy_score(y, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, y_pred), 4),
        "f1_macro":          round(f1_score(y, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted":       round(f1_score(y, y_pred, average="weighted", zero_division=0), 4),
        "f1_class0":         round(f1_score(y, y_pred, pos_label=0, average="binary", zero_division=0), 4),
        "f1_class1":         round(f1_score(y, y_pred, pos_label=1, average="binary", zero_division=0), 4),
        "roc_auc":           round(roc_auc_score(y, y_prob), 4),
        "pr_auc":            round(average_precision_score(y, y_prob), 4),
    }

# ════════════ STEP 1 ════════════
log.info("STEP 1 — Load & Validate")
df = pd.read_csv(DATA_CSV)
n_before = len(df)
df = df.drop_duplicates()
log.info(f"Rows: {n_before} → {len(df)}")
FEATURES = [c for c in df.columns if c != "Potability"]
TARGET   = "Potability"
X = df[FEATURES]; y = df[TARGET]
vc = y.value_counts()
log.info(f"Class dist: {dict(vc)}")

# ════════════ STEP 2 ════════════
log.info("STEP 2 — Split + Preprocess (leakage-safe)")
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/(1-0.15), stratify=y_temp, random_state=SEED)
log.info(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

train_medians = X_train.median()
def impute(Xd): return Xd.fillna(train_medians)
Q1=impute(X_train).quantile(0.25); Q3=impute(X_train).quantile(0.75)
IQR=Q3-Q1; lo=Q1-1.5*IQR; hi=Q3+1.5*IQR
def cap(Xd): return Xd.clip(lower=lo, upper=hi, axis=1)
scaler = StandardScaler().fit(cap(impute(X_train)))
def proc(Xd): return pd.DataFrame(scaler.transform(cap(impute(Xd))), columns=FEATURES)

X_train_s=proc(X_train.copy()); X_val_s=proc(X_val.copy()); X_test_s=proc(X_test.copy())
X_train_s.to_csv(METRICS/"X_train.csv",index=False); X_val_s.to_csv(METRICS/"X_val.csv",index=False)
X_test_s.to_csv(METRICS/"X_test.csv",index=False)
y_train.to_csv(METRICS/"y_train.csv",index=False); y_val.to_csv(METRICS/"y_val.csv",index=False); y_test.to_csv(METRICS/"y_test.csv",index=False)

# ════════════ STEP 3 ════════════
log.info("STEP 3 — Hyperparameter Optimisation (RF, XGB, LGBM)")
class_ratio=float(vc[0]/vc[1])
models_base = {
    "Dummy":             DummyClassifier(strategy="most_frequent",random_state=SEED),
    "LogisticRegression":LogisticRegression(C=1.0,max_iter=1000,class_weight="balanced",solver="lbfgs",random_state=SEED,n_jobs=-1),
    "DecisionTree":      DecisionTreeClassifier(max_depth=6,min_samples_leaf=10,class_weight="balanced",random_state=SEED),
    "RandomForest":      RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=5,max_features=0.5,class_weight="balanced",n_jobs=-1,random_state=SEED),
    "SVM":               SVC(C=1.0,kernel="rbf",gamma="scale",class_weight="balanced",probability=True,random_state=SEED),
    "XGBoost":           XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.01,subsample=0.7,colsample_bytree=1.0,reg_alpha=0,scale_pos_weight=class_ratio,eval_metric="logloss",verbosity=0,random_state=SEED),
    "MLP":               MLPClassifier(hidden_layer_sizes=(128,64,32),activation="relu",max_iter=500,early_stopping=True,validation_fraction=0.1,random_state=SEED),
}
models=dict(models_base)

# ════════════ STEP 4 ════════════
log.info("STEP 4 — Train All Models + 5-Fold CV")
cv_outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
scoring={"accuracy":"accuracy","balanced_accuracy":"balanced_accuracy","f1_macro":"f1_macro","f1_weighted":"f1_weighted","roc_auc":"roc_auc","average_precision":"average_precision"}
all_val_metrics=[]; all_test_metrics=[]; cv_summary=[]; fitted_models={}; roc_data={}

for mname,model in models.items():
    log.info(f"  Training: {mname}")
    model.fit(X_train_s,y_train)
    fitted_models[mname]=model
    all_val_metrics.append(evaluate(model,X_val_s,y_val,mname,"val"))
    cvr=cross_validate(model,X_train_s,y_train,cv=cv_outer,scoring=scoring,n_jobs=-1)
    row={"model":mname}
    for k in scoring: row[f"{k}_mean"]=round(float(np.mean(cvr[f"test_{k}"])),4); row[f"{k}_std"]=round(float(np.std(cvr[f"test_{k}"])),4)
    cv_summary.append(row)
    log.info(f"    CV f1_macro={row['f1_macro_mean']:.4f}±{row['f1_macro_std']:.4f}  roc_auc={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f}")
    with open(MODELS_DIR/f"{mname}.pkl","wb") as f: pickle.dump(model,f)
    if hasattr(model,"predict_proba"):
        fpr,tpr,_=roc_curve(y_test,model.predict_proba(X_test_s)[:,1])
        roc_data[mname]=(fpr,tpr,float(roc_auc_score(y_test,model.predict_proba(X_test_s)[:,1])))

log.info("  Final test evaluation...")
for mname,model in fitted_models.items():
    tm=evaluate(model,X_test_s,y_test,mname,"test")
    all_test_metrics.append(tm)
    log.info(f"    {mname}: acc={tm['accuracy']}  f1_macro={tm['f1_macro']}  roc_auc={tm['roc_auc']}")

pd.DataFrame(all_val_metrics).to_csv(METRICS/"all_val_metrics.csv",index=False)
pd.DataFrame(all_test_metrics).to_csv(METRICS/"all_test_metrics.csv",index=False)
pd.DataFrame(cv_summary).to_csv(METRICS/"all_cv_results.csv",index=False)

# ════════════ STEP 5 ════════════
log.info("STEP 5 — EDA + Model Comparison Figures")
pal=sns.color_palette("viridis",len(FEATURES))
pal10=sns.color_palette("tab10",10)

# class dist
fig,ax=plt.subplots(figsize=(7,5)); counts=y.value_counts().sort_index()
bars=ax.bar(["Not Potable","Potable"],counts.values,color=["#E74C3C","#2ECC71"],width=0.5,edgecolor="white")
for bar,cnt in zip(bars,counts.values): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+15,f"{cnt}\n({100*cnt/len(y):.1f}%)",ha="center",fontsize=12)
ax.set_title("Class Distribution",fontsize=14,fontweight="bold"); ax.set_ylabel("Samples"); ax.set_ylim(0,counts.max()*1.25)
for s in ["top","right"]: ax.spines[s].set_visible(False)
savefig(FIGS_DIR/"eda_class_distribution.png")

# feature dists
fig,axes=plt.subplots(3,3,figsize=(14,10)); axes=axes.flatten()
for i,col in enumerate(FEATURES):
    axes[i].hist(X[col].dropna(),bins=40,color=pal[i],alpha=0.85,edgecolor="white")
    axes[i].set_title(col,fontsize=11,fontweight="bold"); axes[i].set_xlabel("Value",fontsize=8)
    for s in ["top","right"]: axes[i].spines[s].set_visible(False)
fig.suptitle("Feature Distributions",fontsize=14,fontweight="bold")
savefig(FIGS_DIR/"eda_feature_distributions.png")

# correlation
fig,ax=plt.subplots(figsize=(11,9)); corr=X.corr(); mask=np.triu(np.ones_like(corr,dtype=bool))
sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",cmap="coolwarm",center=0,linewidths=0.5,ax=ax,cbar_kws={"shrink":0.8})
ax.set_title("Feature Correlation Matrix",fontsize=14,fontweight="bold")
savefig(FIGS_DIR/"eda_correlation_matrix.png")

# missing values
miss=(df[FEATURES].isnull().sum()/len(df)*100).sort_values(ascending=True); miss=miss[miss>0]
if len(miss)>0:
    fig,ax=plt.subplots(figsize=(9,5))
    colors=["#E74C3C" if v>20 else "#F39C12" if v>10 else "#3498DB" for v in miss.values]
    bars=ax.barh(miss.index,miss.values,color=colors,edgecolor="white")
    for bar,pct in zip(bars,miss.values): ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,f"{pct:.1f}%",va="center",fontsize=10)
    ax.set_xlabel("Missing (%)"); ax.set_title("Missing Values by Feature",fontsize=14,fontweight="bold")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    savefig(FIGS_DIR/"eda_missing_values.png")

# boxplots (skipped due to seaborn NaN compatibility)
# fig,axes=plt.subplots(3,3,figsize=(14,10)); axes=axes.flatten()
# df_bp=X.copy(); df_bp["Class"]=y.map({0:"Not Potable",1:"Potable"})
# for i,feat in enumerate(FEATURES):
#     sns.boxplot(data=df_bp.dropna(subset=[feat]),x="Class",y=feat,palette={"Not Potable":"#E74C3C","Potable":"#2ECC71"},ax=axes[i],width=0.5)
#     axes[i].set_title(feat,fontsize=11,fontweight="bold"); axes[i].set_xlabel("")
#     for s in ["top","right"]: axes[i].spines[s].set_visible(False)
# fig.suptitle("Feature Distributions by Class",fontsize=14,fontweight="bold")
# savefig(FIGS_DIR/"eda_boxplots_by_class.png")

# model comparison bars
test_df=pd.DataFrame(all_test_metrics)
for metric in ["f1_macro","roc_auc","accuracy","balanced_accuracy"]:
    ds=test_df.sort_values(metric,ascending=True); pal2=sns.color_palette("viridis",len(ds))
    fig,ax=plt.subplots(figsize=(10,max(5,len(ds)*0.7)))
    bars=ax.barh(ds["model"],ds[metric],color=pal2,edgecolor="white",height=0.6)
    for bar,val in zip(bars,ds[metric]): ax.text(bar.get_width()+0.002,bar.get_y()+bar.get_height()/2,f"{val:.4f}",va="center",fontsize=10)
    ax.set_xlabel(metric.replace("_"," ").title()); ax.set_title(f"Model Comparison — {metric.replace('_',' ').title()}",fontsize=14,fontweight="bold")
    ax.set_xlim(0,min(1.0,ds[metric].max()*1.15))
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    savefig(FIGS_DIR/f"model_comparison_{metric}.png")

# confusion matrices top 3
for mname in test_df.nlargest(3,"f1_macro")["model"].tolist():
    cm=confusion_matrix(y_test,fitted_models[mname].predict(X_test_s))
    fig,ax=plt.subplots(figsize=(6,5))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Not Potable","Potable"],yticklabels=["Not Potable","Potable"],linewidths=0.5,ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion Matrix — {mname}",fontsize=13,fontweight="bold")
    savefig(FIGS_DIR/f"confusion_matrix_{mname}.png")

# ROC overlay
fig,ax=plt.subplots(figsize=(8,7))
for i,(mname,(fpr,tpr,auc_val)) in enumerate(roc_data.items()):
    if mname=="Dummy": continue
    ax.plot(fpr,tpr,label=f"{mname} (AUC={auc_val:.3f})",color=pal10[i],linewidth=2)
ax.plot([0,1],[0,1],"k--",linewidth=1,label="Random"); ax.set_xlabel("FPR",fontsize=12); ax.set_ylabel("TPR",fontsize=12)
ax.set_title("ROC Curves — All Models",fontsize=14,fontweight="bold"); ax.legend(loc="lower right",fontsize=9)
for s in ["top","right"]: ax.spines[s].set_visible(False)
savefig(FIGS_DIR/"roc_curves_all.png")

# ════════════ STEP 6 ════════════
log.info("STEP 6 — SHAP Explainability")
all_rankings={}
tree_keys=["RandomForest","XGBoost","DecisionTree"]

for mname in tree_keys+["LogisticRegression","SVM","MLP"]:
    if mname not in fitted_models: continue
    model=fitted_models[mname]; log.info(f"  SHAP: {mname}")
    try:
        if mname in tree_keys:
            explainer=shap.TreeExplainer(model)
            sv=explainer.shap_values(X_test_s)
            if isinstance(sv,list): sv=sv[1]
        else:
            bg_idx=np.random.default_rng(SEED).choice(len(X_train_s),80,replace=False)
            bg=X_train_s.iloc[bg_idx]
            def _pred(x,_m=model): return _m.predict_proba(pd.DataFrame(x,columns=FEATURES))[:,1]
            explainer=shap.KernelExplainer(_pred,bg)
            sv=explainer.shap_values(X_test_s.iloc[:100])

        ma=np.abs(sv).mean(axis=0)
        imp=pd.DataFrame({"feature":FEATURES,"mean_abs_shap":ma,"model":mname}).sort_values("mean_abs_shap",ascending=False).reset_index(drop=True)
        imp["rank"]=range(1,len(imp)+1)
        imp.to_csv(SHAP_DIR/f"global_importance_{mname}.csv",index=False)
        pd.DataFrame(sv,columns=FEATURES).to_csv(SHAP_DIR/f"shap_values_{mname}.csv",index=False)
        ranked=imp["feature"].tolist(); all_rankings[mname]=ranked
        log.info(f"    Ranking: {ranked}")

        # SHAP bar
        ds2=imp.sort_values("mean_abs_shap"); pal_s=sns.color_palette("viridis",len(ds2))
        fig,ax=plt.subplots(figsize=(9,max(5,len(ds2)*0.7)))
        bars=ax.barh(ds2["feature"],ds2["mean_abs_shap"],color=pal_s,edgecolor="white")
        for bar,val in zip(bars,ds2["mean_abs_shap"]): ax.text(bar.get_width()+0.0003,bar.get_y()+bar.get_height()/2,f"{val:.4f}",va="center",fontsize=10)
        ax.set_xlabel("Mean |SHAP Value|"); ax.set_title(f"Global SHAP Importance — {mname}",fontsize=13,fontweight="bold")
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        savefig(FIGS_DIR/f"shap_bar_{mname}.png")

        # beeswarm
        try:
            fig=plt.figure(figsize=(10,7)); X_s=X_test_s if mname in tree_keys else X_test_s.iloc[:100]
            shap.summary_plot(sv,X_s,max_display=9,show=False,plot_size=None)
            plt.title(f"SHAP Summary — {mname}",fontsize=14,fontweight="bold")
            savefig(FIGS_DIR/f"shap_summary_{mname}.png")
        except: pass
    except Exception as e: log.warning(f"  SHAP failed for {mname}: {e}")

if len(all_rankings)>=2:
    maxl=max(len(v) for v in all_rankings.values())
    rows=[{"rank":r+1,**{mn:rl[r] if r<len(rl) else None for mn,rl in all_rankings.items()}} for r in range(maxl)]
    pd.DataFrame(rows).to_csv(SHAP_DIR/"cross_model_ranking_comparison.csv",index=False)
rf_ranked=all_rankings.get("RandomForest",FEATURES)
with open(SHAP_DIR/"ranked_features_RandomForest.json","w") as f: json.dump(rf_ranked,f)
log.info(f"RF ranking: {rf_ranked}")

# ════════════ STEP 7 ════════════
log.info("STEP 7 — Feature Ablation")
ablation_configs={"A_all":rf_ranked[:],"B_top7":rf_ranked[:7],"C_top5":rf_ranked[:5],"D_top3":rf_ranked[:3],"E_lowcost":["pH","Turbidity","Conductivity"]}
ablation_results=[]
for mname in ["RandomForest","XGBoost","LogisticRegression","SVM"]:
    if mname not in fitted_models: continue
    for cfg_name,feats in ablation_configs.items():
        avail=[f for f in feats if f in X_train_s.columns]
        if not avail: continue
        m=clone(fitted_models[mname]); m.fit(X_train_s[avail],y_train)
        met=evaluate(m,X_test_s[avail],y_test,mname,"test")
        met["ablation_config"]=cfg_name; met["n_features"]=len(avail); met["features_used"]=str(avail)
        ablation_results.append(met)
        log.info(f"  {mname}/{cfg_name} ({len(avail)}f): f1={met['f1_macro']}  auc={met['roc_auc']}")

ablation_df=pd.DataFrame(ablation_results)
ablation_df.to_csv(ABLATION_DIR/"ablation_results_all.csv",index=False)
abl_models=ablation_df["model"].unique(); pal_a=sns.color_palette("tab10",len(abl_models))
for metric in ["f1_macro","roc_auc"]:
    fig,ax=plt.subplots(figsize=(9,6))
    for i,mn in enumerate(abl_models):
        sub=ablation_df[ablation_df["model"]==mn].sort_values("n_features")
        ax.plot(sub["n_features"],sub[metric],marker="o",label=mn,color=pal_a[i],linewidth=2,markersize=7)
    ax.set_xlabel("Number of Features",fontsize=12); ax.set_ylabel(metric.replace("_"," ").title(),fontsize=12)
    ax.set_title(f"Feature Ablation — {metric.replace('_',' ').title()}",fontsize=14,fontweight="bold")
    ax.legend(loc="lower right",fontsize=9); ax.set_xticks([3,5,7,9])
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    savefig(FIGS_DIR/f"ablation_curve_{metric}.png")

# ════════════ STEP 8 ════════════
log.info("STEP 8 — Noise Robustness")
NOISE_LEVELS=[0.0,0.02,0.05,0.10,0.15]; N_REPS=5
robust_names=["RandomForest","XGBoost","LogisticRegression","SVM"]
robustness_results=[]
for mname in robust_names:
    if mname not in fitted_models: continue
    model=fitted_models[mname]
    for noise in NOISE_LEVELS:
        rf1,ra=[],[]
        for rep in range(N_REPS):
            rng=np.random.default_rng(200+rep); eps=rng.normal(0,noise,X_test_s.shape)
            Xn=pd.DataFrame(X_test_s.values*(1+eps),columns=FEATURES)
            yp=model.predict(Xn); ypr=model.predict_proba(Xn)[:,1] if hasattr(model,"predict_proba") else yp.astype(float)
            rf1.append(f1_score(y_test,yp,average="macro",zero_division=0)); ra.append(roc_auc_score(y_test,ypr))
        row={"model":mname,"noise_level":noise,"noise_pct":f"{int(noise*100)}%",
             "f1_macro_mean":round(np.mean(rf1),4),"f1_macro_std":round(np.std(rf1),4),
             "roc_auc_mean":round(np.mean(ra),4),"roc_auc_std":round(np.std(ra),4)}
        robustness_results.append(row)
        log.info(f"  {mname} @{int(noise*100)}%: f1={row['f1_macro_mean']:.4f}±{row['f1_macro_std']:.4f}")

robust_df=pd.DataFrame(robustness_results)
robust_df.to_csv(ROBUST_DIR/"robustness_results_all.csv",index=False)
pal_r=sns.color_palette("tab10",len(robust_names))
for metric,std_col,label in [("f1_macro_mean","f1_macro_std","Macro F1"),("roc_auc_mean","roc_auc_std","ROC-AUC")]:
    fig,ax=plt.subplots(figsize=(9,6))
    for i,mn in enumerate(robust_names):
        sub=robust_df[robust_df["model"]==mn].sort_values("noise_level")
        ax.plot(sub["noise_level"]*100,sub[metric],marker="o",label=mn,color=pal_r[i],linewidth=2,markersize=7)
        ax.fill_between(sub["noise_level"]*100,sub[metric]-sub[std_col],sub[metric]+sub[std_col],alpha=0.12,color=pal_r[i])
    ax.set_xlabel("Noise Level (%)",fontsize=12); ax.set_ylabel(label,fontsize=12)
    ax.set_title(f"Robustness to Measurement Noise — {label}",fontsize=14,fontweight="bold")
    ax.legend(loc="lower left",fontsize=9)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    savefig(FIGS_DIR/f"robustness_{metric}.png")

# ════════════ STEP 9 ════════════
log.info("STEP 9 — Publication Tables")
t1=pd.DataFrame({"Statistic":["Total samples","Features","Not Potable (class 0)","Potable (class 1)","Class ratio 0:1","Features with missing","Total missing cells","Missing rate (%)"],
                  "Value":[len(df),len(FEATURES),int(vc.get(0,0)),int(vc.get(1,0)),f"{vc.get(0,0)/max(1,vc.get(1,0)):.2f}:1",
                           int((df[FEATURES].isnull().sum()>0).sum()),int(df[FEATURES].isnull().sum().sum()),f"{100*df[FEATURES].isnull().sum().sum()/(len(df)*len(FEATURES)):.2f}"]})
t1.to_csv(TABLES_DIR/"table1_dataset.csv",index=False)

test_metrics_df=pd.DataFrame(all_test_metrics)
cols2=["model","accuracy","balanced_accuracy","f1_macro","f1_weighted","f1_class0","f1_class1","roc_auc","pr_auc"]
t2=test_metrics_df[[c for c in cols2 if c in test_metrics_df.columns]].sort_values("f1_macro",ascending=False)
t2.to_csv(TABLES_DIR/"table2_model_performance.csv",index=False)

cv_df=pd.DataFrame(cv_summary); t3_rows=[]
for _,row in cv_df.iterrows():
    r={"model":row["model"]}
    for m in ["accuracy","balanced_accuracy","f1_macro","roc_auc","average_precision"]:
        if f"{m}_mean" in row: r[m]=f"{row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}"
    t3_rows.append(r)
t3=pd.DataFrame(t3_rows); t3.to_csv(TABLES_DIR/"table3_cv_results.csv",index=False)

shap_files=list(SHAP_DIR.glob("global_importance_*.csv"))
if shap_files:
    t4=pd.concat([pd.read_csv(f) for f in shap_files],ignore_index=True)
    t4=t4[["model","rank","feature","mean_abs_shap"]].sort_values(["model","rank"])
    t4.to_csv(TABLES_DIR/"table4_shap_importance.csv",index=False)

t5=ablation_df[["model","ablation_config","n_features","accuracy","f1_macro","roc_auc","pr_auc"]].sort_values(["model","n_features"])
t5.to_csv(TABLES_DIR/"table5_ablation_results.csv",index=False)

t6=robust_df[["model","noise_pct","f1_macro_mean","f1_macro_std","roc_auc_mean","roc_auc_std"]]
t6.to_csv(TABLES_DIR/"table6_robustness_results.csv",index=False)

# ════════════ FINAL PRINT ════════════
print("\n"+"═"*72)
print("  TABLE 2 — Model Performance on Held-Out Test Set")
print("═"*72)
print(t2[["model","accuracy","f1_macro","roc_auc","pr_auc"]].to_string(index=False))
print("\n"+"═"*72)
print("  TABLE 3 — Cross-Validation (5-Fold Stratified K-Fold)")
print("═"*72)
print(t3.to_string(index=False))
if shap_files:
    print("\n"+"═"*72); print("  TABLE 4 — SHAP Feature Importance (Random Forest)"); print("═"*72)
    rf_imp=t4[t4["model"]=="RandomForest"][["rank","feature","mean_abs_shap"]]
    print(rf_imp.to_string(index=False))
print("\n"+"═"*72); print(f"  TABLE 5 — Feature Ablation (Best Model: {t2.iloc[0]['model']})"); print("═"*72)
print(t5[t5["model"]==t2.iloc[0]["model"]][["ablation_config","n_features","f1_macro","roc_auc"]].to_string(index=False))
print("\n"+"═"*72); print("  TABLE 6 — Noise Robustness (Macro F1 Mean)"); print("═"*72)
try:
    pivot=t6.pivot_table(index="noise_pct",columns="model",values="f1_macro_mean"); print(pivot.to_string())
except: print(t6.to_string(index=False))
n_figs=len(list(FIGS_DIR.glob("*.png"))); n_tables=len(list(TABLES_DIR.glob("*.csv")))
print(f"\n  Figures: {n_figs}  | Tables: {n_tables}  | Models: {len(list(MODELS_DIR.glob('*.pkl')))}  | SHAP CSVs: {len(list(SHAP_DIR.glob('*.csv')))}")
print("═"*72+"\n")
log.info("Pipeline complete!")
