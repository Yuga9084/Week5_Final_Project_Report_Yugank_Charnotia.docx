import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid", palette="deep")
R = {}
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
# target: 0 = malignant, 1 = benign in sklearn; recode to readable label
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
R["shape"] = list(df.shape)
R["missing"] = int(df.isna().sum().sum())
R["duplicates"] = int(df.duplicated().sum())
R["class_counts"] = df["diagnosis"].value_counts().to_dict()
feats = list(raw.feature_names)

# outliers via IQR
Q1, Q3 = df[feats].quantile(.25), df[feats].quantile(.75); IQR = Q3 - Q1
out = ((df[feats] < Q1 - 1.5*IQR) | (df[feats] > Q3 + 1.5*IQR)).sum().sort_values(ascending=False)
R["outlier_top"] = {k: int(v) for k, v in out.head(5).items()}
R["rows_with_outlier"] = int(((df[feats] < Q1-1.5*IQR)|(df[feats] > Q3+1.5*IQR)).any(axis=1).sum())
R["skew_top"] = {k: round(float(v),2) for k,v in df[feats].skew().sort_values(ascending=False).head(5).items()}

# Chart 1 class balance
fig, ax = plt.subplots(figsize=(5.5,3.6))
order=["Benign","Malignant"]; cnt=df["diagnosis"].value_counts()[order]
ax.bar(order, cnt.values, color=["#4C9F70","#C8553D"])
for i,v in enumerate(cnt.values): ax.text(i, v+5, f"{v} ({v/len(df):.1%})", ha="center")
ax.set_ylabel("Patients"); ax.set_title("Class distribution"); plt.tight_layout(); plt.savefig("charts/c1_class.png", dpi=170); plt.close()

# Chart 2 distributions of 4 key features
key = ["mean radius","mean concave points","mean perimeter","worst area"]
fig, axs = plt.subplots(2,2, figsize=(8,5.6))
for a,f in zip(axs.ravel(), key):
    sns.histplot(data=df, x=f, hue="diagnosis", kde=True, ax=a, element="step", palette={"Benign":"#4C9F70","Malignant":"#C8553D"})
    a.set_title(f)
plt.tight_layout(); plt.savefig("charts/c2_dist.png", dpi=170); plt.close()

# Chart 3 correlation of means
mean_cols=[c for c in feats if c.startswith("mean")]
fig, ax = plt.subplots(figsize=(7.2,5.8))
sns.heatmap(df[mean_cols].corr(), cmap="vlag", center=0, annot=True, fmt=".2f", annot_kws={"size":6}, ax=ax, cbar_kws={"shrink":.7})
ax.set_title("Correlation among 'mean' features"); plt.tight_layout(); plt.savefig("charts/c3_corr.png", dpi=170); plt.close()
corr = df[feats].corrwith(df["target"]).abs().sort_values(ascending=False)
R["top_corr_target"] = {k: round(float(v),3) for k,v in corr.head(5).items()}
cm_ = df[mean_cols].corr().abs()
R["radius_perimeter_corr"] = round(float(df["mean radius"].corr(df["mean perimeter"])),3)
R["radius_area_corr"] = round(float(df["mean radius"].corr(df["mean area"])),3)
hc = (df[feats].corr().abs().where(np.triu(np.ones((30,30)),1).astype(bool)) > 0.9).sum().sum()
R["pairs_corr_gt_09"] = int(hc)

# Modelling (malignant = positive class for clinical relevance)
X = df[feats]; y = (df["target"]==0).astype(int)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
models = {
 "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
 "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
 "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
res = {}; probs={}
for n,m in models.items():
    cv = cross_val_score(m, Xtr, ytr, cv=skf, scoring="roc_auc")
    m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]; pr=(p>=.5).astype(int); probs[n]=p
    res[n] = dict(cv_auc=round(cv.mean(),4), cv_std=round(cv.std(),4), acc=round(accuracy_score(yte,pr),4),
      prec=round(precision_score(yte,pr),4), rec=round(recall_score(yte,pr),4), f1=round(f1_score(yte,pr),4),
      auc=round(roc_auc_score(yte,p),4), cm=confusion_matrix(yte,pr).tolist())
R["models"]=res; R["n_train"]=len(Xtr); R["n_test"]=len(Xte)
best = max(res, key=lambda k: res[k]["cv_auc"]); R["best"]=best

# Chart 4 model comparison
fig, ax = plt.subplots(figsize=(7,3.8))
mt=pd.DataFrame(res).T[["acc","prec","rec","f1","auc"]]
mt.columns=["Accuracy","Precision","Recall","F1","ROC-AUC"]
mt.plot.bar(ax=ax, rot=0, width=.8); ax.set_ylim(.85,1.01); ax.legend(ncol=5, fontsize=7, loc="lower center"); ax.set_title("Test-set performance by model")
plt.tight_layout(); plt.savefig("charts/c4_models.png", dpi=170); plt.close()

# Chart 5 ROC
fig, ax = plt.subplots(figsize=(5.2,4.4))
for n,p in probs.items():
    f,t,_=roc_curve(yte,p); ax.plot(f,t,label=f"{n} (AUC {res[n]['auc']:.3f})")
ax.plot([0,1],[0,1],"k--",lw=.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=7); ax.set_title("ROC curves")
plt.tight_layout(); plt.savefig("charts/c5_roc.png", dpi=170); plt.close()

# Chart 6 confusion matrix best
cm = np.array(res[best]["cm"])
fig, ax = plt.subplots(figsize=(4.2,3.6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Benign","Malignant"], yticklabels=["Benign","Malignant"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix – {best}")
plt.tight_layout(); plt.savefig("charts/c6_cm.png", dpi=170); plt.close()

# Chart 7 permutation importance (best model)
pi = permutation_importance(models[best], Xte, yte, n_repeats=20, random_state=42, scoring="roc_auc")
imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False).head(10)
R["top_imp"] = {k: round(float(v),4) for k,v in imp.items()}
fig, ax = plt.subplots(figsize=(6.4,4))
imp[::-1].plot.barh(ax=ax, color="#3A6EA5"); ax.set_xlabel("Drop in ROC-AUC when shuffled"); ax.set_title(f"Top 10 features – {best}")
plt.tight_layout(); plt.savefig("charts/c7_imp.png", dpi=170); plt.close()

# Threshold analysis on best model (cost-sensitive: favour recall on malignant)
p = probs[best]; rows=[]
for t in [0.5,0.4,0.3,0.2,0.1]:
    pr=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pr).ravel()
    rows.append(dict(t=t, rec=round(recall_score(yte,pr),4), prec=round(precision_score(yte,pr),4), fn=int(fn), fp=int(fp)))
R["thresholds"]=rows
json.dump(R, open("results.json","w"), indent=1)
print(json.dumps(R, indent=1))
