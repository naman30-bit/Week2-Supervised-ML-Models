from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import *

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

def regression():
    d = load_diabetes(as_frame=True)
    Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=.2, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
        "KNN": Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=7))])
    }
    rows=[]
    for name,m in models.items():
        m.fit(Xtr,ytr); p=m.predict(Xte)
        rows.append({"Model":name,"MAE":mean_absolute_error(yte,p),"RMSE":np.sqrt(mean_squared_error(yte,p)),"R2 Score":r2_score(yte,p)})
    df=pd.DataFrame(rows).sort_values("RMSE")
    df.to_csv(OUT/"regression_model_comparison.csv",index=False)
    return df

def classification():
    d=load_breast_cancer(as_frame=True)
    Xtr,Xte,ytr,yte=train_test_split(d.data,d.target,test_size=.2,random_state=42,stratify=d.target)
    models={
        "Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=5000))]),
        "Decision Tree":DecisionTreeClassifier(max_depth=4,random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=150,random_state=42),
        "KNN":Pipeline([("scale",StandardScaler()),("model",KNeighborsClassifier(n_neighbors=5))])
    }
    rows=[]; fitted={}
    for name,m in models.items():
        m.fit(Xtr,ytr); p=m.predict(Xte); prob=m.predict_proba(Xte)[:,1]
        rows.append({"Model":name,"Accuracy":accuracy_score(yte,p),"Precision":precision_score(yte,p),"Recall":recall_score(yte,p),"F1 Score":f1_score(yte,p),"ROC-AUC":roc_auc_score(yte,prob)})
        fitted[name]=(m,p)
    df=pd.DataFrame(rows).sort_values("F1 Score",ascending=False)
    df.to_csv(OUT/"classification_model_comparison.csv",index=False)
    best=df.iloc[0]["Model"]; pred=fitted[best][1]
    cm=confusion_matrix(yte,pred)
    fig,ax=plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay(cm,display_labels=d.target_names).plot(ax=ax,cmap="Blues",colorbar=False)
    ax.set_title("Confusion Matrix - "+best); fig.tight_layout()
    fig.savefig(OUT/"best_classification_confusion_matrix.png",dpi=160); plt.close(fig)
    (OUT/"best_classification_report.txt").write_text(classification_report(yte,pred,target_names=d.target_names),encoding="utf-8")
    return df,best,cm

def main():
    reg=regression(); clf,best,cm=classification()
    for df,title,col,file in [(reg,"Regression comparison (lower RMSE is better)","RMSE","regression_model_comparison.png"),(clf,"Classification comparison (higher F1 is better)","F1 Score","classification_model_comparison.png")]:
        fig,ax=plt.subplots(figsize=(8,5)); ax.bar(df["Model"],df[col]); ax.set_title(title); ax.set_ylabel(col); ax.tick_params(axis="x",rotation=15); fig.tight_layout(); fig.savefig(OUT/file,dpi=160); plt.close(fig)
    print("="*65)
    print("YUVaintern WEEK 2 - SUPERVISED MACHINE LEARNING")
    print("="*65)
    print("\nREGRESSION RESULTS (Scikit-learn Diabetes dataset)")
    print(reg.round(4).to_string(index=False))
    print("\nCLASSIFICATION RESULTS (Breast Cancer Wisconsin dataset)")
    print(clf.round(4).to_string(index=False))
    print("\nBest classification model by F1 Score:",best)
    print("Confusion matrix:\n",cm)
    print("\nSaved CSV metrics, report, confusion matrix and charts in outputs/")
    print("="*65)
if __name__=="__main__": main()
