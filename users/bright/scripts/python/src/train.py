# Import Packages
import os
import joblib
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test", type=str, default=os.getenv("SM_CHANNEL_TEST"))
    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR"))
    
    args, _ = parser.parse_known_args()
    
    df_train = pd.read_csv(os.path.join(args.train, "train.csv"))
    df_test = pd.read_csv(os.path.join(args.test, "test.csv"))
    
    #Split Dataset into features(x) and labels(y)
    x_train = df_train.drop("Loan_Status", axis=1)
    x_test = df_test.drop("Loan_Status", axis=1)
    
    y_train = df_train["Loan_Status"]
    y_test = df_test["Loan_Status"]
    
    clf = RandomForestClassifier()
    
    # Fit on training datasets
    clf.fit(x_train, y_train)
    
    # Make prediction
    ypred = clf.predict(x_test)
    
    # Save model into model
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
    
    # Evaluate our models
    print(f"accuracy_score: {round(accuracy_score(y_test, ypred),3)}")
    print(f"f1_score: {round(f1_score(y_test, ypred),3)}")
    print(f"Precision_score: {round(precision_score(y_test, ypred),3)}")
    print(f"recall_score: {round(recall_score(y_test, ypred),3)}")
    print(f"confusion_matrix: {confusion_matrix(y_test, ypred)}")
    print(f"classification_report: {classification_report(y_test, ypred)}")
    