import numpy as np
import pandas as pd 

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == "__main__":

    df = pd.read_csv('/home/lokesh/Desktop/Projects/Mobile price/11167_15520_bundle_archive/train.csv')

    X = df.drop("price_range", axis = 1).values

    y = df.price_range.values

    # defining the model here
    #using RandomForest with n_jobs = -1

    classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    param_grid = {
        "n_estimators" : [100,200,250,300,400,500],
        "max_depth": [1,2,5,7,11,15],
        "criterion": ["gini","entropy"]
    }

    # intiasing grid search
    # estimator is the model that we have defined
    # param_grid is the grid of Parametres
    # we have accuracy as our metric
    # higher value of verbose implies a lot of details
    # cv = 5 means we are using 5 fold cv (not stratified)_

    model = model_selection.GridSearchCV(
        estimator = classifier,
        param_grid = param_grid,
        scoring= "accuracy",
        verbose=10,
        n_jobs = -1,
        cv=5,
    )

    model.fit(X,y)
    print(f"Best Score: {model.best_score_}")

    print("Best paramter set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
