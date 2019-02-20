__author__ = 'kevin'
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--db", help="path to HDF5 database", required=True)
ap.add_argument("-m", "--model", help="path to output model", required=True)
ap.add_argument("-j", "--jobs", help="# of jobs to run when hyperparameters", type=int, default=1)

args = vars(ap.parse_args())

db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)
print("[INFO] tuning hyperparameters....{}".format(i))

params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args["jobs"])
model.fit(db["features"][:i], db["labels"][:i])
print("[INFO] best hyperparameter {}".format(model.best_params_))

print("[INFO] Evaluating {}{}".format(db["labels"][i:], db["label_names"]))

preds = model.predict(db["features"][1009:])
print(classification_report(db["labels"][1009:], preds, target_names=db["label_names"]))

print("[INFO] Saving model")

f = open(args["model"], "wb")

f.write(pickle.dumps(model.best_estimator_))

f.close()

db.close()
