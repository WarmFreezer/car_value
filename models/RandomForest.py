
import pandas as pd  # import pandas library as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # import train test split and randomized search cv
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder  # import ordinal encoder and label encoder
from sklearn.ensemble import RandomForestClassifier  # import random forest classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix  # import evaluation metrics
from scipy.stats import randint  # import randint distribution for hyperparameter search

RANDOM_STATE = 42  # set fixed random seed value

column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]  # define column names list

df = pd.read_csv("car.data", header=None, names=column_names)  # read dataset into dataframe with correct column names
print("first 5 rows of the dataset:")  # print dataset preview header
print(df.head())  # print first 5 rows from dataset
print("dataset shape (rows, columns):", df.shape)  # print dataset shape
print("class distribution:")  # label output as class counts
print(df["class"].value_counts())  # print class frequencies

X = df.drop("class", axis=1)  # create feature matrix without class column
y = df["class"]  # create label vector using class column

feature_encoder = OrdinalEncoder()  # create ordinal encoder for X
X_encoded = feature_encoder.fit_transform(X)  # fit and transform feature matrix to numbers

label_encoder = LabelEncoder()  # create label encoder for y
y_encoded = label_encoder.fit_transform(y)  # fit and transform labels to numbers

print("example of encoded features (first 5 rows):")  # print message for encoded preview
print(X_encoded[:5])  # print first 5 encoded rows
print("example of encoded labels (first 5):", y_encoded[:5])  # print first 5 encoded labels
print("label classes (index -> name):")  # label next mapping output
for idx, class_name in enumerate(label_encoder.classes_):  # loop through class mapping
    print(idx, "->", class_name)  # print mapping

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded)  # split dataset into train and test sets

print("training set size:", X_train.shape[0])  # show training set length
print("test set size:", X_test.shape[0])  # show test set length

baseline_rf = RandomForestClassifier(random_state=RANDOM_STATE)  # create baseline random forest model
baseline_rf.fit(X_train, y_train)  # train baseline model
y_pred_baseline = baseline_rf.predict(X_test)  # get baseline predictions

def evaluate_model(y_true, y_pred, model_name):  # define evaluation function for models
    acc = accuracy_score(y_true, y_pred)  # compute accuracy
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)  # compute weighted precision
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)  # compute weighted recall
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)  # compute weighted f1 score
    print("=== {} evaluation ===".format(model_name))  # print model evaluation header
    print("accuracy:", acc)  # print accuracy
    print("precision (weighted):", prec)  # print precision score
    print("recall (weighted):", rec)  # print recall score
    print("f1-score (weighted):", f1)  # print f1 score
    print("classification report:")  # print label for report
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))  # print detailed class stats
    print("confusion matrix:")  # print label for confusion matrix
    print(confusion_matrix(y_true, y_pred))  # print confusion matrix values

evaluate_model(y_test, y_pred_baseline, "baseline random forest")  # evaluate baseline model

param_distributions = {  # define hyperparameter search space
    "n_estimators": randint(100, 501),  # number of trees
    "max_depth": [None, 5, 10, 15, 20],  # depth choices
    "min_samples_split": randint(2, 11),  # minimum samples to split
    "min_samples_leaf": randint(1, 5),  # minimum leaf samples
    "max_features": ["sqrt", "log2", None],  # how many features at split
}  

rf = RandomForestClassifier(random_state=RANDOM_STATE)  # create model for tuning

random_search = RandomizedSearchCV(  # create randomized hyperparameter search
    estimator=rf,  # model to tune
    param_distributions=param_distributions,  # possible parameter values
    n_iter=20,  # number of random combinations
    cv=5,  # number of cross validation folds
    scoring="accuracy",  # metric for evaluating combinations
    random_state=RANDOM_STATE,  # ensure reproducible randomization
    n_jobs=-1,  # use all cpu cores
    verbose=1,  # show progress in output
)  

print("starting hyperparameter search...")  # notify user search has begun
random_search.fit(X_train, y_train)  # run search procedure

print("best hyperparameters found:")  # label next output
print(random_search.best_params_)  # print best combination found
print("best cross-validation accuracy:", random_search.best_score_)  # print cross-validation result

best_rf = random_search.best_estimator_  # assign best model
best_rf.fit(X_train, y_train)  # train best model
y_pred_best = best_rf.predict(X_test)  # test tuned model

evaluate_model(y_test, y_pred_best, "tuned random forest")  # evaluate tuned model

example_car = pd.DataFrame(  # create a single test car
    {
        "buying": ["med"],  # example buying cost
        "maint": ["low"],  # example maintenance cost
        "doors": ["4"],  # example door count
        "persons": ["4"],  # example passengers allowed
        "lug_boot": ["big"],  # example luggage space
        "safety": ["high"],  # example safety level
    }
)  

example_encoded = feature_encoder.transform(example_car)  # encode example car features
example_pred_num = best_rf.predict(example_encoded)[0]  # numeric prediction result
example_pred_label = label_encoder.inverse_transform([example_pred_num])[0]  # convert numeric result to class name

print("example car:")  # label next output
print(example_car)  # print raw example data
print("predicted class:", example_pred_label)  # print prediction result
