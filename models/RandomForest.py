import pandas as pd  # import pandas library as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # import train test split and randomized search cv
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder  # import ordinal encoder and label encoder
from sklearn.ensemble import RandomForestClassifier  # import random forest classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix  # removed f1_score
from scipy.stats import randint  # import randint distribution for hyperparameter search
import numpy as np  # added: for confusion matrix normalization
import matplotlib.pyplot as plt  # added: for plotting confusion matrix

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

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_encoded,
)  # split dataset into train and test sets

print("training set size:", X_train.shape[0])  # show training set length
print("test set size:", X_test.shape[0])  # show test set length

baseline_rf = RandomForestClassifier(random_state=RANDOM_STATE)  # create baseline random forest model
baseline_rf.fit(X_train, y_train)  # train baseline model
y_pred_baseline = baseline_rf.predict(X_test)  # get baseline predictions


def evaluate_model(y_true, y_pred, model_name):  # define evaluation function for models
    acc = accuracy_score(y_true, y_pred)  # compute accuracy
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)  # compute weighted precision
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)  # compute weighted recall
    
    print("=== {} evaluation ===".format(model_name))  # print model evaluation header
    print("accuracy:", acc)  # print accuracy
    print("precision (weighted):", prec)  # print precision score
    print("recall (weighted):", rec)  # print recall score

    print("classification report:")  # print label for report
    print(
        classification_report(
            y_true, y_pred, target_names=label_encoder.classes_, zero_division=0
        )
    )  # print detailed class stats
    print("confusion matrix:")  # print label for confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)  # compute confusion matrix
    print(conf_matrix)  # print confusion matrix values

    # ==== Plot confusion matrix (decision matrix) ====

    # Normalize confusion matrix to get proportions within each true class
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        conf_norm = conf_matrix / row_sums
    conf_norm = np.nan_to_num(conf_norm)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(conf_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title("Confusion Matrix (proportion within true class)")
    fig.colorbar(cax, ax=ax, label="Proportion within true class")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(label_encoder.classes_)))
    ax.set_yticks(np.arange(len(label_encoder.classes_)))
    ax.set_xticklabels(label_encoder.classes_)
    ax.set_yticklabels(label_encoder.classes_)

    # Annotate each cell with raw count and proportion (count\nprop)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            count = int(conf_matrix[i, j])
            prop = conf_norm[i, j]
            text = f"{count}\n{prop:.2f}"
            color = "white" if prop > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    plt.show()


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
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

print("starting hyperparameter search...")
random_search.fit(X_train, y_train)

print("best hyperparameters found:")
print(random_search.best_params_)
print("best cross-validation accuracy:", random_search.best_score_)

best_rf = random_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred_best = best_rf.predict(X_test)

evaluate_model(y_test, y_pred_best, "tuned random forest")

example_car = pd.DataFrame(
    {
        "buying": ["med"],
        "maint": ["low"],
        "doors": ["4"],
        "persons": ["4"],
        "lug_boot": ["big"],
        "safety": ["high"],
    }
)

example_encoded = feature_encoder.transform(example_car)
example_pred_num = best_rf.predict(example_encoded)[0]
example_pred_label = label_encoder.inverse_transform([example_pred_num])[0]

print("example car:")
print(example_car)
print("predicted class:", example_pred_label)
