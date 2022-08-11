import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

#Read CSV Files
X = pd.read_csv(r"C:\Pycharm\House Prices ML\train.csv")
X_test = pd.read_csv(r"C:\Pycharm\House Prices ML\test.csv")

#Drop rows with missing target (Sale Price)
X.dropna(axis=0, subset = ["SalePrice"], inplace = True)
y = X.SalePrice
X.drop(["SalePrice"], axis=1, inplace=True)
#print(X.isnull().sum(axis=0)) #Show which columns have null values

# #Break off validation set from training data
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state=0)

#Columns
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ["int64", "float"]]
categorical_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and
                        X[cname].dtype == "object"]

#Preprocessing for numerical data
N_transformer = Pipeline(steps=[("impute", SimpleImputer(strategy="mean"))])

#Preprocessing for categorical data
C_transformer = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")), ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False))])

full_processor = ColumnTransformer(transformers=[("number", N_transformer, numerical_cols), ("category", C_transformer, categorical_cols)])

#RandomForest Model
model = XGBRegressor(n_estimators = 500, learning_rate = 0.1, random_state = 0)

my_pipeline = Pipeline(steps=[("preprocessor", full_processor), ("model", model)])
my_pipeline.fit(X, y)
predictions = my_pipeline.predict(X)

#MAE Score, Cross validation score
score = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring = "neg_mean_absolute_error")
print("MAE:", score)
print("MAE Average:", score.mean())
predictions_test = my_pipeline.predict(X_test)

#Save results in CSV
submission = pd.DataFrame({ 'Id': X_test.Id,
                            'SalePrice': predictions_test })
submission.to_csv("submission.csv", index=False)