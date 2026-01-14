import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LinearRegression


def bmi_category(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    else:
        return 3



# Dosyayı oku
df = pd.read_csv(r"C:\Users\User\PycharmProjects\pdf_extract\data_bootcamp_proje_2025_aralık\insurance.csv")

# İlk 5 kayıt
#print(df.head())

# Veri yapısı
#print(df.info())

# İstatistiksel özet
#print(df.describe())


#label encoding
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['sex'] = df['sex'].map({'male': 1, 'female': 0})

#one-hot encoding
df = pd.get_dummies(df, columns=['region'], drop_first=True)

pd.set_option('display.max_columns', None)
#print(df.head())

df["bmi_cat"] = df["bmi"].apply(bmi_category)

df["smoker_age_interaction"] = df["smoker"] * df["age"]
df = df.drop(columns=[col for col in df.columns if "region" in col])
df["smoker_bmi_interaction"] = df["smoker"] * df["bmi"] ##3. feature engineering

#print(df.columns)
#print(df.head())

X = df.drop("charges", axis=1)
y = df["charges"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2 = r2_score(y_test, y_pred_lr)

print("LinearRegression")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)


rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Sonuçları")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)
print("R²:", r2_rf)


