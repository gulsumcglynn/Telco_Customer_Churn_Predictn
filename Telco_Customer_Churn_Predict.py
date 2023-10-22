##############################
# Telco Customer Churn Predict
##############################

# Problem : Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
# Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

# Telco müşteri churn verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan
# hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu içermektedir.

# 21 Değişken 7043 Gözlem

# CustomerId : Müşteri İd’si
# Gender : Cinsiyet
# SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
# Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama
# Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
# tenure : Müşterinin şirkette kaldığı ay sayısı
# PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir
# StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir
# Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges : Müşteriden tahsil edilen toplam tutar
# Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır) - Geçen ay veya çeyreklik içerisinde ayrılan müşteriler

#pip install --upgrade dask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
warnings.simplefilter(action="ignore")


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("datasets/Telco-Customer-Churn.csv")


# GÖREV 1: KEŞİFCİ VERİ ANALİZİ

#Genel resmi inceleyiniz.

df.info()
df.head()
# TotalCharges sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
#Churn değişkenini Yes-No dan 1-0 formatına çevir.
df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)
#Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, car_th=20, cat_th=10):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
#17 categorik değişken, 3 tane nümerik değişken var.CustomerId cat_but_car


#Numerik ve kategorik değişkenlerin analizini yapınız.

# CATEGORICAL/analiz
def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100* dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, True)

# NUMERICAL/analiz
def num_summary(dataframe,numerical_col,plot=False):
    quantiles=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99,1]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.ylabel(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col,True)


# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

##################################
# KORELASYON
##################################

df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

# EKSİK DEĞER ANALİZİ


df.isnull().sum()
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)


# (aylık ödenecek miktarlarıyla totalcharge doldurulailir)
df["TotalCharges"].fillna(df["MonthlyCharges"]*df["tenure"], inplace=True)


# AYKIRI DEĞER ANALİZİ

def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.15)
    quartile3 = dataframe[col_name].quantile(0.85)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#Aykırı değerleri kontrol edip baskıladık.
for col in num_cols:
    print(check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

##################################
# BASE MODEL
##################################
dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["Churn"]
X = dff.drop(["Churn", "customerID"], axis=1)

#Hiperparametre optimizasyonu yapmadan kurduğumuz modeller ve çıkan sonuçlar
models = [("LR", LogisticRegression()),
          ("CART", DecisionTreeClassifier()),
          ("RF", RandomForestClassifier()),
          ("XGB", XGBClassifier()),
          ("LightGBM", LGBMClassifier(verbose=-1)),
          ("CatBoost", CatBoostClassifier(verbose=False))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "recall", "precision", "f1", "roc_auc"])
    print(f"##{name}##")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

'''##LR##
Accuracy: 0.8007
Auc: 0.8396
Recall: 0.4939
Precision: 0.6683
F1: 0.5669
##CART##
Accuracy: 0.7294
Auc: 0.6594
Recall: 0.5056
Precision: 0.4908
F1: 0.4979
##RF##
Accuracy: 0.7947
Auc: 0.8277
Recall: 0.5035
Precision: 0.6455
F1: 0.5655
##XGB##
Accuracy: 0.7865
Auc: 0.8229
Recall: 0.5083
Precision: 0.6194
F1: 0.5581
##LightGBM##
Accuracy: 0.7951
Auc: 0.837
Recall: 0.5217
Precision: 0.6406
F1: 0.5746
##CatBoost##
Accuracy: 0.8014
Auc: 0.8418
Recall: 0.5217
Precision: 0.6586
F1: 0.582'''
# ÖZELLİK ÇIKARIMI

df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "New_Tenure_Year"] = "0-1 year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "New_Tenure_Year"] = "1-2 year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "New_Tenure_Year"] = "2-3 year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "New_Tenure_Year"] = "3-4 year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "New_Tenure_Year"] = "4-5 year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "New_Tenure_Year"] = "5-6 year"

#Kontratı 1 veya 2 yıllık olanları new_Engaged diye adlandırma

df["New_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler

df["New_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes")
                            or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler

df["New_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["SeniorCitizen"] == 0) and (x["New_Engaged"] == 0) else 0, axis=1)

# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# Herhangi bir streaming hizmeti alan kişiler

df["New_Flag_Any_Streaming"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?

df["New_Flag_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in (["Bank transfer (automatic)", "Credit card (automatic)"]) else 0)

# ortalama aylık ödeme

df["New_Avg_Charges"] = df["TotalCharges"]/(df["tenure"] + 1)

# Güncel Fiyatın ortalama fiyata göre artışı

df["NEW_Increase"] = df["New_Avg_Charges"] / df["MonthlyCharges"]

# Servis başına ücret

df["New_Avg_Service_Fee"] = df["MonthlyCharges"] / (df["NEW_TotalServices"] + 1)

df.head()

##################################
# ENCODING
##################################

# LABEL ENCODING

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols #gender, partner, dependents, phoneservice, paperlessbilling

#binary_cols lara label encoder uyguladık.
for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding İşlemi

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()
df.info()

##################################
# MODELLEME
##################################

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

# Random Forests

rf_model = RandomForestClassifier()
rf_model.get_params()
rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 7, "auto"],
             "min_samples_split": [5, 8, 15,],
             "n_estimators": [100,500]}
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_).fit(X,y)

cv_results = cross_validate(rf_final, X, y, cv=10,
                            scoring=["accuracy", "f1","roc_auc"])

cv_results["test_accuracy"].mean()#0.80
cv_results["test_f1"].mean()#0.57
cv_results["test_roc_auc"].mean()#0.84

# XGBoost

xgboost_model = XGBClassifier()
xgboost_model.get_params()
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 15],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}
xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)
xgboost_best_grid.best_params_
xgboost_final = xgboost_model.set_params(max_depth=5, colsample_bytree=0.5,learning_rate=0.01,n_estimators=1000, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()#0.801
cv_results['test_f1'].mean()#0.58
cv_results['test_roc_auc'].mean()#0.844


# LightGBM


lgbm_model = LGBMClassifier()

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1],
               "verbose": [-1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(colsample_bytree=0.5,learning_rate=0.01,n_estimators=500).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# CatBoost

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_best_grid.best_params_

catboost_final = catboost_model.set_params(depth=6, learning_rate=0.01,iterations=500, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)











