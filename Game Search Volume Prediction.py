import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

# 1. Veri Yükleme
game_data = pd.read_csv('game_data.csv')
interest_over_time = pd.read_csv('interest_over_time.csv')
sspy_data = pd.read_csv('sspy_data.csv')
vgi_game_stats = pd.read_csv('vgi_game_stats.csv')

# 2. Veri Entegrasyonu
# Game Data ve Interest Over Time'ı birleştir
df = pd.merge(game_data, interest_over_time, left_on='game_name', right_on='name', how='inner')

# Sosyal Medya Verisi ile birleştirme
df = pd.merge(df, sspy_data, left_on='game_name', right_on='game_name', how='left')

# 3. Veri Temizleme
# Sayısal olmayan karakterleri temizleyip sayısal türdeki verilere dönüştürme
df['initial_price'] = df['initial_price'].replace({',': '', ':': '', ' ': ''}, regex=True)
df['final_price'] = df['final_price'].replace({',': '', ':': '', ' ': ''}, regex=True)
df['followers'] = df['followers'].replace({',': '', ':': '', ' ': ''}, regex=True)
df['num_ytube_vids(one_day)'] = df['num_ytube_vids(one_day)'].replace({',': '', ':': '', ' ': ''}, regex=True)

# Sayısal verilere dönüştürme
df['initial_price'] = pd.to_numeric(df['initial_price'], errors='coerce')
df['final_price'] = pd.to_numeric(df['final_price'], errors='coerce')
df['followers'] = pd.to_numeric(df['followers'], errors='coerce')
df['num_ytube_vids(one_day)'] = pd.to_numeric(df['num_ytube_vids(one_day)'], errors='coerce')

# Eksik verileri kaldırma
df.dropna(subset=['num_searches'], inplace=True)

# 4. Özellik Seçimi
# İlgi oranı (search trends) ve sosyal medya etkisi gibi özellikler
features = ['initial_price', 'final_price', 'num_dlc', 'num_languages', 'followers', 'num_ytube_vids(one_day)', 'num_searches']
target = 'num_searches'

# 5. Makine Öğrenmesi Modeli
X = df[features]
y = df[target]

# Eğitim ve test verisine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model kurma
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Model Değerlendirmesi
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# 6. Sonuçları Görselleştirme
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Gerçek Arama Hacmi')
plt.ylabel('Tahmin Edilen Arama Hacmi')
plt.title('Gerçek vs Tahmin Edilen Arama Hacmi')
plt.show()



# Tarih sütununu uygun formata getirme
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')  # Eğer 'release_date' tarih sütunuysa
prophet_data = df[['release_date', 'num_searches']].dropna()

# Prophet için uygun sütun isimlerini değiştirme
prophet_data = prophet_data.rename(columns={'release_date': 'ds', 'num_searches': 'y'})

# Prophet modelini kurma
prophet_model = Prophet(daily_seasonality=False, yearly_seasonality=True)
prophet_model.fit(prophet_data)

# Gelecekteki veriler için tahmin yapma (365 gün)
future = prophet_model.make_future_dataframe(periods=365)

# Tahmin yapma
forecast = prophet_model.predict(future)

# Sonuçları görselleştirme
prophet_model.plot(forecast)
plt.title('Prophet Tahminleri')
plt.show()



import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# XGBoost modelini kurma
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Tahmin yapma
y_pred_xgb = xgb_model.predict(X_test)

# Model Değerlendirmesi
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print(f"XGBoost MAE: {mae_xgb}")
print(f"XGBoost RMSE: {rmse_xgb}")

# Sonuçları Görselleştirme
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_xgb)
plt.xlabel('Gerçek Arama Hacmi')
plt.ylabel('Tahmin Edilen Arama Hacmi')
plt.title('Gerçek vs XGBoost Tahmin Edilen Arama Hacmi')
plt.show()

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# LightGBM modelini kurma
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)

# Tahmin yapma
y_pred_lgb = lgb_model.predict(X_test)

# Model değerlendirmesi
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))

# Sonuçları yazdırma
print(f"LightGBM MAE: {mae_lgb}")
print(f"LightGBM RMSE: {rmse_lgb}")


# Sonuçları Görselleştirme
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_lgb)
plt.xlabel('Gerçek Arama Hacmi')
plt.ylabel('Tahmin Edilen Arama Hacmi')
plt.title('Gerçek vs LightGBM Tahmin Edilen Arama Hacmi')
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'num_leaves': [31, 50, 100],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500]
}

grid_search = GridSearchCV(lgb.LGBMRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score (MSE): {abs(grid_search.best_score_)}")



