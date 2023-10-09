import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Đọc dữ liệu từ file CSV
data_frame = pd.read_csv("AnThuan-HamLuong.csv")

# Xác định biến độc lập và biến phụ thuộc
X = data_frame[['Date', 'Month']]
y = data_frame['Salinity(ppt)']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Xử lý giá trị bị thiếu bằng cách điền giá trị trung bình của các mẫu trong cùng tháng và năm
neighborhood_distance = 10  # Định rõ khoảng cách lân cận
for index, row in X_test.iterrows():
    date, month = row['Date'], row['Month']
    neighborhood = data_frame[(data_frame['Date'] - date).abs() <= neighborhood_distance]
    neighborhood = neighborhood[neighborhood['Month'] == month]
    mean_salinity = neighborhood['Salinity(ppt)'].mean()
    if not pd.isna(mean_salinity):
        y_test.at[index] = mean_salinity

# Tìm các giá trị NaN trong cột 'Salinity(ppt)' và điền bằng giá trị trung bình của cột này
mean_salinity = y_train.mean()
y_train.fillna(mean_salinity, inplace=True)

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Dự đoán độ mặn trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
