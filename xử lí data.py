# Bài 1
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("/Users/buimanh/Downloads/Bengaluru_House_Data.csv")

# Loại bỏ các cột không cần thiết
df1 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')

# Xử lý các giá trị NA
df1.isnull().sum()  # Kiểm tra số lượng NaN
df2 = df1.dropna()  # Loại bỏ các dòng có giá trị NaN
df2.isnull().sum()  # Kiểm tra lại sau khi loại bỏ NaN

# Tạo cột 'bhk' từ cột 'size'
df2 = df2.assign(bhk=df2['size'].apply(lambda x: int(x.split(' ')[0])))


# Kiểm tra giá trị duy nhất trong 'bhk'
df2.bhk.unique()

# Hàm kiểm tra nếu giá trị là số thực
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# Lọc ra những dòng có giá trị 'total_sqft' không hợp lệ
df2[~df2['total_sqft'].apply(is_float)].head(10)

# Hàm chuyển đổi 'total_sqft' thành giá trị số hợp lệ
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2  # Lấy giá trị trung bình nếu có dấu '-'
    try:
        return float(x)  # Trả về giá trị thực nếu không có dấu '-'
    except:
        return None  # Trả về None nếu không thể chuyển đổi

# Áp dụng chuyển đổi cho cột 'total_sqft'
df3 = df2.copy()
df3['total_sqft'] = df3['total_sqft'].apply(convert_sqft_to_num)

# Loại bỏ các dòng có giá trị NaN trong 'total_sqft'
df3 = df3[df3['total_sqft'].notnull()]

# Tính giá tiền trên mỗi sqft
df3['price_per_sqft'] = df3['price'] * 100000 / df3['total_sqft']

# Kiểm tra thống kê của cột 'price_per_sqft'
df4_stats = df3['price_per_sqft'].describe()

# Lưu kết quả vào file
df3.to_csv("bhp.csv", index=False)

# Xử lý lại cột 'location'
df3['location'] = df3['location'].apply(lambda x: x.strip())  # Xóa khoảng trắng thừa
location_stats = df3['location'].value_counts(ascending=False)

# Chuyển các giá trị location ít xuất hiện thành 'other'
location_stats_less_than_10 = location_stats[location_stats <= 10]
df3['location'] = df3['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

# Lọc outliers (diện tích/số phòng ngủ < 300)
df3 = df3[df3['total_sqft'] / df3['bhk'] >= 300]

# In ra số dòng và kiểm tra lại
print(df3.shape)
print(df3.head(10))

# Lưu DataFrame cuối cùng vào file CSV
df3.to_csv("cleaned_bhp.csv", index=False)


#BÀI 2 
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file
file_path = r"/Users/buimanh/Downloads/Data_Cleaning_Practice.csv"
data = pd.read_csv(file_path)

# Giới thiệu tổng quan dữ liệu
print("Tổng quan dữ liệu ban đầu:")
print(data.info())
print(data.describe())
print("\nSố lượng giá trị thiếu trong từng cột:")
print(data.isnull().sum())

# Bước 1 - Xử lý Giá trị Thiếu
# Điền giá trị thiếu cho cột 'Age' bằng trung vị
median_age = data['Age'].median()
data['Age'] = data['Age'].fillna(median_age)  # Gán lại giá trị cho cột

# Điền giá trị thiếu cho cột 'Annual_Income' bằng trung bình
mean_income = data['Annual_Income'].mean()
data['Annual_Income'] = data['Annual_Income'].fillna(mean_income)  # Gán lại giá trị cho cột

# Hiển thị lại số lượng giá trị thiếu sau khi xử lý
print("\nSố lượng giá trị thiếu sau khi xử lý:")
print(data.isnull().sum())

# Bước 2 - Xử lý Ngoại lệ
# Vẽ biểu đồ boxplot để xác định ngoại lệ trong 'Annual_Income'
plt.boxplot(data['Annual_Income'].dropna())
plt.title('Annual Income Boxplot - Trước khi điều chỉnh ngoại lệ')
plt.show()

# Điều chỉnh ngoại lệ trong 'Annual_Income' (giới hạn trên 99% dữ liệu)
upper_limit = data['Annual_Income'].quantile(0.99)
data.loc[data['Annual_Income'] > upper_limit, 'Annual_Income'] = upper_limit

# Vẽ lại biểu đồ boxplot sau khi điều chỉnh
plt.boxplot(data['Annual_Income'].dropna())
plt.title('Annual Income Boxplot - Sau khi điều chỉnh ngoại lệ')
plt.show()

# Kiểm tra lại dữ liệu sau khi làm sạch
print("\nTổng quan dữ liệu sau khi làm sạch:")
print(data.describe())
print("\nSố lượng giá trị thiếu sau khi làm sạch hoàn toàn:")
print(data.isnull().sum())

# Lưu dữ liệu đã làm sạch ra file mới
output_path = r"D:\Zalo File\KPDL\Data\Xử lý dữ liệu\Data_Cleaning_Practice_Cleaned.csv"
data.to_csv(output_path, index=False)
print(f"\nDữ liệu đã làm sạch được lưu tại: {output_path}")

#Thay vì sử dụng inplace=True, sửa data['Age'] = data['Age'].fillna(median_age) và data['Annual_Income'] = data['Annual_Income'].fillna(mean_income)
#Điều này giúp tránh lỗi về chained assignment và phù hợp với các phiên bản pandas trong tương lai


#BÀI 3 (Sử dụng file Small_Customer.csv để xử lý giá trị thiếu)
import pandas as pd

# Bước 1: Tải dữ liệu và Kiểm tra Giá trị thiếu
file_path = r"/Users/buimanh/Downloads/Small_Customer_Data_Sample.csv"
data = pd.read_csv(file_path)

# Kiểm tra số lượng và tỷ lệ phần trăm giá trị thiếu trong từng cột
print("Số lượng giá trị thiếu trong mỗi cột:")
print(data.isnull().sum())  # Số lượng NaN trong mỗi cột
print("\nTỷ lệ phần trăm giá trị thiếu trong mỗi cột:")
print(data.isnull().mean() * 100)  # Tỷ lệ phần trăm giá trị thiếu

# Bước 2: Xử lý Giá trị thiếu
# Điền giá trị trung bình cho 'Annual_Income'
data['Annual_Income'] = data['Annual_Income'].fillna(data['Annual_Income'].mean())

# Điền giá trị trung vị cho 'Days_Since_Last_Purchase'
data['Days_Since_Last_Purchase'] = data['Days_Since_Last_Purchase'].fillna(data['Days_Since_Last_Purchase'].median())

# Điền giá trị mốt (mode) cho 'Age'
data['Age'] = data['Age'].fillna(data['Age'].mode()[0])

# Nếu cột 'Spending_Score' có giá trị thiếu, xóa những dòng có giá trị thiếu
data.dropna(subset=['Spending_Score'], inplace=True)

# Bước 3: Kiểm tra lại dữ liệu sau khi xử lý
print("\nSố lượng giá trị thiếu sau khi xử lý:")
print(data.isnull().sum())  # Kiểm tra lại số lượng NaN sau khi xử lý


#BÀI 4 (Sử dụng file Normalization_Data_Sample.csv để chuẩn hoá dữ liệu)
import pandas as pd
import matplotlib.pyplot as plt

# Bước 1: Tải dữ liệu
file_path = r"/Users/buimanh/Downloads/Normalization_Data_Sample.csv"
data = pd.read_csv(file_path)

# Bước 2: Trực quan hóa dữ liệu trước khi chuẩn hóa
# Vẽ biểu đồ phân phối cho 'Annual_Income' và 'Age' để thấy được phân bố trước khi chuẩn hóa.
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data['Annual_Income'], bins=30, color='blue', alpha=0.7)
plt.title('Annual Income Distribution')

plt.subplot(1, 2, 2)
plt.hist(data['Age'], bins=30, color='green', alpha=0.7)
plt.title('Age Distribution')

plt.show()

# Bước 3: Áp dụng Chuẩn hóa
# Min-Max Scaling cho 'Annual_Income'
data['Normalized_Income'] = (data['Annual_Income'] - data['Annual_Income'].min()) / (data['Annual_Income'].max() - data['Annual_Income'].min())

# Z-score Scaling cho 'Age'
data['Standardized_Age'] = (data['Age'] - data['Age'].mean()) / data['Age'].std()

# Bước 4: Trực quan hóa dữ liệu sau khi chuẩn hóa
# Vẽ biểu đồ phân phối cho dữ liệu đã được chuẩn hóa để so sánh trước và sau
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(data['Normalized_Income'], bins=30, color='blue', alpha=0.7)
plt.title('Normalized Annual Income Distribution')

plt.subplot(1, 2, 2)
plt.hist(data['Standardized_Age'], bins=30, color='green', alpha=0.7)
plt.title('Standardized Age Distribution')

plt.show()

#BÀI 5 (Sử dụng file Encoding_Data_Sample.csv để Mã hoá dữ liệu)
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Bước 1: Tải dữ liệu
file_path = r"/Users/buimanh/Downloads/Encoding_Data_Sample.csv"
data = pd.read_csv(file_path)

# Bước 2: Khám phá dữ liệu
print(data.head())  # In ra vài dòng đầu tiên để hiểu cấu trúc và các loại biến phân loại.

# Bước 3: Áp dụng Mã hóa Dữ liệu

# One-Hot Encoding cho 'Gender' và 'Product_Category'
data_one_hot_encoded = pd.get_dummies(data, columns=['Gender', 'Product_Category'])

# Label Encoding cho 'Color' và 'Education_Level'
le = LabelEncoder()
data['Color_encoded'] = le.fit_transform(data['Color'])
data['Education_Level_encoded'] = le.fit_transform(data['Education_Level'])

# Bước 4: Trực quan hóa kết quả Mã hóa
print(data_one_hot_encoded.head())  # In ra dữ liệu mới sau khi One-Hot Encoding
print(data[['Color', 'Color_encoded', 'Education_Level', 'Education_Level_encoded']].head())  # In ra dữ liệu với Label Encoding

#BÀI 6 (Sử dụng file PCA_Data_Sample.csv để Giảm chiều dữ liệu)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Bước 1: Tải và Chuẩn bị Dữ liệu
file_path = r"/Users/buimanh/Downloads/PCA_Data_Sample.csv"
data = pd.read_csv(file_path)

# Bước 2: Trực quan hóa Dữ liệu ban đầu
# Xem thống kê cơ bản và trực quan hóa dữ liệu thông qua biểu đồ
data.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Bước 3: Áp dụng PCA

# Chuẩn hóa Dữ liệu: Chuẩn hóa dữ liệu để mỗi biến có trung bình bằng 0 và độ lệch chuẩn bằng 1
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop('Customer_ID', axis=1))  # Loại bỏ cột ID không cần thiết

# Thực hiện PCA: Chọn số lượng thành phần tối đa để giải thích đa số biến động trong dữ liệu
pca = PCA(n_components=2)  # Giảm xuống còn 2 thành phần chính (2 cột)
principal_components = pca.fit_transform(data_scaled)

# In ra tỷ lệ phương sai giải thích bởi từng thành phần chính
print("Explained variance by each component:", pca.explained_variance_ratio_)

# Bước 4: Trực quan hóa kết quả PCA

# Vẽ biểu đồ các điểm dữ liệu trên không gian hai chiều tạo bởi hai thành phần chính đầu tiên
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Result')
plt.grid(True)

plt.show()

# Bước 5: Đánh giá và Thảo luận

# Phân tích các thành phần chính
# In ra các trọng số của các thành phần chính
print("Principal Components (Eigenvectors):", pca.components_)

# Thảo luận về sự thay đổi của dữ liệu và tác dụng của PCA trong việc giảm kích thước dữ liệu.