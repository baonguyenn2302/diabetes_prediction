# Diabetes Prediction System

## 1. Tổng quan hệ thống

Dự án này xây dựng hệ thống phân loại bệnh tiểu đường bằng 5 mô hình học máy: Logistic Regression, Random Forest, SVM, k-NN và Decision Tree. Mục tiêu là phân loại bệnh nhân thành 2 lớp: không mắc tiểu đường (0) và có mắc tiểu đường (1) dựa trên các đặc trưng sức khỏe như tuổi, BMI, HbA1c, nồng độ glucose và tiền sử hút thuốc.

Hệ thống được thiết kế theo quy trình chuẩn của học máy: load dữ liệu, kiểm tra chất lượng, tiền xử lý, đào tạo 5 mô hình, đánh giá bằng các chỉ số phân loại, trực quan hóa ma trận nhầm lẫn và đường ROC. Dữ liệu gốc là tập `diabetes.csv` có 100000 mẫu, trong đó tỷ lệ lớp dương chỉ khoảng 8.5%, khiến bài toán có xu hướng mất cân bằng lớp. Vì vậy, mô hình cần ưu tiên cả độ chính xác tổng thể lẫn khả năng bắt được ca bệnh thực sự.

## 2. Tiền xử lý dữ liệu

Dữ liệu có 2 loại đặc trưng chính:

- Dữ liệu số: `age`, `hypertension`, `heart_disease`, `bmi`, `HbA1c_level`, `blood_glucose_level`.
- Dữ liệu phân loại: `gender`, `smoking_history`.

Các bước xử lý gồm:

- Chuyển đổi biến phân loại về dạng mã hóa phù hợp cho mô hình học máy.
- Điền giá trị thiếu bằng trung vị cho thuộc tính số và giá trị phổ biến cho thuộc tính phân loại.
- Chuẩn hóa đặc trưng số bằng `StandardScaler` để giảm ảnh hưởng của thang đo khác nhau, đặc biệt quan trọng đối với Logistic Regression, SVM và k-NN.
- Chia tập dữ liệu theo tỉ lệ 80/20, giữ nguyên tỷ lệ lớp nhãn bằng `stratify=y` để đảm bảo đánh giá công bằng.

## 3. Chi tiết 5 mô hình

| Mô hình | Cơ chế học tập | Tham số cốt lõi | Ưu điểm | Nhược điểm |
| --- | --- | --- | --- | --- |
| Logistic Regression | Mô hình hóa xác suất của lớp dương bằng hàm sigmoid, tìm đường biên tối ưu trên không gian đặc trưng. | Hệ số trọng số, bias, regularization | Hiểu dễ, nhanh, ổn định với dữ liệu cấu trúc | Khó nắm bắt quan hệ phi tuyến mạnh nếu dữ liệu quá phức tạp |
| Random Forest | Tập hợp nhiều cây quyết định học trên các bootstrap sample và aggregate dự đoán. | Số cây, độ sâu cây, số đặc trưng xét ở mỗi nút | Khá mạnh, ít overfitting, xử lý tốt nhiều biến | Cần thời gian huấn luyện hơn, khó diễn giải nếu cây quá nhiều |
| SVM | Tìm siêu phẳng tối ưu để tách lớp nhãn với khoảng cách lớn nhất; dùng kernel nếu cần. | Trọng số, bias, kernel, C, gamma | Hiệu quả trên không gian đặc trưng cao, robust với dữ liệu số | Phụ thuộc nhiều vào chuẩn hóa, tốn tài nguyên khi dataset lớn |
| k-NN | Dự đoán nhãn của điểm mới theo các điểm gần nhất trong không gian. | Số láng giềng k, khoảng cách | Đơn giản, không cần giả định phân phối dữ liệu | Tốn bộ nhớ, nhạy với thang đo và số chiều |
| Decision Tree | Chia không gian đặc trưng theo các ngưỡng để tạo các nút quyết định. | Độ sâu, số mẫu tối thiểu mỗi nút, tiêu chí chia nhánh | Dễ giải thích, mô hình hóa quy tắc quyết định | Dễ overfit, độ ổn định không cao |

## 4. Thực nghiệm và phân tích hình ảnh

### 4.1. Ma trận nhầm lẫn (Confusion Matrix)

Ma trận nhầm lẫn cho thấy lượng mẫu được dự đoán đúng và sai giữa hai lớp. Trong bài toán y tế, mục tiêu quan trọng là giảm thiểu trường hợp bỏ sót bệnh nhân tiểu đường, tức là `False Negative`. Một bệnh nhân bị chẩn đoán sai là “không bệnh” dù thực tế có bệnh sẽ dẫn đến việc không được điều trị kịp thời, tăng nguy cơ biến chứng tim mạch, thận, và mù lòa. Vì vậy, trong hệ thống này, mô hình tốt không chỉ cần độ chính xác cao, mà còn phải giữ Recall ở mức tốt để phát hiện được càng nhiều ca bệnh càng tốt.

Random Forest đạt Accuracy cao và Recall tốt hơn so với nhiều mô hình khác, nghĩa là nó đánh giá đúng nhiều bệnh nhân có tiểu đường hơn mà vẫn giữ tỷ lệ dương tính giả ở mức chấp nhận được. Đây là lý do mô hình này được chọn làm mô hình tốt nhất cho bài toán này.

### 4.2. Đường ROC - AUC

Đường ROC thể hiện sự thay đổi giữa TPR (True Positive Rate) và FPR (False Positive Rate) khi đổi ngưỡng phân loại. Diện tích dưới đường cong (AUC) phản ánh khả năng phân tách hai lớp của mô hình. Giá trị AUC gần 1 nghĩa là mô hình tách tốt giữa bệnh nhân tiểu đường và không tiểu đường. Trong thực nghiệm, Random Forest và Logistic Regression đều đạt AUC xấp xỉ 0.96, cho thấy khả năng phân loại rất tốt trên dữ liệu cấu trúc này.

## 5. Bảng so sánh hiệu suất tổng hợp

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.9703 | 0.9424 | 0.6929 | 0.7986 |
| SVM | 0.9645 | 0.9688 | 0.6018 | 0.7424 |
| Logistic Regression | 0.9604 | 0.8587 | 0.6400 | 0.7334 |
| k-NN | 0.9606 | 0.8786 | 0.6218 | 0.7282 |
| Decision Tree | 0.9508 | 0.7006 | 0.7365 | 0.7181 |

Mô hình tốt nhất là Random Forest vì nó cân bằng được ba yếu tố quan trọng: độ chính xác tổng thể cao, khả năng phát hiện bệnh rõ ràng (Recall) và F1-score ổn định. Tập dữ liệu chứa nhiều biến số y tế, trong đó các thuộc tính như HbA1c và glucose có ảnh hưởng mạnh đến xác suất bệnh. Random Forest phù hợp với cấu trúc này vì nó học được các quyết định phân đoạn và kết hợp nhiều cây, giúp nắm bắt tốt các tương tác phi tuyến giữa các biến mà Logistic Regression và SVM đơn giản hơn khó biểu diễn hết.

## 6. Kết luận

Hệ thống phân loại tiểu đường xây dựng thành công với mô hình Random Forest là lựa chọn ưu tiên nhất. Nó cho kết quả ổn định, dễ triển khai và có khả năng giảm nguy cơ bỏ sót bệnh nhân trong thực tế lâm sàng. Các hình ảnh được sinh tự động trong thư mục `outputs/` phục vụ trực quan hóa và giải thích mô hình cho người dùng và nhà khoa học dữ liệu.
