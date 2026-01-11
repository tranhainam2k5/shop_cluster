# Rule-Based Transaction Clustering using Association Rules
## 1. Giới thiệu
Trong Mini Project này, nhóm thực hiện bài toán phân khúc giao dịch mua hàng dựa trên hành vi mua kèm, bằng cách kết hợp luật kết hợp (Association Rules) và phân cụm không giám sát (K-Means).

Thay vì phân cụm trực tiếp trên các thuộc tính thô của giao dịch, nhóm xây dựng đặc trưng hành vi từ các luật kết hợp Apriori, sau đó sử dụng các đặc trưng này làm đầu vào cho mô hình phân cụm. Cách tiếp cận này cho phép phản ánh trực tiếp các mối quan hệ mua kèm giữa các sản phẩm trong quá trình phân khúc.

Pipeline tổng quát:

Luật kết hợp → Đặc trưng hành vi mua kèm → Phân cụm → Trực quan hóa → Diễn giải
## 2. Mô tả dữ liệu
### 2.1 Dữ liệu giao dịch
Tập dữ liệu gốc: Online Retail Datasetz
Phạm vi sử dụng: các giao dịch hợp lệ tại United Kingdom
File: cleaned_uk_data.csv
Số bản ghi: 485,123
Số cột: 11

Các thuộc tính chính:

InvoiceNo: mã hóa đơn (Transaction ID)
Description: tên sản phẩm
Quantity: số lượng
UnitPrice: đơn giá
CustomerID
InvoiceDate
### 2.2 Dữ liệu luật kết hợp
Thuật toán sử dụng: Apriori
File: rules_apriori_filtered.csv
Số luật được sử dụng: 1,794
Các chỉ số kèm theo:
-Support
-Confidence
-Lift
Các luật đã được lọc trước để loại bỏ luật yếu, chỉ giữ các luật có ý nghĩa thống kê.
## 3. Phương pháp thực hiện
### 3.1 Chuẩn bị dữ liệu Transaction–Item
Mỗi hóa đơn (InvoiceNo) được xem là một transaction, trong đó tập các sản phẩm (Description) biểu diễn giỏ hàng của giao dịch đó.
Sau khi xử lý:
-Số transaction thu được: 18,021
-Mỗi transaction là một tập các item đã mua
### 3.2 Lựa chọn và sử dụng luật kết hợp
Nhóm sử dụng các luật kết hợp đã được khai thác từ Apriori và tiến hành:

-Sắp xếp luật theo Lift (ưu tiên luật có mức độ phụ thuộc cao)

Sử dụng toàn bộ 1,794 luật làm đầu vào cho bước xây dựng đặc trưng

Mỗi luật có dạng:

Antecedents → Consequents

kèm theo các chỉ số đánh giá chất lượng (support, confidence, lift).
 ### 3.3 Feature Engineering cho phân cụm
3.3.1 Biến thể 1 – Binary Rule Features (Baseline)

Với mỗi transaction và mỗi luật:
-Feature = 1 nếu transaction chứa đầy đủ antecedents và consequents của luật
-Feature = 0 trong trường hợp ngược lại
Kết quả:
-Ma trận đặc trưng kích thước 18,021 × 1,794
3.3.2 Biến thể 2 – Weighted Rule Features (Nâng cao)
Nhằm phản ánh độ mạnh của luật, nhóm xây dựng biến thể nâng cao:
-Feature = lift × confidence nếu transaction thỏa luật
-Feature = 0 nếu không thỏa
Biến thể này cho phép các luật mạnh đóng góp nhiều hơn vào đặc trưng phân cụm.
### 3.4 Chuẩn hóa dữ liệu
Toàn bộ đặc trưng được chuẩn hóa bằng StandardScaler nhằm:
-Đưa các đặc trưng về cùng thang đo
-Tránh việc các feature có giá trị lớn chi phối mô hình K-Means
## 4. Lựa chọn số cụm K
Nhóm sử dụng Silhouette Score để đánh giá chất lượng phân cụm, khảo sát số cụm trong khoảng từ K = 2 đến K = 6.
Kết quả:
K                              Silhouette Score
2                                 0.7714
3                                 0.7665
4                                 0.7626
5                                 0.6449
6                                 0.6467
Dựa trên kết quả:
-K = 2 cho giá trị Silhouette cao nhất
-Đồng thời đảm bảo khả năng diễn giải và ứng dụng thực tế
### 4.1 So sánh các biến thể đặc trưng
Biến thể	                Best K	  Silhouette
Binary Rule Features	       2	   0.771
Weighted Rule Features	       2       0.771
Nhận xét:
-Việc đưa trọng số lift × confidence không cải thiện rõ rệt Silhouette Score trong cấu hình hiện tại
-Biến thể baseline đã đủ hiệu quả để phân tách các transaction
## 5. Huấn luyện mô hình phân cụm
-Thuật toán: K-Means
-Số cụm: K = 2
-Đầu vào: ma trận rule-based features đã chuẩn hóa

Phân bố các cụm:
Cụm	Số        transaction
Cluster 0	   674
Cluster 1	   17,347
## 6. Trực quan hóa kết quả
Để đánh giá trực quan mức độ tách cụm:
-Nhóm sử dụng PCA để giảm chiều dữ liệu về 2D
-Vẽ scatter plot và tô màu theo nhãn cụm
Nhận xét:
-Hai cụm được tách tương đối rõ ràng
-Có sự chồng lấn nhẹ tại vùng ranh giới
-Phù hợp với Silhouette Score đạt được (~0.77)
## 7. Phân tích và diễn giải cụm

Dựa trên mức độ kích hoạt các luật trong từng cụm:
-Cluster 0: số lượng nhỏ, nhưng có mức kích hoạt luật cao hơn → đại diện cho các giao dịch có hành vi mua kèm rõ rệt
-Cluster 1: chiếm đa số, đại diện cho các giao dịch phổ biến với ít hành vi mua kèm phức tạp
Kết quả phân cụm có thể làm nền tảng cho:
-Thiết kế các chương trình bundle hoặc cross-sell
-Phân tích sâu hơn ở mức khách hàng
## 8. Kết luận

Mini Project đã:
-Xây dựng thành công pipeline phân khúc dựa trên luật kết hợp
-Chuyển hóa luật kết hợp thành đặc trưng hành vi có ý nghĩa
-Thực hiện phân cụm với chất lượng tốt và có khả năng diễn giải
-Kết quả cho thấy cách tiếp cận rule-based feature engineering là phù hợp cho bài toán phân tích hành vi mua hàng.