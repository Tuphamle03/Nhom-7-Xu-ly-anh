NHẬN DIỆN ĐỘNG VẬT

⚙️Công nghệ 

Đề tài sử dụng mô hình YOLO, một trong những phương pháp phát hiện đối tượng theo thời gian thực nổi bật hiện nay. Các bước chính bao gồm:

-Thu thập và xử lý dữ liệu hình ảnh động vật.

-Gán nhãn dữ liệu (labeling).

-Huấn luyện mô hình với tập dữ liệu đã gán nhãn.

-Kiểm tra và đánh giá kết quả nhận diện trên tập dữ liệu thử nghiệm.

Ngoài ra, các công cụ hỗ trợ như Python, OpenCV, LabelImg, PyTorch/TensorFlow, và Google Colab cũng được sử dụng để hỗ trợ quá trình xây dựng và triển khai hệ thống.

🧠Thiết kế hệ thống

Hệ thống được chia thành 5 thành phần chính nhằm đảm bảo logic rõ ràng, dễ bảo trì và
linh hoạt trong triển khai:

-Tiền xử lý dữ liệu: Thực hiện các thao tác như resize ảnh, cân bằng sáng, augment
(xoay, lật, thay đổi độ sáng), và chia tập dữ liệu (train/val/test).

-Huấn luyện mô hình: Sử dụng YOLOv5/YOLOv8 để huấn luyện với tập dữ liệu
động vật. Việc huấn luyện có thể điều chỉnh theo số lớp tương ứng với số loài động
vật.

-Suy luận (Inference): Dự đoán động vật từ hình ảnh hoặc video mới sử dụng mô
hình đã huấn luyện.

-Hiển thị kết quả: Gắn bounding boxes lên ảnh/video cùng với nhãn tên loài và
xác suất. Kết quả được hiển thị trực quan hoặc lưu ra tệp ảnh mới.

-Đánh giá mô hình: Tính toán các chỉ số hiệu suất: mAP (mean Average Precision),
Precision, Recall, F1-score,... và trực quan hóa kết quả qua biểu đồ.

💡Kết quả

![xử lý ảnh 5](https://github.com/user-attachments/assets/1ef41a6b-acf2-4611-9a7d-a24fa2eb3add)

![xử lý ảnh 8](https://github.com/user-attachments/assets/28fa8d21-a12c-4cdc-9a78-358bcfbf4000)

💡Mục tiêu

Đề tài hướng đến việc xây dựng một hệ thống có khả năng phát hiện và nhận diện nhiều loài động vật trong hình ảnh hoặc video một cách nhanh chóng, chính xác và hiệu quả. Cụ thể, các mục tiêu chính bao gồm:

-Tìm hiểu và ứng dụng các kỹ thuật học sâu (Deep Learning) trong lĩnh vực thị giác máy tính, đặc biệt là mô hình phát hiện đối tượng YOLO (You Only Look Once).

-Xây dựng tập dữ liệu gồm các hình ảnh của nhiều loài động vật khác nhau và thực hiện gán nhãn (labeling) để phục vụ cho quá trình huấn luyện mô hình.

-Huấn luyện mô hình nhận diện động vật sử dụng YOLO với tập dữ liệu đã chuẩn bị, tối ưu các tham số để cải thiện độ chính xác và tốc độ nhận diện.

-Thử nghiệm và đánh giá hiệu quả mô hình, dựa trên các chỉ số như độ chính xác (accuracy), độ bao phủ (IoU – Intersection over Union), tốc độ xử lý (FPS – Frame per Second), v.v.

-Đề xuất hướng cải tiến nhằm nâng cao hiệu năng hệ thống và khả năng ứng dụng trong thực tế, như giám sát động vật hoang dã, nông trại thông minh, hoặc các hệ thống an ninh.

👨‍💻Tác giả: Phạm Lê Tú
