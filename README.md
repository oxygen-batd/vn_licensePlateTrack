# Multi-Object Tracking with Ultralytics YOLO

- Efficiency: Process video streams in real-time without compromising accuracy.
- Flexibility: Supports multiple tracking algorithms and configurations.
- Ease of Use: Simple Python API and CLI options for quick integration and deployment.
- Customizability: Easy to use with custom trained YOLO models, allowing integration into domain-specific applications.


# Syntax

```
from ultralytics import YOLO

# Load the model and run the tracker with a custom configuration file
model = YOLO("yolov8n.pt")
results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")

```

Kết quả trả về (results):
 1. **boxes**
    
    ![image](https://github.com/oxygen-batd/vn_licensePlateTrack/assets/167840668/45639b16-b5a1-436a-9ca3-467a7ebab071)

     - xyxy: Tọa độ của các hộp giới hạn ở dạng [x1, y1, x2, y2]. Đây là tọa độ của hai góc đối diện của hình chữ nhật (trái trên và phải dưới). Loại giá trị: Tensor hoặc mảng numpy.
     - xywh: Tọa độ của các hộp giới hạn ở dạng [x, y, w, h]. Đây là tọa độ trung tâm của hình chữ nhật, cùng với chiều rộng và chiều cao. Loại giá trị: Tensor hoặc mảng numpy.
     - xyxyn: Tọa độ của các hộp giới hạn ở dạng chuẩn hóa [x1, y1, x2, y2], với các giá trị nằm trong khoảng [0, 1]. Loại giá trị: Tensor hoặc mảng numpy.
     - xywhn: Tọa độ của các hộp giới hạn ở dạng chuẩn hóa [x, y, w, h], với các giá trị nằm trong khoảng [0, 1]. Loại giá trị: Tensor hoặc mảng numpy.
     - conf: Độ tin cậy (confidence) của mỗi hộp giới hạn, cho biết mức độ tin cậy của mô hình rằng hộp chứa một đối tượng. Loại giá trị: Tensor hoặc mảng numpy.
     - cls: Lớp (class) của mỗi hộp giới hạn, cho biết loại đối tượng mà hộp chứa. Loại giá trị: Tensor hoặc mảng numpy.
     - id: ID theo dõi (tracking ID) của mỗi hộp giới hạn, được sử dụng trong việc theo dõi đối tượng qua các khung hình. Loại giá trị: Tensor hoặc mảng numpy, hoặc None nếu không có ID.
2. **masks**
   - Mục đích: Chứa thông tin về mặt nạ (mask) của các đối tượng được phát hiện, nếu mô hình được huấn luyện để tạo ra mặt nạ.
   - Loại giá trị: Thông thường là tensor hoặc danh sách.
3. **probs**
   - Mục đích: Xác suất của các lớp khác nhau cho mỗi đối tượng được phát hiện, hữu ích cho các tác vụ phân loại đa lớp.
   - Loại giá trị: Tensor hoặc mảng numpy.
4. **is_track**
   - Mục đích: Xác định xem kết quả có thuộc về chế độ theo dõi (tracking mode) hay không.
   - Loại giá trị: Boolean.
5. **orig_shape**
   - Mục đích: Kích thước gốc của hình ảnh đầu vào.
   - Loại giá trị: Tuple (chiều cao, chiều rộng)
6. **shape**
   - Mục đích: Kích thước của tensor chứa thông tin các hộp giới hạn.
   - Loại giá trị: Tuple.

# Example







