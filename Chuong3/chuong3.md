# <center> Welcome to Computer Vision</center>
 **<center>Trần Việt Anh - Hoàng Nguyên Phương</center>** 

Trước khi có thể bắt đầu xây dựng bộ phân loại hình ảnh của riêng mình, trước tiên chúng ta cần hiểu hình ảnh là gì. Chúng ta sẽ bắt đầu với các khối của một hình ảnh - pixel.Chúng ta sẽ thảo luận chính xác pixel là gì, cách chúng được sử dụng để tạo hình ảnh và cách truy cập pixel được biểu thị dưới dạng mảng NumPy. Chương này sẽ kết thúc với một cuộc thảo luận về tỷ lệ co của một hình ảnh và mối quan hệ mà nó có khi chuẩn bị tập dữ liệu hình ảnh của chúng tôi để đào tạo một mạng nơ-ron.

### 3.1 Pixel là gì

Điểm ảnh là các khối xây dựng thô của một hình ảnh. Mọi hình ảnh bao gồm một tập hợp các pixel. Thông thường, một pixel được coi là “màu sắc” hoặc “cường độ” của ánh sáng xuất hiện ở một vị trí nhất định trong hình ảnh của chúng ta. . Nếu chúng ta coi một hình ảnh là một lưới, mỗi ô vuông chứa một pixel duy nhất.

Ví dụ một hình ảnh có 1300x757 tức là 1300 pixel chiều dài và 757 chiều rộng. Chúng ta coi hình ảnh là 1 ma trận. Trong trường hợp này, ma trận này có 1300 cột và 757 dòng. Và tổng cộng có 1300x757 = 984100 pixels trong hình ảnh.

<center><img src="https://cdn.tgdd.vn/Files/2021/03/04/1332618/70-status-cau-noi-hay-ve-hoang-hon-khoang-thoi-g-5.jpg" width="300"/></center>
<center><font size="-1">Hình 3.1: Hình ảnh có 1300 chiều dài và 757 chiều rộng</font></center>


Mỗi pixels thường được biểu diễn dưới 2 dạng:
1. Hình ảnh mức xám/ Kênh màu đơn
2. Màu sắc

Trong hình ảnh thang độ xám, mỗi pixel là một giá trị vô hướng từ 0 đến 255, trong đó số 0 tương ứng với "đen" và 255 là "trắng". Các giá trị từ 0 đến 255 là các sắc thái xám khác nhau, trong đó các giá trị gần 0 sẽ tối hơn và các giá trị gần 255 sẽ nhạt hơn. Hình ảnh gradient thang độ xám trong Hình 3.2 cho thấy các điểm ảnh tối hơn ở phía bên trái và các điểm ảnh sáng dần dần ở phía bên phải.

<center><img src="https://images.viblo.asia/56b87230-8060-4861-91f4-1f27170624fa.PNG" /></center>
<center><font size="-1">Hình 3.2: Hình ảnh mức xám</font></center>
