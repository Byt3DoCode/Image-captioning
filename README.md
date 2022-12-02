# Image-captioning
!!! link git ko bao gồm dataset folder và pre-trained GLOVE model !!!<br />
import lỗi gì pip install cái đó <br />
lưu ý: để tải tensorflow mượt nhất, máy tính nên sử dụng intel CPU và NVIDIA GPU
lỗi path ghi gì thì thêm cái đấy vào path <br /> <br />
File IC.ipynb cần folder data, tên folder: <br />
Flicker8k_Dataset <br />
Flicker8k_Text <br />
glove.6B <br />
Nếu chạy IC.ipynb sẽ sinh ra file: <br />
Descriptions.txt <br />
model_30.h5 (file mô hình chỉ có data)<br />
model_ic.h5 (file mô hình có cả data và config)<br />
Pickle/encoded_test_images.pkl <br />
Pickle/encoded_train_images.pkl <br />
Pickle/variables.pkl <br />
