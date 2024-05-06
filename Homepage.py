# import base64
from typing import Union
import av
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, ClientSettings
import threading
import torchvision
from collections import Counter
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image
from unidecode import unidecode
import nltk
from nltk.tokenize import word_tokenize
# Function to get base64 encoding of an image file
nltk.download('punkt')
# def get_base64(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()
#
#
# # Function to set background image
# def set_background(png_file):
#     bin_str = get_base64(png_file)
#     page_bg_img = f'''
#         <style>
#             body {{
#                 background-image: url("data:image/png;base64,{bin_str}");
#                 background-size: cover;
#             }}
#         </style>
#     '''
#     st.markdown(page_bg_img, unsafe_allow_html=True)

def show_numerology_page():
    st.header("Thần số học")
    st.write(
        "Thần số học là nghệ thuật dựa trên việc phân tích các số liên quan đến ngày, tháng và năm sinh của bạn để hiểu về vận mệnh và tính cách.")
    # You can embed your numerology app here

    sochudao = {
        2: ["Nghệ thuật và Sáng tạo", "Truyền thông và Quảng cáo", "Tâm lý và Tư vấn tâm lý", "Giáo dục và Đào tạo"],
        3: ["Nghiên cứu và Phát triển", "Tài chính và Đầu tư", "Doanh nhân và Khởi nghiệp", "Luật sư và Luật phá",
            "Công nghệ thông tin và Lập trình"],
        4: ["Công nghệ thông tin và Lập trình", "Y tế và Chăm sóc sức khỏe", "Tài chính và Đầu tư",
            "Doanh nhân và Khởi nghiệp", "Giáo dục và Đào tạo"],
        22: ["Truyền thông và Quảng cáo", "Kinh doanh và Quản lý", "Nghệ thuật và Văn hóa", "Nghệ thuật và Sáng tạo"],
        5: ["Nghệ thuật và Sáng tạo", "Nghệ thuật và Văn hóa", "Du lịch và Phiêu lưu", "Truyền thông và Quảng cáo"],
        6: ["Y tế và Chăm sóc sức khỏe", "Giáo dục và Đào tạo", "Tâm lý và Tư vấn tâm lý", "Tình nguyện và Cứu trợ",
            "Nghệ thuật và Văn hóa"],
        7: ["Luật sư và Pháp luật", "Nghiên cứu và Phát triển", "Giáo dục và Đào tạo", "Nghệ thuật và Văn hóa"],
        8: ["Doanh nhân và Khởi nghiệp", "Nghệ thuật và Văn hóa", "Du lịch và Quản lý sự kiện", "Du lịch và Phiêu lưu",
            "Xây dựng và Bất động sản"],
        9: ["Tôn giáo và Tâm linh", "Nghiên cứu và Phát triển", "Giáo dục và Đào tạo", "Tâm lý và Tư vấn tâm lý",
            "Y tế và Chăm sóc sức khỏe"],
        10: ["Thể thao và Thể dục", "Nghệ thuật và Văn hóa", "Giáo dục và Đào tạo", "Quảng cáo và Truyền Thông",
             "Du lịch và Phiêu lưu"],
        11: ["Giáo dục và Đào tạo", "Tình nguyện và Cứu trợ", "Nghệ thuật và Sáng tạo", "Tâm lý và Tư vấn tâm lý",
             "Nghiên cứu và Phát triển"]
    }

    sosumenh = {
        1: ["Doanh nhân và Khởi nghiệp", "Công nghệ thông tin và Lập trình", "Nghệ thuật và Sáng tạo",
            "Tâm lý và Tư vấn tâm lý", "Giáo dục và Đào tạo"],
        2: ["Truyền thông và Quảng cáo", "Y tế và Chăm sóc sức khỏe", "Giáo dục và Đào tạo", "Nghệ thuật và Sáng tạo",
            "Tâm lý và Tư vấn tâm lý"],
        3: ["Nghệ thuật và Sáng tạo", "Truyền thông và Quảng cáo", "Giáo dục và Đào tạo", "Tự doanh và Sáng tạo",
            "Tâm lý và Tư vấn tâm lý"],
        4: ["Kinh doanh và Quản lý", "Tài chính và Đầu tư", "Luật sư và Pháp luật", "Y tế và Chăm sóc sức khỏe",
            "Giáo dục và Đào tạo"],
        5: ["Kinh doanh và Quản lý", "Du lịch và Phiêu lưu", "Doanh nhân và Khởi nghiệp", "Nghệ thuật và Sáng tạo"],
        6: ["Y tế và Chăm sóc sức khỏe", "Giáo dục và Đào tạo", "Kinh doanh và Quản lý", "Tâm lý và Tư vấn tâm lý"],
        7: ["Triết học và Nghiên cứu", "Công nghệ thông tin và Lập trình", "Nghệ thuật và Văn hóa",
            "Tâm lý và Tư vấn tâm lý"],
        8: ["Quản lý và Lãnh đạo", "Tài chính và Đầu tư", "Doanh nhân và Khởi nghiệp", "Luật sư và Pháp luật"],
        9: ["Nghệ thuật và Văn hóa", "Tình nguyện và Cứu trợ", "Giáo dục và Đào tạo", "Du lịch và Phiêu lưu"],
    }

    solinhhon = {
        1: ["Nghệ thuật và Văn hóa", "Du lịch và Phiêu lưu", "Nghệ thuật và Sáng tạo", "Giáo dục và đào tạo"],
        2: ["Tâm lý và Tư vấn tâm lý", "Y tế và Chăm sóc sức khỏe", "Giáo dục và đào tạo", "Truyền thông và Quảng cáo"],
        3: ["Doanh nhân và Khởi nghiệp", "Nghiên cứu và Phát triển", "Tài chính và Đầu tư", "Luật sư và Pháp luật"],
        4: ["Tâm lý và Tư vấn tâm lý", "Nghệ thuật và Văn hóa", "Nghệ thuật và Sáng tạo"],
        5: ["Nghệ thuật và Văn hóa", "Nghệ thuật và Sáng tạo", "Tình nguyện và Cứu trợ"],
        6: ["Giáo dục và đào tạo", "Tâm lý và Tư vấn tâm lý", "Y tế và Chăm sóc sức khỏe", "Kinh doanh và Quản lý"],
        7: ["Tâm lý và Tư vấn tâm lý", "Triết học và Nghiên cứu", "Nghệ thuật và Văn hóa"],
        8: ["Doanh nhân và Khởi nghiệp", "Tài chính và Đầu tư", "Luật sư và Pháp luật", "Kinh doanh và Quản lý"],
        9: ["Tình nguyện và Cứu trợ", "Nghệ thuật và Văn hóa", "Tôn giáo và Tâm linh", "Triết học và Nghiên cứu"],
        11: ["Y tế và Chăm sóc sức khỏe", "Tâm lý và Tư vấn tâm lý", "Nghệ thuật và Sáng tạo"],
        22: ["Nghiên cứu và Phát triển", "Kinh doanh và Quản lý", "Tôn giáo và Tâm linh", "Nghệ thuật và Văn hóa"],
    }

    def so_chu_dao(ngay_thang_nam_sinh):
        """Hàm tính số chủ đạo từ ngày tháng năm sinh."""
        # Tách ngày, tháng và năm từ chuỗi ngày tháng năm sinh
        day, month, year = map(int, ngay_thang_nam_sinh.split('/'))

        # Hàm tính tổng các chữ số của một số
        def tong_chu_so(number):
            total = 0
            while number > 0:
                total += number % 10
                number //= 10
            return total

        # Tính tổng các chữ số của ngày, tháng, năm sinh
        tong_chu_so_ngay = tong_chu_so(day)
        tong_chu_so_thang = tong_chu_so(month)
        tong_chu_so_nam = tong_chu_so(year)

        # Tính tổng tổng các chữ số
        tong = tong_chu_so_ngay + tong_chu_so_thang + tong_chu_so_nam

        # Tính số chủ đạo từ tổng
        while tong >= 10:
            if tong == 11 or tong == 22 or tong == 33:
                return tong
            tong = tong_chu_so(tong)

        return tong

    def tinh_chi_so_linh_hon(ten):
        chi_so = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
            'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'O': 6, 'P': 7, 'Q': 8, 'R': 9,
            'S': 1, 'T': 2, 'U': 3, 'V': 4, 'W': 5, 'X': 6, 'Y': 7, 'Z': 8
        }
        nguyen_am = ["A", 'O', 'E', 'I', 'U']

        def tinh_tong_chu_so(so):
            tong = 0
            while so > 0:
                tong += so % 10
                so //= 10
            return tong

        ten = unidecode(ten.upper())

        def check_Y(words, place):
            if words[place] != 'Y':
                return False
            else:
                if place == len(words) - 1:
                    if words[place - 1] not in nguyen_am:

                        return True
                    else:
                        return False
                else:
                    if words[place - 1] not in nguyen_am:
                        if words[place + 1] not in nguyen_am:
                            return True
                    else:
                        return False

        words = word_tokenize(ten)
        temp = []
        chi_so_linh_hon = 0
        for word in words:
            chus = list(word)

            for i in range(len(chus)):
                if check_Y(chus, i) or (chus[i] in nguyen_am):
                    temp.append(chus[i])
                    chi_so_linh_hon += chi_so[chus[i]]

        while chi_so_linh_hon >= 10:
            if chi_so_linh_hon == 11 or chi_so_linh_hon == 22:
                return chi_so_linh_hon
            chi_so_linh_hon = tinh_tong_chu_so(chi_so_linh_hon)

        return chi_so_linh_hon

    def tinh_chi_so_su_menh(ten):
        chi_so = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
            'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'O': 6, 'P': 7, 'Q': 8, 'R': 9,
            'S': 1, 'T': 2, 'U': 3, 'V': 4, 'W': 5, 'X': 6, 'Y': 7, 'Z': 8
        }
        nguyen_am = ["A", 'O', 'E', 'I', 'U']

        def tinh_tong_chu_so(so):
            tong = 0
            while so > 0:
                tong += so % 10
                so //= 10
            return tong

        ten = unidecode(ten.upper())
        words = word_tokenize(ten)
        temp = []
        chi_so_su_menh = 0
        for word in words:
            hold = 0
            chus = list(word)
            for chu in chus:
                hold += chi_so[chu]
            while hold >= 10:
                hold = tinh_tong_chu_so(hold)
            chi_so_su_menh += hold
        while chi_so_su_menh >= 10:
            if chi_so_su_menh == 11 or chi_so_su_menh == 22:
                return chi_so_su_menh
            chi_so_su_menh = tinh_tong_chu_so(chi_so_su_menh)

        return chi_so_su_menh

    def format_date(day, month, year):
        formatted_day = str(day).zfill(2)
        formatted_month = str(month).zfill(2)
        formatted_year = str(year)
        return f"{formatted_day}/{formatted_month}/{formatted_year}"

    def main():
        # Hàng nhập ngày tháng năm
        col_ngay, col_thang, col_nam = st.columns(3)
        with col_ngay:
            ngay = st.number_input("Ngày", min_value=1, max_value=31)
        with col_thang:
            thang = st.number_input("Tháng", min_value=1, max_value=12)
        with col_nam:
            nam = st.number_input("Năm", min_value=1900, max_value=2100)

        # Ô nhập họ tên
        ten = st.text_input("Họ và tên")

        if st.button("Start"):
            if not (ngay and thang and nam and ten):
                st.error("Vui lòng nhập đầy đủ thông tin")
            else:
                so_chu_dao_result = so_chu_dao(format_date(ngay, thang, nam))
                so_linh_hon_result = tinh_chi_so_linh_hon(ten)
                so_su_menh_result = tinh_chi_so_su_menh(ten)

                st.write("Kết quả:")
                st.subheader("Số chủ đạo")
                st.markdown(
                    f"<p style='text-align:center; font-size:80px; color:blue'><strong>{so_chu_dao_result}</strong></p>",
                    unsafe_allow_html=True)
                # st.markdown(f"**Giá trị:** {so_chu_dao_result}")
                st.markdown('**Ngành Nghề Phù Hợp:**')
                for job in sochudao[so_chu_dao_result]:
                    st.markdown(f"- {job}")
                st.markdown('**Đặc điểm tính cách:**')
                st.write("Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của thần số học ! ♥ ♥ ♥")

                st.subheader("Số linh hồn")
                st.markdown(
                    f"<p style='text-align:center; font-size:80px; color:blue'><strong>{so_linh_hon_result}</strong></p>",
                    unsafe_allow_html=True)
                # st.markdown(f"**Giá trị:** {so_linh_hon_result}")
                st.markdown('**Ngành Nghề Phù Hợp:**')
                for job in solinhhon[so_linh_hon_result]:
                    st.markdown(f"- {job}")
                st.markdown('**Đặc điểm tính cách:**')
                st.write("Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của thần số học ! ♥ ♥ ♥")

                st.subheader("Số sứ mệnh")
                st.markdown(
                    f"<p style='text-align:center; font-size:80px; color:blue'><strong>{so_su_menh_result}</strong></p>",
                    unsafe_allow_html=True)

                # st.markdown(f"**Giá trị:** {so_su_menh_result}")
                st.markdown('**Ngành Nghề Phù Hợp:**')
                for job in sosumenh[so_su_menh_result]:
                    st.markdown(f"- {job}")
                st.markdown('**Đặc điểm tính cách:**')
                st.write("Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của thần số học ! ♥ ♥ ♥")

    if __name__ == "__main__":
        main()

# Function for palmistry page
def show_palmistry_page():
    st.header("Sinh trắc học vân tay")
    st.write(
        "Sinh trắc học vân tay là nghiên cứu về các đặc điểm vân tay để xác định tính cách và tương lai của một người.")
    # You can add content related to palmistry here

    classes = ['Hình cung', 'Vòng tròn hướng tâm', 'Vòng lặp Ulnar', 'Vòm lều', 'Vòng xoáy']

    class FingerprintCNN(nn.Module):
        def __init__(self):
            super(FingerprintCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, len(classes))
    
        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = self.pool(nn.functional.relu(self.conv3(x)))
            x = self.pool(nn.functional.relu(self.conv4(x)))
            x = self.pool(nn.functional.relu(self.conv5(x)))
            x = x.view(-1, 128 * 4 * 4)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_fin = FingerprintCNN()
    model_fin.load_state_dict(torch.load(r'fingerprint.pth', map_location=device))
    model_fin.eval()
    
    
    
    # Define class labels and corresponding information
    
    class_info = {
        'Vòng xoáy': {
            'description': 'Những người có dấu vân tay vòng xoáy cực kỳ độc lập và có đặc điểm tính cách nổi trội. Vòng xoáy thường biểu thị mức độ thông minh cao và tính cách có ý chí mạnh mẽ. Những đặc điểm tiêu cực- Bản chất thống trị của họ đôi khi có thể dẫn đến chủ nghĩa hoàn hảo và thiếu sự đồng cảm với người khác.',
            'careers': ['Công nghệ thông tin và Lập trình', 'Kinh doanh và Quản lý', 'Nghệ thuật và Sáng tạo']
        },
        'Hình cung': {
            'description': 'Những người có dấu vân tay hình vòm có đặc điểm phân tích, thực tế và có tổ chức trong hành vi của họ. Họ sẽ tạo nên một sự nghiệp xuất sắc với tư cách là nhà khoa học hoặc bất kỳ lĩnh vực nào cần ứng dụng phương pháp luận. Đặc điểm tiêu cực- Họ là những người ít chấp nhận rủi ro nhất và không muốn đi chệch khỏi con đường cố định của mình.',
            'careers': ['Triết học và Nghiên cứu', 'Tài chính và Đầu tư', 'Công nghệ thông tin và Lập trình']
        },
        'Vòm lều': {
            'description': 'Một trong những đặc điểm tính cách của dấu vân tay bốc đồng là vòm lều. Họ thường thô lỗ và dường như không giới hạn bất kỳ hành vi cụ thể nào. Họ có thể chào đón một ngày và hoàn toàn không quan tâm vào ngày khác. Một lần nữa mái vòm hình lều là một dấu vân tay hiếm có thể tìm thấy.',
            'careers': ['Nghệ thuật và Sáng tạo', 'Truyền thông và Quảng cáo', 'Du lịch và Phiêu lưu']
        },
        'Vòng lặp Ulnar': {
            'description': 'Hạnh phúc khi đi theo dòng chảy và nói chung là hài lòng với cuộc sống, Hãy đối tác và nhân viên xuất sắc, Dễ gần và vui vẻ hòa nhập với dòng chảy, Thoải mái lãnh đạo một nhóm, Không giỏi tổ chức, Chấp nhận sự thay đổi, Có đạo đức làm việc tốt.',
            'careers': ['Truyền thông và Quảng cáo', 'Giáo dục và Đào tạo', 'Du lịch và Phiêu lưu']
        },
        'Vòng tròn hướng tâm': {
            'description': 'Những người có kiểu vòng tròn hướng tâm có xu hướng tự cho mình là trung tâm và ích kỷ. Họ thích đi ngược lại số đông, thắc mắc và chỉ trích. Họ yêu thích sự độc lập và thường rất thông minh.',
            'careers': ['Nghiên cứu và Phát triển', 'Nghệ thuật và Văn hóa', 'Nghệ thuật và Sáng tạo']
        }
    }
    
    
  
    
    
    # Function to predict label for input image
    def predict_label(img):
        # img = cv2.imread(image_path)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (128, 128))  # Resize the image to 128x128
        img.reshape(-1, 128, 128, 3)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model_fin(img)
            _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]
        return predicted_class

    uploaded_file = st.file_uploader("Nhập ảnh vân tay của bạn", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', width=200, use_column_width=False)

        # Start prediction when "Start" button is clicked
        if st.button('Start'):
            # Save the uploaded file locally
            # with open(uploaded_file.name, "wb") as f:
            #     f.write(uploaded_file.getbuffer())
            image = Image.open(uploaded_file)
            # Predict label
            predicted_label = predict_label(image)

            # Display prediction result
            st.header('Predicted Label:')
            st.write(predicted_label)
            st.header('Personality Traits:')
            st.write('Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của sinh trắc học vân tay ! ♥ ♥ ♥')
            st.header('Suitable Careers:')
            st.write(class_info[predicted_label]['careers'])

# Function for physiognomy page
def show_physiognomy_page():
    st.header("Nhân tướng học")
    st.write(
        "Nhân tướng học là việc đọc về tính cách và vận mệnh từ các đặc điểm về khuôn mặt, đặc biệt là khuôn mặt và dáng vẻ của mắt, mũi và miệng.")
    # You can add content related to physiognomy here

    # Load lại mô hình đã được huấn luyện
    model_path = r"face_shape_classifier.pth"
    train_dataset = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}

    class MyNormalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            # Kiểm tra số kênh của tensor
            if tensor.size(0) == 1:  # Nếu là ảnh xám
                # Thêm một kênh để đảm bảo phù hợp với normalize
                tensor = torch.cat([tensor, tensor, tensor], 0)

            # Normalize tensor
            tensor = transforms.functional.normalize(tensor, self.mean, self.std)
            return tensor

    # Load lại mô hình đã được huấn luyện
    device = torch.device('cpu')  # Sử dụng CPU
    model = torchvision.models.efficientnet_b4(pretrained=False)
    num_classes = len(train_dataset)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Định nghĩa biến đổi cho ảnh đầu vào
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        MyNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load mô hình nhận diện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Định nghĩa hàm dự đoán qua ảnh
    def predict_from_image(image):
        # Chuyển ảnh sang grayscale nếu cần thiết
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Chuyển ảnh sang numpy array
        image_np = np.array(image)

        # Chuyển ảnh sang grayscale để sử dụng mô hình nhận diện khuôn mặt
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Nhận diện khuôn mặt trong ảnh
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Nếu tìm thấy khuôn mặt, lấy ảnh khuôn mặt và thực hiện dự đoán
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Giả sử chỉ lấy khuôn mặt đầu tiên
            face_img = image.crop((x, y, x + w, y + h))  # Cắt ảnh khuôn mặt từ ảnh gốc

            # Áp dụng biến đổi cho ảnh khuôn mặt
            input_image = transform(face_img).unsqueeze(0)  # Thêm chiều batch (batch size = 1)

            # Thực hiện dự đoán
            with torch.no_grad():
                output = model(input_image)

            # Lấy chỉ số có giá trị lớn nhất là nhãn dự đoán
            predicted_class_idx = torch.argmax(output).item()

            train_dataset = {0: 'Khuôn mặt trái tim', 1: 'Khuôn mặt hình chữ nhật/Khuôn mặt dài',
                             2: 'Khuôn mặt trái xoan', 3: 'Khuôn mặt tròn', 4: 'Khuôn mặt vuông'}
            # Lấy tên của nhãn dự đoán từ tập dữ liệu
            predicted_label = train_dataset[predicted_class_idx]

            return predicted_label
        else:
            return "No face detected."

    def predict_from_list(images):
        predicted_labels = []
        for image in images:

            # Chuyển ảnh sang numpy array
            image_np = np.array(image)

            # Chuyển ảnh sang grayscale để sử dụng mô hình nhận diện khuôn mặt
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            try:
                gray = Image.fromarray(gray)
                input_image = transform(gray).unsqueeze(0)  # Thêm chiều batch (batch size = 1)

                # Thực hiện dự đoán
                with torch.no_grad():
                    output = model(input_image)

                # Lấy chỉ số có giá trị lớn nhất là nhãn dự đoán
                predicted_class_idx = torch.argmax(output).item()

                train_dataset = {0: 'Khuôn mặt trái tim', 1: 'Khuôn mặt hình chữ nhật/Khuôn mặt dài',
                                 2: 'Khuôn mặt trái xoan', 3: 'Khuôn mặt tròn', 4: 'Khuôn mặt vuông'}
                # Lấy tên của nhãn dự đoán từ tập dữ liệu
                predicted_label = train_dataset[predicted_class_idx]

                predicted_labels.append(predicted_label)
            except:
                predicted_labels.append("No face detected")

        # Đếm số lần xuất hiện của mỗi nhãn dự đoán
        label_counts = Counter(predicted_labels)

        # Lấy nhãn có số lần xuất hiện nhiều nhất
        most_common_label = label_counts.most_common(1)[0][0]

        return most_common_label

    class_info = {
        'Khuôn mặt trái xoan': {
            'description': 'Những người có khuôn mặt hình trái xoan không bao giờ sai lời nói. Họ luôn biết dùng từ ngữ phù hợp trong mọi tình huống – nghiêm túc hay vui vẻ. Mọi người tôn trọng họ về cách ăn nói và họ cũng có thể hòa hợp với các nhóm tuổi khác nhau nhờ kỹ năng giao tiếp hiệu quả. Đôi khi họ có thể quá tập trung vào việc nói tất cả những điều đúng đắn, điều này có thể khiến họ mất đi những cuộc trò chuyện không được lọc và những khoảnh khắc gắn kết',
            'careers': ['Truyền thông và Quảng cáo', 'Nghệ thuật và Văn hóa', 'Giáo dục và Đào tạo']
        },
        'Khuôn mặt trái tim': {
            'description': 'Những người có khuôn mặt hình trái tim là người có tinh thần mạnh mẽ. Đôi khi họ có thể quá bướng bỉnh, chỉ muốn mọi việc được thực hiện theo một cách cụ thể. Về mặt tích cực, họ lắng nghe trực giác của mình, điều này bảo vệ họ khỏi rơi vào những tình huống nguy hiểm. Họ cũng rất sáng tạo trong bất cứ điều gì họ làm.',
            'careers': ['Kinh doanh và Quản lý', 'Nghệ thuật và Sáng tạo']
        },
        'Khuôn mặt hình chữ nhật/Khuôn mặt dài': {
            'description': 'Bạn đã bao giờ nghe nói về việc đọc khuôn mặt và lòng bàn tay chưa? Vâng, ngay cả hình dạng khuôn mặt cũng có thể tiết lộ rất nhiều điều về tính cách của bạn. Nếu bạn có khuôn mặt hình chữ nhật, bạn tin tưởng nhiều vào suy nghĩ. Bạn dành thời gian suy nghĩ trước khi đưa ra bất kỳ quyết định quan trọng nào. Kết quả là bạn có thể suy nghĩ quá nhiều.',
            'careers': ['Luật sư và Pháp luật', 'Nghiên cứu và Phát triển', 'Tài chính và Đầu tư']
        },
        'Khuôn mặt tròn': {
            'description': 'Những người có khuôn mặt tròn là những người có trái tim nhân hậu. Họ tin vào việc giúp đỡ người khác và làm từ thiện. Do có tấm lòng bao dung nên đôi khi họ không ưu tiên bản thân mình, điều này có thể dẫn đến những kết quả không mấy tốt đẹp cho bản thân họ',
            'careers': ['Y tế và Chăm sóc sức khỏe', 'Tình nguyện và Cứu trợ', 'Tình nguyện và Cứu trợ']
        },
        'Khuôn mặt vuông': {
            'description': 'Những người có khuôn mặt này thường khá mạnh mẽ - cả về thể chất cũng như tình cảm. Tuy nhiên, hãy đảm bảo rằng bạn tiếp tục nuôi dưỡng những điểm mạnh của mình, nếu không chúng sẽ chỉ ở mức bề nổi trong tương lai.',
            'careers': ['Xây dựng và Bất động sản', 'Thể thao và Thể dục', 'Kinh doanh và Quản lý']
        },
        'No face detected': {
            'description': '',
            'careers': ['']
        }
    }

    def main():
        # Lựa chọn giữa webcam và tải ảnh
        option = st.radio("Chọn cách thức:", ("Webcam", "Image"))

        if option == "Webcam":

            WEBRTC_CLIENT_SETTINGS = ClientSettings(
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
            )

            class VideoTransformer(VideoProcessorBase):

                frame_lock: threading.Lock  # transform() is running in another thread, then a lock object is used here for thread-safety.

                in_image: Union[np.ndarray, None]

                def __init__(self) -> None:
                    self.frame_lock = threading.Lock()
                    self.in_image = None
                    self.img_list = []

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    in_image = frame.to_ndarray(format="bgr24")

                    global img_counter

                    with self.frame_lock:
                        self.in_image = in_image

                        gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                        # Draw rectangles around the detected faces
                        for (x, y, w, h) in faces:
                            cv2.rectangle(in_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            face = in_image[y:y + h, x:x + w]
                            if len(self.img_list) <= 10:
                                self.img_list.append(face)
                    return av.VideoFrame.from_ndarray(in_image, format="bgr24")

            ctx = webrtc_streamer(key="snapshot", client_settings=WEBRTC_CLIENT_SETTINGS,
                                  video_processor_factory=VideoTransformer)

            if ctx.video_transformer:
                if st.button("Predict"):
                    with ctx.video_transformer.frame_lock:

                        img_list = ctx.video_transformer.img_list

                    if img_list is not []:  # put in column form 5 images in a row
                        predicted_label = predict_from_list(img_list)
                        st.subheader("Hình Dạng Khuôn mặt:")
                        st.markdown(
                            f"<p style='text-align:center; font-size:60px; color:blue'><strong>{predicted_label}</strong></p>",
                            unsafe_allow_html=True)

                        st.markdown('**Ngành Nghề Phù Hợp:**')
                        for career in class_info[predicted_label]['careers']:
                            st.markdown(f"- {career}")
                        st.markdown('**Đặc điểm tính cách:**')
                        st.write("Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của thần số học ! ♥ ♥ ♥")
                    else:
                        st.warning("No faces available yet. Press predict again")


        elif option == "Image":
            st.write("Upload Image:")
            image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
            if image_file is not None:
                image = Image.open(image_file)
                predicted_label = predict_from_image(image)
                st.subheader("Hình Dạng Khuôn mặt:")
                st.markdown(
                    f"<p style='text-align:center; font-size:60px; color:blue'><strong>{predicted_label}</strong></p>",
                    unsafe_allow_html=True)

                st.markdown('**Ngành Nghề Phù Hợp:**')
                for career in class_info[predicted_label]['careers']:
                    st.markdown(f"- {career}")
                st.markdown('**Đặc điểm tính cách:**')
                st.write("Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của thần số học ! ♥ ♥ ♥")

    if __name__ == "__main__":
        main()

def homepage():
    st.title("Chào mừng đến với Direction-Pathway")
    st.write(
        "Chúng ta hãy khám phá những khía cạnh thú vị về vận mệnh và tính cách của bạn thông qua thần số học, sinh trắc học và nhân tướng học")
    st.title("Chào mừng đến với Trang chủ Thần số học, Sinh trắc học vân tay và Nhân tướng học")
    st.write("Chúng ta hãy khám phá những khía cạnh thú vị về vận mệnh, tính cách và vân tay của bạn!")

    st.header("Thần số học")
    st.write(
        "Thần số học là nghệ thuật dựa trên việc phân tích các số liên quan đến ngày, tháng và năm sinh của bạn để hiểu về vận mệnh và tính cách.")

    st.header("Sinh trắc học vân tay")
    st.write(
        "Sinh trắc học vân tay là nghiên cứu về các đặc điểm vân tay để xác định tính cách và tương lai của một người.")

    st.header("Nhân tướng học")
    st.write(
        "Nhân tướng học là việc đọc về tính cách và vận mệnh từ các đặc điểm về khuôn mặt, đặc biệt là khuôn mặt và dáng vẻ của mắt, mũi và miệng.")
# Main function
def main():
    # set_background('background.jpg')
    #
    # st.set_page_config(
    #     page_title="Direction-Pathway",
    #     page_icon=":))"
    # )
    # Sidebar navigation
    page_options = ["Trang chủ","Thần số học", "Sinh trắc học vân tay", "Nhân tướng học"]
    selected_page = st.sidebar.radio("Chọn trang", page_options)

    # Display selected page
    if selected_page == "Thần số học":
        show_numerology_page()
    elif selected_page == "Sinh trắc học vân tay":
        show_palmistry_page()
    elif selected_page == "Nhân tướng học":
        show_physiognomy_page()
    elif selected_page == "Trang chủ":
        homepage()



if __name__ == "__main__":
    main()
