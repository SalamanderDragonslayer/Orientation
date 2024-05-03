# Loại bỏ import và tạo model từ TensorFlow và EfficientNet
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# from efficientnet_pytorch import EfficientNet
import streamlit as st
from PIL import Image
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
# model = torch.load(r'D:\Final1\boi\Finger_print\fingerprint.pth', map_location=device)
model = FingerprintCNN()
model.load_state_dict(torch.load(r'D:\Final1\boi\Finger_print\fingerprint.pth', map_location=device))
model.eval()



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


# Function to preprocess image for prediction

# Function to preprocess image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    return img.reshape(-1, 128, 128, 3)


# Function to predict label for input image
def predict_label(img):
    # img = cv2.imread(image_path)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    predicted_class = classes[predicted.item()]
    return predicted_class






# Streamlit App
st.title('Ứng Dụng Sinh Trắc Học Vân Tay')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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