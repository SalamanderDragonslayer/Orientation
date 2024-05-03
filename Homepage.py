import streamlit as st
import base64


# Function to get base64 encoding of an image file
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Function to set background image
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
        <style>
            body {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: cover;
            }}
        </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Function to define the main content
def main():
    # Set background image
    set_background('background.jpg')

    # URL for the numerology app
    numerology_app_url = "D:\Duy\EXE\Orientation\Pages\app_num.py"  # Check this path

    # Configure Streamlit page
    # st.set_page_config(
    #     page_title="Direction-Pathway",
    #     page_icon=":))"
    # )

    # Main title
    st.title("Chào mừng đến với Direction-Pathway")
    st.write(
        "Chúng ta hãy khám phá những khía cạnh thú vị về vận mệnh và tính cách của bạn thông qua thần số học, sinh trắc học và nhân tướng học")

    # Numerology section
    st.header("Thần số học")
    st.write(
        "Thần số học là nghệ thuật dựa trên việc phân tích các số liên quan đến ngày, tháng và năm sinh của bạn để hiểu về vận mệnh và tính cách.")
    # st.write(f'<iframe src="{numerology_app_url}" width="700" height="500" frameborder="0"></iframe>',
    #          unsafe_allow_html=True)

    # Palmistry section
    st.header("Sinh trắc học vân tay")
    st.write(
        "Sinh trắc học vân tay là nghiên cứu về các đặc điểm vân tay để xác định tính cách và tương lai của một người.")

    # Physiognomy section
    st.header("Nhân tướng học")
    st.write(
        "Nhân tướng học là việc đọc về tính cách và vận mệnh từ các đặc điểm về khuôn mặt, đặc biệt là khuôn mặt và dáng vẻ của mắt, mũi và miệng.")


# Call the main function if this script is executed
if __name__ == "__main__":
    main()
