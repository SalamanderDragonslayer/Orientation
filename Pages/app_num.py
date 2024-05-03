from unidecode import unidecode
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
sochudao = {
    2: ["Nghệ thuật và Sáng tạo", "Truyền thông và Quảng cáo", "Tâm lý và Tư vấn tâm lý", "Giáo dục và Đào tạo"],
    3: ["Nghiên cứu và Phát triển", "Tài chính và Đầu tư", "Doanh nhân và Khởi nghiệp", "Luật sư và Luật phá", "Công nghệ thông tin và Lập trình"],
    4: ["Công nghệ thông tin và Lập trình", "Y tế và Chăm sóc sức khỏe", "Tài chính và Đầu tư", "Doanh nhân và Khởi nghiệp", "Giáo dục và Đào tạo"],
    22: ["Truyền thông và Quảng cáo", "Kinh doanh và Quản lý", "Nghệ thuật và Văn hóa", "Nghệ thuật và Sáng tạo"],
    5: ["Nghệ thuật và Sáng tạo", "Nghệ thuật và Văn hóa", "Du lịch và Phiêu lưu", "Truyền thông và Quảng cáo"],
    6: ["Y tế và Chăm sóc sức khỏe", "Giáo dục và Đào tạo", "Tâm lý và Tư vấn tâm lý", "Tình nguyện và Cứu trợ", "Nghệ thuật và Văn hóa"],
    7: ["Luật sư và Pháp luật", "Nghiên cứu và Phát triển", "Giáo dục và Đào tạo", "Nghệ thuật và Văn hóa"],
    8: ["Doanh nhân và Khởi nghiệp", "Nghệ thuật và Văn hóa", "Du lịch và Quản lý sự kiện", "Du lịch và Phiêu lưu", "Xây dựng và Bất động sản"],
    9: ["Tôn giáo và Tâm linh", "Nghiên cứu và Phát triển", "Giáo dục và Đào tạo", "Tâm lý và Tư vấn tâm lý", "Y tế và Chăm sóc sức khỏe"],
    10: ["Thể thao và Thể dục", "Nghệ thuật và Văn hóa", "Giáo dục và Đào tạo", "Quảng cáo và Truyền Thông", "Du lịch và Phiêu lưu"],
    11: ["Giáo dục và Đào tạo", "Tình nguyện và Cứu trợ", "Nghệ thuật và Sáng tạo", "Tâm lý và Tư vấn tâm lý", "Nghiên cứu và Phát triển"]
}

sosumenh = {
    1: ["Doanh nhân và Khởi nghiệp", "Công nghệ thông tin và Lập trình", "Nghệ thuật và Sáng tạo", "Tâm lý và Tư vấn tâm lý", "Giáo dục và Đào tạo"],
    2: ["Truyền thông và Quảng cáo", "Y tế và Chăm sóc sức khỏe", "Giáo dục và Đào tạo", "Nghệ thuật và Sáng tạo", "Tâm lý và Tư vấn tâm lý"],
    3: ["Nghệ thuật và Sáng tạo", "Truyền thông và Quảng cáo", "Giáo dục và Đào tạo", "Tự doanh và Sáng tạo", "Tâm lý và Tư vấn tâm lý"],
    4: ["Kinh doanh và Quản lý", "Tài chính và Đầu tư", "Luật sư và Pháp luật", "Y tế và Chăm sóc sức khỏe", "Giáo dục và Đào tạo"],
    5: ["Kinh doanh và Quản lý", "Du lịch và Phiêu lưu", "Doanh nhân và Khởi nghiệp", "Nghệ thuật và Sáng tạo"],
    6: ["Y tế và Chăm sóc sức khỏe", "Giáo dục và Đào tạo", "Kinh doanh và Quản lý", "Tâm lý và Tư vấn tâm lý"],
    7: ["Triết học và Nghiên cứu", "Công nghệ thông tin và Lập trình", "Nghệ thuật và Văn hóa", "Tâm lý và Tư vấn tâm lý"],
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
        if tong == 11 or tong == 22 or tong==33:
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

    def check_Y(words,place):
        if words[place] !='Y':
            return False
        else:
            if place == len(words)-1:
                if words[place-1] not in nguyen_am:

                    return True
                else:
                    return False
            else:
                if words[place - 1] not in nguyen_am:
                    if words[place+1] not in nguyen_am:
                        return True
                else:
                    return False
    words = word_tokenize(ten)
    temp=[]
    chi_so_linh_hon=0
    for word in words:
        chus = list(word)

        for i in range(len(chus)):
            if check_Y(chus,i) or (chus[i] in nguyen_am):
                temp.append(chus[i])
                chi_so_linh_hon += chi_so[chus[i]]

    while chi_so_linh_hon >=10:
        if chi_so_linh_hon ==11 or chi_so_linh_hon ==22:
            return chi_so_linh_hon
        chi_so_linh_hon= tinh_tong_chu_so(chi_so_linh_hon)

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
    st.title("Ứng dụng Tính Thần Số Học")

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
            st.markdown(f"<p style='text-align:center; font-size:80px; color:blue'><strong>{so_su_menh_result}</strong></p>",
                        unsafe_allow_html=True)

            # st.markdown(f"**Giá trị:** {so_su_menh_result}")
            st.markdown('**Ngành Nghề Phù Hợp:**')
            for job in sosumenh[so_su_menh_result]:
                st.markdown(f"- {job}")
            st.markdown('**Đặc điểm tính cách:**')
            st.write("Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của thần số học ! ♥ ♥ ♥")

if __name__ == "__main__":
    main()