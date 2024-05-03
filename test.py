import streamlit as st

def main():
    st.title("Trang chủ")

    st.write("""
    ## Chọn trang để xem
    - [Trang 1](http://localhost:8501/?page=trang_1)
    - [Trang 2](http://localhost:8501/?page=trang_2)
    - [Trang 3](http://localhost:8501/?page=trang_3)
    """)

    page = st.experimental_get_query_params().get("page", [""])[0]

    if page == "trang_1":
        trang_1()
    elif page == "trang_2":
        trang_2()
    elif page == "trang_3":
        trang_3()

def trang_1():
    st.title("Trang 1")
    st.write("Nội dung của trang 1")

def trang_2():
    st.title("Trang 2")
    st.write("Nội dung của trang 2")

def trang_3():
    st.title("Trang 3")
    st.write("Nội dung của trang 3")

if __name__ == "__main__":
    main()
