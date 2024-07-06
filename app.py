import streamlit as st
from streamlit_option_menu import option_menu
from src.main import main_interface

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    body {
        zoom: 90%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
from PIL import Image
import os


with open('style/final.css') as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
    
imcol1, imcol2, imcol3 = st.columns((7,5,2))
with imcol1:
    st.write("")
with imcol2:
    st.image('image/default_logo.png')
    #st.markdown("")
with imcol3:
    st.write("")

st.markdown("<p style='text-align: center; color: black;margin-top: -10px ;font-size:40px;'><span style='font-weight: bold'>Working with Vector Databases </span></p>", unsafe_allow_html=True)
st.markdown("<hr style='height:2.5px; margin-top:0px; width:80%; background-color:gray; margin-left:auto; margin-right:auto;'>", unsafe_allow_html=True)

#---------Side bar-------#

with st.sidebar:
    selected = st.selectbox(" ",
                     ['Vector DB'],key='text')
    Library = st.selectbox("",
                     ["Library Used","Streamlit","Image","Pandas"],key='text1')
    Gcp_cloud = st.selectbox(" ",
                     ["GCP Services Used","VM Instance","Computer Engine","Cloud Storage"],key='text2')
    GPT_TOOL =  st.selectbox(" ",['Models Used'],key='text3')
    st.markdown("## ")
    href = """<form action="#">
            <input type="submit" value="Clear/Reset" />
            </form>"""
    st.sidebar.markdown(href, unsafe_allow_html=True)
    st.markdown("# ")
    st.markdown("# ")
    st.markdown("<p style='text-align: center; color: White; font-size:20px;'>Build & Deployed on<span style='font-weight: bold'></span></p>", unsafe_allow_html=True)
    s1,s2=st.columns((2,2))
    with s1:
        st.markdown("### ")
        st.image('image/002.png')
    with s2:    
        st.markdown("### ")
        st.image("image/oie_png.png")


#--------------function calling-----------#
if __name__ == "__main__":
    # try:
        if selected == "Vector DB":
            main_interface()
    # except BaseException as error:
    #     st.error(error)