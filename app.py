import streamlit as st
import requests
from PIL import Image
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
st.set_page_config(page_title="KatYos",page_icon=":tada:",layout="wide")
from model import teachable_machine_classification
from image_detec import object_detection_image
from image_detec import object_detection_video
import matplotlib.pyplot as plt
#asssets

def load_lottie1(url):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()
glasses=load_lottie1("https://assets4.lottiefiles.com/packages/lf20_i0ZzAQ.json")
person=load_lottie1("https://assets4.lottiefiles.com/packages/lf20_2cbmucbb.json")
logo=Image.open("images/logo.png")

#navigation bar
with st.sidebar:
    selected=option_menu(menu_title=None,options=["Home","Image","Video"],icons=["house-door-fill","camera-fill","camera-reels-fill"],
                    )
    
#using css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

#nheader

        
#body
if selected=="Home":
    with st.container():
        image_column,test_column=st.columns((1,2))
        with image_column:
            st_lottie(person,height=200,key="person")
        with test_column:
            st.title("Welcome to KatYos Virtual Assistant :wave:")
            st.subheader("I will help you choose your perfect Eye-Frames")
if selected=="Image":
    with st.container():
        
        st.title("Please Upload an image of your face ")
        left,right=st.columns(2)
        with left:
            uploaded_file = st.file_uploader('')
        with right:
            st.empty()
        if uploaded_file is not None:
            label,image,image2=object_detection_image(uploaded_file)
            left,right=st.columns(2)
            with left:
                st.image(image)
            with right:
                st.image(image2)
                st.subheader("Your face shape is "+label)
            
            st.title("Please select some additional options")
            #rules
            left,right=st.columns(2)
            with left:
                            option = st.selectbox(
     'How would you like to be contacted?',
     ('Email', 'Home phone', 'Mobile phone'))
            with right:
                st_lottie(glasses,height=200,key="glasses")

            
if selected=="Video":
    with st.container():
        st.write("---")
        left,right=st.columns(2)
        with left:
            st.header("Coming soon")
            st.header("##")
        with right:
            st_lottie(glasses,height=200)
            
            
            
            
            
            
            
            
            
            
            
            
            
