import cv2
import streamlit as st
import torch
from detect_streamlit import *
import subprocess


st.title("Analytics")
inference_msg = st.empty()
st.sidebar.title("Configurations")
input_source = st.sidebar.radio("Select input source", ('Local video', 'None'))
### Video module
if input_source == "Local video":
    video = st.sidebar.file_uploader("Select input image",
                                     type=["jpg", "png"], accept_multiple_files=False)

    if st.sidebar.button("StartDetection"):
        stframe = st.empty()
        st.subheader("Inference Stats")
        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")
            fps_warn = st.empty()

        with kpi2:
            st.markdown("**Detected objects in current Frame**")
            kpi2_text = st.markdown("0")

        #p = subprocess.Popen("python detect_streamlit.py --weights yolov7-tiny.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg", stdout=subprocess.PIPE, shell=True)
        #print(p.communicate())

        detect(
            weights='yolov7.pt',
            source='inference/images/' + video.name,
            stframe=stframe
        )
