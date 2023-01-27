from detect_streamlit import *
import tempfile
import cv2
import yaml
from yaml.loader import SafeLoader

import streamlit as st

def main():

    ## 1)
    st.title('Monitoramento de tráfego rodoviário usando o YOLOv7')

    ## 2)
    st.sidebar.title('Configurações')

    ## 3)
    confidence = st.sidebar.slider('Nível de confiança', min_value=0.0, max_value=1.0, value=0.25)
    st.sidebar.markdown('---')

    ## 4)
    # Checkboxes
    save_img = st.sidebar.checkbox('Salvar Vídeo')
    enable_GPU = st.sidebar.checkbox('Habilitar GPU')
    custom_classes = st.sidebar.checkbox('Selecionar classes')
    st.sidebar.markdown('---')

    if enable_GPU:
        device = 'gpu'
    else:
        device = 'cpu'

    ## 5)
    # Custom classes
    # Open the file and load the file
    with open('data/coco.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
        print(data)

    names = data['names']

    assigned_class_id = []

    if custom_classes:
        assigned_class = st.sidebar.multiselect('Selecionar classes', list(names), default=list(names))
        for each in assigned_class:
            assigned_class_id.append(names.index(each))

    ## 6)
    # Uploading out video
    video_file_buffer = st.sidebar.file_uploader("Upload do vídeo", type=["mp4", "mov", "avi"])
    DEMO_VIDEO = 'test.mp4'
    tfflie = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

    # We get out input video here
    if not video_file_buffer:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        tfflie.name = DEMO_VIDEO
        dem_vid = open(tfflie.name, 'rb')
        demo_bytes = dem_vid.read()

        st.sidebar.text('Input Video')
        st.sidebar.video(demo_bytes)
    else:
        tfflie.write(video_file_buffer.read())
        dem_vid = open(tfflie.name, 'rb')
        demo_bytes = dem_vid.read()

        st.sidebar.text('Input Video')
        st.sidebar.video(demo_bytes)

    print(tfflie.name)

    stframe = st.empty()
    st.sidebar.markdown('---')

    ## 7)
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Tracked Object**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Dimensão**")
        kpi3_text = st.markdown("0")

    ## 8)
    detect(
        weights='yolov7-tiny.pt',
        source=tfflie.name,
        conf_thres=confidence,
        classes=assigned_class_id,
        kpi1_text=kpi1_text,
        kpi2_text=kpi2_text,
        kpi3_text=kpi3_text,
        device=device,
        stframe=stframe
    )

    st.text('Processamento finalizado')
    vid.release()

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass