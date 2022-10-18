from detect_streamlit import *
import tempfile
import cv2

import streamlit as st

def main():
    st.title('Detecção de Objectos usando o YOLOv7')

    st.sidebar.title('Configurações')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{ width: 500px;}
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{ width: 500px; margin-left: -200px}
        </style>
        """,
        unsafe_allow_html = True,
    )

    st.sidebar.markdown('---')
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
    st.sidebar.markdown('---')

    # Checkboxes
    save_img = st.sidebar.checkbox('Salvar Vídeo')
    enable_GPU = st.sidebar.checkbox('Habilitar GPU')
    custom_classes = st.sidebar.checkbox('Usar classes')
    assigned_class_id = []
    st.sidebar.markdown('---')

    # Custom classes
    if custom_classes:
        assigned_class = st.sidebar.multiselect('Selecionar classes', list(names), default='person')
        for each in assigned_class:
            assigned_class_id.append(names.index(each))

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

    #kp1, kp2, kp3 = st.columns(3)

    detect(
        weights='yolov7-tiny.pt',
        source=tfflie.name,
        stframe=stframe
    )

    st.text('Processamento finalizado')
    vid.release()







if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass