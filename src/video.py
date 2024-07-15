import streamlit as st
import numpy as np
import faiss
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import pandas as pd
import io
import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load a pre-trained ResNet50 model for generating frame embeddings
model = resnet50(pretrained=True)
model.eval()

# Function to generate an embedding from a video file path
@st.cache_resource
def generate_embedding(video_path, frame_skip=5):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frame_embeddings = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames to reduce computation
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_skip != 0:
            continue

        # Preprocess the frame
        frame_tensor = preprocess(frame).unsqueeze(0)

        # Generate embedding for the frame
        with torch.no_grad():
            frame_embedding = model(frame_tensor).numpy()
        frame_embeddings.append(frame_embedding)

    cap.release()

    # Aggregate frame embeddings (e.g., by taking the mean)
    video_embedding = np.mean(frame_embeddings, axis=0)

    return video_embedding

# Default video files
default_video_files = ["result/css.mp4", "result/opus_ad.mp4"]

# Generate embeddings for the default video data
default_embeddings = np.vstack([generate_embedding(video_path) for video_path in default_video_files])

# Initialize video paths and embeddings list
video_paths = default_video_files.copy()
embeddings = default_embeddings.copy()

# Create a FAISS index
d = embeddings.shape[1]  # dimension of the embeddings
index = faiss.IndexFlatL2(d)  # using L2 distance
index.add(embeddings)  # add embeddings to the index

# Function to save the video file
def save_video_file(video_bytes, filename):
    with open(filename, 'wb') as f:
        f.write(video_bytes)

# Function to get the video file path
def get_video_file_path(index):
    return f"result/video_{index}.mp4"

# Function to add a new video
def add_video(video_bytes):
    global embeddings
    video_file = io.BytesIO(video_bytes)
    video_path = get_video_file_path(len(video_paths))
    save_video_file(video_bytes, video_path)
    new_embedding = generate_embedding(video_path)
    if embeddings.size == 0:
        embeddings = new_embedding
    else:
        embeddings = np.vstack([embeddings, new_embedding])
    index.add(new_embedding)
    video_paths.append(video_path)

# Function to modify an existing video
def modify_video(index_to_modify, video_bytes):
    global embeddings
    video_file = io.BytesIO(video_bytes)
    video_path = get_video_file_path(index_to_modify)
    save_video_file(video_bytes, video_path)
    embeddings = np.delete(embeddings, index_to_modify, axis=0)
    index.reset()
    index.add(embeddings)
    new_embedding = generate_embedding(video_path)
    embeddings = np.vstack([embeddings, new_embedding])
    index.add(new_embedding)
    video_paths[index_to_modify] = video_path

# Function to delete a video
def delete_video(index_to_delete):
    global embeddings
    embeddings = np.delete(embeddings, index_to_delete, axis=0)
    index.reset()
    index.add(embeddings)
    video_paths.pop(index_to_delete)

# Function to create a DataFrame with the index and embeddings
def create_dataframe():
    vector_values = [emb.tolist() for emb in embeddings]
    return pd.DataFrame({'id': range(len(video_paths)), 'vector': vector_values})

def video_operations(vAR_manipulate):
    w1,c1,w2=st.columns([1.7,5,0.1])
    w1, col1, col2, w2 = st.columns([1, 2.5, 2.9, 0.1])
    w1,col3,col4,w2=st.columns([1,2.5,2.9,0.1])
    w1,c2,w2=st.columns([1.7,5,0.1])
    w1,col5,col6,w2=st.columns([1,2.5,2.9,0.1])
    w1,c3,w2=st.columns([1,5,0.1])

    if 'flag21' not in st.session_state:
        st.session_state['flag21'] = False
    if 'flag22' not in st.session_state:
        st.session_state['flag22'] = False
    if 'flag23' not in st.session_state:
        st.session_state['flag23'] = False

    videodf = create_dataframe()
    with c1:
        st.write("# ")
        st.markdown("<p style='text-align: center; color: black; font-size:16px;'><span style='font-weight: bold'>Vector DB data (before DB operation)</span></p>", unsafe_allow_html=True)
        st.dataframe(videodf)
    if vAR_manipulate != "Delete":
        with col1:
            st.write("### ")
            st.write("")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload the Video File</span></p>", unsafe_allow_html=True)
        with col2:
            video_file = st.file_uploader("", type=["mp4"], key="videoupload")
        if video_file:
            video_bytes = video_file.read()
            with c2:
                st.video(video_file)
            if vAR_manipulate == "Insert":
                if st.session_state['flag21'] == False:
                    add_video(video_bytes)
                    videodf = create_dataframe()
                    st.session_state['flag21'] = True
                with c2:
                    st.write("# ")
                    st.markdown("<p style='text-align: center; color: black; font-size:16px;'><span style='font-weight: bold'>Vector DB data (after DB operation)</span></p>", unsafe_allow_html=True)
                    videodf = create_dataframe()
                    st.dataframe(videodf)
                with col5:
                    st.write("# ")
                    st.write("")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Similarity Search</span></p>", unsafe_allow_html=True)
                with col6:
                    st.write("")
                    uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi", "mkv"])
                    if uploaded_file is not None:
                        with c3:
                            st.video(uploaded_file)
                        # Save the uploaded file to the 'result' folder
                        with open(os.path.join("result", uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        # Read and display the video from the saved path
                        video_path = os.path.join("result", uploaded_file.name)
                    st.write("")
                    if st.button("Search"):
                        with c3:
                            query_embedding = generate_embedding(video_path)
                            D, I = index.search(query_embedding, k=2)
                            st.write("")
                            st.video(video_paths[I[0][0]])

            elif vAR_manipulate == "Modify":
                with col3:
                    st.write("# ")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Enter index to modify</span></p>", unsafe_allow_html=True)
                with col4:
                    index_to_modify = st.number_input("", min_value=0, max_value=len(video_paths)-1, step=1, key="modifyindex")
                    if st.button("Modify"):
                        if st.session_state['flag2'] == False:
                            modify_video(index_to_modify, video_bytes)
                            df = create_dataframe()
                            st.session_state['flag2'] = True
                        with c2:
                            st.write("# ")
                            st.markdown("<p style='text-align: center; color: black; font-size:16px;'><span style='font-weight: bold'>Vector DB data (after DB operation)</span></p>", unsafe_allow_html=True)
                            st.dataframe(df)
                    with col5:
                        st.write("# ")
                        st.write("")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Similarity Search</span></p>", unsafe_allow_html=True)
                    with col6:
                        st.write("")
                        uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi", "mkv"])
                        if uploaded_file is not None:
                            with c3:
                                st.video(uploaded_file)
                            # Save the uploaded file to the 'result' folder
                            with open(os.path.join("result", uploaded_file.name), "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            # Read and display the video from the saved path
                            video_path = os.path.join("result", uploaded_file.name)
                        st.write("")
                        if st.button("Search"):
                            with c3:
                                query_embedding = generate_embedding(video_path)
                                D, I = index.search(query_embedding, k=2)
                                st.write("")
                                st.video(video_paths[I[0][0]])
    
    if vAR_manipulate == "Delete":
        with col1:
            st.write("## ")
            # st.write("")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Enter index to delete</span></p>", unsafe_allow_html=True)
        with col2:
            index_to_delete = st.number_input("", min_value=0, max_value=len(video_paths)-1, step=1, key="deleteindex")
            if st.button("Delete"):
                if st.session_state['flag23'] == False:
                    delete_video(index_to_delete)
                    df = create_dataframe()
                    st.session_state['flag23'] = True
                with c2:
                    st.write("# ")
                    st.markdown("<p style='text-align: center; color: black; font-size:16px;'><span style='font-weight: bold'>Vector DB data (after DB operation)</span></p>", unsafe_allow_html=True)
                    st.dataframe(df)
        with col5:
            st.write("# ")
            st.write("")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Similarity Search</span></p>", unsafe_allow_html=True)
        with col6:
            st.write("")
            # File uploader for video file
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
            if uploaded_file is not None:
                with c3:
                    st.video(uploaded_file)
                # Save the uploaded file to the 'result' folder
                with open(os.path.join("result", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # Read and display the video from the saved path
                video_path = os.path.join("result", uploaded_file.name)
            st.write("")
            if st.button("Search"):
                with c3:
                    query_embedding = generate_embedding(video_path)
                    D, I = index.search(query_embedding, k=2)
                    st.write("")
                    st.video(video_paths[I[0][0]])

