

import streamlit as st
import numpy as np
import faiss
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import pandas as pd

# Load a pre-trained ResNet50 model for generating image embeddings
resnet_model = resnet50(pretrained=True)
resnet_model.eval()

# Function to generate an embedding from an image file
def generate_embedding(image_file):
    # Load and preprocess the image
    image = Image.open(image_file).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)

    # Generate embedding
    with torch.no_grad():
        embedding = resnet_model(image_tensor).numpy()

    return embedding

# Initialize image paths list and embeddings array
image_paths = ["result/oranges.png","result/watermelon.png"]
image_embeddings = np.array([])



# Function to create a FAISS index for images
def create_image_index(embeddings):
    d = embeddings.shape[1]  # dimension of the embeddings
    index = faiss.IndexFlatL2(d)  # using L2 distance
    index.add(embeddings)  # add embeddings to the index
    return index

# Function to add a new image
def add_image(image_file):
    global image_embeddings
    new_embedding = generate_embedding(image_file)
    if image_embeddings.size == 0:
        image_embeddings = new_embedding
    else:
        image_embeddings = np.vstack([image_embeddings, new_embedding])
    image_index.add(new_embedding)
    image_paths.append(image_file)
    # st.write(image_file)

# Function to modify an existing image
def modify_image(index_to_modify, image_file):
    global image_embeddings
    image_embeddings = np.delete(image_embeddings, index_to_modify, axis=0)
    image_index.reset()
    image_index.add(image_embeddings)
    new_embedding = generate_embedding(image_file)
    image_embeddings = np.vstack([image_embeddings, new_embedding])
    image_index.add(new_embedding)
    image_paths[index_to_modify] = image_file

# Function to delete an image
def delete_image(index_to_delete):
    global image_embeddings
    image_embeddings = np.delete(image_embeddings, index_to_delete, axis=0)
    image_index.reset()
    image_index.add(image_embeddings)
    image_paths.pop(index_to_delete)

# Function to create a DataFrame with the index and embeddings
def create_image_dataframe():
    vector_values = [emb.tolist() for emb in image_embeddings]
    return pd.DataFrame({'id': range(len(image_paths)), 'vector': vector_values})


# Initial embeddings and index creation for images
if image_paths:
    image_embeddings = np.vstack([generate_embedding(image_path) for image_path in image_paths])
    image_index = create_image_index(image_embeddings)
else:
    image_embeddings = np.array([])
    image_index = create_image_index(image_embeddings)


def image_operations(vAR_manipulate):
    w1,c1,w2=st.columns([1.7,5,0.1])
    w1, col1, col2, w2 = st.columns([1, 2.5, 2.9, 0.1])
    w1,col3,col4,w2=st.columns([1,2.5,2.9,0.1])
    w1,c2,w2=st.columns([1.7,5,0.1])
    w1,col5,col6,w2=st.columns([1,2.5,2.9,0.1])
    w1,c3,w2=st.columns([1,5,0.1])

    if 'flag5' not in st.session_state:
        st.session_state['flag5'] = False
    if 'flag6' not in st.session_state:
        st.session_state['flag6'] = False
    # if 'flag7' not in st.session_state:
    #     st.session_state['flag7'] = False
    # if 'flag8' not in st.session_state:
    #     st.session_state['flag8'] = False


    imgdf = create_image_dataframe()
    with c1:
        st.write("# ")
        st.markdown("<p style='text-align: center; color: black; font-size:16px;'><span style='font-weight: bold'>Vector DB data (before DB operation)</span></p>", unsafe_allow_html=True)
        
        st.dataframe(imgdf)
    if vAR_manipulate != "Delete":
        with col1:
            st.write("### ")
            st.write("")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload the Image File</span></p>", unsafe_allow_html=True)
        with col2:
            vAR_imageinput = st.file_uploader("", key="imageupload1")
        if vAR_imageinput:
            image = Image.open(vAR_imageinput)
            resized_image = image.resize((400, 300))
            with c2:
                st.write("## ")
                st.image(resized_image, caption='Uploaded Image')
            if vAR_manipulate == "Insert":
                if st.session_state['flag5'] == False:
                    add_image(vAR_imageinput)
                    imgdf = create_image_dataframe()
                    st.session_state['flag5']= True
                with c2:
                    st.write("# ")
                    st.markdown("<p style='text-align: center; color: black; font-size:16px;'><span style='font-weight: bold'>Vector DB data (after DB operation)</span></p>", unsafe_allow_html=True)
                    st.dataframe(imgdf)
                with col5:
                    st.write("# ")
                    st.write("")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Similarity Search</span></p>", unsafe_allow_html=True)
                with col6:
                    st.write("")
                    vAR_search_input=st.file_uploader("",key="imgsearchinput2")
                    if vAR_search_input != None:
                        image = Image.open(vAR_search_input)
                        resized_image = image.resize((400, 300))
                        with c3:
                            st.write("## ")
                            st.image(resized_image, caption='Uploaded Image')
                    st.write("")
                    if st.button("Search"):
                        with c3:
                            query_embedding = generate_embedding(vAR_search_input)
                            D, I = image_index.search(query_embedding, k=1)  # search for the top 1 nearest neighbor
                            st.write("Search results:")
                            st.image(image_paths[I[0][0]], caption=f"Result Image with Distance: {D[0][0]}")

            elif vAR_manipulate == "Modify":
                
                with col3:
                    st.write("# ")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Enter index to update</span></p>", unsafe_allow_html=True)
                with col4:
                    idx = st.number_input("", min_value=0, max_value=len(image_paths)-1, step=1)
                if st.session_state['flag6'] == False:
                    with col4:
                        if st.button("Modify"):
                            modify_image(idx, vAR_imageinput)
                            imgdf = create_image_dataframe()
                            st.session_state['flag6']= True
                            with c2:
                                st.write("# ")
                                st.markdown("<p style='text-align: center; color: black; font-size:16px;'><span style='font-weight: bold'>Vector DB data (after DB operation)</span></p>", unsafe_allow_html=True)
                                st.dataframe(imgdf)
                with col5:
                    st.write("# ")
                    st.write("")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Similarity Search</span></p>", unsafe_allow_html=True)
                with col6:
                    st.write("")
                    vAR_search_input=st.file_uploader("",key="imgsearchinput3")
                    if vAR_search_input != None:
                        image = Image.open(vAR_search_input)
                        resized_image = image.resize((400, 300))
                        with c3:
                            st.write("## ")
                            st.image(resized_image, caption='Uploaded Image')
                    st.write("")
                    if st.button("Search"):
                        
                        with c3:
                            query_embedding = generate_embedding(vAR_search_input)
                            D, I = image_index.search(query_embedding, k=1)  # search for the top 1 nearest neighbor
                            st.write("Search results:")
                            st.image(image_paths[I[0][0]], caption=f"Result Image with Distance: {D[0][0]}")

    if vAR_manipulate == "Delete":
        with col1:
            st.write("## ")
            # st.write("")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Enter index to delete</span></p>", unsafe_allow_html=True)
        with col2:
            idx = st.number_input("", min_value=0, max_value=len(image_paths)-1, step=1)
        
            if st.button("Delete"):

                delete_image(idx)
                imgdf = create_image_dataframe()
                with c2:
                    st.write("# ")
                    st.markdown("<p style='text-align: center; color: black; font-size:16px;'><span style='font-weight: bold'>Vector DB data (after DB operation)</span></p>", unsafe_allow_html=True)
                    st.dataframe(imgdf)
            with col5:
                st.write("# ")
                st.write("")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Similarity Search</span></p>", unsafe_allow_html=True)
            with col6:
                st.write("")
                vAR_search_input=st.file_uploader("",key="imgsearchinput4")
                if vAR_search_input != None:
                    image = Image.open(vAR_search_input)
                    resized_image = image.resize((400, 300))
                    with c3:
                        st.write("## ")
                        st.image(resized_image, caption='Uploaded Image')
                st.write("")
                with col6:
                    if st.button("Search"):
                        with c3:
                            query_embedding = generate_embedding(vAR_search_input)
                            D, I = image_index.search(query_embedding, k=1)  # search for the top 1 nearest neighbor
                            st.write("Search results:")
                            st.image(image_paths[I[0][0]], caption=f"Result Image with Distance: {D[0][0]}")

    # image_df = create_image_dataframe()
    # st.write("Image Data")
    # st.write(image_df)



