import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import io
import speech_recognition as sr
from pydub import AudioSegment

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to convert audio to text
def convert_audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text

# Function to save the audio file
def save_audio_file(audio_bytes, filename):
    with open(filename, 'wb') as f:
        f.write(audio_bytes)

# Function to get the audio file path
def get_audio_file_path(index):
    return f"result/audio_{index}.wav"


# Default audio files
default_audio_files = ["result/economics.wav", "result/sports.wav"]

# Convert default audio files to text
default_texts = []
for audio_path in default_audio_files:
    audio = AudioSegment.from_wav(audio_path)
    audio_bytes = io.BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)
    audio_file = io.BytesIO(audio_bytes.read())
    text = convert_audio_to_text(audio_file)
    default_texts.append(text)

# Initialize texts list
texts = default_texts.copy()
audio_paths = default_audio_files.copy()
embeddings = np.array([])

# Function to generate embeddings
def generate_embeddings(texts):
    return model.encode(texts)

# Function to create a FAISS index
def create_index(embeddings):
    d = embeddings.shape[1]  # dimension of the embeddings
    index = faiss.IndexFlatL2(d)  # using L2 distance
    index.add(embeddings)  # add embeddings to the index
    return index

# Function to add a new text and audio file
def add_text(new_text, audio_bytes):
    global embeddings
    new_embedding = model.encode([new_text])
    if embeddings.size == 0:
        embeddings = new_embedding
    else:
        embeddings = np.vstack([embeddings, new_embedding])
    index.add(new_embedding)
    texts.append(new_text)
    audio_index = len(texts) - 1
    audio_path = get_audio_file_path(audio_index)
    save_audio_file(audio_bytes, audio_path)
    audio_paths.append(audio_path)

# Function to modify an existing text and audio file
def modify_text(index_to_modify, new_text, audio_bytes):
    global embeddings
    embeddings = np.delete(embeddings, index_to_modify, axis=0)
    index.reset()
    index.add(embeddings)
    new_embedding = model.encode([new_text])
    embeddings = np.vstack([embeddings, new_embedding])
    index.add(new_embedding)
    texts[index_to_modify] = new_text
    audio_path = get_audio_file_path(index_to_modify)
    save_audio_file(audio_bytes, audio_path)
    audio_paths[index_to_modify] = audio_path

# Function to delete a text and audio file
def delete_text(index_to_delete):
    global embeddings
    embeddings = np.delete(embeddings, index_to_delete, axis=0)
    index.reset()
    index.add(embeddings)
    texts.pop(index_to_delete)
    audio_paths.pop(index_to_delete)

# Function to create a DataFrame with the index, texts, and embeddings
def create_dataframe():
    vector_values = [emb.tolist() for emb in embeddings]
    return pd.DataFrame({'id': range(len(texts)), 'vector': vector_values})

# Initial embeddings and index creation
if texts:
    embeddings = generate_embeddings(texts)
    index = create_index(embeddings)
else:
    embeddings = np.array([])
    index = create_index(embeddings)

def voice_operations(vAR_manipulate):
    w1,c1,w2=st.columns([1.7,5,0.1])
    w1, col1, col2, w2 = st.columns([1, 2.5, 2.9, 0.1])
    w1,col3,col4,w2=st.columns([1,2.5,2.9,0.1])
    w1,c2,w2=st.columns([1.7,5,0.1])
    w1,col5,col6,w2=st.columns([1,2.5,2.9,0.1])
    w1,c3,w2=st.columns([1,5,0.1])
    w1,col7,col8,w2=st.columns([1,2.5,2.9,0.1])
    w1,c4,w2=st.columns([1,5,0.1])

    if 'flag11' not in st.session_state:
        st.session_state['flag11'] = False
    if 'flag12' not in st.session_state:
        st.session_state['flag12'] = False
    if 'flag13' not in st.session_state:
        st.session_state['flag13'] = False
    if 'flag14' not in st.session_state:
        st.session_state['flag14'] = False

    voicedf = create_dataframe()
    with c1:
        st.write("# ")
        st.markdown("<p style='text-align: center; color: black; font-size:16px;'><span style='font-weight: bold'>Vector DB data (before DB operation)</span></p>", unsafe_allow_html=True)
        st.dataframe(voicedf)

    if vAR_manipulate != "Delete":
        with col1:
            st.write("### ")
            st.write("")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload the Audio File</span></p>", unsafe_allow_html=True)
        with col2:
            audio_file = st.file_uploader("", type=["wav"], key="voiceupload2")
        if audio_file:
            audio_bytes = audio_file.read()
            audio_file = io.BytesIO(audio_bytes)
            audio_text = convert_audio_to_text(audio_file)
            with c2:
                st.audio(audio_bytes, format='audio/wav')
            if audio_text:
                if vAR_manipulate == "Insert":
                    if st.session_state['flag11'] == False:
                        add_text(audio_text, audio_bytes)
                        voicedf = create_dataframe()
                        st.session_state['flag11'] = True
                    with c2:
                        st.write("# ")
                        st.markdown("<p style='text-align: center; color: black; font-size:16px;'><span style='font-weight: bold'>Vector DB data (after DB operation)</span></p>", unsafe_allow_html=True)
                        voicedf = create_dataframe()
                        st.dataframe(voicedf)
                    with col5:
                        st.write("# ")
                        st.write("")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Similarity Search</span></p>", unsafe_allow_html=True)
                    with col6:
                        st.write("")
                        vAR_search_input = st.file_uploader("", type=["wav"], key="voicesearchinput3")

                        st.write("")
                        if vAR_search_input:
                            audio_bytes = vAR_search_input.read()
                            audio_file = io.BytesIO(audio_bytes)
                            audio_text = convert_audio_to_text(audio_file)
                            with c3:
                                st.audio(audio_bytes, format='audio/wav')
                        if st.button("Search"):
                            with c3:
                                query_embedding = model.encode([audio_text])
                                D, I = index.search(query_embedding, k=2)
                                st.write("")
                                st.write("Search result")
                                # st.info("Rank 1 :\n\n" + texts[I[0][0]] + "\n\n Rank 2 : \n\n" + texts[I[0][1]])
                                st.audio(audio_paths[I[0][0]], format='audio/wav')
                                
                elif vAR_manipulate == "Modify":
                    with col5:
                        st.write("# ")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Enter index to modify</span></p>", unsafe_allow_html=True)
                    with col6:
                        index_to_modify = st.number_input("", min_value=0, max_value=len(texts)-1, step=1)
                        if st.session_state['flag2'] == False:
                            if st.button("Modify"):
                                modify_text(index_to_modify, audio_text, audio_bytes)
                                df = create_dataframe()
                                st.session_state['flag2'] = True
                                with c3:
                                    st.write("# ")
                                    st.markdown("<p style='text-align: center; color: black; font-size:16px;'><span style='font-weight: bold'>Vector DB data (after DB operation)</span></p>", unsafe_allow_html=True)
                                    st.dataframe(df)
                        with col7:
                            st.write("# ")
                            st.write("")
                            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Similarity Search</span></p>", unsafe_allow_html=True)
                        with col8:
                            st.write("")
                            vAR_search_input = st.file_uploader("", type=["wav"],key="voicesearchinput4")
                        st.write("")
                        if vAR_search_input:
                            audio_bytes = vAR_search_input.read()
                            audio_file = io.BytesIO(audio_bytes)
                            audio_text = convert_audio_to_text(audio_file)
                            with c4:
                                st.write("")
                                st.audio(audio_bytes, format='audio/wav')
                        with col8:
                            if st.button("Search"):
                                with c4:
                                    query_embedding = model.encode([audio_text])
                                    D, I = index.search(query_embedding, k=2)
                                    st.write("")
                                    st.write("Search result")
                                    # st.info("Rank 1 :\n\n" + texts[I[0][0]] + "\n\n Rank 2 : \n\n" + texts[I[0][1]])
                                    st.audio(audio_paths[I[0][0]], format='audio/wav')

    if vAR_manipulate == "Delete":
        with col1:
            st.write("## ")
            # st.write("")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Enter index to delete</span></p>", unsafe_allow_html=True)
        with col2:
            index_to_delete = st.number_input("", min_value=0, max_value=len(texts)-1, step=1)
            if st.button("Delete"):
                if st.session_state['flag13'] == False:
                    delete_text(index_to_delete)
                    df = create_dataframe()
                    st.session_state['flag13'] = True
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
                vAR_search_input = st.file_uploader("", type=["wav"],key="voicesearchinput1")
                st.write("")
                if vAR_search_input:
                    audio_bytes = vAR_search_input.read()
                    audio_file = io.BytesIO(audio_bytes)
                    audio_text = convert_audio_to_text(audio_file)
                    with c4:
                        st.write("")
                        st.audio(audio_bytes, format='audio/wav')
                with col8:
                    if st.button("Search"):
                        with c4:
                            query_embedding = model.encode([audio_text])
                            D, I = index.search(query_embedding, k=2)
                            st.write("")
                            st.write("Search result")
                            # st.info("Rank 1 :\n\n" + texts[I[0][0]] + "\n\n Rank 2 : \n\n" + texts[I[0][1]])
                            st.audio(audio_paths[I[0][0]], format='audio/wav')
