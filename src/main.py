import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize texts list
texts = ["The athlete's rigorous training regimen and disciplined lifestyle are key factors in their consistent performance and success.",
    "Inflation affects purchasing power, causing prices to rise and reducing the value of money over time.",
    "Quantum computing could revolutionize encryption and processing, solving complex problems in seconds"]
embeddings = np.array([])


# if 'flag4' not in st.session_state:
#     st.session_state['flag4'] = False

# I do run daily to maintain my fitness

# Function to generate embeddings
def generate_embeddings(texts):
    return model.encode(texts)

# Function to create a FAISS index
def create_index(embeddings):
    d = embeddings.shape[1]  # dimension of the embeddings
    index = faiss.IndexFlatL2(d)  # using L2 distance
    index.add(embeddings)  # add embeddings to the index
    return index

# Function to add a new text
def add_text(new_text):
    global embeddings
    if type(new_text)==str:
        new_embedding = model.encode([new_text])
        if embeddings.size == 0:
            embeddings = new_embedding
        else:
            embeddings = np.vstack([embeddings, new_embedding])
        index.add(new_embedding)
        texts.append(new_text)
    if type(new_text)==list:
        for each in new_text:
            new_embedding = model.encode([each])
            if embeddings.size == 0:
                embeddings = new_embedding
            else:
                embeddings = np.vstack([embeddings, new_embedding])
            index.add(new_embedding)
            texts.append(each)

# Function to modify an existing text
def modify_text(index_to_modify, new_text):
    global embeddings
    embeddings = np.delete(embeddings, index_to_modify, axis=0)
    index.reset()
    index.add(embeddings)
    new_embedding = model.encode([new_text])
    embeddings = np.vstack([embeddings, new_embedding])
    index.add(new_embedding)
    texts[index_to_modify] = new_text

# Function to delete a text
def delete_text(index_to_delete):
    global embeddings
    embeddings = np.delete(embeddings, index_to_delete, axis=0)
    index.reset()
    index.add(embeddings)
    texts.pop(index_to_delete)

# Function to create a DataFrame with the index, texts, and embeddings
def create_dataframe():
    vector_values = [emb.tolist() for emb in embeddings]
    return pd.DataFrame({'id': range(len(texts)), 'text': texts, 'vector': vector_values})

                    
# Initial embeddings and index creation
if texts:
    embeddings = generate_embeddings(texts)
    index = create_index(embeddings)
    print("running")
else:
    embeddings = np.array([])
    index = create_index(embeddings)


def main_interface():
    w1,col1,col2,w2=st.columns([1,2.5,2.9,0.1])
    w1,c1,w2=st.columns([1,5,0.1])
    w1,col3,col4,w2=st.columns([1,2.5,2.9,0.1])
    w1,c2,w2=st.columns([1,5,0.1])
    w1,col5,col6,w2=st.columns([1,2.5,2.9,0.1])
    w1,c3,w2=st.columns([1,5,0.1])

    if 'flag1' not in st.session_state:
        st.session_state['flag1'] = False
    if 'flag2' not in st.session_state:
        st.session_state['flag2'] = False
    if 'flag3' not in st.session_state:
        st.session_state['flag3'] = False
    if 'flag4' not in st.session_state:
        st.session_state['flag4'] = False

    with col1:
        st.write("")
        st.write("## ")
        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Select Database</span></p>", unsafe_allow_html=True)
    with col2:
        st.write("")
        vAR_DB=st.selectbox("",["Select", "Faiss", "Milvus"],key="db")

    if vAR_DB != "Select":
        with col1:
            st.write("")
            st.write("## ")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Database Operation</span></p>", unsafe_allow_html=True)
        with col2:
            st.write("")
            vAR_manipulate=st.selectbox("",["Select", "Insert", "Modify", "Delete"],key="edit")

        if vAR_manipulate != "Select":
            with col1:
                st.write("## ")
                st.write("")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Data Type</span></p>", unsafe_allow_html=True)
            with col2:
                vAR_type=st.selectbox("",["Select", "Text", "Image", "Voice", "Video"], key="type")
            
            if vAR_type=="Text" and vAR_manipulate=="Insert":
                with col1:
                    st.write("### ")
                    st.write("")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Type of input</span></p>", unsafe_allow_html=True)
                with col2:
                    vAR_type_input=st.selectbox("",["Select","Raw text","Text file"], key="typeofinput")
                
                if vAR_type_input == "Raw text":
                    df = create_dataframe()
                    with c1:
                        st.write("# ")
                        
                        st.dataframe(df)
                    with col3:
                        st.write("### ")
                        st.write("")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Enter the Text</span></p>", unsafe_allow_html=True)
                    with col4:
                        vAR_input=st.text_input("",key="textinput")
                    if vAR_input:
                        if st.session_state['flag1'] == False:
                            add_text(vAR_input)
                            df = create_dataframe()
                            st.session_state['flag1'] = True
                        with c2:
                            st.write("# ")
                            st.dataframe(df)
                        with col5:
                            st.write("# ")
                            st.write("")
                            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Similarity Search</span></p>", unsafe_allow_html=True)
                        with col6:
                            st.write("")
                            vAR_search_input=st.text_input("",key="textsearchinput")
                            st.write("")
                            if st.button("Search"):
                                with c3:
                                    query_embedding = model.encode([vAR_search_input])
                                    D, I = index.search(query_embedding, k=2)
                                    st.write("")
                                    st.info("Rank 1 :\n\n"+texts[I[0][0]]+"\n\n Rank 2 : \n\n"+texts[I[0][1]])

                elif vAR_type_input == "Text file":
                    with col3:
                        st.write("# ")
                        st.write("")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload the Text File</span></p>", unsafe_allow_html=True)
                    with col4:
                        vAR_fileinput=st.file_uploader("",key="textupload")
                    if vAR_fileinput:
                        file_text = vAR_fileinput.read().decode('utf-8')
                        sentences = file_text.split('.')
                        sentences = [s.strip() for s in sentences if s.strip()]
                        if st.session_state["flag4"]==False:
                            add_text(sentences)
                            df = create_dataframe()
                            with c2:
                                st.write("## ")
                                st.dataframe(df)
                            st.session_state['flag4'] = True
                        with col5:
                            st.write("# ")
                            st.write("")
                            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Similarity Search</span></p>", unsafe_allow_html=True)
                        with col6:
                            st.write("")
                            vAR_search_input=st.text_input("",key="textsearchinput")
                            st.write("")
                            if st.button("Search"):
                                with c3:
                                    query_embedding = model.encode([vAR_search_input])
                                    D, I = index.search(query_embedding, k=2)
                                    st.write("")
                                    st.info("Rank 1 :\n\n"+texts[I[0][0]]+"\n\n Rank 2 : \n\n"+texts[I[0][1]])
                        
            elif vAR_type=="Text" and vAR_manipulate=="Modify":
                df = create_dataframe()
                with c1:
                    st.write("# ")
                    st.dataframe(df)
                with col3:
                    st.write("### ")
                    st.write("")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Enter the Text</span></p>", unsafe_allow_html=True)
                with col4:
                    vAR_input_to_m = st.text_input("", key="modifytextinput")
                if vAR_input_to_m:
                    with col3:
                        st.write("### ")
                        st.write("")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Which Index</span></p>", unsafe_allow_html=True)
                    with col4:
                        idx=st.number_input("", min_value=0, max_value=len(texts)-1, step=1, key="mod")
                        st.write("")
                        if st.button("Modify"):
                            if st.session_state['flag2'] == False:
                                modify_text(idx, vAR_input_to_m)
                                st.session_state['flag2'] = True
                            df = create_dataframe()
                            with c2:
                                st.write("# ")
                                st.dataframe(df)
                        with col5:
                            st.write("# ")
                            st.write("")
                            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Similarity Search</span></p>", unsafe_allow_html=True)
                        with col6:
                            st.write("")
                            vAR_search_input=st.text_input("",key="textsearchinput")
                            st.write("")
                            if st.button("Search"):
                                with c3:
                                    query_embedding = model.encode([vAR_search_input])
                                    D, I = index.search(query_embedding, k=2)
                                    st.write("")
                                    st.info("Rank 1 :\n\n"+texts[I[0][0]]+"\n\n Rank 2 : \n\n"+texts[I[0][1]])

            elif vAR_type=="Text" and vAR_manipulate=="Delete":
                df = create_dataframe()
                with c1:
                    st.write("# ")
                    st.dataframe(df)
                with col3:
                    st.write("### ")
                    st.write("")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Index to delete</span></p>", unsafe_allow_html=True)
                with col4:
                    idx=st.number_input("", min_value=0, max_value=len(texts)-1, step=1, key="del")
                    st.write("")
                    if st.button("Delete"):
                        if st.session_state['flag3']==False:
                            delete_text(idx)
                            st.session_state['flag3']=True
                        df = create_dataframe()
                        with c2:
                            st.write("# ")
                            st.dataframe(df)
                    with col5:
                        st.write("# ")
                        st.write("")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Similarity Search</span></p>", unsafe_allow_html=True)
                    with col6:
                        st.write("")
                        vAR_search_input=st.text_input("",key="textsearchinput")
                        st.write("")
                        if st.button("Search"):
                            with c3:
                                query_embedding = model.encode([vAR_search_input])
                                D, I = index.search(query_embedding, k=2)
                                st.write("")
                                st.info("Rank 1 :\n\n"+texts[I[0][0]]+"\n\n Rank 2 : \n\n"+texts[I[0][1]])






        # elif vAR_type =="Image":
        #     pass

                
                




