from langchain_openai import OpenAIEmbeddings
from pdfminer.high_level import extract_pages, extract_text
import pinecone
import re
from langchain.schema import Document

from tqdm.auto import tqdm
from uuid import uuid4
from pinecone.exceptions import PineconeException

from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import tiktoken

import time

# Add a delay of 5 seconds

note = '''
If you are getting an error that says "ModuleNotFoundError: 
No module named 'altair.vegalite.v4'" - you have to downgrade your 
Altair to version 4.1.0 as version 5 is giving the error. You can do 
this by running the command "pip install altair==4.1.0"
'''




# Create the length function
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')

    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=70,
        length_function=tiktoken_len,
        separators=['\n\n', '\n', ' ', '']
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=st.secrets["open_api_key"]
)

def extract_pages_content(path):
    docs =[]
    ppages = []

    text = extract_text(path)
    pages = text.split('\x0c')

    if len(pages[-1]) == 0:
        pages = pages[:-1]

    for page in pages:
        pattern = r"ARC Dental Station\d+"

        if re.match(pattern, page.split('\n\n')[-1]):
            page = '\n\n'.join(page.split('\n\n')[:-1])

            ppages.append(page)
        else:
            ppages.append(page)

    content = '\n'.join(ppages)


    docs.append(Document(
            page_content=content,
            metadata={"title": 'dental.pdf'}
        ))

    return docs


def update_record(pkl_file, index_name, p_key, p_env):
    pinecone.init(
        api_key=p_key,
        environment=p_env
    )

    # Delete all records
    index = pinecone.Index(index_name)
    index.delete(delete_all=True)


    index = pinecone.GRPCIndex(index_name)

    batch_limit = 100 # I don't want to upset or add any more than 100 records at any one time
    # 2 reasons
    # 1. The API request to OPEN AI and you can only send and recieve so much data
    # 2. The API request to Pinecone that for the exact same reason can only send so much data

    # Initialize this lists
    metadatas = []
    texts = []

    for i, record in enumerate(pkl_file):
        # first get metadata fields for this record
        metadata = {
            'title': str(record.metadata['title'])
        }

        # now we create chunks from the record text
        record_texts = get_text_chunks(record.page_content)

        # # create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]

        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)

        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit: # len(texts) here refers to the number of elements in a list, it's not a string, so technically it's 100 batches of chunks
            ids = [str(uuid4()) for _ in range(len(texts))] # We just make a unique identifier
            embeds = embed.embed_documents(texts)
            # print(len(embeds))
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))

    return'Success in updating Knowledge base'



def main():
    #load_dotenv("C:/Users/Stephen/source/repos/kb_feature/kb_feature/.env") # You can specify the path or leave it blank
    st.set_page_config(page_title="Update Knowledge Base")
    st.header("Update Knowledge Base")
 
    # Upload the file 
    # allowed_types = ["jpg", "jpeg", "png", "pdf", "docx", "txt"]
    allowed_types = ["pdf"]

    uploaded_file = st.file_uploader("Upload your PDF file", type=allowed_types, accept_multiple_files=False)


    # Button to process and show file groups
    if st.button('Update'):
        if uploaded_file:
            file_name = uploaded_file.name
            # Check tSe file extension to determine the group
            if file_name.lower().endswith(('.pdf')):

                with st.spinner('Please wait for a few seconds...'):
                    
                    f_dental = extract_pages_content(uploaded_file)
                    update_record(f_dental, 
                                  index_name=st.secrets["index_name"], 
                                  p_key=st.secrets["pinecone_api_key"], 
                                  p_env=st.secrets["pinecone_env"])
                    time.sleep(10)

                st.success('Knowledge base Successfully updated', icon="✅")
            else:
                st.write("Please upload a PDF file")
        else:
            st.write("No PDF file uploaded. Please upload a PDF file.")




if __name__ == '__main__':
    main()







