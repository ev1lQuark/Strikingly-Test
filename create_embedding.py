import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from constant import VECTOR_OUTPUT_PATH


def create_embedding_from_pages(pages, chunk_size):
    print("trunk size: ", chunk_size)
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page["text"])
        docs.extend(splits)
        metadatas.extend([{"source": page["source"]}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")

    store = FAISS.from_texts(
        docs, OpenAIEmbeddings(), metadatas=metadatas
    )
    with open(VECTOR_OUTPUT_PATH, "wb") as f:
        pickle.dump(store, f)
    print("Done!")
