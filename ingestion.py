import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("./mediumblog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    # 2. Initialize Google Embeddings
    # Default model is "models/embedding-001"
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",  # This is the current stable 2026 ID
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        # This is the critical line that forces the 3072 vector down to 768
        output_dimensionality=768
    )

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")