import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
import gradio as gr

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = "GEMINI_API_KEY"

# Load books data
books = pd.read_csv("data_test/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].fillna("images.jfif") + "&fife=w800"

# Load documents
loader = DirectoryLoader("data_test", glob="./*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
raw_documents = loader.load()

# Split text
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
text = text_splitter.split_documents(raw_documents)

# Initialize embedding model
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize vector store
persist_directory = "db_books"
vectordb = Chroma.from_documents(text, embedding, persist_directory=persist_directory)

# Initialize retriever
retriever = vectordb.as_retriever()

# Function to retrieve recommendations
def retrieve_semantic_recommendations(query, category="All", tone="All", initial_top_k=50, final_top_k=16):
    recs = retriever.get_relevant_documents(query, k=initial_top_k)
    books_list = []

    for rec in recs:
        content = rec.page_content.strip('"')
        if content.split()[0].isdigit():
            books_list.append(int(content.split()[0]))

    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)

    if tone != "All":
        emotion_map = {"Happy": "joy", "Surprising": "surprise", "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"}
        book_recs = book_recs.sort_values(by=emotion_map.get(tone, "joy"), ascending=False)

    return book_recs

# Function to display recommendations in Gradio
def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        authors = row["authors"] if pd.notna(row["authors"]) else "Unknown"
        description = " ".join(row["description"].split()[:30]) + "..."
        caption = f"{row['title']} by {authors}: {description}"
        results.append((row["large_thumbnail"], caption))

    return results

# Gradio App
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")
    user_query = gr.Textbox(label="Enter a book description:")
    category_dropdown = gr.Dropdown(categories, label="Category:", value="All")
    tone_dropdown = gr.Dropdown(tones, label="Tone:", value="All")
    submit_button = gr.Button("Find recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    dashboard.launch()
