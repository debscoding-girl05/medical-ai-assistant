from matplotlib import pyplot as plt
import os
import fitz
from PIL import Image
import warnings
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import base64
import re
import io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from typing import List, Dict, Tuple, Optional

# Load environment variables and suppress warnings
load_dotenv()
warnings.filterwarnings("ignore")

# Load spaCy model for text processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model not found. Installing now...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


class EnhancedFirstAidRAG:
    def __init__(self, pdf_folder="./data/pdfs", images_folder="./data/first_aid_images", db_path="./data/first_aid.db"):
        self.pdf_folder = pdf_folder
        self.images_folder = images_folder
        self.db_path = db_path
        
        os.makedirs(self.images_folder, exist_ok=True)
        
        # Initialize ChromaDB components
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.image_loader = ImageLoader()
        self.embedding_function = OpenCLIPEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            "first_aid_collection",
            embedding_function=self.embedding_function,
            data_loader=self.image_loader,
        )
        
        # Initialize LangChain components
        self.vision_model = ChatOpenAI(model="gpt-4o", temperature=0.0)
        self.parser = StrOutputParser()
        self.setup_prompt()

    def setup_prompt(self):
        """Sets up the prompt template for the vision model."""
        self.first_aid_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a compassionate and knowledgeable first aid expert. Your goal is to provide clear, actionable first aid instructions based on the provided text and images. Use the context to answer the user's question, emphasizing safety and care. If the situation is severe, advise them to seek professional medical help immediately. Maintain a reassuring, informative tone. Focus on the most relevant information from the provided context."
            ),
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": "Based on the provided information, how should I handle the following situation: {user_query}. Here is the most relevant text and image for context."
                    },
                    {
                        "type": "text",
                        "text": "Context: {document_text}"
                    },
                    {
                        "type": "image_url",
                        "image_url": "data:image/jpeg;base64,{image_data}",
                    },
                ],
            ),
        ])
        self.vision_chain = self.first_aid_prompt | self.vision_model | self.parser
        
    def extract_text_sections(self, text: str) -> List[Dict]:
        """Extracts meaningful text sections from page text."""
        if not text.strip():
            return []
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 1 and nlp:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
            paragraphs = [' '.join(sentences[i:i+4]) for i in range(0, len(sentences), 4)]
        
        meaningful_sections = []
        for paragraph in paragraphs:
            if len(paragraph) > 50 and not re.match(r'^\s*\d+\s*$', paragraph):
                meaningful_sections.append({
                    'text': paragraph,
                    'word_count': len(paragraph.split())
                })
        return meaningful_sections
        
    def calculate_image_text_relevance(self, image_path: str, text_sections: List[Dict]) -> List[Tuple[Dict, float]]:
        """Calculates relevance scores between an image and text sections using a vision model and TF-IDF."""
        if not text_sections:
            return []
        
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert at analyzing first aid images. Describe what you see in this image in 2-3 sentences, focusing on the medical condition, procedure, or equipment shown."),
                ("user", [
                    {"type": "text", "text": "Analyze this first aid image and describe what medical situation or procedure it shows:"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"},
                ]),
            ])
            analysis_chain = analysis_prompt | self.vision_model | self.parser
            image_description = analysis_chain.invoke({})
            
            text_contents = [section['text'] for section in text_sections]
            all_texts = text_contents + [image_description]
            
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            image_vector = tfidf_matrix[-1:]
            text_vectors = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(image_vector, text_vectors)[0]
            
            scored_sections = []
            for i, section in enumerate(text_sections):
                length_bonus = min(section['word_count'] / 100, 0.2)
                final_score = similarities[i] + length_bonus
                scored_sections.append((section, final_score))
            
            return scored_sections
            
        except Exception as e:
            print(f"Error calculating relevance for {image_path}: {e}")
            return [(section, 1.0) for section in text_sections]


    def extract_info_from_pdfs(self) -> List[Dict]:
        """
        Extracts and processes data from all PDFs in the specified folder.
        This method now ensures a consistent data structure is created.
        """
        extracted_data = []
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        if not pdf_files:
            print("No PDF files found.")
            return []

        global_id_counter = 0

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            print(f"Processing {pdf_path}...")
            
            try:
                doc = fitz.open(pdf_path)
                for page_num, page in enumerate(doc):
                    text = page.get_text().strip()
                    image_list = page.get_images(full=True)
                    
                    if not text and not image_list:
                        continue
                    
                    text_sections = self.extract_text_sections(text)
                    valid_images = []
                    
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        try:
                            image_bytes = doc.extract_image(xref)["image"]
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            if pil_image.size[0] > 100 and pil_image.size[1] > 100:
                                if pil_image.mode in ('CMYK', 'P'):
                                    pil_image = pil_image.convert('RGB')
                                
                                image_path = os.path.join(self.images_folder, f"{global_id_counter}_{page_num}.png")
                                pil_image.save(image_path, quality=95)
                                valid_images.append(image_path)
                                global_id_counter += 1
                        except Exception as e:
                            print(f"Error extracting image {img_index} on page {page_num}: {e}")
                    
                    if valid_images and text_sections:
                        best_score = -1
                        best_image = None
                        best_text = None
                        
                        for image_path in valid_images:
                            scored_sections = self.calculate_image_text_relevance(image_path, text_sections)
                            if scored_sections:
                                best_section, score = max(scored_sections, key=lambda x: x[1])
                                if score > best_score:
                                    best_score = score
                                    best_image = image_path
                                    best_text = best_section['text']
                                    
                        if best_image and best_text:
                            extracted_data.append({
                                "id": str(global_id_counter),
                                "uris": [best_image],
                                "text": best_text,
                                "metadata": {
                                    "pdf_file": pdf_file,
                                    "page_number": page_num,
                                    "relevance_score": best_score
                                }
                            })
                            global_id_counter += 1
                        
                    elif not valid_images and text_sections:
                        placeholder_path = os.path.join(self.images_folder, f"placeholder_{global_id_counter}.png")
                        Image.new('RGB', (100, 100), color='gray').save(placeholder_path)
                        extracted_data.append({
                            "id": str(global_id_counter),
                            "uris": [placeholder_path],
                            "text": ' '.join([s['text'] for s in text_sections]),
                            "metadata": {
                                "pdf_file": pdf_file,
                                "page_number": page_num,
                                "relevance_score": 1.0,
                                "is_placeholder": True
                            }
                        })
                        global_id_counter += 1
                        
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                
        return extracted_data

    def build_database(self):
     """Extract information and populate the database."""
     extracted_info = self.extract_info_from_pdfs()

     if extracted_info:
        try:
            self.chroma_client.delete_collection("first_aid_collection")
            self.collection = self.chroma_client.create_collection(
                "first_aid_collection",
                embedding_function=self.embedding_function,
                data_loader=self.image_loader,
            )
        except Exception:
            pass
        
        print(f"Adding {len(extracted_info)} items to the database...")

        ids = [item["id"] for item in extracted_info]
        uris = [item["uris"][0] for item in extracted_info]  # Extract the first URI from each list
        documents = [item["text"] for item in extracted_info]
        metadatas = []
        
        # Add text content to metadata so it's preserved with the image
        for i, item in enumerate(extracted_info):
            metadata = item["metadata"].copy()
            metadata["text_content"] = documents[i]  # Store text in metadata
            metadatas.append(metadata)

        # Add only images with metadata containing the text
        batch_size = 50
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            
            self.collection.add(
                ids=ids[i:batch_end],
                uris=uris[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        
        print(f"Added {len(extracted_info)} items to the database.")
        print(f"Total documents in database: {self.collection.count()}")

     return len(extracted_info)

    def query_db(self, query, results=3):
        print(f"Querying the database for: {query}")
        results = self.collection.query(
            query_texts=[query],
            n_results=results,
            include=["uris", "distances", "documents", "metadatas"]
        )
        return results

    def generate_response(self, user_query):
        results = self.query_db(user_query, results=1)
        
        if not results["uris"][0]:
            return {"success": False, "message": "No relevant information found."}
        
        best_image_path = results["uris"][0][0]
        best_document = results["metadatas"][0][0].get("text_content", "No text available")
    
        with open(best_image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        prompt_input = {
            "user_query": user_query,
            "document_text": best_document,
            "image_data": image_data
        }
        
        response = self.vision_chain.invoke(prompt_input)
        
        return {
            "success": True,
            "response": response,
            "image_path": best_image_path,
        }

    def show_image_from_uri(self, uri):
        try:
            img = Image.open(uri)
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"Could not display image: {e}")

if __name__ == "__main__":
    rag_system = EnhancedFirstAidRAG()
    
    # Check if database needs to be built. DELETE existing database files for a clean build.
    if rag_system.collection.count() == 0:
        print("Database is empty. Building database...")
        rag_system.build_database()
        
    print(f"Database contains {rag_system.collection.count()} documents.")
    print("Welcome to the First Aid RAG System!")

    query = input("Enter your first aid query: \n")
    if query:
        result = rag_system.generate_response(query)
        if result["success"]:
            print("\n--- Response ---")
            print(result["response"])
            print(f"\nImage Path: {result['image_path']}")
            rag_system.show_image_from_uri(result['image_path'])
        else:
            print(result["message"])