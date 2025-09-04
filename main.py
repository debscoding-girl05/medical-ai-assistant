import os
import shutil
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image as PILImage

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# AI Analysis imports
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

# Medical Chatbot imports
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

# First Aid RAG import
from multimodal_first_aid import EnhancedFirstAidRAG

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Unified Medical Analysis API",
    description="Comprehensive medical analysis system with CNN models, AI fallback, chatbot, and first aid assistance",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for first aid images
if not os.path.exists("./data/first_aid_images"):
    os.makedirs("./data/first_aid_images", exist_ok=True)
app.mount("/images", StaticFiles(directory="./data/first_aid_images"), name="images")

# =============================================
# PYDANTIC MODELS
# =============================================

class ImageAnalysisRequest(BaseModel):
    analysis_type: str  # "brain_mri", "chest_xray"

class ImageAnalysisResponse(BaseModel):
    success: bool
    analysis_type: str
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    ai_analysis: Optional[str] = None
    message: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None  # For diagnosis elaboration

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

class FirstAidRequest(BaseModel):
    query: str
    max_results: Optional[int] = 3

class FirstAidResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    image_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class HealthStatus(BaseModel):
    status: str
    services: Dict[str, bool]
    message: str

# =============================================
# GLOBAL VARIABLES & CONFIGURATION
# =============================================

# Model configurations
class Config:
    IMAGE_SIZE = 224  # Adjust based on your models
    BRAIN_MRI_LABELS = ['pituitary', 'glioma', 'notumor', 'meningioma']
    CHEST_XRAY_LABELS = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA', 'COVID19']
    
    # Model paths
    BRAIN_MRI_MODEL_PATH = '/models/brain_mri_model.h5'
    CHEST_XRAY_MODEL_PATH = '/models/chestXray_model.keras'

cfg = Config()

# Global service instances
brain_mri_model = None
chest_xray_model = None
medical_agent = None
rag_chain = None
first_aid_rag = None

# =============================================
# INITIALIZATION FUNCTIONS
# =============================================

def load_cnn_models():
    """Load CNN models for brain MRI and chest X-ray analysis"""
    global brain_mri_model, chest_xray_model
    
    try:
        if os.path.exists(cfg.BRAIN_MRI_MODEL_PATH):
            brain_mri_model = load_model(cfg.BRAIN_MRI_MODEL_PATH)
            logger.info("Brain MRI model loaded successfully")
        else:
            logger.warning(f"Brain MRI model not found at {cfg.BRAIN_MRI_MODEL_PATH}")
            
        if os.path.exists(cfg.CHEST_XRAY_MODEL_PATH):
            chest_xray_model = load_model(cfg.CHEST_XRAY_MODEL_PATH)
            logger.info("Chest X-ray model loaded successfully")
        else:
            logger.warning(f"Chest X-ray model not found at {cfg.CHEST_XRAY_MODEL_PATH}")
            
    except Exception as e:
        logger.error(f"Error loading CNN models: {e}")

def initialize_ai_agent():
    """Initialize the AI fallback agent"""
    global medical_agent
    
    try:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            logger.warning("Google API Key not found. AI fallback will be disabled.")
            return
            
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        
        medical_agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[DuckDuckGoTools()],
            markdown=True
        )
        logger.info("Medical AI agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing AI agent: {e}")

def initialize_medical_chatbot():
    """Initialize the medical chatbot RAG system"""
    global rag_chain
    
    try:
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        
        if not PINECONE_API_KEY or not OPENAI_API_KEY:
            logger.warning("Pinecone or OpenAI API keys missing. Medical chatbot will be disabled.")
            return
            
        embeddings = download_hugging_face_embeddings()
        index_name = "medical-chatbot"
        
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        chat_model = ChatOpenAI(model="gpt-4o")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        logger.info("Medical chatbot RAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing medical chatbot: {e}")

def initialize_first_aid_rag():
    """Initialize the first aid RAG system"""
    global first_aid_rag
    
    try:
        first_aid_rag = EnhancedFirstAidRAG()
        logger.info(f"First Aid RAG initialized with {first_aid_rag.collection.count()} documents")
        
    except Exception as e:
        logger.error(f"Error initializing First Aid RAG: {e}")

# =============================================
# ANALYSIS FUNCTIONS
# =============================================

def analyze_brain_mri(image_path: str) -> Dict[str, Any]:
    """Analyze brain MRI image"""
    if brain_mri_model is None:
        raise HTTPException(status_code=503, detail="Brain MRI model not available")
    
    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = brain_mri_model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]
        
        # Prepare results
        probabilities = {label: float(prob) for label, prob in 
                        zip(cfg.BRAIN_MRI_LABELS, predictions[0])}
        
        predicted_label = cfg.BRAIN_MRI_LABELS[predicted_class_index]
        
        # Format result
        if predicted_label == 'notumor':
            result = "No Tumor Detected"
        else:
            result = f"Tumor Detected: {predicted_label.capitalize()}"
            
        return {
            "prediction": result,
            "confidence": float(confidence_score),
            "probabilities": probabilities,
            "raw_prediction": predicted_label
        }
        
    except Exception as e:
        logger.error(f"Error analyzing brain MRI: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

def analyze_chest_xray(image_path: str) -> Dict[str, Any]:
    """Analyze chest X-ray image"""
    if chest_xray_model is None:
        raise HTTPException(status_code=503, detail="Chest X-ray model not available")
    
    try:
        # Load and preprocess image
        image = load_img(image_path, target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        image = np.array(image)
        image = image / image.max()  # Normalize
        image = image.reshape(-1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)
        
        # Make prediction
        probabilities = chest_xray_model.predict(image).reshape(-1)
        predicted_class_index = np.argmax(probabilities)
        predicted_label = cfg.CHEST_XRAY_LABELS[predicted_class_index]
        confidence_score = float(probabilities[predicted_class_index])
        
        # Prepare results
        probabilities_dict = {label: float(prob) for label, prob in 
                             zip(cfg.CHEST_XRAY_LABELS, probabilities)}
        
        return {
            "prediction": predicted_label,
            "confidence": confidence_score,
            "probabilities": probabilities_dict,
            "raw_prediction": predicted_label
        }
        
    except Exception as e:
        logger.error(f"Error analyzing chest X-ray: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

def get_ai_analysis(image_path: str, analysis_type: str) -> str:
    """Get AI analysis as fallback"""
    if medical_agent is None:
        return "AI analysis not available"
    
    try:
        # Resize image for better processing
        image = PILImage.open(image_path)
        width, height = image.size
        aspect_ratio = width / height
        new_width = 500
        new_height = int(new_width / aspect_ratio)
        resized_image = image.resize((new_width, new_height))
        
        # Save resized image temporarily
        temp_resized = "temp_resized_image.png"
        resized_image.save(temp_resized)
        
        # Create AgnoImage object
        agno_image = AgnoImage(filepath=temp_resized)
        
        # Customize query based on analysis type
        if analysis_type == "brain_mri":
            query = """
            You are a radiologist analyzing a brain MRI scan. Provide a detailed analysis including:
            1. Image quality and technical adequacy
            2. Key anatomical structures visible
            3. Any abnormalities or pathological findings
            4. Potential diagnosis with confidence level
            5. Recommendations for further evaluation if needed
            Format your response in clear, professional medical language.
            """
        else:  # chest_xray
            query = """
            You are a radiologist analyzing a chest X-ray. Provide a detailed analysis including:
            1. Image quality and positioning
            2. Cardiothoracic structures assessment
            3. Lung fields evaluation
            4. Any pathological findings
            5. Potential diagnosis with confidence level
            6. Clinical recommendations
            Format your response in clear, professional medical language.
            """
        
        # Run AI analysis
        response = medical_agent.run(query, images=[agno_image])
        return response.content
        
    except Exception as e:
        logger.error(f"Error in AI analysis: {e}")
        return f"AI analysis failed: {str(e)}"
    finally:
        if os.path.exists("temp_resized_image.png"):
            os.remove("temp_resized_image.png")

# =============================================
# STARTUP EVENT
# =============================================

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    logger.info("Starting Unified Medical Analysis API...")
    
    load_cnn_models()
    initialize_ai_agent()
    initialize_medical_chatbot()
    initialize_first_aid_rag()
    
    logger.info("All services initialization completed")

# =============================================
# API ENDPOINTS
# =============================================

@app.get("/", response_model=Dict[str, Any])
async def root():
    return {
        "message": "Unified Medical Analysis API",
        "version": "2.0.0",
        "services": {
            "image_analysis": "CNN-based medical image analysis with AI fallback",
            "medical_chatbot": "RAG-based medical assistance with diagnosis elaboration",
            "first_aid": "Emergency first aid guidance with multimodal RAG"
        },
        "endpoints": {
            "health": {
                "url": "/health",
                "method": "GET",
                "description": "Check all services status"
            },
            "image_analysis": {
                "url": "/analyze-image",
                "method": "POST", 
                "description": "Analyze medical images (brain MRI or chest X-ray)",
                "parameters": {
                    "file": "Image file to analyze",
                    "analysis_type": "'brain_mri' or 'chest_xray'"
                }
            },
            "combined_analysis": {
                "url": "/analyze-and-elaborate", 
                "method": "POST",
                "description": "🔥 COMPLETE WORKFLOW: Analyze image + Get detailed medical explanation",
                "parameters": {
                    "file": "Image file to analyze",
                    "analysis_type": "'brain_mri' or 'chest_xray'",
                    "elaboration_question": "Custom question for elaboration (optional)"
                }
            },
            "elaborate_diagnosis": {
                "url": "/elaborate-diagnosis",
                "method": "POST", 
                "description": "Get detailed explanation for a specific diagnosis",
                "parameters": {
                    "diagnosis": "The diagnosis to elaborate on",
                    "analysis_type": "Type of analysis that produced the diagnosis",
                    "custom_question": "Specific question about the diagnosis (optional)"
                }
            },
            "medical_chat": {
                "url": "/medical-chat",
                "method": "POST",
                "description": "Chat with medical assistant",
                "parameters": {
                    "message": "Your medical question",
                    "context": "Previous diagnosis for context (optional)"
                }
            },
            "first_aid": {
                "url": "/first-aid",
                "method": "POST",
                "description": "Get first aid guidance",
                "parameters": {
                    "query": "First aid situation/question",
                    "max_results": "Number of results to return (default: 3)"
                }
            },
            "first_aid_search": {
                "url": "/first-aid-search", 
                "method": "POST",
                "description": "Search first aid database",
                "parameters": {
                    "query": "Search query",
                    "max_results": "Number of results (default: 3)"
                }
            }
        },
        "workflows": {
            "complete_diagnosis": {
                "description": "Complete medical diagnosis workflow",
                "steps": [
                    "1. Use /analyze-and-elaborate for complete analysis + explanation",
                    "2. Or use /analyze-image then /elaborate-diagnosis for step-by-step approach"
                ]
            },
            "model_selection": {
                "brain_mri": "Detects: pituitary, glioma, notumor, meningioma",
                "chest_xray": "Detects: NORMAL, TUBERCULOSIS, PNEUMONIA, COVID19"
            }
        }
    }

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Comprehensive health check for all services"""
    services_status = {
        "brain_mri_model": brain_mri_model is not None,
        "chest_xray_model": chest_xray_model is not None,
        "ai_agent": medical_agent is not None,
        "medical_chatbot": rag_chain is not None,
        "first_aid_rag": first_aid_rag is not None
    }
    
    all_healthy = all(services_status.values())
    
    return HealthStatus(
        status="healthy" if all_healthy else "partial",
        services=services_status,
        message="All services operational" if all_healthy else "Some services may be unavailable"
    )

@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_medical_image(
    file: UploadFile = File(...),
    analysis_type: str = "chest_xray"  # or "brain_mri"
):
    """Analyze medical images using CNN models with AI fallback"""
    
    if analysis_type not in ["brain_mri", "chest_xray"]:
        raise HTTPException(status_code=400, detail="Invalid analysis type. Use 'brain_mri' or 'chest_xray'")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Try CNN analysis first
        cnn_result = None
        try:
            if analysis_type == "brain_mri":
                cnn_result = analyze_brain_mri(temp_path)
            else:
                cnn_result = analyze_chest_xray(temp_path)
        except Exception as e:
            logger.warning(f"CNN analysis failed: {e}")
        
        # Get AI analysis as supplementary information
        ai_analysis = get_ai_analysis(temp_path, analysis_type)
        
        if cnn_result:
            return ImageAnalysisResponse(
                success=True,
                analysis_type=analysis_type,
                prediction=cnn_result["prediction"],
                confidence=cnn_result["confidence"],
                probabilities=cnn_result["probabilities"],
                ai_analysis=ai_analysis
            )
        else:
            return ImageAnalysisResponse(
                success=True,
                analysis_type=analysis_type,
                ai_analysis=ai_analysis,
                message="CNN analysis unavailable, using AI analysis"
            )
            
    except Exception as e:
        logger.error(f"Error in image analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/medical-chat", response_model=ChatResponse)
async def medical_chat(request: ChatRequest):
    """Medical chatbot with optional diagnosis elaboration"""
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Medical chatbot service unavailable")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Prepare the input message
        input_message = request.message
        
        # If context (diagnosis) is provided, enhance the message
        if request.context:
            input_message = f"""
            Based on the following medical diagnosis: "{request.context}"
            
            Please elaborate and provide detailed information about: {request.message}
            
            Include information about:
            - What this condition means
            - Possible causes and risk factors
            - Symptoms and progression
            - Treatment options
            - Prognosis and outlook
            - When to seek immediate medical attention
            - Lifestyle recommendations
            """
        
        logger.info(f"Processing medical chat: {input_message}")
        
        # Get response from RAG chain
        response = rag_chain.invoke({"input": input_message})
        
        # Extract sources
        sources = []
        if "context" in response:
            sources = [doc.metadata.get("source", "Unknown") for doc in response["context"]]
        
        return ChatResponse(
            answer=response["answer"],
            sources=list(set(sources))
        )
        
    except Exception as e:
        logger.error(f"Error in medical chat: {e}")
        raise HTTPException(status_code=500, detail="Medical chat service error")

@app.post("/first-aid", response_model=FirstAidResponse)
async def first_aid_query(request: FirstAidRequest):
    """First aid guidance using multimodal RAG"""
    
    if first_aid_rag is None:
        raise HTTPException(status_code=503, detail="First aid service unavailable")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        logger.info(f"Processing first aid query: {request.query}")
        result = first_aid_rag.generate_response(request.query)
        
        if not result["success"]:
            return FirstAidResponse(
                success=False,
                message=result.get("message", "No relevant first aid information found")
            )
        
        # Convert image path to URL
        image_url = None
        if result.get("image_path"):
            image_filename = os.path.basename(result["image_path"])
            image_url = f"/images/{image_filename}"
        
        return FirstAidResponse(
            success=True,
            response=result["response"],
            image_url=image_url,
            metadata={
                "query": request.query,
                "processing_time": "processed"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in first aid query: {e}")
        raise HTTPException(status_code=500, detail="First aid service error")

@app.post("/first-aid-search")
async def first_aid_search(request: FirstAidRequest):
    """Search first aid database without generating full response"""
    
    if first_aid_rag is None:
        raise HTTPException(status_code=503, detail="First aid service unavailable")
    
    try:
        results = first_aid_rag.query_db(request.query, results=request.max_results)
        
        search_results = []
        for i in range(len(results["uris"][0])):
            image_path = results["uris"][0][i]
            image_filename = os.path.basename(image_path)
            image_url = f"/images/{image_filename}"
            
            search_results.append({
                "image_url": image_url,
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "text_content": results["metadatas"][0][i].get("text_content", "No text available")
            })
        
        return {
            "results": search_results,
            "query": request.query,
            "total_found": len(search_results)
        }
        
    except Exception as e:
        logger.error(f"Error in first aid search: {e}")
        raise HTTPException(status_code=500, detail="First aid search error")

# =============================================
# UTILITY ENDPOINTS
# =============================================

@app.get("/image/{filename}")
async def get_first_aid_image(filename: str):
    """Serve first aid images"""
    if first_aid_rag is None:
        raise HTTPException(status_code=503, detail="First aid service unavailable")
    
    image_path = Path(first_aid_rag.images_folder) / filename
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )