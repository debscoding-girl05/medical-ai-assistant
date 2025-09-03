import os
from pathlib import Path

class Config:
    # =============================================
    # MODEL CONFIGURATIONS
    # =============================================
    
    # Image processing settings
    IMAGE_SIZE = 224
    BRAIN_MRI_SIZE = 128
    CHEST_XRAY_SIZE = 224
    
    # Model labels
    BRAIN_MRI_LABELS = ['pituitary', 'glioma', 'notumor', 'meningioma']
    CHEST_XRAY_LABELS = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA', 'COVID19']
    
    # Model paths - these will work on Render
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    BRAIN_MRI_MODEL_PATH = MODELS_DIR / "brain_mri_model.h5"
    CHEST_XRAY_MODEL_PATH = MODELS_DIR / "chestXray_model.keras"
    
    # =============================================
    # API CONFIGURATIONS (From Environment Variables)
    # =============================================
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Pinecone settings
    PINECONE_INDEX_NAME = "medical-chatbot"
    
    # =============================================
    # PRODUCTION SETTINGS
    # =============================================
    
    # Environment detection
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    IS_PRODUCTION = ENVIRONMENT == "production"
    
    # Frontend URL for CORS
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # CORS settings
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        FRONTEND_URL,  # Your Next.js app URL
    ]
    
    # Directories
    FIRST_AID_DATA_DIR = BASE_DIR / "data" / "first_aid_images"
    FIRST_AID_DB_PATH = BASE_DIR / "data" / "first_aid_db"
    TEMP_DIR = BASE_DIR / "temp"
    LOGS_DIR = BASE_DIR / "logs"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.MODELS_DIR,
            cls.FIRST_AID_DATA_DIR,
            cls.FIRST_AID_DB_PATH,
            cls.TEMP_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate configuration and return issues"""
        issues = []
        
        # Check model files
        if not cls.BRAIN_MRI_MODEL_PATH.exists():
            issues.append(f"Brain MRI model not found: {cls.BRAIN_MRI_MODEL_PATH}")
        
        if not cls.CHEST_XRAY_MODEL_PATH.exists():
            issues.append(f"Chest X-ray model not found: {cls.CHEST_XRAY_MODEL_PATH}")
        
        # Check API keys (only warn, don't fail)
        if not cls.GOOGLE_API_KEY:
            issues.append("GOOGLE_API_KEY not set - AI fallback will be disabled")
            
        if not cls.PINECONE_API_KEY:
            issues.append("PINECONE_API_KEY not set - Medical chatbot will be disabled")
            
        if not cls.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY not set - Medical chatbot will be disabled")
        
        return issues