from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
import joblib
import io
from PIL import Image
import logging
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
from passlib.context import CryptContext
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Melanoma Detection API",
    description="AI-powered melanoma detection system",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Database setup (use PostgreSQL in production)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./melanoma_api.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    display_name = Column(String)
    bio = Column(Text)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow)

    # Settings
    email_notifications = Column(Boolean, default=True)
    analysis_reminders = Column(Boolean, default=False)
    system_updates = Column(Boolean, default=True)
    data_sharing = Column(Boolean, default=False)
    analytics = Column(Boolean, default=True)

    # Relationship
    image_records = relationship("ImageRecord", back_populates="user")


class ImageRecord(Base):
    __tablename__ = "image_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String)
    notes = Column(Text)
    prediction = Column(String)
    confidence = Column(Float)
    date = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String)

    # Relationship
    user = relationship("User", back_populates="image_records")


# Create tables
Base.metadata.create_all(bind=engine)


# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    display_name: Optional[str]
    bio: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


class PredictionResponse(BaseModel):
    result: str
    confidence: str
    timestamp: str
    record_id: int


class ImageRecordResponse(BaseModel):
    id: int
    name: Optional[str]
    prediction: str
    confidence: float
    date: datetime
    notes: Optional[str]

    class Config:
        from_attributes = True


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    try:
        payload = jwt.decode(
            credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM]
        )
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=401, detail="Invalid authentication credentials"
            )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# Load only the voting ensemble model
def load_voting_model():
    try:
        model = joblib.load("voting_ensemble.pkl")
        logger.info("Loaded voting model successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load voting model: {e}")
        return None


# Load model at startup
voting_model = load_voting_model()


def preprocess_image_for_api(image_data: bytes, img_size=(32, 32)):
    """Preprocess image for model prediction"""
    try:
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert PIL to numpy array
        img_array = np.array(pil_image)

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Resize image
        img_resized = cv2.resize(img_bgr, img_size)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Normalize and flatten
        img_normalized = img_gray.astype(np.float32) / 255.0
        img_flat = img_normalized.reshape(1, -1)

        return img_flat
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")


# API Routes


@app.get("/")
async def root():
    return {"message": "Melanoma Detection API", "version": "1.0.0", "status": "active"}


@app.post("/auth/register", response_model=Token)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.from_orm(db_user),
    }


@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    # Update last login
    db_user.last_login = datetime.utcnow()
    db.commit()

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.from_orm(db_user),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_melanoma(
    image: UploadFile = File(...),
    name: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Predict melanoma from uploaded image"""

    # Validate file type
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload JPEG or PNG image.",
        )

    # Ensure model is loaded
    if voting_model is None:
        raise HTTPException(status_code=503, detail="Voting model not available")

    try:
        # Read image data
        image_data = await image.read()

        # Preprocess image
        processed_image = preprocess_image_for_api(image_data)

        # Make prediction using the voting model
        model = voting_model
        prediction = model.predict(processed_image)

        # Get prediction probability if available
        confidence = 85.0  # Default confidence
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(processed_image)
                confidence = float(np.max(proba) * 100)
            except:
                pass

        # Map prediction to class
        class_map = {0: "Benign", 1: "Malignant"}
        result = class_map.get(int(prediction[0]), "Unknown")

        # Save record to database
        image_record = ImageRecord(
            user_id=current_user.id,
            name=name or f"Image_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            notes=notes or "",
            prediction=result,
            confidence=confidence,
            image_path=f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image.filename}",
        )

        db.add(image_record)
        db.commit()
        db.refresh(image_record)

        return PredictionResponse(
            result=result,
            confidence=f"{confidence:.2f}%",
            timestamp=datetime.now().isoformat(),
            record_id=image_record.id,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction"
        )


@app.get("/dashboard")
async def get_dashboard(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get dashboard statistics"""
    total = db.query(ImageRecord).filter(ImageRecord.user_id == current_user.id).count()
    benign = (
        db.query(ImageRecord)
        .filter(
            ImageRecord.user_id == current_user.id, ImageRecord.prediction == "Benign"
        )
        .count()
    )
    malignant = (
        db.query(ImageRecord)
        .filter(
            ImageRecord.user_id == current_user.id,
            ImageRecord.prediction == "Malignant",
        )
        .count()
    )

    return {"total": total, "benign": benign, "malignant": malignant}


@app.get("/history", response_model=List[ImageRecordResponse])
async def get_history(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get user's prediction history"""
    records = (
        db.query(ImageRecord)
        .filter(ImageRecord.user_id == current_user.id)
        .order_by(ImageRecord.date.desc())
        .all()
    )

    return [ImageRecordResponse.from_orm(record) for record in records]


@app.get("/models")
async def get_available_models():
    """Get list of available models (now fixed to voting only)"""
    available = ["voting"] if voting_model is not None else []
    return {"available_models": available, "total_models": len(available)}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": 1 if voting_model is not None else 0,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
