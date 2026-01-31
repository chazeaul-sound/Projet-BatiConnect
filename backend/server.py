from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import bcrypt
import jwt
import httpx

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'baticonnect-secret-key-change-in-production')
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 168  # 7 days

# Create the main app
app = FastAPI(title="BâtiConnect API", description="Plateforme de mise en relation travaux")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

security = HTTPBearer(auto_error=False)

# ============== MODELS ==============

class UserType:
    PARTICULIER = "particulier"
    PROFESSIONNEL = "professionnel"
    ENTREPRISE = "entreprise"

class UserBase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    email: EmailStr
    name: str
    user_type: str  # particulier, professionnel, entreprise

class UserCreate(UserBase):
    password: str
    # Pro fields
    siret: Optional[str] = None
    profession: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    description: Optional[str] = None
    years_experience: Optional[int] = None
    # Enterprise fields
    company_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    email: str
    name: str
    user_type: str
    picture: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    created_at: datetime
    # Pro specific
    siret: Optional[str] = None
    profession: Optional[str] = None
    description: Optional[str] = None
    years_experience: Optional[int] = None
    is_verified: bool = False
    verification_status: str = "pending"  # pending, approved, rejected
    # Enterprise specific
    company_name: Optional[str] = None
    # Stats
    rating_average: float = 0.0
    rating_count: int = 0
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class Professional(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    email: str
    name: str
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    siret: str
    profession: str
    description: Optional[str] = None
    years_experience: Optional[int] = None
    is_verified: bool = False
    verification_status: str = "pending"
    rating_average: float = 0.0
    rating_count: int = 0
    picture: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    created_at: datetime

class ReviewCreate(BaseModel):
    professional_id: str
    rating: int = Field(ge=1, le=5)
    comment: str
    project_type: Optional[str] = None

class Review(BaseModel):
    model_config = ConfigDict(extra="ignore")
    review_id: str
    professional_id: str
    reviewer_id: str
    reviewer_name: str
    rating: int
    comment: str
    project_type: Optional[str] = None
    created_at: datetime

class MessageCreate(BaseModel):
    receiver_id: str
    content: str

class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    message_id: str
    conversation_id: str
    sender_id: str
    sender_name: str
    receiver_id: str
    content: str
    is_read: bool = False
    created_at: datetime

class Conversation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    conversation_id: str
    participants: List[str]
    participant_names: dict
    last_message: Optional[str] = None
    last_message_at: Optional[datetime] = None
    unread_count: int = 0
    created_at: datetime

class ContactFormCreate(BaseModel):
    professional_id: str
    name: str
    email: EmailStr
    phone: Optional[str] = None
    message: str
    project_type: Optional[str] = None

class ContactForm(BaseModel):
    model_config = ConfigDict(extra="ignore")
    contact_id: str
    professional_id: str
    name: str
    email: str
    phone: Optional[str] = None
    message: str
    project_type: Optional[str] = None
    status: str = "pending"  # pending, contacted, completed
    created_at: datetime

class AvailabilitySlot(BaseModel):
    day: str  # monday, tuesday, etc.
    start_time: str
    end_time: str
    is_available: bool = True

class AvailabilityUpdate(BaseModel):
    slots: List[AvailabilitySlot]

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User

# ============== AUTH HELPERS ==============

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_jwt_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(request: Request) -> User:
    # Check session_token cookie first (for Google OAuth)
    session_token = request.cookies.get("session_token")
    if session_token:
        session = await db.user_sessions.find_one(
            {"session_token": session_token},
            {"_id": 0}
        )
        if session:
            expires_at = session.get("expires_at")
            if isinstance(expires_at, str):
                expires_at = datetime.fromisoformat(expires_at)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at > datetime.now(timezone.utc):
                user_doc = await db.users.find_one(
                    {"user_id": session["user_id"]},
                    {"_id": 0}
                )
                if user_doc:
                    if isinstance(user_doc.get('created_at'), str):
                        user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])
                    return User(**user_doc)
    
    # Check Authorization header (for JWT)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = decode_jwt_token(token)
            user_doc = await db.users.find_one(
                {"user_id": payload["user_id"]},
                {"_id": 0}
            )
            if user_doc:
                if isinstance(user_doc.get('created_at'), str):
                    user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])
                return User(**user_doc)
        except Exception:
            pass
    
    raise HTTPException(status_code=401, detail="Not authenticated")

async def get_optional_user(request: Request) -> Optional[User]:
    try:
        return await get_current_user(request)
    except HTTPException:
        return None

# ============== AUTH ENDPOINTS ==============

@api_router.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserCreate):
    # Check if email exists
    existing = await db.users.find_one({"email": user_data.email}, {"_id": 0})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Validate pro fields
    if user_data.user_type == UserType.PROFESSIONNEL:
        if not user_data.siret:
            raise HTTPException(status_code=400, detail="SIRET required for professionals")
        if not user_data.profession:
            raise HTTPException(status_code=400, detail="Profession required for professionals")
    
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    
    user_doc = {
        "user_id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "password_hash": hash_password(user_data.password),
        "user_type": user_data.user_type,
        "phone": user_data.phone,
        "address": user_data.address,
        "city": user_data.city,
        "postal_code": user_data.postal_code,
        "created_at": now.isoformat(),
        "is_verified": False,
        "verification_status": "pending" if user_data.user_type == UserType.PROFESSIONNEL else "approved",
        "rating_average": 0.0,
        "rating_count": 0,
    }
    
    if user_data.user_type == UserType.PROFESSIONNEL:
        user_doc.update({
            "siret": user_data.siret,
            "profession": user_data.profession,
            "description": user_data.description,
            "years_experience": user_data.years_experience,
        })
    
    if user_data.user_type == UserType.ENTREPRISE:
        user_doc["company_name"] = user_data.company_name
    
    await db.users.insert_one(user_doc)
    
    # Remove password_hash before returning
    del user_doc["password_hash"]
    user_doc["created_at"] = now
    
    token = create_jwt_token(user_id)
    return TokenResponse(access_token=token, user=User(**user_doc))

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    user_doc = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not verify_password(credentials.password, user_doc.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if isinstance(user_doc.get('created_at'), str):
        user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])
    
    # Remove password_hash
    user_doc.pop("password_hash", None)
    
    token = create_jwt_token(user_doc["user_id"])
    return TokenResponse(access_token=token, user=User(**user_doc))

@api_router.post("/auth/session")
async def create_session_from_google(request: Request, response: Response):
    """Exchange Google session_id for local session"""
    body = await request.json()
    session_id = body.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    
    # Fetch user data from Emergent Auth
    async with httpx.AsyncClient() as client:
        auth_response = await client.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": session_id}
        )
        
        if auth_response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid session")
        
        auth_data = auth_response.json()
    
    email = auth_data.get("email")
    name = auth_data.get("name")
    picture = auth_data.get("picture")
    session_token = auth_data.get("session_token")
    
    # Check if user exists
    existing_user = await db.users.find_one({"email": email}, {"_id": 0})
    
    if existing_user:
        user_id = existing_user["user_id"]
        # Update picture if changed
        if picture and picture != existing_user.get("picture"):
            await db.users.update_one(
                {"user_id": user_id},
                {"$set": {"picture": picture}}
            )
    else:
        # Create new user (default to particulier)
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        
        user_doc = {
            "user_id": user_id,
            "email": email,
            "name": name,
            "picture": picture,
            "user_type": UserType.PARTICULIER,
            "created_at": now.isoformat(),
            "is_verified": False,
            "verification_status": "approved",
            "rating_average": 0.0,
            "rating_count": 0,
        }
        await db.users.insert_one(user_doc)
    
    # Store session
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    await db.user_sessions.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "session_token": session_token,
                "expires_at": expires_at.isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        },
        upsert=True
    )
    
    # Set cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=7 * 24 * 60 * 60
    )
    
    # Return user data
    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    if isinstance(user_doc.get('created_at'), str):
        user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])
    user_doc.pop("password_hash", None)
    
    return User(**user_doc)

@api_router.get("/auth/me", response_model=User)
async def get_me(user: User = Depends(get_current_user)):
    return user

@api_router.post("/auth/logout")
async def logout(request: Request, response: Response):
    session_token = request.cookies.get("session_token")
    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})
    
    response.delete_cookie(key="session_token", path="/")
    return {"message": "Logged out successfully"}

# ============== USER ENDPOINTS ==============

@api_router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0, "password_hash": 0})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    if isinstance(user_doc.get('created_at'), str):
        user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])
    
    return User(**user_doc)

@api_router.put("/users/me", response_model=User)
async def update_user(request: Request, user: User = Depends(get_current_user)):
    body = await request.json()
    
    allowed_fields = ["name", "phone", "address", "city", "postal_code", "description", "years_experience", "company_name"]
    update_data = {k: v for k, v in body.items() if k in allowed_fields}
    
    if update_data:
        await db.users.update_one(
            {"user_id": user.user_id},
            {"$set": update_data}
        )
    
    user_doc = await db.users.find_one({"user_id": user.user_id}, {"_id": 0, "password_hash": 0})
    if isinstance(user_doc.get('created_at'), str):
        user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])
    
    return User(**user_doc)

# ============== PROFESSIONALS ENDPOINTS ==============

@api_router.get("/professionals", response_model=List[Professional])
async def get_professionals(
    profession: Optional[str] = None,
    city: Optional[str] = None,
    postal_code: Optional[str] = None,
    verified_only: bool = False,
    min_rating: Optional[float] = None,
    search: Optional[str] = None,
    limit: int = 50,
    skip: int = 0
):
    query = {"user_type": UserType.PROFESSIONNEL}
    
    if verified_only:
        query["is_verified"] = True
        query["verification_status"] = "approved"
    
    if profession:
        query["profession"] = {"$regex": profession, "$options": "i"}
    
    if city:
        query["city"] = {"$regex": city, "$options": "i"}
    
    if postal_code:
        query["postal_code"] = {"$regex": f"^{postal_code[:2]}", "$options": "i"}
    
    if min_rating:
        query["rating_average"] = {"$gte": min_rating}
    
    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"profession": {"$regex": search, "$options": "i"}},
            {"description": {"$regex": search, "$options": "i"}}
        ]
    
    professionals = await db.users.find(
        query,
        {"_id": 0, "password_hash": 0}
    ).skip(skip).limit(limit).to_list(limit)
    
    result = []
    for pro in professionals:
        if isinstance(pro.get('created_at'), str):
            pro['created_at'] = datetime.fromisoformat(pro['created_at'])
        result.append(Professional(**pro))
    
    return result

@api_router.get("/professionals/{user_id}", response_model=Professional)
async def get_professional(user_id: str):
    pro = await db.users.find_one(
        {"user_id": user_id, "user_type": UserType.PROFESSIONNEL},
        {"_id": 0, "password_hash": 0}
    )
    if not pro:
        raise HTTPException(status_code=404, detail="Professional not found")
    
    if isinstance(pro.get('created_at'), str):
        pro['created_at'] = datetime.fromisoformat(pro['created_at'])
    
    return Professional(**pro)

@api_router.get("/professions", response_model=List[str])
async def get_professions():
    """Get list of unique professions"""
    professions = await db.users.distinct("profession", {"user_type": UserType.PROFESSIONNEL})
    return [p for p in professions if p]

# ============== REVIEWS ENDPOINTS ==============

@api_router.post("/reviews", response_model=Review)
async def create_review(review_data: ReviewCreate, user: User = Depends(get_current_user)):
    # Check if professional exists
    pro = await db.users.find_one(
        {"user_id": review_data.professional_id, "user_type": UserType.PROFESSIONNEL},
        {"_id": 0}
    )
    if not pro:
        raise HTTPException(status_code=404, detail="Professional not found")
    
    # Check if user already reviewed this pro
    existing = await db.reviews.find_one({
        "professional_id": review_data.professional_id,
        "reviewer_id": user.user_id
    })
    if existing:
        raise HTTPException(status_code=400, detail="You already reviewed this professional")
    
    review_id = f"review_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    
    review_doc = {
        "review_id": review_id,
        "professional_id": review_data.professional_id,
        "reviewer_id": user.user_id,
        "reviewer_name": user.name,
        "rating": review_data.rating,
        "comment": review_data.comment,
        "project_type": review_data.project_type,
        "created_at": now.isoformat()
    }
    
    await db.reviews.insert_one(review_doc)
    
    # Update professional rating
    all_reviews = await db.reviews.find(
        {"professional_id": review_data.professional_id},
        {"_id": 0, "rating": 1}
    ).to_list(1000)
    
    avg_rating = sum(r["rating"] for r in all_reviews) / len(all_reviews)
    
    await db.users.update_one(
        {"user_id": review_data.professional_id},
        {
            "$set": {
                "rating_average": round(avg_rating, 2),
                "rating_count": len(all_reviews)
            }
        }
    )
    
    review_doc["created_at"] = now
    return Review(**review_doc)

@api_router.get("/reviews/{professional_id}", response_model=List[Review])
async def get_reviews(professional_id: str, limit: int = 50, skip: int = 0):
    reviews = await db.reviews.find(
        {"professional_id": professional_id},
        {"_id": 0}
    ).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    
    for review in reviews:
        if isinstance(review.get('created_at'), str):
            review['created_at'] = datetime.fromisoformat(review['created_at'])
    
    return [Review(**r) for r in reviews]

# ============== MESSAGES ENDPOINTS ==============

@api_router.post("/messages", response_model=Message)
async def send_message(msg_data: MessageCreate, user: User = Depends(get_current_user)):
    # Check receiver exists
    receiver = await db.users.find_one({"user_id": msg_data.receiver_id}, {"_id": 0})
    if not receiver:
        raise HTTPException(status_code=404, detail="Receiver not found")
    
    # Get or create conversation
    participants = sorted([user.user_id, msg_data.receiver_id])
    conv = await db.conversations.find_one({"participants": participants}, {"_id": 0})
    
    if not conv:
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        conv = {
            "conversation_id": conv_id,
            "participants": participants,
            "participant_names": {
                user.user_id: user.name,
                msg_data.receiver_id: receiver["name"]
            },
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.conversations.insert_one(conv)
    else:
        conv_id = conv["conversation_id"]
    
    # Create message
    message_id = f"msg_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    
    message_doc = {
        "message_id": message_id,
        "conversation_id": conv_id,
        "sender_id": user.user_id,
        "sender_name": user.name,
        "receiver_id": msg_data.receiver_id,
        "content": msg_data.content,
        "is_read": False,
        "created_at": now.isoformat()
    }
    
    await db.messages.insert_one(message_doc)
    
    # Update conversation
    await db.conversations.update_one(
        {"conversation_id": conv_id},
        {
            "$set": {
                "last_message": msg_data.content[:100],
                "last_message_at": now.isoformat()
            },
            "$inc": {"unread_count": 1}
        }
    )
    
    message_doc["created_at"] = now
    return Message(**message_doc)

@api_router.get("/conversations", response_model=List[Conversation])
async def get_conversations(user: User = Depends(get_current_user)):
    convs = await db.conversations.find(
        {"participants": user.user_id},
        {"_id": 0}
    ).sort("last_message_at", -1).to_list(100)
    
    result = []
    for conv in convs:
        if isinstance(conv.get('created_at'), str):
            conv['created_at'] = datetime.fromisoformat(conv['created_at'])
        if isinstance(conv.get('last_message_at'), str):
            conv['last_message_at'] = datetime.fromisoformat(conv['last_message_at'])
        result.append(Conversation(**conv))
    
    return result

@api_router.get("/conversations/{conversation_id}/messages", response_model=List[Message])
async def get_conversation_messages(
    conversation_id: str,
    user: User = Depends(get_current_user),
    limit: int = 100,
    skip: int = 0
):
    # Check user is participant
    conv = await db.conversations.find_one(
        {"conversation_id": conversation_id, "participants": user.user_id},
        {"_id": 0}
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = await db.messages.find(
        {"conversation_id": conversation_id},
        {"_id": 0}
    ).sort("created_at", 1).skip(skip).limit(limit).to_list(limit)
    
    # Mark as read
    await db.messages.update_many(
        {"conversation_id": conversation_id, "receiver_id": user.user_id, "is_read": False},
        {"$set": {"is_read": True}}
    )
    
    result = []
    for msg in messages:
        if isinstance(msg.get('created_at'), str):
            msg['created_at'] = datetime.fromisoformat(msg['created_at'])
        result.append(Message(**msg))
    
    return result

# ============== CONTACT FORM ENDPOINTS ==============

@api_router.post("/contact", response_model=ContactForm)
async def submit_contact_form(contact_data: ContactFormCreate):
    # Check professional exists
    pro = await db.users.find_one(
        {"user_id": contact_data.professional_id, "user_type": UserType.PROFESSIONNEL},
        {"_id": 0}
    )
    if not pro:
        raise HTTPException(status_code=404, detail="Professional not found")
    
    contact_id = f"contact_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    
    contact_doc = {
        "contact_id": contact_id,
        "professional_id": contact_data.professional_id,
        "name": contact_data.name,
        "email": contact_data.email,
        "phone": contact_data.phone,
        "message": contact_data.message,
        "project_type": contact_data.project_type,
        "status": "pending",
        "created_at": now.isoformat()
    }
    
    await db.contact_forms.insert_one(contact_doc)
    contact_doc["created_at"] = now
    
    return ContactForm(**contact_doc)

@api_router.get("/contact/received", response_model=List[ContactForm])
async def get_received_contacts(user: User = Depends(get_current_user)):
    if user.user_type != UserType.PROFESSIONNEL:
        raise HTTPException(status_code=403, detail="Only professionals can access this")
    
    contacts = await db.contact_forms.find(
        {"professional_id": user.user_id},
        {"_id": 0}
    ).sort("created_at", -1).to_list(100)
    
    for contact in contacts:
        if isinstance(contact.get('created_at'), str):
            contact['created_at'] = datetime.fromisoformat(contact['created_at'])
    
    return [ContactForm(**c) for c in contacts]

# ============== AVAILABILITY ENDPOINTS ==============

@api_router.put("/availability")
async def update_availability(availability: AvailabilityUpdate, user: User = Depends(get_current_user)):
    if user.user_type != UserType.PROFESSIONNEL:
        raise HTTPException(status_code=403, detail="Only professionals can update availability")
    
    slots = [slot.model_dump() for slot in availability.slots]
    
    await db.users.update_one(
        {"user_id": user.user_id},
        {"$set": {"availability": slots}}
    )
    
    return {"message": "Availability updated"}

@api_router.get("/availability/{user_id}")
async def get_availability(user_id: str):
    user_doc = await db.users.find_one(
        {"user_id": user_id, "user_type": UserType.PROFESSIONNEL},
        {"_id": 0, "availability": 1}
    )
    if not user_doc:
        raise HTTPException(status_code=404, detail="Professional not found")
    
    return user_doc.get("availability", [])

# ============== ADMIN ENDPOINTS ==============

@api_router.get("/admin/pending-professionals", response_model=List[Professional])
async def get_pending_professionals(user: User = Depends(get_current_user)):
    # Simple admin check - in production, use proper role-based auth
    if user.email not in ["admin@baticonnect.fr"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    pros = await db.users.find(
        {"user_type": UserType.PROFESSIONNEL, "verification_status": "pending"},
        {"_id": 0, "password_hash": 0}
    ).to_list(100)
    
    result = []
    for pro in pros:
        if isinstance(pro.get('created_at'), str):
            pro['created_at'] = datetime.fromisoformat(pro['created_at'])
        result.append(Professional(**pro))
    
    return result

@api_router.post("/admin/verify-professional/{user_id}")
async def verify_professional(user_id: str, request: Request, user: User = Depends(get_current_user)):
    if user.email not in ["admin@baticonnect.fr"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    body = await request.json()
    status = body.get("status", "approved")  # approved or rejected
    
    if status not in ["approved", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    await db.users.update_one(
        {"user_id": user_id, "user_type": UserType.PROFESSIONNEL},
        {
            "$set": {
                "verification_status": status,
                "is_verified": status == "approved"
            }
        }
    )
    
    return {"message": f"Professional {status}"}

# ============== FAVORITES ENDPOINTS ==============

@api_router.post("/favorites/{professional_id}")
async def add_favorite(professional_id: str, user: User = Depends(get_current_user)):
    # Check professional exists
    pro = await db.users.find_one(
        {"user_id": professional_id, "user_type": UserType.PROFESSIONNEL},
        {"_id": 0}
    )
    if not pro:
        raise HTTPException(status_code=404, detail="Professional not found")
    
    await db.favorites.update_one(
        {"user_id": user.user_id, "professional_id": professional_id},
        {
            "$set": {
                "user_id": user.user_id,
                "professional_id": professional_id,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        },
        upsert=True
    )
    
    return {"message": "Added to favorites"}

@api_router.delete("/favorites/{professional_id}")
async def remove_favorite(professional_id: str, user: User = Depends(get_current_user)):
    await db.favorites.delete_one(
        {"user_id": user.user_id, "professional_id": professional_id}
    )
    return {"message": "Removed from favorites"}

@api_router.get("/favorites", response_model=List[Professional])
async def get_favorites(user: User = Depends(get_current_user)):
    favorites = await db.favorites.find(
        {"user_id": user.user_id},
        {"_id": 0}
    ).to_list(100)
    
    pro_ids = [f["professional_id"] for f in favorites]
    
    if not pro_ids:
        return []
    
    pros = await db.users.find(
        {"user_id": {"$in": pro_ids}, "user_type": UserType.PROFESSIONNEL},
        {"_id": 0, "password_hash": 0}
    ).to_list(100)
    
    result = []
    for pro in pros:
        if isinstance(pro.get('created_at'), str):
            pro['created_at'] = datetime.fromisoformat(pro['created_at'])
        result.append(Professional(**pro))
    
    return result

# ============== STATS ==============

@api_router.get("/stats")
async def get_stats():
    total_pros = await db.users.count_documents({"user_type": UserType.PROFESSIONNEL})
    verified_pros = await db.users.count_documents({
        "user_type": UserType.PROFESSIONNEL,
        "is_verified": True
    })
    total_reviews = await db.reviews.count_documents({})
    total_users = await db.users.count_documents({})
    
    return {
        "total_professionals": total_pros,
        "verified_professionals": verified_pros,
        "total_reviews": total_reviews,
        "total_users": total_users
    }

# ============== ROOT ENDPOINT ==============

@api_router.get("/")
async def root():
    return {"message": "BâtiConnect API", "version": "1.0.0"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
