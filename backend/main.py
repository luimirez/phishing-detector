import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware



sys.path.append(os.path.dirname(__file__))
from app.detector import is_phishing

# WE must create the instance to access to the FastAPI
app = FastAPI(
    title="Phishing Awareness Detector API",
    description="API for detecting Phishing messages using a trained ML model",
    version="1.0.0"
)

#We are going to provide allowances to the browser extension CORS (Essential!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #in the production area, we must restrict the specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#We are going to define the request Schema in the system
class EmailContent(BaseModel):
    subject: str
    body: str
    
    
#Creating the Root Endpoint
@app.get("/")
def read_root():
    return {"message": "Phishing awareness detector is running."}

#Detecting the  Detection endpoint
@app.post("/scan")
def scan_email(email: EmailContent):
    full_text = f"{email.subject} {email.body}"
    result = is_phishing(full_text)
    return {
        "phishing": result,
        "message": "Phishing detected!" if result else "Message is safe"
    }
    
