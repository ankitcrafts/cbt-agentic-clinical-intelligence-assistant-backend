from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    ADMIN = "ADMIN"
    DOCTOR = "DOCTOR"
    PATIENT = "PATIENT"


class SendOTPRequest(BaseModel):
    email: str = Field(..., description="User Gmail/Email address")
    role: UserRole = Field(default=UserRole.DOCTOR, description="Target role (ADMIN, DOCTOR, PATIENT)")
    purpose: str = Field(default="SIGNIN", description="SIGNIN or SIGNUP")
    full_name: Optional[str] = Field(default=None, description="Optional full name for signup")


class SendOTPResponse(BaseModel):
    success: bool
    message: str
    email: str
    role: str
    expires_in_seconds: int
    dev_otp: Optional[str] = None


class VerifyOTPRequest(BaseModel):
    email: str = Field(..., description="User Gmail/Email address")
    otp_code: str = Field(..., min_length=4, max_length=10, description="One-Time Password code")
    role: Optional[UserRole] = Field(default=None, description="Selected role")
    full_name: Optional[str] = Field(default=None, description="Optional name for new account")
    department: Optional[str] = Field(default=None, description="Department for medical staff")
    specialization: Optional[str] = Field(default=None, description="Specialization")


class UserProfileResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    role: str
    department: Optional[str] = None
    specialization: Optional[str] = None
    designation: Optional[str] = None
    phone_number: Optional[str] = None
    avatar_initials: Optional[str] = None
    is_active: bool = True
    is_profile_completed: bool = False
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = Field(None, description="Full Name of the user")
    department: Optional[str] = Field(None, description="Department (e.g. Cardiology)")
    specialization: Optional[str] = Field(None, description="Specialization or Role Title")
    designation: Optional[str] = Field(None, description="Designation")
    phone_number: Optional[str] = Field(None, description="Contact Phone Number")


class AuthResponse(BaseModel):
    success: bool
    message: str
    access_token: str
    token_type: str = "bearer"
    user: UserProfileResponse
    target_dashboard_url: str
    is_new_user: bool = False
