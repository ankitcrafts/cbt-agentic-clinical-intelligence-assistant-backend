from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import User
from app.schemas.auth import (
    SendOTPRequest,
    SendOTPResponse,
    VerifyOTPRequest,
    AuthResponse,
    UserProfileResponse,
)
from app.services.auth_service import (
    generate_and_send_otp,
    verify_otp_and_authenticate,
    get_current_user,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication & OTP"])


@router.post(
    "/send-otp",
    response_model=SendOTPResponse,
    status_code=status.HTTP_200_OK,
    summary="Send OTP code to Gmail / Email",
    description="Generates a 6-digit OTP code, saves to PostgreSQL, and sends via Gmail SMTP to the user.",
)
def send_otp_endpoint(req: SendOTPRequest, db: Session = Depends(get_db)):
    return generate_and_send_otp(req=req, db=db)


@router.post(
    "/verify-otp",
    response_model=AuthResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify OTP and Sign in / Sign up",
    description="Validates the OTP code, registers or fetches the user in PostgreSQL, and returns a JWT token with their target dashboard URL.",
)
def verify_otp_endpoint(req: VerifyOTPRequest, db: Session = Depends(get_db)):
    return verify_otp_and_authenticate(req=req, db=db)


@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get current authenticated user profile",
    description="Requires Bearer JWT token in Authorization header. Returns user data from PostgreSQL.",
)
def get_me_endpoint(current_user: User = Depends(get_current_user)):
    return UserProfileResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        department=current_user.department,
        specialization=current_user.specialization,
        designation=current_user.designation,
        phone_number=current_user.phone_number,
        avatar_initials=current_user.get_initials(),
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )


@router.get(
    "/users",
    response_model=List[UserProfileResponse],
    summary="List registered users from PostgreSQL",
    description="Optional filter by role (ADMIN, DOCTOR, PATIENT).",
)
def list_users_endpoint(
    role: Optional[str] = Query(None, description="Filter by role: ADMIN, DOCTOR, PATIENT"),
    db: Session = Depends(get_db),
):
    query = db.query(User)
    if role:
        query = query.filter(User.role == role.upper())
    users = query.all()
    return [
        UserProfileResponse(
            id=u.id,
            email=u.email,
            full_name=u.full_name,
            role=u.role,
            department=u.department,
            specialization=u.specialization,
            designation=u.designation,
            phone_number=u.phone_number,
            avatar_initials=u.get_initials(),
            is_active=u.is_active,
            created_at=u.created_at,
        )
        for u in users
    ]
