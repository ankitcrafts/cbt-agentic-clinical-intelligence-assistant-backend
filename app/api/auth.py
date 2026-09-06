from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import User
from app.schemas.auth import (
    SendOTPRequest,
    SendOTPResponse,
    VerifyOTPRequest,
    UpdateProfileRequest,
    AuthResponse,
    UserProfileResponse,
)
from app.services.auth_service import (
    generate_and_send_otp,
    verify_otp_and_authenticate,
    update_user_profile,
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
        id=str(current_user.id),
        email=str(current_user.email),
        full_name=str(current_user.full_name) if current_user.full_name else None,
        role=str(current_user.role),
        department=str(current_user.department) if current_user.department else None,
        specialization=str(current_user.specialization) if current_user.specialization else None,
        designation=str(current_user.designation) if current_user.designation else None,
        phone_number=str(current_user.phone_number) if current_user.phone_number else None,
        avatar_initials=current_user.get_initials(),
        is_active=bool(current_user.is_active),
        is_profile_completed=bool(current_user.full_name and current_user.full_name.strip()),
        created_at=current_user.created_at if isinstance(current_user.created_at, datetime) else None,
    )


@router.put(
    "/profile",
    response_model=UserProfileResponse,
    summary="Update current authenticated user profile",
    description="Updates full name, department, specialization, designation, or phone number in PostgreSQL.",
)
def update_profile_endpoint(
    req: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return update_user_profile(current_user=current_user, req=req, db=db)


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
            id=str(u.id),
            email=str(u.email),
            full_name=str(u.full_name) if u.full_name else None,
            role=str(u.role),
            department=str(u.department) if u.department else None,
            specialization=str(u.specialization) if u.specialization else None,
            designation=str(u.designation) if u.designation else None,
            phone_number=str(u.phone_number) if u.phone_number else None,
            avatar_initials=u.get_initials(),
            is_active=bool(u.is_active),
            is_profile_completed=bool(u.full_name and u.full_name.strip()),
            created_at=u.created_at if isinstance(u.created_at, datetime) else None,
        )
        for u in users
    ]
