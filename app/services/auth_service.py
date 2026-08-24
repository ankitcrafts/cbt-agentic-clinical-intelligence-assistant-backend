import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.config import JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, OTP_EXPIRE_MINUTES, SMTP_EMAIL, SMTP_PASSWORD
from app.core.database import get_db
from app.models.models import User, OTPVerification
from app.schemas.auth import SendOTPRequest, SendOTPResponse, VerifyOTPRequest, AuthResponse, UserProfileResponse, UserRole
from app.services.email_service import send_otp_email

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


def generate_and_send_otp(req: SendOTPRequest, db: Session) -> SendOTPResponse:
    """Generate 6-digit OTP, store in PostgreSQL, and send via Gmail SMTP."""
    email_clean = req.email.strip().lower()
    role_str = req.role.value if isinstance(req.role, UserRole) else str(req.role).upper()
    
    # Strict RBAC: Check if account already exists with a different role
    existing_user = db.query(User).filter(User.email == email_clean).first()
    if existing_user:
        if existing_user.role != role_str:
            role_labels = {
                "ADMIN": "Administrator",
                "DOCTOR": "Doctor / Clinician",
                "PATIENT": "Patient",
            }
            actual_role_label = role_labels.get(existing_user.role, existing_user.role)
            target_role_label = role_labels.get(role_str, role_str)
            logger.warning(
                f"RBAC Violation: {email_clean} ({existing_user.role}) attempted login as {role_str}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access Denied: Account '{email_clean}' is registered as an {actual_role_label}. You cannot sign in to the {target_role_label} portal with this account.",
            )

    # Invalidate previous unused OTPs for this email to prevent collisions
    db.query(OTPVerification).filter(
        OTPVerification.email == email_clean,
        OTPVerification.is_used == False,
    ).update({"is_used": True})
    db.commit()

    # Generate cryptographically secure 6-digit OTP
    otp_code = str(secrets.randbelow(900000) + 100000)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRE_MINUTES)

    # Save OTP to database
    otp_record = OTPVerification(
        email=email_clean,
        otp_code=otp_code,
        role=role_str,
        purpose=req.purpose.upper(),
        expires_at=expires_at,
        is_used=False,
    )
    db.add(otp_record)
    db.commit()
    db.refresh(otp_record)

    # Dispatch email
    send_otp_email(to_email=email_clean, otp_code=otp_code, role=role_str, purpose=req.purpose)

    # Include dev_otp if SMTP is unconfigured for local developer ease
    dev_otp = otp_code if (not SMTP_EMAIL or not SMTP_PASSWORD) else None

    return SendOTPResponse(
        success=True,
        message=f"A 6-digit verification code has been sent to {email_clean}",
        email=email_clean,
        role=role_str,
        expires_in_seconds=OTP_EXPIRE_MINUTES * 60,
        dev_otp=dev_otp,
    )


def verify_otp_and_authenticate(req: VerifyOTPRequest, db: Session) -> AuthResponse:
    """Verify submitted OTP code, provision/fetch User in PostgreSQL, and issue JWT."""
    email_clean = req.email.strip().lower()
    submitted_otp = req.otp_code.strip()

    # Determine target role
    target_role = (
        (req.role.value if isinstance(req.role, UserRole) else str(req.role).upper())
        if req.role
        else None
    )

    # Find the matching OTP record
    otp_record = db.query(OTPVerification).filter(
        OTPVerification.email == email_clean,
        OTPVerification.otp_code == submitted_otp,
        OTPVerification.is_used == False,
    ).order_by(OTPVerification.created_at.desc()).first()

    if not otp_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP verification code. Please request a new code.",
        )

    if otp_record.is_expired():
        otp_record.is_used = True
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This verification code has expired. Please request a new OTP.",
        )

    # Enforce role match with OTP record
    if target_role and otp_record.role != target_role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Security Alert: OTP was requested for {otp_record.role} portal but verification was attempted for {target_role}.",
        )

    active_role = target_role or otp_record.role

    # Mark OTP as used
    otp_record.is_used = True
    db.commit()

    # Find or create user in PostgreSQL
    user = db.query(User).filter(User.email == email_clean).first()
    
    if not user:
        # Create new user account permanently locked to active_role
        default_name = req.full_name or (
            "Hospital Administrator" if active_role == "ADMIN"
            else ("Dr. Yuki Tanaka" if active_role == "DOCTOR" else "Patient User")
        )
        
        user = User(
            email=email_clean,
            full_name=default_name,
            role=active_role,
            department=req.department or ("Cardiology" if active_role == "DOCTOR" else None),
            specialization=req.specialization or ("Senior Cardiologist" if active_role == "DOCTOR" else None),
            designation="System Admin" if active_role == "ADMIN" else ("Attending Physician" if active_role == "DOCTOR" else "Patient"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Created new {active_role} user: {email_clean}")
    else:
        # Strict RBAC Check on existing user: DO NOT allow cross-role hijacking
        if user.role != active_role:
            role_labels = {
                "ADMIN": "Administrator",
                "DOCTOR": "Doctor / Clinician",
                "PATIENT": "Patient",
            }
            actual_role_label = role_labels.get(user.role, user.role)
            target_role_label = role_labels.get(active_role, active_role)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access Denied: Account '{email_clean}' is registered as an {actual_role_label}. You cannot sign in to the {target_role_label} portal.",
            )

        # Update non-security metadata if provided
        if req.full_name:
            user.full_name = req.full_name
        if req.department:
            user.department = req.department
        if req.specialization:
            user.specialization = req.specialization
        db.commit()
        db.refresh(user)

    # Generate JWT Token
    access_token = create_access_token(
        data={
            "sub": user.id,
            "email": user.email,
            "role": user.role,
            "name": user.full_name,
        }
    )

    # Target dashboard routing URL
    dashboard_urls = {
        "ADMIN": "/admin",
        "DOCTOR": "/doctor",
        "PATIENT": "/patient",
    }
    target_url = dashboard_urls.get(user.role, "/doctor")

    user_profile = UserProfileResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        department=user.department,
        specialization=user.specialization,
        designation=user.designation,
        phone_number=user.phone_number,
        avatar_initials=user.get_initials(),
        is_active=user.is_active,
        created_at=user.created_at,
    )

    return AuthResponse(
        success=True,
        message=f"Welcome, {user.full_name or user.email}! Authentication successful.",
        access_token=access_token,
        token_type="bearer",
        user=user_profile,
        target_dashboard_url=target_url,
    )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Dependency to retrieve currently authenticated user from Bearer JWT token."""
    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is missing. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = auth.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload.",
            )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired. Please sign in again.",
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authenticated user record not found.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    return user
