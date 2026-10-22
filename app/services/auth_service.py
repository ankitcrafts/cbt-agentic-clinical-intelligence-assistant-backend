import uuid
import secrets
import logging
import re
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.config import JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, OTP_EXPIRE_MINUTES, SMTP_EMAIL, SMTP_PASSWORD
from app.core.database import get_db
from app.models.models import User, OTPVerification, PatientModel
from app.schemas.auth import SendOTPRequest, SendOTPResponse, VerifyOTPRequest, UpdateProfileRequest, AuthResponse, UserProfileResponse, UserRole
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
            user_role_str = str(existing_user.role)
            actual_role_label = role_labels.get(user_role_str, user_role_str)
            target_role_label = role_labels.get(role_str, role_str)
            logger.warning(
                f"RBAC Violation: {email_clean} ({user_role_str}) attempted login as {role_str}"
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

    # Dispatch email via Gmail SMTP (strictly deliver OTP to recipient's email)
    email_sent = send_otp_email(to_email=email_clean, otp_code=otp_code, role=role_str, purpose=req.purpose)

    if email_sent:
        msg_text = f"A 6-digit verification code has been sent to {email_clean} (Check your inbox & Spam/Promotions folder)"
        dev_otp_val = None
    else:
        msg_text = f"Email delivery unavailable (SMTP credentials rejected). Verification Code: {otp_code}"
        dev_otp_val = otp_code

    return SendOTPResponse(
        success=True,
        message=msg_text,
        email=email_clean,
        role=role_str,
        expires_in_seconds=OTP_EXPIRE_MINUTES * 60,
        dev_otp=dev_otp_val,
    )


def generate_unique_clinical_id(role: str, db: Session) -> str:
    """Generate a clean, memorable clinical identifier (DOC-XXXX, PAT-XXXX, ADM-XXXX)."""
    prefix = "DOC" if role == "DOCTOR" else ("PAT" if role == "PATIENT" else "ADM")
    for _ in range(20):
        random_num = secrets.randbelow(9000) + 1000  # 4-digit number (1000 - 9999)
        candidate_id = f"{prefix}-{random_num}"
        existing = db.query(User).filter(User.id == candidate_id).first()
        if not existing:
            return candidate_id
    return f"{prefix}-{uuid.uuid4().hex[:6].upper()}"


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

    active_role = str(target_role or otp_record.role)

    # Mark OTP as used
    otp_record.is_used = True
    db.commit()

    # Find or create user in PostgreSQL
    user = db.query(User).filter(User.email == email_clean).first()
    is_new_user = False

    if not user:
        is_new_user = True
        user_id = generate_unique_clinical_id(active_role, db)
        
        # Determine full_name for new user
        if req.full_name and req.full_name.strip():
            full_name = req.full_name.strip()
        elif active_role == "PATIENT":
            raw_username = email_clean.split('@')[0]
            clean_name = re.sub(r'[\d_.]+', ' ', raw_username).strip().title()
            full_name = clean_name if len(clean_name) >= 2 else "New Patient"
        else:
            full_name = None

        user = User(
            id=user_id,
            email=email_clean,
            full_name=full_name,
            role=active_role,
            department=req.department or ("Cardiology" if active_role == "DOCTOR" else None),
            specialization=req.specialization or ("Cardiologist" if active_role == "DOCTOR" else None),
            designation="System Admin" if active_role == "ADMIN" else ("Attending Physician" if active_role == "DOCTOR" else "Patient"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Created new {active_role} user with ID {user.id}: {email_clean}")

        # If PATIENT, also ensure patient record exists in PostgreSQL patients table
        if active_role == "PATIENT":
            existing_pat = db.query(PatientModel).filter(
                (PatientModel.email == email_clean) | (PatientModel.id == user_id)
            ).first()
            if not existing_pat:
                id_num = f"{uuid.uuid4().int % 9000000 + 1000000}"
                pat_id = f"#{id_num}"
                pat_name = full_name or "New Patient"
                initials = "".join([part[0] for part in pat_name.split() if part])[:2].upper() or "PT"
                new_patient = PatientModel(
                    id=pat_id,
                    pid=f"P-{id_num}",
                    name=pat_name,
                    initials=initials,
                    age=35,
                    gender="M",
                    phone="+1 (555) 000-0000",
                    email=email_clean,
                    demographics=f"35M · {pat_id}",
                    arrival_time=datetime.now().strftime("%I:%M %p"),
                    arrival_status="ARRIVED",
                    reason="General Health Consultation",
                    condition="Triage Assessment",
                    last_visit="—",
                    vitals_json=json.dumps({
                        "bp": "120/80 mmHg",
                        "hr": "72 bpm",
                        "pain": "0/10",
                        "spo2": "98%",
                        "temp": "36.6 °C"
                    }),
                    visit_type="New Patient",
                    booking_notes="Self-registered via Patient OTP Sign-in."
                )
                db.add(new_patient)
                db.commit()
                logger.info(f"Created PatientModel {pat_id} for {email_clean}")
    else:
        # Strict RBAC Check on existing user: DO NOT allow cross-role hijacking
        if user.role != active_role:
            role_labels = {
                "ADMIN": "Administrator",
                "DOCTOR": "Doctor / Clinician",
                "PATIENT": "Patient",
            }
            actual_role_label = role_labels.get(str(user.role), str(user.role))
            target_role_label = role_labels.get(active_role, active_role)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access Denied: Account '{email_clean}' is registered as an {actual_role_label}. You cannot sign in to the {target_role_label} portal.",
            )

        # Update non-security metadata if provided
        if req.full_name and req.full_name.strip():
            user.full_name = req.full_name.strip()
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
            "name": user.full_name or user.id,
        }
    )

    # Target dashboard routing URL
    dashboard_urls = {
        "ADMIN": "/admin",
        "DOCTOR": "/doctor",
        "PATIENT": "/patient",
    }
    target_url = dashboard_urls.get(str(user.role), "/doctor")

    is_profile_completed = bool(user.full_name and user.full_name.strip())

    user_profile = UserProfileResponse(
        id=str(user.id),
        email=str(user.email),
        full_name=str(user.full_name) if user.full_name else None,
        role=str(user.role),
        department=str(user.department) if user.department else None,
        specialization=str(user.specialization) if user.specialization else None,
        designation=str(user.designation) if user.designation else None,
        phone_number=str(user.phone_number) if user.phone_number else None,
        avatar_initials=user.get_initials(),
        is_active=bool(user.is_active),
        is_profile_completed=is_profile_completed,
        created_at=user.created_at if isinstance(user.created_at, datetime) else None,
    )

    return AuthResponse(
        success=True,
        message=f"Welcome, {user.full_name or user.id}! Authentication successful.",
        access_token=access_token,
        token_type="bearer",
        user=user_profile,
        target_dashboard_url=target_url,
        is_new_user=is_new_user,
    )


def update_user_profile(current_user: User, req: UpdateProfileRequest, db: Session) -> UserProfileResponse:
    """Updates user profile information in PostgreSQL."""
    if req.full_name is not None:
        current_user.full_name = req.full_name.strip()
    if req.department is not None:
        current_user.department = req.department.strip()
    if req.specialization is not None:
        current_user.specialization = req.specialization.strip()
    if req.designation is not None:
        current_user.designation = req.designation.strip()
    if req.phone_number is not None:
        current_user.phone_number = req.phone_number.strip()

    db.commit()
    db.refresh(current_user)

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
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
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
