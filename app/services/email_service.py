import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from app.core.config import SMTP_HOST, SMTP_PORT, SMTP_EMAIL, SMTP_PASSWORD, SMTP_FROM_NAME, OTP_EXPIRE_MINUTES

logger = logging.getLogger(__name__)


def send_otp_email(to_email: str, otp_code: str, role: str, purpose: str = "SIGNIN") -> bool:
    """
    Send OTP verification email via Gmail SMTP.
    If SMTP credentials are not configured, prints OTP to console for seamless local development.
    """
    subject = f"Your ClinIQ Verification Code: {otp_code}"
    role_display = {
        "ADMIN": "Hospital Administrator",
        "DOCTOR": "Clinician / Doctor",
        "PATIENT": "Patient Portal",
    }.get(role.upper(), role.title())

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f9f8; margin: 0; padding: 20px; color: #0f172a; }}
        .email-container {{ max-width: 520px; margin: 0 auto; background: #ffffff; border-radius: 12px; border: 1px solid #e2ede9; overflow: hidden; box-shadow: 0 4px 16px rgba(11, 60, 53, 0.05); }}
        .header {{ background: #0b3c35; padding: 24px; text-align: center; }}
        .logo-badge {{ display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; background: #0d9488; color: #ffffff; border-radius: 8px; font-weight: 800; font-size: 16px; margin-bottom: 8px; }}
        .header h1 {{ color: #ffffff; margin: 0; font-size: 20px; font-weight: 700; }}
        .header p {{ color: #a7f3d0; margin: 4px 0 0 0; font-size: 13px; }}
        .content {{ padding: 32px 24px; text-align: center; }}
        .role-pill {{ display: inline-block; background: #e6f4f1; color: #0d9488; font-weight: 600; font-size: 12px; padding: 4px 12px; border-radius: 20px; margin-bottom: 16px; }}
        .otp-box {{ background: #f4f9f8; border: 2px dashed #0d9488; border-radius: 12px; padding: 18px; margin: 24px 0; }}
        .otp-code {{ font-family: 'Courier New', monospace; font-size: 32px; font-weight: 800; letter-spacing: 8px; color: #0b3c35; margin: 0; }}
        .expiry-text {{ color: #64748b; font-size: 13px; margin-top: 8px; }}
        .footer {{ background: #fafdfc; padding: 16px 24px; text-align: center; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2ede9; }}
      </style>
    </head>
    <body>
      <div class="email-container">
        <div class="header">
          <div class="logo-badge">CI</div>
          <h1>ClinIQ Clinical Intelligence</h1>
          <p>Agentic Medical Decision Support Platform</p>
        </div>
        <div class="content">
          <span class="role-pill">{role_display} Access</span>
          <h2 style="margin: 0 0 8px 0; color: #0f172a; font-size: 18px;">Your One-Time Password (OTP)</h2>
          <p style="color: #64748b; font-size: 14px; margin: 0;">Use the 6-digit code below to sign in or verify your account.</p>
          
          <div class="otp-box">
            <div class="otp-code">{otp_code}</div>
            <div class="expiry-text">Valid for {OTP_EXPIRE_MINUTES} minutes · Do not share with anyone</div>
          </div>
          
          <p style="font-size: 13px; color: #64748b; line-height: 1.5;">
            If you did not request this verification code, please ignore this email or contact hospital security.
          </p>
        </div>
        <div class="footer">
          &copy; 2026 ClinIQ Agentic Health Intelligence. All medical rights reserved.
        </div>
      </div>
    </body>
    </html>
    """

    text_content = f"""ClinIQ Clinical Intelligence Assistant
Your One-Time Password (OTP) for {role_display} access is: {otp_code}

This code is valid for {OTP_EXPIRE_MINUTES} minutes.
If you did not request this code, please ignore this message.
"""

    # Check if Gmail SMTP credentials are provided
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        logger.warning(
            f"[DEV MODE - NO SMTP CREDENTIALS] OTP for {to_email} ({role_display}): {otp_code}"
        )
        print(f"\n=======================================================")
        print(f" [ClinIQ Gmail OTP Service - Dev Mode]")
        print(f" To:      {to_email} ({role_display})")
        print(f" OTP:     >>> {otp_code} <<<")
        print(f" Expires: {OTP_EXPIRE_MINUTES} minutes")
        print(f"=======================================================\n")
        return True

    # Send real email via Gmail SMTP
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_EMAIL}>"
        msg["To"] = to_email

        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, [to_email], msg.as_string())

        logger.info(f"Successfully sent OTP email to {to_email} via Gmail SMTP")
        return True
    except Exception as e:
        logger.error(f"Failed to send email via SMTP ({e}). Logging OTP for local dev.")
        print(f"\n=======================================================")
        print(f" [ClinIQ Gmail OTP Service - Fallback]")
        print(f" To:      {to_email}")
        print(f" OTP:     >>> {otp_code} <<<")
        print(f"=======================================================\n")
        return False
