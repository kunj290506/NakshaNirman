"""Security middleware for NakshaNirman application.

Implements:
- Rate limiting
- File upload validation
- Input sanitization
- Security headers
- Request logging
"""

import hashlib
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Configuration
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".dxf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_PER_HOUR = 1000

# Rate limiting storage (use Redis in production)
rate_limit_store = defaultdict(lambda: {"minute": [], "hour": []})


class SecurityMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        # Log request
        logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")

        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data:; "
            "connect-src 'self'"
        )

        return response


def check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limits.
    
    Args:
        client_ip: Client IP address
        
    Returns:
        True if within limits, False if exceeded
    """
    now = time.time()
    client_data = rate_limit_store[client_ip]

    # Clean old entries
    client_data["minute"] = [t for t in client_data["minute"] if now - t < 60]
    client_data["hour"] = [t for t in client_data["hour"] if now - t < 3600]

    # Check limits
    if len(client_data["minute"]) >= RATE_LIMIT_PER_MINUTE:
        logger.warning(f"Rate limit exceeded (minute): {client_ip}")
        return False

    if len(client_data["hour"]) >= RATE_LIMIT_PER_HOUR:
        logger.warning(f"Rate limit exceeded (hour): {client_ip}")
        return False

    # Record request
    client_data["minute"].append(now)
    client_data["hour"].append(now)

    return True


async def validate_file_upload(file: UploadFile) -> None:
    """Validate uploaded file for security.
    
    Args:
        file: Uploaded file
        
    Raises:
        HTTPException: If file is invalid or dangerous
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Sanitize filename
    safe_filename = Path(file.filename).name
    if safe_filename != file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename characters")

    # Check extension
    ext = Path(safe_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check file size
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Reset to start

    max_size = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB}MB"
        )

    if size == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    # Validate file magic numbers (first few bytes)
    header = await file.read(12)
    await file.seek(0)

    if ext == ".dxf":
        # DXF files start with "0\r\nSECTION" or similar
        if not (header.startswith(b"0\r\n") or header.startswith(b"0\n")):
            raise HTTPException(status_code=400, detail="Invalid DXF file format")

    elif ext in {".png"}:
        # PNG magic number: 89 50 4E 47 0D 0A 1A 0A
        if not header.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(status_code=400, detail="Invalid PNG file format")

    elif ext in {".jpg", ".jpeg"}:
        # JPEG magic number: FF D8 FF
        if not header.startswith(b"\xff\xd8\xff"):
            raise HTTPException(status_code=400, detail="Invalid JPEG file format")

    elif ext == ".bmp":
        # BMP magic number: 42 4D
        if not header.startswith(b"BM"):
            raise HTTPException(status_code=400, detail="Invalid BMP file format")

    logger.info(f"File validated: {safe_filename} ({size} bytes)")


def sanitize_string(value: str, max_length: int = 255) -> str:
    """Sanitize string input.
    
    Args:
        value: Input string
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not value:
        return ""

    # Remove null bytes
    value = value.replace("\x00", "")

    # Limit length
    value = value[:max_length]

    # Remove control characters except newline/tab
    value = "".join(char for char in value if char.isprintable() or char in "\n\t")

    return value.strip()


def sanitize_numeric(value: any, min_val: float = None, max_val: float = None) -> Optional[float]:
    """Sanitize numeric input.
    
    Args:
        value: Input value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Sanitized float or None if invalid
    """
    try:
        num = float(value)

        # Check for infinity/NaN
        if not (num == num and abs(num) != float("inf")):
            return None

        # Apply bounds
        if min_val is not None and num < min_val:
            num = min_val
        if max_val is not None and num > max_val:
            num = max_val

        return num
    except (ValueError, TypeError):
        return None


def hash_password(password: str) -> str:
    """Hash password using SHA-256 (use bcrypt in production).
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()


def generate_secure_token() -> str:
    """Generate a secure random token.
    
    Returns:
        Hex token string
    """
    import secrets
    return secrets.token_hex(32)


def validate_project_access(user_id: str, project_id: str) -> bool:
    """Validate user has access to project (placeholder).
    
    Args:
        user_id: User identifier
        project_id: Project identifier
        
    Returns:
        True if authorized
    """
    # TODO: Implement actual authorization logic
    return True


# Security audit logging
def log_security_event(event_type: str, details: dict, severity: str = "INFO"):
    """Log security-related events.
    
    Args:
        event_type: Type of security event
        details: Event details
        severity: Log severity (INFO, WARNING, ERROR, CRITICAL)
    """
    log_entry = {
        "timestamp": time.time(),
        "event_type": event_type,
        "severity": severity,
        **details
    }

    if severity == "CRITICAL":
        logger.critical(f"Security Event: {log_entry}")
    elif severity == "ERROR":
        logger.error(f"Security Event: {log_entry}")
    elif severity == "WARNING":
        logger.warning(f"Security Event: {log_entry}")
    else:
        logger.info(f"Security Event: {log_entry}")
