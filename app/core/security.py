import hashlib
import hmac
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings


class BearerAuth(HTTPBearer):
    """Custom Bearer authentication with constant-time comparison"""

    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(
        self,
        request: Request,
    ) -> Optional[HTTPAuthorizationCredentials]:
        credentials = await super().__call__(request)
        if credentials:
            if not self.verify_token(credentials.credentials):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return credentials
        return None

    @staticmethod
    def verify_token(token: str) -> bool:
        """
        Verify token using constant-time comparison to prevent timing attacks
        """
        if not token:
            return False

        # Hash the provided token for constant-time comparison
        token_hash = hashlib.sha256(token.encode()).digest()

        # Check against each valid API key
        for api_key in settings.api_keys_list:
            api_key_hash = hashlib.sha256(api_key.encode()).digest()
            if hmac.compare_digest(token_hash, api_key_hash):
                return True

        return False


class APIKeyHeader:
    """API Key authentication via header"""

    def __init__(self, auto_error: bool = True):
        self.auto_error = auto_error
        self.header_name = settings.API_KEY_HEADER

    async def __call__(self, request: Request) -> Optional[str]:
        api_key = request.headers.get(self.header_name)

        if not api_key:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key missing",
                    headers={"WWW-Authenticate": "ApiKey"},
                )
            return None

        if not self.verify_api_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return api_key

    @staticmethod
    def verify_api_key(api_key: str) -> bool:
        """
        Verify API key using constant-time comparison
        """
        if not api_key:
            return False

        # Use constant-time comparison for each valid key
        for valid_key in settings.api_keys_list:
            if secrets.compare_digest(api_key, valid_key):
                return True

        return False


# Dependency instances
bearer_auth = BearerAuth()
api_key_auth = APIKeyHeader()


async def get_current_api_key(
    bearer_credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        BearerAuth(auto_error=False)
    ),
    header_api_key: Optional[str] = None,
) -> str:
    """
    Flexible authentication that accepts either Bearer token or API key header.
    Tries Bearer token first, then falls back to API key header.
    """
    # Check for Bearer token first
    if bearer_credentials:
        return bearer_credentials.credentials

    # Then check for API key in header
    if header_api_key is None:
        # Manually check for API key header
        from fastapi import Request
        from starlette.requests import Request as StarletteRequest

        # This will be injected by FastAPI
        request: Request = StarletteRequest

        header_api_key = request.headers.get(settings.API_KEY_HEADER)

    if header_api_key and APIKeyHeader.verify_api_key(header_api_key):
        return header_api_key

    # No valid authentication found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_api_key(
    api_key: str = Depends(api_key_auth),
) -> str:
    """
    Strict dependency that requires API key in header
    """
    return api_key


async def require_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_auth),
) -> str:
    """
    Strict dependency that requires Bearer token
    """
    return credentials.credentials


async def optional_api_key(
    api_key: Optional[str] = Depends(APIKeyHeader(auto_error=False)),
) -> Optional[str]:
    """
    Optional API key authentication for public endpoints with rate limiting
    """
    return api_key