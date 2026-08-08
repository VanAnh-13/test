# Standard libraries
import os
from datetime import datetime, timedelta, timezone

# Third party libraries
from dotenv import load_dotenv
from jwt import PyJWTError, decode, encode

# Load .env file
load_dotenv()


class JWTService:
    def __init__(self) -> None:
        self.__secret_key: str = os.getenv('SECRET_KEY', '')
        self.__algorithm: str = os.getenv('ALGORITHM', 'HS256')
        self.__access_exp: timedelta = timedelta(minutes=int(os.getenv('ACCESS_EXPIRE', 1)))
        self.__refresh_exp: timedelta = timedelta(days=int(os.getenv('REFRESH_EXPIRE', 1)))

        self.__verify_exp: timedelta = timedelta(minutes=int(5)) # verification email expire
        self.__password_reset_exp: timedelta = timedelta(
            minutes=int(os.getenv('PASSWORD_RESET_EXPIRE_MINUTES', 5))
        )

    def create_access_token(self, data: dict) -> str:
        """
        Create access token
        """
        to_encode = data.copy()
        exp = datetime.now(timezone.utc) + self.__access_exp

        to_encode.update({
            'exp': exp.timestamp(), 
            'type': 'access'
        })

        return encode(to_encode, self.__secret_key, self.__algorithm)
    
    def create_refresh_token(self, data: dict) -> str:
        """
        Create refresh token
        """
        to_encode = data.copy()
        exp = datetime.now(timezone.utc) + self.__refresh_exp
        to_encode.update({
            'exp': exp.timestamp(), 
            'type': 'refresh'
        })

        return encode(to_encode, self.__secret_key, self.__algorithm)

    def create_verification_token(self, data: dict) -> str:
        """
        Create email verification token
        """
        to_encode = data.copy()
        exp = datetime.now(timezone.utc) + self.__verify_exp
        
        to_encode.update({
            'exp': exp.timestamp(), 
            'type': 'verification'
        })

        return encode(to_encode, self.__secret_key, self.__algorithm)

    @property
    def password_reset_expires_in(self) -> int:
        return int(self.__password_reset_exp.total_seconds())

    def create_password_reset_token(self, data: dict) -> str:
        """Tạo token ngắn hạn chỉ dùng cho luồng đặt lại mật khẩu."""
        to_encode = data.copy()
        exp = datetime.now(timezone.utc) + self.__password_reset_exp
        to_encode.update({
            'exp': exp.timestamp(),
            'type': 'password_reset'
        })
        return encode(to_encode, self.__secret_key, self.__algorithm)

    def verify_token(self, token: str) -> dict | None:
        """
        Verify token
        """
        try:
            payload = decode(token, self.__secret_key, algorithms=[self.__algorithm])

            return payload
        except PyJWTError:
            return None

    def verify_password_reset_token(self, token: str) -> dict | None:
        """Chỉ chấp nhận token còn hạn và đúng mục đích reset mật khẩu."""
        payload = self.verify_token(token)
        if not payload or payload.get('type') != 'password_reset':
            return None
        return payload
        

jwt_service = JWTService()
