import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from authlib.integrations.starlette_client import OAuth
from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import EmailStr
from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase
from starlette.requests import Request

from config.providers import get_db
from config.server_runtime import load_cookie_runtime_policy
from users.engine import handle_send_otp, verify_stored_password
from users.schema import (
    OAuthCodeExchangeRequest,
    PasswordResetRequest,
    RefreshRequest,
    ResendEmailRequest,
    Token,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
    VerifyEmailRequest,
    VerifyOtp,
)
from users.utils.authentication import jwt_service
from users.utils.email_service import email_service
from users.utils.security import HashHelper

router = APIRouter(tags=["Authentication"])

OAUTH_CODE_TTL_SECONDS = int(os.getenv('OAUTH_CODE_TTL_SECONDS', '60'))
_REFRESH_COOKIE_KEY = "refresh_token"
_REFRESH_COOKIE_PATH = "/"
_REFRESH_COOKIE_SAMESITE = "lax"


def _is_session_cookie_secure() -> bool:
    return load_cookie_runtime_policy().session_https_only


def _is_skip_email_verification_enabled() -> bool:
    return os.getenv("SKIP_EMAIL_VERIFICATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@router.post("/signup", response_model=UserResponse)
async def register(user_data: UserRegisterRequest, background_tasks: BackgroundTasks, db: AsyncDatabase = Depends(get_db)) -> UserResponse:
    # Check email và username
    if await db.tbl_User.find_one({
        "$or": [
            {"email": user_data.email},
            {"username": user_data.username}
        ]
    }):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username or Email already registed")
    
    skip_email_verification = _is_skip_email_verification_enabled()
    new_user_doc = {
        "username": user_data.username,
        "email": user_data.email,
        "gender": user_data.gender,
        "date": user_data.date,
        "number": user_data.number,
        "fullName": user_data.fullName,
        "role": "user",
        "avatar": None,
        "is_verified": skip_email_verification,
        "created_at": datetime.now(timezone.utc).timestamp()
    }

    user_insert_result = await db.tbl_User.insert_one(new_user_doc)
    user_id = user_insert_result.inserted_id

    linked_account_doc = {
        "user_id": user_id,
        "provider": "local",
        "provider_id": user_data.email,
        "password": HashHelper.get_password_hash(user_data.password),
        "created_at": datetime.now(timezone.utc).timestamp()
    }

    await db.linked_accounts.insert_one(linked_account_doc)

    if not skip_email_verification:
        verification_token = jwt_service.create_verification_token({
            'sub': str(user_id),
            'email': user_data.email
        })

        # QR code
        verify_link = email_service.get_verify_link(verification_token)
        qr_base64 = email_service.generate_qr_base64(verify_link)

        # Offload email sending to a background worker
        background_tasks.add_task(
            email_service.send_verification_email,
            user_data.email,
            verification_token,
            qr_base64
        )

    created_user = await db.tbl_User.find_one({'_id': user_id})
    created_user['_id'] = str(created_user['_id'])

    return created_user


@router.post("/login", response_model=Token)
async def login(user_login: UserLoginRequest, response: Response, db: AsyncDatabase = Depends(get_db)) -> Token:
    user = await db.tbl_User.find_one({
        "$or": [
            {"email": user_login.username},
            {"username": user_login.username}
        ]
    })

    invalid_credentials_exception = HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid username or password",
        headers={"WWW-Authenticate": "Bearer"}
    )

    if not user:
        raise invalid_credentials_exception

    if not user.get('is_verified', True):
        if _is_skip_email_verification_enabled():
            await db.tbl_User.update_one(
                {'_id': user['_id']},
                {'$set': {'is_verified': True}}
            )
            user['is_verified'] = True
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is not verified. Please check your email to verify your account."
            )

    if user.get('password'):
        is_valid, needs_upgrade = verify_stored_password(
            user_login.password, user.get('password')
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect login information",
                headers={"WWW-Authenticate": "Bearer"}
            )
        canonical_password_hash = user['password']
        if needs_upgrade:
            canonical_password_hash = HashHelper.get_password_hash(user_login.password)
            await db.tbl_User.update_one(
                {'_id': user['_id']},
                {'$set': {'password': canonical_password_hash}}
            )
        await db.linked_accounts.update_one(
            {'user_id': user['_id'], 'provider': 'local'},
            {
                '$set': {'password': canonical_password_hash},
                '$setOnInsert': {
                    'provider_id': user['email'],
                    'created_at': datetime.now(timezone.utc).timestamp(),
                },
            },
            upsert=True,
        )
    else:
        account = await db.linked_accounts.find_one({
            'user_id': user['_id'],
            'provider': 'local'
        })

        is_valid, needs_upgrade = verify_stored_password(
            user_login.password,
            account.get('password') if account else None,
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect login information",
                headers={"WWW-Authenticate": "Bearer"}
            )
        if needs_upgrade:
            await db.linked_accounts.update_one(
                {'_id': account['_id']},
                {'$set': {'password': HashHelper.get_password_hash(user_login.password)}}
            )
        
    access_token = jwt_service.create_access_token({
        'sub': str(user['_id']),
        'role': user.get('role', 'user'),
        'email': user['email']
    })

    refresh_token = jwt_service.create_refresh_token({
        'sub': str(user['_id'])
    })

    return Token(access_token=access_token, refresh_token=refresh_token, token_type='bearer')


@router.post('/refresh', response_model=Token)
async def refresh_token(response: Response, refresh_request: RefreshRequest, db: AsyncDatabase = Depends(get_db)) -> Token:
    cookie_secure = _is_session_cookie_secure()
    refresh_token = refresh_request.refresh_token

    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing",
            headers={"WWW-Authenticate": "Bearer"}
        )

    payload = jwt_service.verify_token(refresh_token)
    if not payload or payload.get('type') != 'refresh':
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    
    user = await db.tbl_User.find_one({'_id': ObjectId(payload.get('sub'))})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists",
        )
    
    new_access_token = jwt_service.create_access_token({
        'sub': str(user['_id']),
        'role': user.get('role', 'user'),
        'email': user['email']
    })

    new_refresh_token = jwt_service.create_refresh_token({
        'sub': str(user['_id'])
    })

    refresh_expire_days = int(os.getenv('REFRESH_EXPIRE', 1))
    response.set_cookie(
        key=_REFRESH_COOKIE_KEY,
        value=new_refresh_token,
        httponly=True,
        max_age=refresh_expire_days * 24 * 60 * 60,
        path=_REFRESH_COOKIE_PATH,
        samesite=_REFRESH_COOKIE_SAMESITE,
        secure=cookie_secure,
    )

    return Token(access_token=new_access_token, refresh_token=new_refresh_token, token_type='bearer')


oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/login')

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncDatabase = Depends(get_db)):
    payload = jwt_service.verify_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    
    user = await db.tbl_User.find_one({'_id': ObjectId(payload['sub'])})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    user['_id'] = str(user['_id'])
    return user


@router.get('/me', response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)) -> UserResponse:
    return current_user


@router.post('/logout')
async def logout(response: Response, current_user: dict = Depends(get_current_user)):
    cookie_secure = _is_session_cookie_secure()
    response.delete_cookie(
        key=_REFRESH_COOKIE_KEY,
        httponly=True,
        path=_REFRESH_COOKIE_PATH,
        samesite=_REFRESH_COOKIE_SAMESITE,
        secure=cookie_secure,
    )

    return {'message': 'Logged out successfully'}


oauth = OAuth()

oauth.register(
    name = 'google',
    client_id = os.getenv('GOOGLE_CLIENT_ID'),
    client_secret = os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url = 'https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs = {
        'scope': 'openid email profile'
    }
)

@router.get('/google/login')
async def google_login(request: Request):
    redirect_uri = f"{os.getenv('REDIRECT_URI')}/google/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get('/google/callback')
async def google_callback(request: Request, response: Response, db: AsyncDatabase = Depends(get_db)):
    frontend_url = os.getenv('FRONTEND_URL', '').rstrip('/')
    if not frontend_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='FRONTEND_URL chưa được cấu hình',
        )

    token = await oauth.google.authorize_access_token(request)

    user_info = token.get('userinfo')
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to get user info from Google",
        )
    
    google_email = user_info.get('email')
    google_sub = user_info.get('sub')
    google_name = user_info.get('name')
    google_picture = user_info.get('picture')

    existing_user = await db.tbl_User.find_one({'email': google_email})

    user_id = None
    if existing_user:
        user_id = existing_user['_id']

        linked_acc = await db.linked_accounts.find_one({
            'user_id': user_id,
            'provider': 'google'
        })

        if not linked_acc:
            await db.linked_accounts.insert_one({
                'user_id': user_id,
                'provider': 'google',
                'provider_id': google_sub,
                'created_at': datetime.now(timezone.utc).timestamp()
            })

            if not existing_user.get('avatar'):
                await db.tbl_User.update_one(
                    {'_id': user_id},
                    {'$set': {'avatar': google_picture}}
                )
            
            if not existing_user.get('is_verified'):
                await db.tbl_User.update_one(
                    {'_id': user_id},
                    {'$set': {'is_verified': True}}
                )
    else:
        new_user_doc = {
            "username": google_name,
            "email": google_email,
            "gender": None,
            "date": None,
            "number": None,
            "fullName": google_name,
            "role": "user",
            "avatar": google_picture,
            "is_verified": True,
            "created_at": datetime.now(timezone.utc).timestamp()
        }
        insert_result = await db.tbl_User.insert_one(new_user_doc)
        user_id = insert_result.inserted_id

        await db.linked_accounts.insert_one({
            'user_id': user_id,
            'provider': 'google',
            'provider_id': google_sub,
            'created_at': datetime.now(timezone.utc).timestamp()
        })
    
    authorization_code = secrets.token_urlsafe(32)
    await db.oauth_login_codes.insert_one({
        'code_hash': hashlib.sha256(authorization_code.encode('utf-8')).hexdigest(),
        'user_id': user_id,
        'expires_at': datetime.now(timezone.utc) + timedelta(
            seconds=OAUTH_CODE_TTL_SECONDS
        ),
    })

    query = urlencode({'code': authorization_code})
    return RedirectResponse(url=f"{frontend_url}/google?{query}")


@router.post('/auth/oauth/exchange', response_model=Token)
async def exchange_oauth_code(
    request: OAuthCodeExchangeRequest,
    db: AsyncDatabase = Depends(get_db),
) -> Token:
    code_hash = hashlib.sha256(request.code.encode('utf-8')).hexdigest()
    login_code = await db.oauth_login_codes.find_one_and_delete({
        'code_hash': code_hash,
        'expires_at': {'$gte': datetime.now(timezone.utc)},
    })
    if not login_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Mã ủy quyền không hợp lệ hoặc đã hết hạn',
        )

    user = await db.tbl_User.find_one({'_id': login_code['user_id']})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Mã ủy quyền không hợp lệ hoặc đã hết hạn',
        )

    access_token = jwt_service.create_access_token({
        'sub': str(user['_id']),
        'role': user.get('role', 'user'),
        'email': user['email'],
    })
    refresh_token = jwt_service.create_refresh_token({'sub': str(user['_id'])})
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type='bearer',
    )


@router.post('/auth/verifications', response_model=Token)
async def verify_user_email(request: VerifyEmailRequest, db: AsyncDatabase = Depends(get_db)) -> Token:
    payload = jwt_service.verify_token(request.token)
    if not payload or payload.get('type') != 'verification':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid or expired verification code"
        )

    user_id = payload.get('sub')

    user = await db.tbl_User.find_one({'_id': ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account not found")

    if user.get('is_verified'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="This account has already been verified. Please log in with your password"
        )

    await db.tbl_User.update_one(
        {'_id': ObjectId(user_id)},
        {'$set': {'is_verified': True}}
    )

    access_token = jwt_service.create_access_token({
        'sub': str(user['_id']),
        'role': user.get('role', 'user'),
        'email': user['email']
    })

    refresh_token = jwt_service.create_refresh_token({
        'sub': str(user['_id'])
    })

    return Token(access_token=access_token, refresh_token=refresh_token, token_type='bearer')


@router.post("/auth/token/verifications")
async def request_email_verification_token(
    request: ResendEmailRequest,
    background_tasks: BackgroundTasks,
    db: AsyncDatabase = Depends(get_db)
):
    user = await db.tbl_User.find_one({"email": request.email})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found with this email"
        )

    if user.get('is_verified', True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account has already been verified"
        )

    verification_token = jwt_service.create_verification_token({
        'sub': str(user['_id']),
        'email': user['email']
    })
    background_tasks.add_task(email_service.send_verification_email, user['email'], verification_token)

    return {"detail": "A new verification link has been sent. Please check your email"}


@router.post("/auth/otp/verifications")
async def request_new_otp(
    request: ResendEmailRequest,
    background_tasks: BackgroundTasks,
    db: AsyncDatabase = Depends(get_db)
):
    user = await db.tbl_User.find_one({"email": request.email})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found with this email"
        )

    if user.get('is_verified', True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account has already been verified"
        )
    
    await db.tbl_User.update_one(
        {"_id": user["_id"]},
        {"$unset": {"otp": "", "createAtOTP": ""}}
    )

    background_tasks.add_task(handle_send_otp, user['email'])

    return {"detail": "A new otp has been sent. Please check your email"}


@router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(
    email: EmailStr,
    background_tasks: BackgroundTasks, 
    db: AsyncDatabase = Depends(get_db)
):
    user = await db.tbl_User.find_one({"email": email})

    if not user:
        return {
            "status": "success",
            "message": "If this email is registered, you will receive an OTP shortly."
        }

    if not user.get('is_verified', True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Account is not verified. Please verify your email first."
        )

    background_tasks.add_task(handle_send_otp, email, db)

    return {
        "status": "success",
        "message": "A password reset OTP has been sent to your email address."
    }


@router.post("/auth/verify-otp", status_code=status.HTTP_200_OK)
async def verify_reset_otp(
    payload: VerifyOtp, 
    db: AsyncDatabase = Depends(get_db)
):
    now = datetime.now(timezone.utc).timestamp()
    nonce = secrets.token_urlsafe(32)
    nonce_hash = hashlib.sha256(nonce.encode("utf-8")).hexdigest()
    expires_in = jwt_service.password_reset_expires_in
    user = await db.tbl_User.find_one_and_update(
        {
            "email": payload.email,
            "otp": payload.otp,
            "createAtOTP": {"$gte": now},
        },
        {
            "$set": {
                "password_reset_nonce_hash": nonce_hash,
                "password_reset_nonce_expires_at": now + expires_in,
            },
            "$unset": {"otp": "", "createAtOTP": ""},
        },
        return_document=ReturnDocument.AFTER,
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP code.",
        )

    return {
        "reset_token": jwt_service.create_password_reset_token({
            "sub": str(user["_id"]),
            "email": user["email"],
            "nonce": nonce,
        }),
        "expires_in": expires_in,
    }


@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(
    payload: PasswordResetRequest,
    db: AsyncDatabase = Depends(get_db)
):
    if payload.new_password != payload.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match. Please try again"
        )

    token_payload = jwt_service.verify_password_reset_token(payload.reset_token)
    if not token_payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Password reset authorization is invalid or expired",
        )
    user_id = token_payload.get("sub")
    email = token_payload.get("email")
    nonce = token_payload.get("nonce")
    if not all(isinstance(value, str) and value for value in (user_id, email, nonce)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Password reset authorization is invalid or expired",
        )
    try:
        object_id = ObjectId(user_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Password reset authorization is invalid or expired",
        ) from exc

    nonce_hash = hashlib.sha256(nonce.encode("utf-8")).hexdigest()
    password_hash = HashHelper.get_password_hash(payload.new_password)
    user = await db.tbl_User.find_one_and_update(
        {
            "_id": object_id,
            "email": email,
            "password_reset_nonce_hash": nonce_hash,
            "password_reset_nonce_expires_at": {
                "$gte": datetime.now(timezone.utc).timestamp()
            },
        },
        {
            "$set": {"password": password_hash},
            "$unset": {
                "password_reset_nonce_hash": "",
                "password_reset_nonce_expires_at": "",
            },
        },
        return_document=ReturnDocument.AFTER,
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Password reset authorization is invalid or expired",
        )

    await db.linked_accounts.update_one(
        {"user_id": object_id, "provider": "local"},
        {"$set": {"password": password_hash}},
        upsert=True,
    )

    return {
        "status": "success",
        "message": "Password updated successfully. You can now login with your new password."
    }
