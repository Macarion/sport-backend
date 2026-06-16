import requests
import logging
from django.conf import settings
from django.core.cache import cache
from rest_framework import exceptions
from rest_framework_simplejwt.authentication import JWTAuthentication

logger = logging.getLogger(__name__)


class RemoteIntrospectJWTAuthentication(JWTAuthentication):
    """
    Validate JWT locally, then require a short-lived positive introspection
    result from the remote auth service. Results are cached briefly to balance
    availability and the requirement to stay online.
    """

    def authenticate(self, request):
        result = super().authenticate(request)
        if result is None:
            return None

        user, validated_token = result

        # 开发/调试场景可通过 AUTH_REMOTE_ENABLED 关闭远端内省，仅本地验证 JWT
        if getattr(settings, "AUTH_REMOTE_ENABLED", True) is False:
            return user, validated_token

        if hasattr(user, "is_authorized") and not user.is_authorized:
            raise exceptions.AuthenticationFailed("User not authorized")

        jti = validated_token.get("jti")
        cache_key = f"introspect:{jti}" if jti else None

        cached_positive = False
        if cache_key:
            cached = cache.get(cache_key)
            # honor negative cache; allow positive cache as a grace window
            if cached is False:
                raise exceptions.AuthenticationFailed("Token revoked or expired")
            if cached is True:
                cached_positive = True

        raw_auth = request.META.get("HTTP_AUTHORIZATION", "")
        token = raw_auth.split(" ", 1)[1] if " " in raw_auth else raw_auth

        try:
            # 将前端传入的设备标识一并转发给远端内省服务，避免 device_mismatch
            device_id = request.headers.get("X-Client-Device", "")
            resp = requests.post(
                settings.AUTH_REMOTE_INTROSPECT,
                data={"token": token},
                timeout=settings.AUTH_REMOTE_TIMEOUT,
                headers={"X-Client-Device": device_id} if device_id else {},
                # Bypass any system proxy to reach local auth service directly.
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
            data = resp.json()
            logger.debug(
                "Introspect result: active=%s revoked=%s jti=%s sub=%s reason=%s",
                data.get("active"),
                data.get("revoked"),
                jti,
                validated_token.get("sub"),
                data.get("reason"),
            )
        except Exception:
            # 如果远端暂时不可用且近期有正向缓存，则放行，避免误踢
            if cached_positive:
                return user, validated_token
            raise exceptions.AuthenticationFailed("Introspection failed")

        active = bool(data.get("active")) and not data.get("revoked")
        if cache_key:
            ttl = getattr(settings, "AUTH_CACHE_TTL", 90)
            cache.set(cache_key, active, ttl)

        if not active:
            raise exceptions.AuthenticationFailed("Token invalid or revoked")

        return user, validated_token
