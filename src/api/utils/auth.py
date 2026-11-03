# utils/auth.py
from rest_framework.permissions import BasePermission
from rest_framework.exceptions import PermissionDenied
from pyotp import TOTP
from rest_framework import status
import security
from rest_framework.exceptions import ValidationError, JsonResponse, PermissionDenied
class JWTAuthenticationMiddleware:
    """JWT验证中间件 - 用于所有操作"""
    def __init__(self, get_response):
        self.get_response = get_response
        
    def __call__(self, request):
        # 验证JWT令牌
        if not request.user.is_authenticated:
            return JsonResponse({"error": "身份验证失败"}, status=401)
        return self.get_response(request)

class TOTPPermission(BasePermission):
    def has_permission(self, request, view):
        # 统一通过视图名称判断敏感性
        view_name = view.__class__.__name__
        if not security.is_sensitive_action(view_name):
            return True
        
        # 返回结构化错误信息
        if not request.user.totp_secret:
            return True
        
        totp_token = request.headers.get('X-TOTP-Token', '')
        if not totp_token:
            raise PermissionDenied({
                "code": "totp_required",
                "detail": "此操作需要双重验证",
                "requires_totp": True
            }, status=status.HTTP_403_FORBIDDEN)
        
        # 4. 验证TOTP有效性
        totp = TOTP(request.user.totp_secret)
        if not totp.verify(totp_token, valid_window=1):

            raise PermissionDenied({
                "code": "totp_invalid",
                "detail": "双重验证码无效",
                "requires_totp": True
            }, status=status.HTTP_403_FORBIDDEN)
        
        # 5. 验证成功，记录使用时间
        request.user.record_totp_usage()
        return True