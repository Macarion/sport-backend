# security/permissions.py
from rest_framework import permissions
from rest_framework.exceptions import PermissionDenied
from .totp_utils import TOTPVerifier

class TOTPVerified(permissions.BasePermission):
    """要求用户已通过TOTP验证的权限类"""
    
    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False
        
        # 检查是否已完成TOTP验证
        if TOTPVerifier.is_verified(request.user):
            return True
        
        # 尝试从请求获取TOTP代码
        totp_code = request.data.get('totp_code')
        
        if not totp_code:
            # 返回结构化错误信息
            raise PermissionDenied({
                "detail": "TOTP verification required",
                "code": "totp_required"
            })
        
        # 验证TOTP
        verified, msg = TOTPVerifier.verify_totp(request.user, totp_code)
        if not verified:
            raise PermissionDenied({
                "detail": f"Invalid TOTP code: {msg}",
                "code": "invalid_totp"
            })
        
        return True