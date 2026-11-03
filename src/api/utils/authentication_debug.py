# utils/authentication_debug.py
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.exceptions import AuthenticationFailed
import logging

logger = logging.getLogger(__name__)

class DebugJWTAuthentication(JWTAuthentication):
    def authenticate(self, request):
        logger.info(f"认证请求: {request.path}")
        logger.info(f"Authorization头: {request.headers.get('Authorization')}")
        
        try:
            # 调用父类认证逻辑
            return super().authenticate(request)
        except AuthenticationFailed as e:
            logger.error(f"JWT认证失败: {str(e)}")
            raise