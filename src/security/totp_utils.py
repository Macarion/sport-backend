# security/totp_utils.py
import pyotp
from django.utils import timezone
from django.conf import settings
from api.models import UserTOTP, UsedTOTPCode

class TOTPVerifier:
    """
    通用的TOTP验证服务类
    """
    
    @staticmethod
    def verify_totp(user, code):
        """
        验证TOTP代码
        :param user: 用户对象
        :param code: TOTP验证码
        :return: (是否成功, 错误消息)
        """
        # 检查用户是否启用了TOTP
        try:
            totp_profile = UserTOTP.objects.get(user=user)
            if not totp_profile.enabled:
                return True, "TOTP not enabled"  # 用户未启用TOTP，跳过验证
        except UserTOTP.DoesNotExist:
            return True, "TOTP not configured"  # 用户未配置TOTP，跳过验证
        
        # 检查验证码格式
        if not code or len(code) != 6 or not code.isdigit():
            return False, "Invalid TOTP format"
        
        # 防止重放攻击
        recent_time = timezone.now() - timezone.timedelta(minutes=2)
        if UsedTOTPCode.objects.filter(
            user=user, 
            code=code,
            used_at__gte=recent_time
        ).exists():
            return False, "TOTP code already used"
        
        # 验证TOTP
        totp = pyotp.TOTP(totp_profile.secret_key)
        if totp.verify(code, valid_window=1):  # 允许1步容差（前后30秒）
            # 更新最后验证时间
            totp_profile.last_verified = timezone.now()
            totp_profile.save()
            
            # 记录已使用的代码
            UsedTOTPCode.objects.create(user=user, code=code)
            
            return True, "TOTP verified successfully"
        return False, "Invalid TOTP code"

    @staticmethod
    def requires_recent_verification(user):
        """
        检查用户是否需要重新验证TOTP
        :param user: 用户对象
        :return: 是否需要验证
        """
        try:
            totp_profile = UserTOTP.objects.get(user=user)
            if not totp_profile.enabled:
                return False  # 未启用TOTP，不需要验证
            
            # 检查最后验证时间是否在5分钟内
            return (timezone.now() - totp_profile.last_verified) > timezone.timedelta(minutes=5)
        except UserTOTP.DoesNotExist:
            return False  # 未配置TOTP，不需要验证