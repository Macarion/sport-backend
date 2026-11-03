# backend/api/security/__init__.py
import pyotp
from django.conf import settings
from django.utils import timezone
from api.models import Users  # 确保正确导入您的用户模型
from django.shortcuts import render

def generate_totp_secret():
    """生成新的 TOTP 密钥"""
    return pyotp.random_base32()

def verify_totp(user, code):
    """
    验证 TOTP 代码
    :param user: 用户对象
    :param code: 用户输入的 TOTP 代码
    :return: 验证是否成功
    """
    if not user.totp_secret:
        return False
        
    totp = pyotp.TOTP(user.totp_secret)
    return totp.verify(code, valid_window=1)

def require_totp_verification(user):
    """
    检查用户是否需要进行 TOTP 验证
    :param user: 用户对象
    :return: 是否需要验证
    """
    # 只有当用户完成设置后才要求验证
    return user.totp_enabled and user.totp_secret and user.totp_setup_completed

# 安全设置视图
def security_settings(request):
    user = request.user
    # 展示当前TOTP状态
    totp_status = "已启用" if user.totp_enabled else "未启用"
    
    # 如果用户从未设置过，生成预备密钥
    if not user.totp_secret:
        user.totp_secret = generate_totp_secret()
        user.save(update_fields=['totp_secret'])
    
    # 生成二维码URI
    totp_uri = pyotp.totp.TOTP(user.totp_secret).provisioning_uri(
        name=user.email,
        issuer_name="智慧体测系统"  # 替换为您的应用名称
    )
    
    return render(request, 'security.html', {
        'totp_status': totp_status,
        'totp_uri': totp_uri
    })