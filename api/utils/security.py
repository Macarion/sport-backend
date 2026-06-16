# security.py
SENSITIVE_ACTIONS = {
    'change_password': True,
    'transfer_funds': True,
    'change_email': True,
    'delete_account': True,
    'modify_payment_method': True,
    'export_data': True,
    # 添加其他敏感操作...
}

def is_sensitive_action(view_name):
    """检查操作是否敏感"""
    return SENSITIVE_ACTIONS.get(view_name, False)