from django.apps import AppConfig


class ApiConfig(AppConfig):
    # default_auto_field = "django.db.models.BigAutoField"
    name = "api"
    
    def ready(self):
        # 确保模型正确注册
        from . import models  # noqa