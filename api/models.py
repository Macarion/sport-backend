from django.db import models
from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone
from django.conf import settings    
from django.contrib.auth.models import AbstractUser, Group, Permission,PermissionsMixin, AbstractBaseUser
from django.utils.translation import gettext_lazy as _
from .managers import CustomUserManager

class Users(AbstractBaseUser, PermissionsMixin):
    id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=30, unique=True)
    password = models.CharField(max_length=128)  # 长度增加到128以适应Django密码哈希
    email = models.CharField(max_length=50, blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=150)
    last_login = models.DateTimeField(blank=True, null=True)  # 确保有这个字段
    createtime = models.DateTimeField(default=timezone.now)
    updatetime = models.DateTimeField(auto_now=True)
    isdeleted = models.BooleanField(default=False)  # 改为BooleanField
    remark = models.CharField(max_length=255, blank=True, null=True)
    totp_setup_completed=models.BooleanField(default=False) # 建议添加此状态字段，用于记录用户是否完成TOTP设置
    # 添加用户类型字段（根据实际需要）
    USER_TYPE_CHOICES = (
        ('teacher', '教师'),
        ('student', '学生'),
        ('admin', '管理员'),
    )
    usertype = models.CharField(
        max_length=50, 
        choices=USER_TYPE_CHOICES,
        default='teacher',
        verbose_name="用户类型"
    )
    
    # TOTP 相关字段
    totp_secret = models.CharField(
        max_length=32, 
        blank=True, 
        null=True,
        default=None,
        verbose_name="TOTP 密钥"
    )
    
    totp_enabled = models.BooleanField(
        default=False,
        verbose_name="启用双重验证"
    )

    last_totp_used = models.DateTimeField(
        blank=True, 
        null=True, 
        verbose_name="上次使用 TOTP 的时间"
    )
    # 授权开关：远端/管理员可用此字段整体启用/禁用账号
    is_authorized = models.BooleanField(default=True, verbose_name="已授权")
    # 必需字段
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    date_joined = models.DateTimeField(default=timezone.now)

    # 登录追踪（用于设备校验与单会话）
    last_device_id = models.CharField(max_length=128, blank=True, null=True)
    last_access_jti = models.CharField(max_length=64, blank=True, null=True)
    last_login_at = models.DateTimeField(blank=True, null=True)
    last_access_exp = models.DateTimeField(blank=True, null=True)

    # 权限相关字段
    groups = models.ManyToManyField(
        Group,
        verbose_name='用户组',
        blank=True,
        help_text='用户所属的用户组',
        related_name="custom_users",
        related_query_name="user",
    )
    
    user_permissions = models.ManyToManyField(
        Permission,
        verbose_name='用户权限',
        blank=True,
        help_text='用户的特定权限',
        related_name="custom_users",
        related_query_name="user",
    )

    # 设置用户名字段
    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['email']  # 创建超级用户时需要

    class Meta:
        db_table = 'users'
        verbose_name = '用户'
        verbose_name_plural = '用户'

    def __str__(self):
        return self.username

    def record_totp_usage(self):
        """记录TOTP使用时间（在验证成功后调用）"""
        self.last_totp_used = timezone.now()
        self.save(update_fields=['last_totp_used'])    
        
    def enable_totp(self, secret):
        self.totp_secret = secret
        self.totp_enabled = True
        self.save()
    
    def disable_totp(self):
        self.totp_secret = None
        self.totp_enabled = False
        self.save()
    
    def has_totp(self):
        """检查用户是否绑定了TOTP设备"""
        return bool(self.totp_secret)
    
    # 实现AbstractBaseUser要求的方法
    def get_full_name(self):
        return self.username

    def get_short_name(self):
        return self.username
    
    objects = CustomUserManager()  # 添加这一行
  
# class CustomUser(AbstractUser):
#     totp_secret = models.CharField(
#         max_length=32, 
#         blank=True, 
#         null=True,
#         verbose_name="TOTP 密钥"
#     )
    
#     totp_enabled = models.BooleanField(
#         default=False,
#         verbose_name="启用双重验证"
#     )
    
#         # 解决groups字段冲突
#     groups = models.ManyToManyField(
#         Group,
#         verbose_name='groups',
#         blank=True,
#         help_text='用户所属的用户组',
#         related_name="customuser_set",  # 添加唯一反向名称
#         related_query_name="customuser",
#     )
    
#     # 解决user_permissions字段冲突
#     user_permissions = models.ManyToManyField(
#         Permission,
#         verbose_name='user permissions',
#         blank=True,
#         help_text='用户的特定权限',
#         related_name="customuser_set",  # 添加唯一反向名称
#         related_query_name="customuser",
#     )
#     def enable_totp(self, secret):
#         self.totp_secret = secret
#         self.totp_enabled = True
#         self.save()
    
#     def disable_totp(self):
#         self.totp_secret = None
#         self.totp_enabled = False
#         self.save()
    
    # class Meta:
    #     # managed = False
    #     db_table = 'customuser'
            
class Class(models.Model):
    classid = models.AutoField(primary_key=True, db_comment='班级 ID')
    classname = models.CharField(max_length=50, blank=True, null=True, db_comment='班级名称')
    teacherid = models.IntegerField(db_comment='对应教师 ID')
    createtime = models.DateTimeField(db_comment='创建时间')
    updatetime = models.DateTimeField(db_comment='修改时间')
    isdeleted = models.PositiveIntegerField(db_comment='是否删除')
    remark = models.CharField(max_length=255, blank=True, null=True, db_comment='备注')

    class Meta:
        # managed = False
        db_table = 'class'


class Department(models.Model):
    departmentid = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50)
    createtime = models.DateTimeField()
    updatetime = models.DateTimeField()
    isdeleted = models.IntegerField()
    remark = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        # managed = False
        db_table = 'department'


class Inform(models.Model):
    informid = models.AutoField(primary_key=True, db_comment='通知 ID')
    content = models.TextField(db_comment='通知内容')
    classid = models.IntegerField(db_comment='通知班级')
    teacherid = models.IntegerField(db_comment='上传教师')
    uploadtime = models.DateTimeField(db_comment='上传时间')
    isdeleted = models.PositiveIntegerField(db_comment='是否删除')
    remark = models.CharField(max_length=255, blank=True, null=True, db_comment='备注')

    class Meta:
        # managed = False
        db_table = 'inform'


class Manager(models.Model):
    managerid = models.IntegerField(primary_key=True)
    userid = models.CharField(max_length=30)
    name = models.CharField(max_length=30)
    gender = models.CharField(max_length=10)
    email = models.CharField(max_length=50, blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    permissionid = models.ForeignKey('Permission', models.DO_NOTHING, db_column='permissionid', blank=True, null=True)
    hiredate = models.DateTimeField(blank=True, null=True)
    createtime = models.DateTimeField()
    updatetime = models.DateTimeField()
    isdeleted = models.IntegerField()
    remark = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        # managed = False
        db_table = 'manager'


class Permission(models.Model):
    permissionid = models.AutoField(primary_key=True)
    permission = models.TextField(blank=True, null=True)
    createtime = models.DateTimeField()
    updatetime = models.DateTimeField()
    isdeleted = models.IntegerField()
    remark = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        # managed = False
        db_table = 'permission'

# class Student(models.Model):
#     numid = models.CharField(max_length=30, unique=True, verbose_name="学号")
#     name = models.CharField(max_length=30, db_comment='姓名')
#     gender = models.CharField(max_length=10, db_comment='性别')
#     birth = models.DateField(blank=True, null=True, db_comment='出生日期')
#     phone = models.CharField(max_length=20, blank=True, null=True, db_comment='电话')
#     email = models.CharField(max_length=50, blank=True, null=True, db_comment='邮箱')
#     address = models.CharField(max_length=255, blank=True, null=True, verbose_name="地址")
#     departid = models.IntegerField(blank=True, null=True, db_comment='学院信息 ID')
    
#     # 学校信息
#     Universityid = models.CharField(max_length=50, verbose_name="大学ID")
#     departid = models.CharField(max_length=50, verbose_name="系所ID")
#     enrollment = models.DateField(verbose_name="入学日期", blank=True, null=True)
#     nationality = models.CharField(max_length=50, blank=True, null=True, verbose_name="国籍")
#     major = models.CharField(max_length=100, verbose_name="专业")
    
#     user = models.OneToOneField(
#         'Users', 
#         on_delete=models.CASCADE, 
#         related_name='student_profile',
#         blank=True, 
#         null=True,
#         verbose_name="关联用户"
#     )
    
#     class Meta:
#         # managed = False
#         db_table = 'student'

class Student(models.Model):
    numid = models.CharField(max_length=30, unique=True, verbose_name="学号")
    name = models.CharField(max_length=30, db_comment='姓名')
    gender = models.CharField(max_length=10, db_comment='性别')
    birth = models.DateField(blank=True, null=True, db_comment='出生日期')
    phone = models.CharField(max_length=20, blank=True, null=True, db_comment='电话')
    email = models.CharField(max_length=50, blank=True, null=True, db_comment='邮箱')
    address = models.CharField(max_length=255, blank=True, null=True, verbose_name="地址")
    Universityid = models.CharField(max_length=50, verbose_name="大学ID")
    departid = models.CharField(max_length=50, verbose_name="系所ID")
    enrollment = models.DateField(verbose_name="入学日期", blank=True, null=True)
    nationality = models.CharField(max_length=50, blank=True, null=True, verbose_name="国籍")
    major = models.CharField(max_length=100, verbose_name="专业")
    
    # 新增文件字段
    # avatar = models.ImageField(upload_to='avatars/%Y/%m/%d/', blank=True, null=True, verbose_name="头像")
    # resume = models.FileField(upload_to='resumes/%Y/%m/%d/', blank=True, null=True, verbose_name="简历")
    # video = models.FileField(upload_to='videos/%Y/%m/%d/', blank=True, null=True, verbose_name="介绍视频")
    avatar = models.CharField(max_length=255, blank=True, null=True, verbose_name="头像URL")
    resume = models.CharField(max_length=255, blank=True, null=True, verbose_name="简历URL")
    video = models.CharField(max_length=255, blank=True, null=True, verbose_name="视频URL")
    data_file = models.CharField(max_length=255, blank=True, null=True, verbose_name="数据文件URL")
    report_pdf = models.CharField(max_length=255, blank=True, null=True, verbose_name="报告PDF URL")    
    user = models.OneToOneField(
        'Users', 
        on_delete=models.CASCADE, 
        related_name='student_profile',
        blank=True, 
        null=True,
        verbose_name="关联用户"
    )
    
    updatetime = models.DateTimeField(blank=True, null=True, verbose_name="更新时间")
    
    class Meta:
        db_table = 'student'

    def __str__(self):
        return f"{self.name} ({self.numid})"

# 2025-12-02 批量导入专用学生表，对应 MySQL: student_bulk
class BulkStudent(models.Model):
    id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=30)   # 登录名 / 学号
    name = models.CharField(max_length=30)
    gender = models.CharField(max_length=10, blank=True, null=True)
    # 2025-12-05 批量导入：学生年龄
    age = models.IntegerField(blank=True, null=True)
    birth = models.DateField(blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)
    address = models.CharField(max_length=255, blank=True, null=True)
    university = models.CharField(max_length=50, blank=True, null=True)
    depart = models.CharField(max_length=50, blank=True, null=True)
    nationality = models.CharField(max_length=50, blank=True, null=True)  # 2025-12-02 批量导入：民族
    major = models.CharField(max_length=100, blank=True, null=True)
    enrollment = models.DateField(blank=True, null=True)
    # 2025-12-05 新增：班级名称
    class_name = models.CharField(max_length=50, blank=True, null=True) 

    class Meta:
        db_table = 'student_bulk'
        managed = False  # 不让 Django 通过迁移改这张表

    def __str__(self):
        return f"{self.name} ({self.username})"

class Teacher(models.Model):
    teacherid = models.IntegerField(primary_key=True)
    userid = models.ForeignKey('Users', models.DO_NOTHING, db_column='userid')
    name = models.CharField(max_length=30)
    gender = models.CharField(max_length=10)
    birth = models.DateField(blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)
    departid = models.IntegerField(blank=True, null=True)
    hiredate = models.DateTimeField(blank=True, null=True)
    createtime = models.DateTimeField()
    updatetime = models.DateTimeField()
    isdeleted = models.IntegerField()
    remark = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        # managed = False
        db_table = 'teacher'


class TestLog(models.Model):
    test_id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=255)
    time = models.DateTimeField()
    address = models.CharField(max_length=255)

    class Meta:
        # managed = False
        db_table = 'test_log'


class Testitem(models.Model):
    itemid = models.AutoField(primary_key=True, db_comment='项目 ID')
    name = models.CharField(max_length=50, db_comment='项目名称（如听力）')
    description = models.TextField(db_comment='描述（评分细则）')
    uploaderid = models.IntegerField(db_comment='上传者 ID（教师）')
    createtime = models.DateTimeField(db_comment='创建时间')
    updatetime = models.DateTimeField(db_comment='修改时间')
    isdeleted = models.PositiveIntegerField(db_comment='是否删除')

    class Meta:
        # managed = False
        db_table = 'testitem'


class Testrecord(models.Model):
    testid = models.AutoField(primary_key=True, db_comment='检测记录 ID')
    userid = models.IntegerField(db_comment='检测者 ID')
    itemid = models.IntegerField(db_comment='检测项目 ID')
    score = models.DecimalField(max_digits=5, decimal_places=2, db_comment='成绩')
    videourl = models.CharField(max_length=255, db_comment='视频地址')
    dataurl = models.CharField(max_length=255, db_comment='数据地址')
    testtime = models.DateTimeField(db_comment='检测时间')
    createtime = models.DateTimeField(db_comment='创建时间')
    updatetime = models.DateTimeField(db_comment='修改时间')
    isdeleted = models.PositiveIntegerField(db_comment='是否删除')
    remark = models.CharField(max_length=255, blank=True, null=True, db_comment='备注')

    class Meta:
        # managed = True
        db_table = 'testrecord'


# class Users(models.Model):
#     id = models.AutoField(primary_key=True)  # 显式定义主键字段
#     username = models.CharField(max_length=30)
#     password = models.CharField(max_length=60)
#     usertype = models.CharField(max_length=10)
#     # userid = models.AutoField(primary_key=True)
#     email = models.CharField(max_length=50)
#     createtime = models.DateTimeField()
#     phone = models.CharField(max_length=20)
#     updatetime = models.DateTimeField()
#     isdeleted = models.IntegerField(blank=True, null=True)
#     remark = models.CharField(max_length=255, blank=True, null=True)

#     class Meta:
#         # managed = False
#         db_table = 'users'
        
class UserTOTP(models.Model):
    user = models.ForeignKey(Users, on_delete=models.CASCADE)
    secret_key = models.CharField(max_length=32, unique=True)
    last_verified = models.DateTimeField(null=True, blank=True)
    enabled = models.BooleanField(default=False)


# class UserTOTP(models.Model):
#     user = models.OneToOneField(
#         settings.AUTH_USER_MODEL,  # 使用自定义用户模型
#         on_delete=models.CASCADE
#     )
#     secret_key = models.CharField(max_length=32, unique=True)
#     last_verified = models.DateTimeField(null=True, blank=True)
#     enabled = models.BooleanField(default=False)

class UsedTOTPCode(models.Model):
    """
    存储已使用的TOTP验证码，防止重放攻击
    """
    # user = models.ForeignKey(
    #     settings.AUTH_USER_MODEL,
    #     on_delete=models.CASCADE,
    #     related_name='used_totp_codes',
    #     verbose_name="用户"
    # )
    user = models.ForeignKey(Users, on_delete=models.CASCADE)
    code = models.CharField(
        max_length=6,
        verbose_name="验证码",
        help_text="6位TOTP验证码"
    )
    used_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="使用时间"
    )
    ip_address = models.GenericIPAddressField(
        verbose_name="IP地址",
        help_text="使用该验证码的客户端IP地址"
    )
    user_agent = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        verbose_name="用户代理",
        help_text="客户端的User-Agent信息"
    )
    purpose = models.CharField(
        max_length=50,
        choices=[
            ('login', '登录'),
            ('transaction', '交易'),
            ('security_change', '安全设置变更'),
            ('other', '其他')
        ],
        default='login',
        verbose_name="用途"
    )
    
    class Meta:
        verbose_name = "已使用的TOTP验证码"
        verbose_name_plural = "已使用的TOTP验证码"
        indexes = [
            models.Index(fields=['user', 'used_at']),
            models.Index(fields=['code', 'used_at']),
        ]
        constraints = [
            # 确保同一验证码对同一用户不会在短时间内重复记录
            models.UniqueConstraint(
                fields=['user', 'code'],
                condition=models.Q(used_at__gte=timezone.now() - timezone.timedelta(minutes=10)),
                name='unique_code_per_user_10min'
            )
        ]
    
    def __str__(self):
        return f"{self.user} - {self.code} at {self.used_at.strftime('%Y-%m-%d %H:%M')}"
    
    def clean(self):
        """
        验证模型数据
        """
        # 验证码必须是6位数字
        if len(self.code) != 6 or not self.code.isdigit():
            raise ValidationError("验证码必须是6位数字")
        
        # 防止添加过时的记录
        if self.used_at < timezone.now() - timezone.timedelta(minutes=10):
            raise ValidationError("无法添加超过10分钟前的使用记录")
        
        super().clean()
    
    def save(self, *args, **kwargs):
        """
        保存前自动清理数据
        """
        self.full_clean()
        super().save(*args, **kwargs)
