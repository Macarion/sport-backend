"""
student_bulk 表操作模块
独立文件，不影响原有逻辑
"""
from django.db import models
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json


# ========== 模型定义 ==========
class StudentBulk(models.Model):
    """对应 student_bulk 表"""
    id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=30)      # 登录名/学号
    name = models.CharField(max_length=30)          # 姓名
    gender = models.CharField(max_length=10, blank=True, null=True)
    birth = models.DateField(blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)
    address = models.CharField(max_length=255, blank=True, null=True)
    university = models.CharField(max_length=50, blank=True, null=True)
    depart = models.CharField(max_length=50, blank=True, null=True)
    nationality = models.CharField(max_length=50, blank=True, null=True)
    major = models.CharField(max_length=100, blank=True, null=True)
    enrollment = models.DateField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'student_bulk'
        managed = False  # 不让 Django 管理表结构


# ========== 字段映射 ==========
def map_frontend_to_db(frontend_data, username):
    """前端字段 -> 数据库字段映射"""
    return {
        'username': username,                              # 学号 = 登录名
        'name': frontend_data.get('name'),
        'gender': frontend_data.get('gender'),
        'birth': frontend_data.get('birth') or None,
        'phone': frontend_data.get('phone') or None,
        'email': frontend_data.get('email') or None,
        'address': frontend_data.get('address') or None,
        'university': frontend_data.get('Universityid'),   # 前端 Universityid -> 数据库 university
        'depart': frontend_data.get('departid'),           # 前端 departid -> 数据库 depart
        'nationality': frontend_data.get('nationality') or None,
        'major': frontend_data.get('major'),
        'enrollment': frontend_data.get('enrollment') or None,
    }


def map_db_to_frontend(db_record):
    """数据库字段 -> 前端字段映射"""
    return {
        'numid': db_record.username,                       # 数据库 username -> 前端 numid
        'name': db_record.name or '',
        'gender': db_record.gender or '',
        'birth': str(db_record.birth) if db_record.birth else '',
        'phone': db_record.phone or '',
        'email': db_record.email or '',
        'address': db_record.address or '',
        'Universityid': db_record.university or '',        # 数据库 university -> 前端 Universityid
        'departid': db_record.depart or '',                # 数据库 depart -> 前端 departid
        'nationality': db_record.nationality or '',
        'major': db_record.major or '',
        'enrollment': str(db_record.enrollment) if db_record.enrollment else '',
    }


# ========== 接口视图 ==========
class StudentBulkSaveView(APIView):
    """保存学生信息到 student_bulk 表"""
    
    def post(self, request):
        try:
            username = request.data.get('username')
            student_form = request.data.get('studentForm')
            
            if not username:
                return Response({'error': '缺少用户名'}, status=status.HTTP_400_BAD_REQUEST)
            if not student_form:
                return Response({'error': '缺少学生信息'}, status=status.HTTP_400_BAD_REQUEST)
            
            # 字段映射
            db_data = map_frontend_to_db(student_form, username)
            
            # 更新或创建记录
            obj, created = StudentBulk.objects.update_or_create(
                username=username,
                defaults=db_data
            )
            
            action = '创建' if created else '更新'
            return Response({
                'success': True, 
                'message': f'学生信息{action}成功'
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class StudentBulkGetView(APIView):
    """从 student_bulk 表获取学生信息"""
    
    def post(self, request):
        try:
            username = request.data.get('username')
            
            if not username:
                return Response({'error': '缺少用户名'}, status=status.HTTP_400_BAD_REQUEST)
            
            try:
                student = StudentBulk.objects.get(username=username)
                data = map_db_to_frontend(student)
                return Response(data, status=status.HTTP_200_OK)
            except StudentBulk.DoesNotExist:
                # 新用户，返回空数据，但学号填充 username
                return Response({
                    'numid': username,
                    'name': '',
                    'gender': '',
                    'birth': '',
                    'phone': '',
                    'email': '',
                    'address': '',
                    'Universityid': '',
                    'departid': '',
                    'nationality': '',
                    'major': '',
                    'enrollment': '',
                }, status=status.HTTP_200_OK)
                
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def get_student_info_for_report(username):
    """
    供 PDF 报告生成模块调用
    返回：姓名、性别、学号、学校（班级取默认）
    """
    try:
        student = StudentBulk.objects.get(username=username)
        return {
            'name': student.name or username,
            'gender': student.gender or '男',
            'student_id': student.username,
            'university': student.university or '北京交通大学',
            'class_name': '学硕1班',  # 班级取默认
        }
    except StudentBulk.DoesNotExist:
        # 没有填写信息，返回默认值
        return {
            'name': username,
            'gender': '男',
            'student_id': username,
            'university': '北京交通大学',
            'class_name': '学硕1班',
        }

