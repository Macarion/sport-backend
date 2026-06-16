# api/serializers.py
from rest_framework import serializers
from .models import Teacher, Class, Inform, Users, Manager
class TeacherSerializer(serializers.ModelSerializer):
    class Meta:
        model = Teacher
        fields = [
            'teacherid', 'userid', 'name', 'gender', 'birth', 'phone', 'email',
            'departid', 'hiredate', 'createtime', 'updatetime', 'isdeleted', 'remark'
        ]

class ClassSerializer(serializers.ModelSerializer):
    class Meta:
        model = Class
        fields = ['classid', 'classname', 'teacherid', 'createtime', 'updatetime', 'isdeleted', 'remark']

class InformSerializer(serializers.ModelSerializer):
    class Meta:
        model = Inform
        fields = ['informid', 'content', 'classid', 'teacherid', 'uploadtime', 'isdeleted', 'remark']
        extra_kwargs = {
            'uploadtime': {'format': '%Y-%m-%d %H:%M:%S'}
        }

class ManagerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Manager
        fields = [
            'managerid', 'userid', 'name', 'gender', 'email', 'phone',
            'permissionid', 'hiredate', 'createtime', 'updatetime', 'isdeleted', 'remark'
        ]
        extra_kwargs = {
            'createtime': {'format': '%Y-%m-%d %H:%M:%S'},
            'updatetime': {'format': '%Y-%m-%d %H:%M:%S'},
            'hiredate': {'format': '%Y-%m-%d %H:%M:%S'}
        }
        
class ChangePasswordSerializer(serializers.Serializer):
    userid = serializers.IntegerField()  # 改为 IntegerField，与 Users.userid 匹配
    newpassword = serializers.CharField(min_length=8, max_length=128)

    def validate_newpassword(self, value):
        if len(value) < 8:
            raise serializers.ValidationError("Password must be at least 8 characters long")
        return value