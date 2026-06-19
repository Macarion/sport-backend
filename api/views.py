from django.shortcuts import render
from django.http import JsonResponse,HttpResponseBadRequest
# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.handlers.asgi import ASGIRequest
from rest_framework.decorators import api_view

from api.models import  Class,Student   # 假设你有一个Student模型
# api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from .models import Teacher, Class, Inform, Users, Manager, BulkStudent
from .serializers import TeacherSerializer, ClassSerializer, InformSerializer, ManagerSerializer, ChangePasswordSerializer
from django.utils import timezone
import subprocess, signal, os, sys, time, psutil,threading,bcrypt,json,shutil,requests
from typing import Optional, List, Dict, Any
from . import process_manager as pm
from pathlib import Path
from typing import Optional
from .pdf_report import PDFGenerator
from .situp_pdf import SitupPDFGenerator
from .pullup_pdf import PullupPDFGenerator
from django.http import FileResponse, Http404
from django.conf import settings
from typing import Union
import pyotp
from rest_framework import status
from django.contrib.auth import authenticate
from .models import UserTOTP,Users
from rest_framework_simplejwt.tokens import RefreshToken
import jwt
from django.conf import settings
from datetime import datetime, timedelta
from django.contrib.auth.models import User
from security.permissions import TOTPVerified
from rest_framework.exceptions import PermissionDenied
from pyotp import TOTP
import logging
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.authentication import JWTAuthentication
from .utils.auth import TOTPPermission
from .security import require_totp_verification, verify_totp  # 导入安全模块
from rest_framework.exceptions import PermissionDenied
from api.security import generate_totp_secret, require_totp_verification
import qrcode
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.authentication import JWTAuthentication
from io import BytesIO
from django.db import connection  # 2025-12-20 成绩分析：使用原生 SQL 查询 testrecord 成绩记录
from django.db import transaction
from api.authentication import RemoteIntrospectJWTAuthentication
import requests  # 2026-01-08 视频下载：后端通过 HTTP 调用媒体服务器 best-video 接口
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from django.core.validators import validate_email

from api.user_manager import UserManager
from api.config import CAMERA_INDEX
from django.http import JsonResponse, HttpResponse

from .sitreach import (
    sitreach_start,
    sitreach_process_frame,
    sitreach_fetch_inc_data,
    sitreach_stop,
    sitreach_get_latest_frame,
    sitreach_start_local_camera,
    sitreach_stop_local_camera,
)

# import oss2
_auth_http = requests.Session()
_auth_http.trust_env = False  # 不使用系统代理

User = get_user_model()
def get_device_id(request) -> Optional[str]:
    """
    Extract device identifier from headers for embedding into JWT claims.
    """
    return request.headers.get('X-Client-Device') or request.META.get('HTTP_X_CLIENT_DEVICE')


def _resolve_student_upload_userid(raw_value: Optional[str]) -> Optional[int]:
    value = str(raw_value).strip() if raw_value is not None else ""
    if not value:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        pass

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM users
                WHERE username = %s
                  AND COALESCE(isdeleted, 0) = 0
                  AND usertype = 'student'
                LIMIT 1
                """,
                [value],
            )
            row = cursor.fetchone()
            return row[0] if row else None
    except Exception:
        return None
def make_password(password):
    # 使用bcrypt加密密码
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
DATA_FILE: Path = Path(__file__).resolve().parent / "runtime_data.json"
PULLUP_DATA_FILE = Path(__file__).resolve().parent / "pullup_runtime_data.json"
_JSON_LOCK = threading.Lock()         # 并发写锁
BASE_DIR: Path = Path(__file__).resolve().parent.parent
REPORT_DIR: Path = BASE_DIR / "report_pdf/"

# === 需要启动的脚本位置（务必使用绝对路径） ===
SCRIPT = Path(r"..\backend\api\new_situp.py").resolve()
PULLUP_SCRIPT = Path(r"..\backend\api\new_pullup.py").resolve()
TTS_SCRIPT = Path(r"..\backend\api\tts.py").resolve()

_JSON_LOCK = threading.Lock()
# 记录子进程 PID 的文件
PID_FILE = SCRIPT.with_suffix(".pid")
PULLUP_PID_FILE = PULLUP_SCRIPT.with_suffix(".pid")
LOG_FILE  = SCRIPT.with_suffix(".log")

# ---------- 辅助函数 ----------
def _write_pid(pid: int, pid_file: Path = PID_FILE) -> None:
    pid_file.write_text(str(pid), encoding="utf-8")

def _read_pid(pid_file: Path = PID_FILE) -> Optional[int]:
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        return None

def _pid_alive(pid: Optional[int]) -> bool:
    return bool(pid) and psutil.pid_exists(int(pid))

def _read_latest() -> Optional[Dict[str, Any]]:
    """读取 JSON 文件最后一条记录。"""
    if not DATA_FILE.exists():
        return None
    try:
        data: List[Dict[str, Any]] = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        print(data)
        return data[-1] if data else None
    except json.JSONDecodeError:
        return None

def open_pdf_locally(path: Union[str, Path]) -> None:
    """
    在服务器本机打开指定 PDF（阻塞式；失败将抛异常）。
    Windows → os.startfile
    """
    pdf = Path(path).resolve()

    if not pdf.exists():
        raise FileNotFoundError(pdf)
    os.startfile(pdf)                      

# def _append_record(record: Dict[str, Any]) -> None:
#     """用新记录覆盖原有 JSON 内容（线程安全，仅保留最新一条）。"""
#     with _JSON_LOCK:
#         # 仍以数组形式保存，便于后续统一解析
#         DATA_FILE.write_text(
#             json.dumps([record], ensure_ascii=False, indent=2),
#             encoding="utf-8",
#         )
#
def _append_record(record: Dict[str, Any], file: Path = DATA_FILE) -> None:
    """
    用新记录覆盖原有 JSON 内容（线程安全，仅保留最新一条）。
    file: 可指定写入的文件，默认为 situp 的 DATA_FILE
    """
    with _JSON_LOCK:
        file.write_text(
            json.dumps([record], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

@csrf_exempt
def get_img(request):
    file_path = os.path.join(BASE_DIR, 'api', 'img1.jpeg')
    print(file_path)
    if not os.path.exists(file_path):
        raise Http404("Image not found")
    return FileResponse(
        open(file_path, 'rb'),
        content_type='image/jpeg',  # 明确指定 MIME 类型
        headers={'Cache-Control': 'no-cache'}
    )

def latest_frame(request):
    """
    直接把磁盘中最新生成的图片文件原封不动发给前端。
    假设文件名为 latest.jpg，存放在 settings.MEDIA_ROOT 目录里。
    """
    img_path = os.path.join('latest.jpg')
    if not os.path.exists(img_path):
        raise Http404("Image not found")

    # FileResponse 会以流式方式输出文件，节省内存
    return FileResponse(
        open(img_path, 'rb'),
        content_type='image/jpeg',      # 或 image/png
        filename='frame.jpg',           # 可选：Content-Disposition 中的文件名
        headers={'Cache-Control': 'no-cache'}  # 防浏览器缓存旧图
    )



@csrf_exempt
def start_script(request):
    """启动脚本（若已在运行则返回 'running'）。"""

    print(f"DEBUG: Request method: {request.method}")
    print(f"DEBUG: Request headers: {dict(request.headers)}")
    print(f"DEBUG: Raw request body: {request.body}")

    # ===== 新增 1：读取 username =====
    try:
        body = json.loads(request.body) if request.body else {}
        print(f"DEBUG: Received request body: {body}")
    except Exception as e:
        print(f"DEBUG: Failed to parse request body: {e}")
        body = {}

    username = body.get("username")
    print(f"DEBUG: Extracted username: {username}")
    if not username:
        print("DEBUG: Username is missing or empty, returning error")
        return JsonResponse(
            {"status": "error", "message": "username is required"},
            status=400,
        )

    pid = _read_pid()

    # 若脚本已在运行，直接返回
    if pid and _pid_alive(pid):
        return JsonResponse(
            {"status": "running", "pid": pid, "username": username}
        )



    DETACHED_PROCESS       = subprocess.DETACHED_PROCESS
    CREATE_NEW_PROCESS_GRP = subprocess.CREATE_NEW_PROCESS_GROUP
    creation_flags         = DETACHED_PROCESS | CREATE_NEW_PROCESS_GRP

    try:
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT), f"--username={username}"],
            cwd=str(SCRIPT.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        _write_pid(proc.pid)
        return JsonResponse(
            {"status": "started", "pid": proc.pid, "username": username}
        )

    except FileNotFoundError as exc:
        return JsonResponse(
            {"status": "error", "message": f"脚本文件不存在: {exc}"},
            status=500,
        )

    except PermissionError:
        return JsonResponse(
            {"status": "error", "message": "权限不足，无法执行脚本"},
            status=500,
        )

    except subprocess.SubprocessError as exc:
        return JsonResponse(
            {"status": "error", "message": f"启动子进程失败: {exc}"},
            status=500,
        )

    except Exception as exc:
        return JsonResponse(
            {"status": "error", "message": f"未知错误: {exc}"},
            status=500,
        )
    

def stop_script(request):
    pid = _read_pid()
    # record = {
    #     "nums": [],
    #     "num": 0,
    #     "num_all": 0,
    #     "timestamps": 0,
    #     "angles": [],
    # }
    # _append_record(record, file=DATA_FILE)
    

    if not _pid_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        return JsonResponse({"status": "not running"})

    flag = SCRIPT.with_suffix(".stop.flag")

    # A) 外部停机信号：写标记文件 + 尝试发控制台中断
    try:
        flag.write_text("1", encoding="utf-8")
    except Exception:
        pass

    if os.name == "nt":
        try:
            # 发送 CTRL_BREAK；是否送达取决于启动方式/控制台
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        except Exception:
            pass
    else:
        try:
            os.kill(pid, signal.SIGINT)
        except Exception:
            pass

    # B) 等待脚本自行 release()/写尾（建议 8~12 秒）
    forced = False
    try:
        psutil.Process(pid).wait(timeout=12)
    except Exception:
        forced = True
        # C) 兜底强杀（仍可能损坏；所以关键还是靠 stop.flag + 脚本里 finally）
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
        else:
            os.kill(pid, signal.SIGKILL)

    # D) 清理
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        flag.unlink(missing_ok=True)
    except Exception:
        pass

    return JsonResponse({"status": "stopped", "forced": forced})
    
def latest_data(request):
    """
    返回 runtime_data.json 中最新记录。
    返回示例:
        {
          "temperature": 23.5,
          "humidity": 0.62,
          "timestamp": "2025-04-30T13:25:41"
        }
    若无数据:
        {"status": "empty"}
    """
    
    latest = _read_latest()

    if latest is None:
        return JsonResponse({"status": "empty"})
    return JsonResponse(latest)


# ---------- 引体向上 相关接口 ----------
def _read_pullup_pid():
    if PULLUP_PID_FILE.exists():
        return int(PULLUP_PID_FILE.read_text())
    return None


def _write_pullup_pid(pid: int):
    PULLUP_PID_FILE.write_text(str(pid))


def _read_pullup_latest():
    if PULLUP_DATA_FILE.exists():
        import json
        with PULLUP_DATA_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list) and data:
                return data[-1]
    return None


def start_pullup(request):
    """启动 pullup 脚本（若已在运行则返回 'running'）。"""
    pid = _read_pullup_pid()
    username = request.GET.get("username") or request.POST.get("username")
    upload_userid = _resolve_student_upload_userid(username)
    if pid and _pid_alive(pid):
        return JsonResponse({"status": "running", "pid": pid, "userid": upload_userid})

    DETACHED_PROCESS = subprocess.DETACHED_PROCESS
    CREATE_NEW_PROCESS_GRP = subprocess.CREATE_NEW_PROCESS_GROUP
    creation_flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GRP

    try:
        command = [sys.executable, str(PULLUP_SCRIPT)]
        if upload_userid is not None:
            command.extend(["--userid", str(upload_userid)])
        proc = subprocess.Popen(
            command,
            cwd=str(PULLUP_SCRIPT.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        _write_pullup_pid(proc.pid)
        return JsonResponse({"status": "started", "pid": proc.pid, "userid": upload_userid})

    except FileNotFoundError as exc:
        return JsonResponse(
            {"status": "error", "message": f"脚本文件不存在: {exc}"},
            status=500,
        )
    except PermissionError as exc:
        return JsonResponse(
            {"status": "error", "message": "权限不足，无法执行脚本"},
            status=500,
        )
    except subprocess.SubprocessError as exc:
        return JsonResponse(
            {"status": "error", "message": f"启动子进程失败: {exc}"},
            status=500,
        )
    except Exception as exc:
        return JsonResponse(
            {"status": "error", "message": f"未知错误: {exc}"},
            status=500,
        )


def stop_pullup(request):
    """优雅终止 pullup 脚本；若无法优雅退出则强制 Kill。"""
    pid = _read_pullup_pid()
    flag = PULLUP_SCRIPT.with_suffix(".stop.flag")
    

    if not pid or not _pid_alive(pid):
        PULLUP_PID_FILE.unlink(missing_ok=True)
        flag.unlink(missing_ok=True)
        return JsonResponse({"status": "not running"})

    try:
        try:
            flag.write_text("1", encoding="utf-8")
        except Exception:
            pass
        os.kill(pid, signal.CTRL_BREAK_EVENT)
        for _ in range(150):
            if not _pid_alive(pid):
                break
            time.sleep(0.1)
        if _pid_alive(pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=True)
    except Exception as exc:
        return JsonResponse({"status": "error", "detail": str(exc)}, status=500)
    finally:
        PULLUP_PID_FILE.unlink(missing_ok=True)
        flag.unlink(missing_ok=True)

    return JsonResponse({"status": "stopped"})


def latest_pullup_data(request):
    """返回 pullup_runtime_data.json 中最新记录。"""
    latest = _read_pullup_latest()
    if latest is None:
        return JsonResponse({"status": "empty"})
    return JsonResponse(latest)

def open_latest_report(request):
    """
    POST /api/reports/open_latest/
    在服务器本地打开 report_pdf/ 里最近修改的一份 PDF。
    客户端仅收到 204；真正的 PDF 会在服务器桌面弹出。
    """
    try:
        latest = max(
            REPORT_DIR.glob("*.pdf"),
            key=lambda p: p.stat().st_mtime
        )
    except ValueError:          # 目录为空
        return HttpResponseBadRequest("No report available")

    try:
        open_pdf_locally(latest)
    except Exception as exc:
        return HttpResponseBadRequest(f"Failed to open: {exc}")

    return JsonResponse(status=200)              # 无内容


@csrf_exempt
def gen_pdf(request):
    """生成 PDF 报告"""
    # 如果 username 非必填，可给默认值或允许 None
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            username = body.get('username')
            if not username:
                return JsonResponse({'code': 400, 'message': 'Missing username'})
            # 1. 读取最新数据
            latest = _read_latest()
            if latest is None:
                print("meiyou 数据为空")
                return JsonResponse({"status": "empty"})

            num: Optional[int] = latest.get("num")
            # 2. 生成 PDF 报告
            PDFReport = PDFGenerator()  # 实例化 PDFGenerator 对象
            PDFReport.pdf_report_gen(username, num)

            try:
                latest = max(
                    REPORT_DIR.glob("*.pdf"),
                    key=lambda p: p.stat().st_mtime
                )
            except ValueError:  # 目录为空
                return HttpResponseBadRequest("No report available")

            pdf_path = Path(latest).resolve()
            if not pdf_path.exists():
                print("文件不存在")
                return HttpResponseBadRequest("No pdf available")
            print("文件存在")
            return FileResponse(
                open(pdf_path, 'rb'),
                content_type='application/pdf',
                as_attachment=False  # 关键：允许浏览器直接预览而非下载
            )

            # try:
            #     open_pdf_locally(latest)
            # except Exception as exc:
            #     return HttpResponseBadRequest(f"Failed to open: {exc}")

            # return JsonResponse({"status": "success", "message": "PDF report generated"})
        except Exception as e:
            return JsonResponse({'code': 400, 'message': f'Invalid JSON: {str(e)}'})


@csrf_exempt
def gen_situp_pdf(request):
    """生成仰卧起坐单项 PDF 报告"""
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            username = body.get('username')
            if not username:
                return JsonResponse({'code': 400, 'message': 'Missing username'})
            
            # 读取仰卧起坐最新数据
            latest = _read_latest()
            if latest is None:
                return JsonResponse({"status": "empty", "message": "No situp data available"})

            num: Optional[int] = latest.get("num")
            
            # 生成单项 PDF 报告
            pdf_generator = SitupPDFGenerator()
            pdf_path = pdf_generator.generate_report(username, num)

            # 返回 PDF 文件
            pdf_file = Path(pdf_path).resolve()
            if not pdf_file.exists():
                return HttpResponseBadRequest("PDF file not found")
            
            return FileResponse(
                open(pdf_file, 'rb'),
                content_type='application/pdf',
                as_attachment=False
            )
        except Exception as e:
            return JsonResponse({'code': 400, 'message': f'Error: {str(e)}'})


@csrf_exempt
def gen_pullup_pdf(request):
    """生成引体向上单项 PDF 报告"""
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            username = body.get('username')
            if not username:
                return JsonResponse({'code': 400, 'message': 'Missing username'})
            
            # 读取引体向上最新数据
            latest = _read_pullup_latest()
            if latest is None:
                return JsonResponse({"status": "empty", "message": "No pullup data available"})

            num: Optional[int] = latest.get("num")
            
            # 生成单项 PDF 报告
            pdf_generator = PullupPDFGenerator()
            pdf_path = pdf_generator.generate_report(username, num)

            # 返回 PDF 文件
            pdf_file = Path(pdf_path).resolve()
            if not pdf_file.exists():
                return HttpResponseBadRequest("PDF file not found")
            
            return FileResponse(
                open(pdf_file, 'rb'),
                content_type='application/pdf',
                as_attachment=False
            )
        except Exception as e:
            return JsonResponse({'code': 400, 'message': f'Error: {str(e)}'})


def get_classes(request):
    try:
        classes = Class.objects.all().values()  # 获取所有班级信息
        return JsonResponse({
            'status': 'success',
            'data': list(classes)  # 转为list，JsonResponse可以返回
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })

def get_students(request):
    try:
        students = Student.objects.all().values()  # 获取所有学生信息
        return JsonResponse({
            'status': 'success',
            'data': list(students)  # 转为list，JsonResponse可以返回
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })

# # JWT生成辅助函数
# def generate_final_jwt_response(user):
#     """
#     生成最终访问JWT
#     """
#     payload = {
#         'user_id': user.userid,
#         'exp': timezone.now() + timedelta(minutes=60),  # 60分钟有效期
#         'iat': timezone.now()
#     }
    
#     access_token = jwt.encode(
#         payload,
#         settings.SECRET_KEY,  # 使用Django项目密钥
#         algorithm='HS256'
#     )
    
#     refresh_payload = {
#         'user_id': user.userid,
#         'exp': timezone.now() + timedelta(days=1),  # 1天有效期
#         'iat': timezone.now()
#     }
    
#     refresh_token = jwt.encode(
#         refresh_payload,
#         settings.SECRET_KEY,  # 使用Django项目密钥
#         algorithm='HS256'
#     )
    
#     return Response({
#         "access": access_token,
#         "refresh": refresh_token
#     })


# def generate_temp_jwt(user):
#     """
#     生成用于TOTP验证阶段的短期临时JWT
#     有效时间短（5分钟），仅包含必要的最小用户信息
#     """
#     payload = {
#         'user_id': user.userid,
#         'exp': timezone.now() + timedelta(minutes=5),  # 5分钟有效期
#         'iat': timezone.now(),
#         'purpose': 'totp_verification'  # 明确令牌用途
#     }
    
#     return jwt.encode(
#         payload,
#         settings.SECRET_KEY,  # 使用Django项目密钥
#         algorithm='HS256'
#     )

def generate_temp_jwt(user):
    """
    生成临时 JWT Token，用于 TOTP 验证阶段
    """
    refresh = RefreshToken.for_user(user)
    return str(refresh.access_token)

def generate_final_jwt_response(user):
    """
    生成最终的 JWT Token 响应
    """
    refresh = RefreshToken.for_user(user)
    return {
        "access": str(refresh.access_token),
        "refresh": str(refresh)
    }


class TOTPSetupInfoView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        获取 TOTP 设置信息，包括二维码 URL 和临时 Token。
        """
        print("SetupTOTPView")
        user = request.user
        print(user)
        try:
            # 检查用户是否已启用 TOTP
            print('jiancha用户是否已启用 TOTP')
            if user.totp_enabled and user.totp_secret and user.totp_setup_completed:
                return Response({
                    'error': 'TOTP 已启用，无需重新设置'
                }, status=status.HTTP_400_BAD_REQUEST)
            print('jiancha用户是否已启用 TOTP')
            # 生成新的 TOTP 密钥（如果尚未设置）
            if not user.totp_secret:
                user.totp_secret = pyotp.random_base32()
                user.save(update_fields=['totp_secret'])

            # 生成 TOTP URI
            totp = pyotp.TOTP(user.totp_secret)
            totp_uri = totp.provisioning_uri(
                name=user.username,
                issuer_name="智慧体测@BJTU"  # 替换为你的应用名称
            )

            # 生成二维码
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(totp_uri)
            qr.make(fit=True)
            img = qr.make_image(fill="black", back_color="white")

            # 将二维码转换为 base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            qr_code_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            qr_code_url = f"data:image/png;base64,{qr_code_base64}"

            # 生成临时 JWT Token
            temp_token = generate_temp_jwt(user)

            return Response({
                'qr_code_url': qr_code_url,
                'temp_token': temp_token,
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
from rest_framework.permissions import IsAuthenticated, AllowAny
class VerifyTOTPView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        """
        验证 TOTP 码并生成最终 JWT Token。
        """
        temp_token = request.data.get('temp_token')
        code = request.data.get('code')

        if not temp_token or not code:
            return Response({
                'error': 'temp_token 和 code 不能为空'
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 验证临时 JWT Token
            payload = jwt.decode(
                temp_token,
                settings.SECRET_KEY,
                algorithms=['HS256']
            )
            user = Users.objects.get(id=payload['user_id'])

            # 检查是否启用 TOTP
            if not user.totp_enabled or not user.totp_secret:
                return Response({
                    'error': 'TOTP 未启用或密钥未设置'
                }, status=status.HTTP_400_BAD_REQUEST)

            # 验证 TOTP 码
            totp = pyotp.TOTP(user.totp_secret)
            if totp.verify(code, valid_window=1):
                # 记录 TOTP 使用时间
                user.record_totp_usage()
                # 生成最终 JWT Token
                jwt_response = generate_final_jwt_response(user)
                return Response(jwt_response, status=status.HTTP_200_OK)
            else:
                return Response({
                    'error': '无效的 TOTP 码'
                }, status=status.HTTP_400_BAD_REQUEST)
        except jwt.ExpiredSignatureError:
            return Response({
                'error': '临时 Token 已过期'
            }, status=status.HTTP_401_UNAUTHORIZED)
        except jwt.InvalidTokenError:
            return Response({
                'error': '无效的 Token'
            }, status=status.HTTP_401_UNAUTHORIZED)
        except Users.DoesNotExist:
            return Response({
                'error': '用户不存在'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)       

import base64
import secrets
class SetupTOTPView(APIView):
    # permission_classes = [IsAuthenticated]
    # authentication_classes = [JWTAuthentication]


    def post(self, request):
        """
        验证 TOTP 码并启用 TOTP。
        """
        print("SetupTOTPView")
        temp_token = request.data.get('temp_token')
        code = request.data.get('code')
        print(temp_token, code)
        if not temp_token or not code:
            print('temp_token 和 code 不能为空')
            return Response({
                'error': 'temp_token 和 code 不能为空'
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 验证临时 JWT Token
            payload = jwt.decode(
                temp_token,
                settings.SECRET_KEY,
                algorithms=['HS256']
            )
            user = Users.objects.get(id=payload['user_id'])
            print("jwt验证成功")
            # 检查 TOTP 密钥
            if not user.totp_secret:
                print('TOTP 密钥未设置')
                return Response({
                    'error': 'TOTP 密钥未设置'
                }, status=status.HTTP_400_BAD_REQUEST)

            # 验证 TOTP 码
            print('验证 TOTP 码')
            print(user)
            totp = pyotp.TOTP(user.totp_secret)
            # 获取当前时间（确保使用服务器时间）
            current_time = timezone.now()

            if totp.verify(code, valid_window=1):
                user.enable_totp(user.totp_secret)  # 使用 Users 模型的 enable_totp 方法
                user.totp_setup_completed = True  # 标记 TOTP 设置完成
                user.save(update_fields=['totp_enabled', 'totp_setup_completed'])
                return Response({
                    'message': 'TOTP 设置成功'
                }, status=status.HTTP_200_OK)
            else:
                print('无效的 TOTP 码')
                return Response({
                    'error': '无效的 TOTP 码'
                }, status=status.HTTP_400_BAD_REQUEST)
        except jwt.ExpiredSignatureError:
            return Response({
                'error': '临时 Token 已过期'
            }, status=status.HTTP_401_UNAUTHORIZED)
        except jwt.InvalidTokenError:
            return Response({
                'error': '无效的 Token'
            }, status=status.HTTP_401_UNAUTHORIZED)
        except Users.DoesNotExist:
            return Response({
                'error': '用户不存在'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class RegisterView(APIView):
    """
    POST /api/register/
    请求体:
        {
          "username": "alice",
          "password": "PlainTextPwd!",
          "email":    "alice@example.com",
          "phone":    "13800000000",
          "usertype": "normal"          # 可选，默认 "normal"
        }
    成功返回:
        {"userid": 12, "message": "register success"}
    """
    permission_classes = [AllowAny]
    authentication_classes = []
    def post(self, request):
        # 1. 读取参数
        username = request.data.get("username")
        password = request.data.get("password")
        usertype = request.data.get("role")
        print(username, password, usertype)

        # 2. 必填项检查
        if not all([username, password]):
            return Response(
                {"error": "username、password、 均为必填参数"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 4. 用户名 唯一性检查
        if Users.objects.filter(username=username, isdeleted=0).exists():
            return Response(
                {"error": "用户名已存在"},
                status=status.HTTP_409_CONFLICT
            )

        now = timezone.now()
        user = Users.objects.create(
            username    = username,
            password    = make_password(password),   # 加密
            usertype    = usertype,
            createtime  = now,
            updatetime  = now,
            isdeleted   = 0,
        )
        user.enable_totp(None)
        # 6. 返回成功
        return Response(
            {"userid": user.id, "message": "register success"},
            status=status.HTTP_201_CREATED
        )

class TeacherInfoView(APIView):
    def get(self, request):
        userid = request.query_params.get('userid')
        if not userid:
            return Response({"error": "userid is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            teacher = get_object_or_404(Teacher, userid=userid)
            serializer = TeacherSerializer(teacher)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Teacher.DoesNotExist:
            return Response({"error": "Teacher not found"}, status=status.HTTP_404_NOT_FOUND)

class TeacherUpdateView(APIView):
    def post(self, request):
        userid = request.data.get('userid')
        if not userid:
            return Response({"error": "userid is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            teacher = get_object_or_404(Teacher, userid=userid)
            serializer = TeacherSerializer(teacher, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save(updatetime=timezone.now())
                return Response({"success": True, "message": "Teacher info updated"}, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Teacher.DoesNotExist:
            return Response({"error": "Teacher not found"}, status=status.HTTP_404_NOT_FOUND)

#安全验证test
class TeacherJoinClassView(APIView):
    permission_classes = [IsAuthenticated, TOTPVerified]
    def post(self, request):
        class_id = request.data.get('classId')
        teacher_id = request.data.get('teacherid')
        if not class_id or not teacher_id:
            return Response({"error": "classId and teacherid are required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            teacher = get_object_or_404(Teacher, teacherid=teacher_id)
            if Class.objects.filter(classid=class_id, isdeleted=0).exists():
                return Response({"error": "Class already exists"}, status=status.HTTP_400_BAD_REQUEST)
            
            new_class = Class(
                classid=class_id,
                classname=request.data.get('classname', f"Class {class_id}"),
                teacherid=teacher.teacherid,
                createtime=timezone.now(),
                updatetime=timezone.now(),
                isdeleted=0,
                remark=request.data.get('remark', '')
            )
            new_class.save()
            
            serializer = ClassSerializer(new_class)
            return Response({"success": True, "message": f"Class {class_id} created", "data": serializer.data}, status=status.HTTP_201_CREATED)
        except Teacher.DoesNotExist:
            return Response({"error": "Teacher not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


logger = logging.getLogger(__name__)
#安全验证test
# from .utils.authentication_debug import DebugJWTAuthentication
class TeacherInfoIDView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [RemoteIntrospectJWTAuthentication]

    def post(self, request):
        print("处理加入班级请求")
        # 1. 检查用户是否需要进行 TOTP 验证
        if require_totp_verification(request.user):
            # 2. 检查请求头中是否有 TOTP 令牌
            totp_token = request.headers.get('X-TOTP-Code')
            print(f"TOTP 令牌: {totp_token}")
            if not totp_token:
                # 返回需要 TOTP 验证的响应
                print("""需要TOTP验证""")
                return Response(
                    {
                        "code": "require_totp",
                        "message": "需要双重验证"
                    },
                    status=status.HTTP_403_FORBIDDEN
                )
            
            # 3. 验证 TOTP 令牌
            if not verify_totp(request.user, totp_token):
                print("""验证码无效""")
                return Response(
                    {
                        "code": "invalid_totp",
                        "message": "验证码无效"
                    },
                    status=status.HTTP_403_FORBIDDEN
                )
            
            # 4. 记录 TOTP 使用时间
            request.user.record_totp_usage()
        
        # 5. 执行业务逻辑（这里以加入班级为例）
        class_id = request.data.get('classId')
        print(f"加入班级 {class_id}")
        
        # 这里实现实际的业务逻辑
        # 例如: class_obj = Class.objects.get(id=class_id)
        #        class_obj.teachers.add(request.user)
        
        return Response(
            {
                "success": True,
                "message": "成功加入班级",
                "classId": class_id
            },
            status=status.HTTP_200_OK
        )          
    
class AddInformView(APIView):
    def post(self, request):
        teacher_id = request.data.get('teacherid')
        class_id = request.data.get('classid')
        content = request.data.get('informcontent')

        if not all([teacher_id, class_id, content]):
            return Response({"error": "teacherid, classid, and informcontent are required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            teacher = get_object_or_404(Teacher, teacherid=teacher_id, isdeleted=0)
            class_obj = get_object_or_404(Class, classid=class_id, isdeleted=0)

            new_inform = Inform(
                content=content,
                classid=class_id,
                teacherid=teacher_id,
                uploadtime=timezone.now(),
                isdeleted=0,
                remark=''
            )
            new_inform.save()

            serializer = InformSerializer(new_inform)
            return Response({"success": True, "message": "Notification added successfully", "data": serializer.data}, status=status.HTTP_201_CREATED)
        except Teacher.DoesNotExist:
            return Response({"error": "Teacher not found or has been deleted"}, status=status.HTTP_404_NOT_FOUND)
        except Class.DoesNotExist:
            return Response({"error": "Class not found or has been deleted"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# 管理员信息查询
class ManagerInfoView(APIView):
    def get(self, request):
        userid = request.query_params.get('userid')
        if not userid:
            return Response({"error": "userid is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            manager = get_object_or_404(Manager, userid=userid, isdeleted=0)
            serializer = ManagerSerializer(manager)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Manager.DoesNotExist:
            return Response({"error": "Manager not found or has been deleted"}, status=status.HTTP_404_NOT_FOUND)

# 管理员信息更新
class ManagerUpdateView(APIView):
    def post(self, request):
        userid = request.data.get('userid')
        if not userid:
            return Response({"error": "userid is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            manager = get_object_or_404(Manager, userid=userid, isdeleted=0)
            serializer = ManagerSerializer(manager, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save(updatetime=timezone.now())
                return Response({"success": True, "message": "Manager info updated"}, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Manager.DoesNotExist:
            return Response({"error": "Manager not found or has been deleted"}, status=status.HTTP_404_NOT_FOUND)
        
# 管理员强改密码
class ManagerChangePasswordView(APIView):
    def post(self, request):
        serializer = ChangePasswordSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        userid = serializer.validated_data['userid']
        newpassword = serializer.validated_data['newpassword']

        # 根据 userid 查询 Users 表
        try:
            user = get_object_or_404(Users, userid=userid, isdeleted=0)
        except Users.DoesNotExist:
            return Response({"error": "User not found or has been deleted"}, status=status.HTTP_404_NOT_FOUND)

        # 对新密码进行 bcrypt 哈希
        try:
            hashed_password = make_password(newpassword)
        except Exception as e:
            return Response({"error": f"Password hashing failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 更新 users 表的密码
        try:
            user.password = hashed_password
            user.updatetime = timezone.now()
            user.save()
            return Response({"success": True, "message": "Password updated successfully"}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


from rest_framework.throttling import UserRateThrottle

class TOTPStatusThrottle(UserRateThrottle):
    rate = '30/hour'  # 每个用户每小时最多30次

# backend/api/views.py
from django.core.cache import cache

class TOTPStatusView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """获取用户的 TOTP 状态"""
        return Response({
            'totp_enabled': request.user.totp_enabled,
            'has_secret': bool(request.user.totp_secret),
            'last_used': request.user.last_totp_used
        })
        
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework.permissions import AllowAny        
class TokenRefreshView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        refresh_token = request.data.get('refresh')
        if not refresh_token:
            return Response(
                {"error": "刷新令牌缺失"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # 验证并刷新令牌
            refresh = RefreshToken(refresh_token)
            new_access_token = str(refresh.access_token)

            # 可选: 如果需要返回新的刷新令牌
            # new_refresh_token = str(refresh)

            return Response({
                "access": new_access_token,
                # "refresh": new_refresh_token  # 如果需要同时返回新的刷新令牌
            }, status=status.HTTP_200_OK)

        except TokenError as e:
            # 处理令牌无效或过期的情况
            return Response(
                {"error": "令牌无效或已过期", "detail": str(e)},
                status=status.HTTP_401_UNAUTHORIZED
            )
        except Exception as e:
            # 处理其他意外错误
            return Response(
                {"error": "服务器错误", "detail": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# 远端鉴权版 TokenRefreshView（保留备用）
class RemoteTokenRefreshView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        refresh_token = request.data.get("refresh")
        if not refresh_token:
            return Response(
                {"error": "刷新令牌缺失"}, status=status.HTTP_400_BAD_REQUEST
            )

        device_id = get_device_id(request)
        if not device_id:
            return Response({"error": "缺少设备指纹"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            resp = _auth_http.post(
                settings.AUTH_REMOTE_REFRESH,
                data={"refresh": refresh_token},
                headers={"X-Client-Device": device_id},
                timeout=settings.AUTH_REMOTE_TIMEOUT,
                proxies={"http": None, "https": None},
            )
        except Exception as exc:
            return Response({"error": f"远端刷新失败: {exc}"}, status=status.HTTP_502_BAD_GATEWAY)

        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = {"error": resp.text or "remote refresh error"}
            return Response(detail, status=resp.status_code)

        payload = resp.json()
        user_id = payload.get("user_id")
        try:
            user = Users.objects.filter(id=user_id).first() if user_id else None
            if user:
                exp_ts = payload.get("access_exp")
                user.last_access_jti = payload.get("access_jti")
                user.last_access_exp = (
                    datetime.fromtimestamp(exp_ts, tz=timezone.utc) if exp_ts else None
                )
                user.last_login_at = timezone.now()
                user.save(
                    update_fields=[
                        "last_access_jti",
                        "last_access_exp",
                        "last_login_at",
                    ]
                )
        except Exception:
            pass

        return Response({"access": payload.get("access"), "refresh": payload.get("refresh")}, status=status.HTTP_200_OK)


class UserInfoView(APIView):
    def post(self, request):
        # 1. 获取用户名和 studentForm 数据
        username = request.data.get('username')
        student_form = request.data.get('studentForm')

        # 2. 验证用户名和 studentForm 是否存在
        if not username or not student_form:
            return Response(
                {"error": "缺少用户名或学生信息"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 3. 验证用户是否存在且未被删除
        try:
            user = Users.objects.get(username=username, isdeleted=0)
        except Users.DoesNotExist:
            return Response(
                {"error": "用户不存在或已被删除"},
                status=status.HTTP_404_NOT_FOUND
            )

        # 4. 获取或创建关联的 Student 记录
        try:
            student = user.student_profile  # 通过 OneToOneField 获取关联的 Student
        except Student.DoesNotExist:
            # 如果 Student 不存在，创建新记录
            student = Student(user=user)

        # 5. 更新 Student 模型的字段
        valid_fields = [
            'numid', 'name', 'gender', 'birth', 'phone', 'email', 'address',
            'Universityid', 'departid', 'enrollment', 'nationality', 'major'
        ]
        for field, value in student_form.items():
            if field in valid_fields and hasattr(student, field):
                setattr(student, field, value or None)  # 将空值转换为 None 以符合模型定义

        # 6. 验证 numid 的唯一性
        if student_form.get('numid'):
            existing_student = Student.objects.filter(numid=student_form['numid']).exclude(user=user)
            if existing_student.exists():
                return Response(
                    {"error": "学号已存在"},
                    status=status.HTTP_400_BAD_REQUEST
                )

        # 7. 更新修改时间并保存
        student.updatetime = timezone.now()
        student.save()

        # 8. 返回成功响应
        return Response({
            "message": "学生信息更新成功",
            "updatetime": student.updatetime
        }, status=status.HTTP_200_OK)
        
#上传测试数据文件        
class UserTestingInfoView(APIView):
    def post(self, request):
        # 1. 获取用户名
        username = request.data.get('username')
        
        # 2. 验证用户是否存在且未被删除
        try:
            user = Users.objects.get(username=username, isdeleted=0)
        except Users.DoesNotExist:
            return Response({"error": "用户不存在或已被删除"}, status=status.HTTP_404_NOT_FOUND)
        
        # 3. 移除不需要更新的字段（如username）
        update_data = request.data.copy()
        update_data.pop('username', None)
        
        # 4. 更新所有提供的字段
        for field, value in update_data.items():
            if hasattr(user, field):
                setattr(user, field, value)
        
        # 5. 更新修改时间并保存
        user.updatetime = timezone.now()
        user.save()
        
        # 6. 返回成功响应
        return Response({
            "message": "用户信息更新成功",
            "updatetime": user.updatetime
        }, status=status.HTTP_200_OK)

#上传测试图像文件        
class UserTestingImgView(APIView):
    def post(self, request):
        # 1. 获取用户名
        username = request.data.get('username')
        
        # 2. 验证用户是否存在且未被删除
        try:
            user = Users.objects.get(username=username, isdeleted=0)
        except Users.DoesNotExist:
            return Response({"error": "用户不存在或已被删除"}, status=status.HTTP_404_NOT_FOUND)
        
        # 3. 移除不需要更新的字段（如username）
        update_data = request.data.copy()
        update_data.pop('username', None)
        
        # 4. 更新所有提供的字段
        for field, value in update_data.items():
            if hasattr(user, field):
                setattr(user, field, value)
        
        # 5. 更新修改时间并保存
        user.updatetime = timezone.now()
        user.save()
        
        # 6. 返回成功响应
        return Response({
            "message": "用户信息更新成功",
            "updatetime": user.updatetime
        }, status=status.HTTP_200_OK)
        
class getStudentInfoView(APIView):
    def post(self, request):
        # 1. 获取用户名
        username = request.data.get('username')
        print(username)
        # 2. 验证用户是否存在且未被删除
        try:
            user = Users.objects.get(username=username, isdeleted=False)
        except Users.DoesNotExist:
            return Response(
                {"error": "用户不存在或已被删除"}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        # 3. 检查用户类型是否为学生
        if user.usertype != 'student':
            return Response(
                {"error": "该用户不是学生类型"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 4. 获取学生信息
        try:
            # 通过一对一关系获取学生信息
            student = Student.objects.get(user=user)
            
            # 5. 构建返回数据
            student_data = {
                # 用户基本信息
                "username": user.username,
                "email": student.email,
                "phone": student.phone,
                "first_name": user.first_name,
                "last_name": user.last_name,
                
                # 学生特有信息
                "numid": student.numid,
                "name": student.name,
                "gender": student.gender,
                "birth": student.birth.isoformat() if student.birth else None,
                "address": student.address,
                "Universityid": student.Universityid,
                "departid": student.departid,
                "enrollment": student.enrollment.isoformat() if student.enrollment else None,
                "nationality": student.nationality,
                "major": student.major,
            }
            
            return Response(student_data, status=status.HTTP_200_OK)
            
        except Student.DoesNotExist:
            return Response(
                {"error": "未找到对应的学生信息"}, 
                status=status.HTTP_404_NOT_FOUND
            )

# 2025-12-02 批量导入：写入 student_bulk 表
class BulkStudentImportView(APIView):
    # 统一显式声明认证与权限，确保需要携带 JWT 访问
    authentication_classes = [RemoteIntrospectJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        students = request.data.get('students', [])
        success = 0
        errors = []

        for idx, s in enumerate(students, start=1):
            username = s.get('username')
            name = s.get('name')
            if not username or not name:
                errors.append(f'第 {idx} 条：缺少 username 或 name')
                continue

            try:
                # 2025-12-05 覆盖写：按 username update_or_create，避免重复导入产生多条记录
                # 新增class_name列，学生班级信息写入此列
                # 旧实现（单纯 create）保留在此供参考：
                # BulkStudent.objects.create(...)
                # 2025-12-05 批量导入：安全解析年龄字段
                age_raw = s.get('age')
                try:
                    age_value = int(age_raw) if age_raw not in (None, '',) else None
                except (TypeError, ValueError):
                    age_value = None
                obj, created = BulkStudent.objects.update_or_create(
                    username=username,
                    defaults={
                        'name': name,
                        'gender': s.get('gender') or '',
                        'age': age_value,
                        'birth': s.get('birth') or None,
                        'phone': s.get('phone') or '',
                        'email': s.get('email') or '',
                        'address': s.get('address') or '',
                        'university': s.get('university') or '',
                        'depart': s.get('depart') or '',
                        'nationality': s.get('nationality') or '',
                        'major': s.get('major') or '',
                        'enrollment': s.get('enrollment') or None,
                        'class_name': s.get('class_name') or '',
                    },
                )
                success += 1
            except Exception as e:
                errors.append(f'第 {idx} 条：{str(e)}')

        return Response(
            {
                'success_count': success,
                'failed_count': len(errors),
                'errors': errors,
            },
            status=status.HTTP_200_OK,
        )

# 2025-12-02 批量导入：按 username 查询 student_bulk，用于前端展示
class BulkStudentInfoView(APIView):
    authentication_classes = [RemoteIntrospectJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        username = request.query_params.get('username')
        if not username:
            return Response({'error': '缺少 username'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            s = BulkStudent.objects.get(username=username)
        except BulkStudent.DoesNotExist:
            # 没有记录就返回空对象
            return Response({}, status=status.HTTP_200_OK)

        return Response(
            {
                'username': s.username,
                'name': s.name,
                'gender': s.gender,
                'age': s.age,
                'birth': s.birth,
                'phone': s.phone,
                'email': s.email,
                'address': s.address,
                'university': s.university,
                'depart': s.depart,
                'nationality': s.nationality,
                'major': s.major,
                'enrollment': s.enrollment,
                # 2025-12-05 新增：班级名称
                'class_name': s.class_name,
            },
            status=status.HTTP_200_OK,
        )

# 2025-12-05 教师端学生列表：从 student_bulk 读取全部学生
class BulkStudentListView(APIView):
    authentication_classes = [RemoteIntrospectJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        qs = BulkStudent.objects.all().order_by('id')
        data = []
        for s in qs:
            data.append(
                {
                    "studentId": s.username,         # 学号（与登录名一致）
                    "name": s.name,
                    "gender": s.gender,
                    "age": s.age,                    # 2025-12-05：从 student_bulk 读取年龄
                    "class": s.class_name or "",
                }
            )
        return Response(data, status=status.HTTP_200_OK)

# 2026-01-27 教师端学生管理：单个学生信息修改 / 创建（基于 student_bulk）
class TeacherBulkStudentUpdateView(APIView):
    """
    教师端：新增 / 编辑单个学生的基础信息（基于 student_bulk 表）。

    约定：
    - studentId 等同于 username，作为唯一键；
    - 维护 student_bulk 中的完整基础字段（姓名 / 性别 / 年龄 / 民族 / 出生日期 / 电话 / 邮箱 / 家庭住址 / 学校 / 学院 / 专业 / 入学日期 / 班级）；
    - 若记录不存在则创建，存在则更新。
    """

    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def post(self, request):
        payload = request.data or {}
        student_id = payload.get("studentId") or payload.get("username")

        if not student_id:
            return Response(
                {"error": "缺少 studentId 参数"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        username = str(student_id).strip()
        name = (payload.get("name") or "").strip()
        gender = (payload.get("gender") or "").strip()
        nationality = (payload.get("nationality") or "").strip()
        class_name = (payload.get("class") or payload.get("class_name") or "").strip()
        phone = (payload.get("phone") or "").strip()
        email = (payload.get("email") or "").strip()
        address = (payload.get("address") or "").strip()
        university = (payload.get("university") or "").strip()
        depart = (payload.get("depart") or "").strip()
        major = (payload.get("major") or "").strip()
        enrollment_raw = payload.get("enrollment")
        birth_raw = payload.get("birth")
        age_raw = payload.get("age")

        try:
            age_value = int(age_raw) if age_raw not in (None, "",) else None
        except (TypeError, ValueError):
            age_value = None

        # 简单日期处理：空字符串视为 None，其余交给 Django 解析
        enrollment_value = enrollment_raw or None
        birth_value = birth_raw or None

        try:
            obj, created = BulkStudent.objects.update_or_create(
                username=username,
                defaults={
                    "name": name,
                    "gender": gender,
                    "age": age_value,
                    "class_name": class_name,
                    "nationality": nationality,
                    "birth": birth_value,
                    "phone": phone,
                    "email": email,
                    "address": address,
                    "university": university,
                    "depart": depart,
                    "major": major,
                    "enrollment": enrollment_value,
                },
            )
        except Exception as exc:
            return Response(
                {"error": f"保存学生信息失败: {str(exc)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "created": created,
                "student": {
                    "studentId": obj.username,
                    "name": obj.name,
                    "gender": obj.gender,
                    "age": obj.age,
                    "class": obj.class_name or "",
                    "nationality": obj.nationality or "",
                    "birth": obj.birth,
                    "phone": obj.phone or "",
                    "email": obj.email or "",
                    "address": obj.address or "",
                    "university": obj.university or "",
                    "depart": obj.depart or "",
                    "major": obj.major or "",
                    "enrollment": obj.enrollment,
                },
            },
            status=status.HTTP_200_OK,
        )


# 2026-01-27 教师端学生管理：删除单个 / 多个学生（仅从 student_bulk 表中删除）
class TeacherBulkStudentDeleteView(APIView):
    """
    教师端：删除学生基础信息记录。

    请求体示例：
        {"studentIds": ["20260001", "20260002"]}
    """

    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def post(self, request):
        ids = request.data.get("studentIds") or []

        if isinstance(ids, str):
            ids = [ids]

        ids = [str(i).strip() for i in ids if str(i).strip()]

        if not ids:
            return Response(
                {"error": "缺少 studentIds 参数"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            deleted_count, _ = BulkStudent.objects.filter(username__in=ids).delete()
        except Exception as exc:
            return Response(
                {"error": f"删除学生信息失败: {str(exc)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {"deleted": deleted_count, "studentIds": ids},
            status=status.HTTP_200_OK,
        )


class HeartbeatView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response({"status": "ok"})


# ==== Login/Refresh（支持本地和远程鉴权） ====
class LoginView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")
        if not username or not password:
            return Response(
                {"error": "username 和 password 均为必填项"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 检查是否启用远端鉴权
        use_remote = getattr(settings, "AUTH_REMOTE_ENABLED", False)

        if use_remote:
            # 远端鉴权模式
            device_id = get_device_id(request)
            if not device_id:
                return Response({"error": "缺少设备指纹"}, status=status.HTTP_400_BAD_REQUEST)

            try:
                print(f"[DEBUG] 远端鉴权 URL: {settings.AUTH_REMOTE_TOKEN}")
                resp = _auth_http.post(
                    settings.AUTH_REMOTE_TOKEN,
                    data={"username": username, "password": password},
                    headers={"X-Client-Device": device_id},
                    timeout=settings.AUTH_REMOTE_TIMEOUT,
                    proxies={"http": None, "https": None},
                )
                print(f"[DEBUG] 远端响应状态: {resp.status_code}")
                print(f"[DEBUG] 远端响应内容: {resp.text[:500] if resp.text else 'empty'}")
            except Exception as exc:
                return Response({"error": f"远端鉴权失败: {exc}"}, status=status.HTTP_502_BAD_GATEWAY)

            if resp.status_code != 200:
                try:
                    detail = resp.json()
                except Exception:
                    detail = {"error": resp.text or "remote auth error"}
                return Response(detail, status=resp.status_code)

            jwt_response = resp.json()
            user_id = jwt_response.get("user_id")
            user = None
            try:
                user = Users.objects.filter(id=user_id).first() if user_id else None
            except Exception:
                user = None
            if not user:
                user = Users.objects.filter(username=username, isdeleted=0).first()
            if user:
                try:
                    exp_ts = jwt_response.get("access_exp")
                    user.last_device_id = device_id
                    user.last_access_jti = jwt_response.get("access_jti")
                    user.last_access_exp = (
                        datetime.fromtimestamp(exp_ts, tz=timezone.utc) if exp_ts else None
                    )
                    user.last_login_at = timezone.now()
                    user.save(
                        update_fields=[
                            "last_device_id",
                            "last_access_jti",
                            "last_access_exp",
                            "last_login_at",
                        ]
                    )
                except Exception:
                    pass

        else:
            # 本地鉴权模式 - 使用原生 SQL 避免 Django ORM 字段问题
            from django.db import connection
            
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id, username, password, usertype FROM users WHERE username = %s AND isdeleted = 0",
                    [username]
                )
                row = cursor.fetchone()
            
            if not row:
                return Response(
                    {"error": "用户名或密码错误"},
                    status=status.HTTP_401_UNAUTHORIZED,
                )
            
            user_id, username, password_hash, usertype = row
            from django.contrib.auth import get_user_model
            User = get_user_model()
            user = User(id=user_id, username=username, usertype=usertype)
            # 使用 bcrypt 验证密码（数据库中存储的是 bcrypt 格式）
            if not bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
                return Response(
                    {"error": "用户名或密码错误"},
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            # 生成 JWT
            refresh = RefreshToken.for_user(user)

            # 尝试更新登录信息（使用原生 SQL 避免字段问题）
            try:
                from django.db import connection
                with connection.cursor() as cursor:
                    cursor.execute(
                        """UPDATE users SET last_device_id = %s, last_login_at = NOW() 
                           WHERE id = %s AND isdeleted = 0""",
                        [get_device_id(request) or "default-device", user.id]
                    )
            except Exception:
                pass

            return Response({
                "userid": user.id,
                "usertype": user.usertype,
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "message": "login success",
            }, status=status.HTTP_200_OK)

        return Response(
            {
                "userid": user.id if user else user_id,
                "usertype": user.usertype if user else None,
                "access": jwt_response.get("access"),
                "refresh": jwt_response.get("refresh"),
                "message": "login success",
            },
            status=status.HTTP_200_OK,
        )


# 2025-12-20 成绩分析：按 username 查询 testrecord 成绩历史，供前端"学生成绩分析"页面使用
class StudentScoreHistoryView(APIView):
    def get(self, request):
        """
        2025-12-20 成绩分析：
        - 按照用户名（username）从 testrecord 表中查询该学生的所有成绩记录；
        - 其中 testrecord.userid 存储的是 username 本身，itemid 为整型项目 ID；
        - 通过 join testitem 获取项目名称（如"仰卧起坐"、"引体向上"），按项目名称分组返回，便于前端直接使用按钮文案映射。
        """
        username = request.query_params.get("username")
        if not username:
            return Response(
                {"error": "缺少 username 参数"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 2025-12-20 成绩分析：使用原生 SQL 直接访问 testrecord + testitem，避免改动既有 ORM 模型
        sql = """
            SELECT tr.itemid,
                   ti.name   AS item_name,
                   tr.score0,
                   tr.testtime,
                   tr.videourl
            FROM testrecord AS tr
            LEFT JOIN testitem AS ti
              ON tr.itemid = ti.itemid
            WHERE tr.userid = %s
              AND (tr.isdeleted IS NULL OR tr.isdeleted = 0)
            ORDER BY tr.itemid ASC, tr.testtime ASC
        """

        records_by_item = {}
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, [username])
                rows = cursor.fetchall()
        except Exception as e:
            # 2025-12-20 成绩分析：数据库查询异常时返回 500，便于前端和调试定位
            return Response(
                {"error": f"查询成绩记录失败: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        for itemid, item_name, score, testtime, videourl in rows:
            # 2025-12-20 成绩分析：优先使用 testitem.name 作为项目键，缺失时退回 itemid 字符串
            item_key = item_name or str(itemid) or ""
            if item_key not in records_by_item:
                records_by_item[item_key] = []

            # 2025-12-20 成绩分析：将成绩与时间整理为简单结构，便于前端直接绘制折线图
            records_by_item[item_key].append(
                {
                    "item": item_key,
                    # 2025-12-20 成绩分析：按前端要求将成绩字段命名为 score0
                    "score0": float(score) if score is not None else None,
                    "testtime": (
                        testtime.isoformat()
                        if hasattr(testtime, "isoformat") and testtime is not None
                        else None
                    ),
                     # 2026-01-06 成绩分析/展示：携带每次测试的视频地址，供教师端“视频下载”展示使用
                    "videourl": videourl,
                }
            )

        return Response(
            {
                "username": username,
                "items": records_by_item,
            },
            status=status.HTTP_200_OK,
        )
        
# 2026-01-08 视频下载：按 username + itemid 代理请求媒体服务器 best-video 接口，返回 mp4 流供前端下载
class StudentBestVideoView(APIView):
    def get(self, request):
        """
        2026-01-08 视频下载：
        - 前端调用: GET /api/student/score-video/?username=xxx&itemid=0|1
        - 后端将参数作为 userid / itemid 原样 POST 到媒体服务器 /api/student/best 接口；
        - 将媒体服务器返回的 mp4 流直接转发给前端，以实现下载“最佳”测试视频。
        """
        username = request.query_params.get("username")
        itemid = request.query_params.get("itemid", "0")

        if not username:
            return Response({"error": "缺少 username 参数"}, status=status.HTTP_400_BAD_REQUEST)

        # 当前约定：itemid 0=仰卧起坐，1=引体向上
        if str(itemid) not in {"0", "1"}:
            return Response({"error": "itemid 参数非法，仅支持 0 或 1"}, status=status.HTTP_400_BAD_REQUEST)

        # 2026-01-13 视频下载：对齐线下调试使用的 cURL 地址，直接调用远端媒体服务器 /media/best-video/ 接口
        media_url = "http://121.196.163.155:8081/media/best-video/"
        try:
            # 向媒体服务器发送 POST 请求，请求对应学生在指定项目的最佳动作视频
            # 2026-01-13 视频下载：显式禁用环境代理，避免被 127.0.0.1:7890 等本地代理劫持
            media_resp = requests.post(
                media_url,
                data={"userid": str(username), "itemid": str(itemid)},
                stream=True,
                timeout=30,
                proxies={"http": None, "https": None},
            )
        except Exception as exc:
            return Response(
                {"error": f"请求媒体服务器失败: {str(exc)}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        if media_resp.status_code != 200:
            # 2026-01-13 视频下载：直接透传媒体服务器返回的错误信息，便于排查问题
            error_text = media_resp.text
            # 也在服务端打印一份，方便查看日志
            print(f"[StudentScoreHistoryView] 媒体服务器返回错误 {media_resp.status_code}: {error_text}")
            return Response(
                {
                    "error": "媒体服务器返回错误",
                    "status": media_resp.status_code,
                    "detail": error_text,
                },
                status=media_resp.status_code,
            )

        from django.http import StreamingHttpResponse

        def file_iterator(chunk_iter):
            for chunk in chunk_iter:
                if chunk:
                    yield chunk

        # 根据约定生成一个简易的下载文件名：username + itemid 组合
        filename = f"{username}_item{itemid}.mp4"

        response = StreamingHttpResponse(
            file_iterator(media_resp.iter_content(chunk_size=8192)),
            content_type="video/mp4",
        )
        response["Content-Disposition"] = f'attachment; filename=\"{filename}\"'
        return response

        # =====================================================
        # 2026-01 班级维度查询相关接口
        # =====================================================


def _fetch_class_students_with_latest_scores(class_name: str):
    """
    2026-01-27 班级功能：
    - 给定 class_name，查询该班级下所有学生的基本信息 + 各项目最近一次成绩。
    - 数据来源：
        - student_class：userid -> class_name
        - student_bulk：username -> 学生基础信息（姓名 / 性别 / 年龄 / 班级等）
        - testrecord + testitem：各项目成绩记录，取“每个项目最近一次”
    - 返回 Python list，元素结构：
        {
            "studentId": "...",
            "name": "...",
            "gender": "...",
            "age": 18,
            "class": "学硕1班",
            "scores": {
                "仰卧起坐": {"score0": 30.0, "testtime": "...", "videourl": "..."},
                "引体向上": {"score0": 10.0, "testtime": "...", "videourl": "..."}
            }
        }
    """

    students: list[dict] = []

    # 1）查出班级里所有学生 + 基本信息
    sql_students = """
        SELECT sc.userid,
               bs.name,
               bs.gender,
               bs.age,
               bs.class_name
        FROM student_class AS sc
        LEFT JOIN student_bulk AS bs
               ON bs.username = sc.userid
        WHERE sc.class_name = %s
        ORDER BY sc.userid ASC
    """

    with connection.cursor() as cursor:
        cursor.execute(sql_students, [class_name])
        rows = cursor.fetchall()

    if not rows:
        return []

    for userid, name, gender, age, cls_name in rows:
        students.append(
            {
                "studentId": userid,
                "name": name,
                "gender": gender,
                "age": age,
                "class": cls_name or class_name,
                "scores": {},  # 后面补成绩
            }
        )

    # 2）一次性查询这些学生所有成绩记录，按 item 取最近一次
    user_ids = [s["studentId"] for s in students]

    # 动态拼接 IN (%s, %s, ...) 占位符
    in_placeholder = ", ".join(["%s"] * len(user_ids))

    sql_scores = f"""
        SELECT tr.userid,
               tr.itemid,
               ti.name   AS item_name,
               tr.score0,
               tr.testtime,
               tr.videourl
        FROM testrecord AS tr
        LEFT JOIN testitem AS ti
               ON tr.itemid = ti.itemid
        WHERE tr.userid IN ({in_placeholder})
          AND (tr.isdeleted IS NULL OR tr.isdeleted = 0)
        ORDER BY tr.userid ASC, tr.itemid ASC, tr.testtime ASC
    """

    scores_by_user: dict[str, dict[str, dict]] = {}
    with connection.cursor() as cursor:
        cursor.execute(sql_scores, user_ids)
        score_rows = cursor.fetchall()

    for userid, itemid, item_name, score, testtime, videourl in score_rows:
        user_key = str(userid)
        item_key = item_name or str(itemid) or ""
        if not item_key:
            continue

        if user_key not in scores_by_user:
            scores_by_user[user_key] = {}

        # 由于已按 testtime 升序排序，循环过程中“最后一次覆盖”就是最近一次记录
        scores_by_user[user_key][item_key] = {
            "item": item_key,
            "score0": float(score) if score is not None else None,
            "testtime": (
                testtime.isoformat()
                if hasattr(testtime, "isoformat") and testtime is not None
                else None
            ),
            "videourl": videourl,
        }

    # 3）将成绩挂到学生列表上
    for s in students:
        uid = str(s["studentId"])
        s["scores"] = scores_by_user.get(uid, {})

    return students


class TeacherClassListView(APIView):
    """
    教师端：获取当前教师管理的班级列表（供前端下拉选择使用）。

    2026-01-27 班级功能：
    - 使用 JWT 中的用户身份（Users.username）作为 teacher_id；
    - 从 teacher_class / school_class 中查询该教师负责的所有班级。
    """

    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request):
        username = getattr(request.user, "username", None)
        if not username:
            return Response({"error": "未获取到教师账号"}, status=status.HTTP_401_UNAUTHORIZED)

        sql = """
            SELECT c.id, c.name
            FROM teacher_class AS tc
            JOIN school_class AS c
              ON tc.class_name = c.name
            WHERE tc.teacher_id = %s
            ORDER BY c.name ASC
        """
        with connection.cursor() as cursor:
            cursor.execute(sql, [username])
            rows = cursor.fetchall()

        classes = [{"id": cid, "name": name} for cid, name in rows]

        return Response(
            {
                "teacher": username,
                "classes": classes,
            },
            status=status.HTTP_200_OK,
        )


class TeacherClassStudentScoreView(APIView):
    """
    教师端：按班级查看学生信息/成绩页面使用的接口。

    2026-01-27 班级功能：
    - 支持通过 query 参数传入 class_name 或 class_id；
    - 自动校验当前教师是否有权查看该班级（teacher_class 表）；
    - 返回该班下学生的基本信息 + 各项目最近一次成绩。
    """

    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request):
        username = getattr(request.user, "username", None)
        if not username:
            return Response({"error": "未获取到教师账号"}, status=status.HTTP_401_UNAUTHORIZED)

        class_name = request.query_params.get("class_name")
        class_id = request.query_params.get("class_id")

        if not class_name and not class_id:
            return Response(
                {"error": "缺少 class_name 或 class_id 参数"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 如果只给了 class_id，则先到 school_class 查出 name
        if not class_name and class_id:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT name FROM school_class WHERE id = %s", [class_id]
                    )
                    row = cursor.fetchone()
            except Exception as exc:
                return Response(
                    {"error": f"查询班级信息失败: {str(exc)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            if not row:
                return Response(
                    {"error": "未找到对应的班级"}, status=status.HTTP_404_NOT_FOUND
                )
            class_name = row[0]

        # 校验教师是否有权查看该班级
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM teacher_class
                WHERE teacher_id = %s AND class_name = %s
                LIMIT 1
                """,
                [username, class_name],
            )
            if cursor.fetchone() is None:
                return Response(
                    {"error": "当前教师无权查看该班级"},
                    status=status.HTTP_403_FORBIDDEN,
                )

        # 查询该班级下学生 + 最近一次成绩
        try:
            students = _fetch_class_students_with_latest_scores(class_name)
        except Exception as exc:
            return Response(
                {"error": f"查询学生及成绩失败: {str(exc)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "class": {
                    "id": class_id,
                    "name": class_name,
                },
                "students": students,
            },
            status=status.HTTP_200_OK,
        )


class StudentClassmateScoreView(APIView):
    """
    学生端：查看自己所在班级其他学生成绩。

    2026-01-27 班级功能：
    - 使用当前登录学生的 username，从 student_class 查出其班级；
    - 返回该班级全部同学（含自己）的最近一次成绩列表。
    """

    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request):
        username = getattr(request.user, "username", None)
        if not username:
            return Response({"error": "未获取到学生账号"}, status=status.HTTP_401_UNAUTHORIZED)

        # 可选：限制只有学生账号可以调用
        if getattr(request.user, "usertype", None) != "student":
            return Response({"error": "当前用户不是学生账号"}, status=status.HTTP_403_FORBIDDEN)

        # 查询该学生所在班级
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT class_name
                FROM student_class
                WHERE userid = %s
                LIMIT 1
                """,
                [username],
            )
            row = cursor.fetchone()

        if not row:
            return Response(
                {
                    "username": username,
                    "class": {"id": None, "name": None},
                    "classmates": [],
                    "warning": "当前学生尚未绑定班级",
                },
                status=status.HTTP_200_OK,
            )

        class_name = row[0]

        # 查询该班级同学 + 最近一次成绩
        try:
            classmates = _fetch_class_students_with_latest_scores(class_name)
        except Exception as exc:
            return Response(
                {"error": f"查询同班同学及成绩失败: {str(exc)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # 尝试查出 class_id（便于前端展示/后续联动）
        class_id = None
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM school_class WHERE name = %s LIMIT 1",
                    [class_name],
                )
                row = cursor.fetchone()
                if row:
                    class_id = row[0]
        except Exception:
            # 查询失败时不影响主流程
            class_id = None

        return Response(
            {
                "username": username,
                "class": {
                    "id": class_id,
                    "name": class_name,
                },
                "classmates": classmates,
            },
            status=status.HTTP_200_OK,
        )


# =====================================================
# 2026-03 师生私信：学生 <-> 班级教师 一对一消息
# =====================================================


def _resolve_student_class_and_teacher(username: str) -> tuple[Optional[str], Optional[str]]:
    """
    根据学生账号找到其班级名称和对应教师账号：
    - 先从 student_class 查出 class_name
    - 再从 teacher_class 中按 class_name 查询第一个 teacher_id
    返回 (class_name, teacher_id)，任意一步查不到时返回 (None, None)
    """
    class_name: Optional[str] = None
    teacher_id: Optional[str] = None

    # 1）学生所在班级
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT class_name
            FROM student_class
            WHERE userid = %s
            LIMIT 1
            """,
            [username],
        )
        row = cursor.fetchone()
        if not row:
            return None, None
        class_name = row[0]

    # 2）该班对应的教师（若有多位教师，取第一位）
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT teacher_id
            FROM teacher_class
            WHERE class_name = %s
            ORDER BY id ASC
            LIMIT 1
            """,
            [class_name],
        )
        row = cursor.fetchone()
        if not row:
            return None, None
        teacher_id = row[0]

    return class_name, teacher_id


def _teacher_has_access_to_student(teacher_id: str, student_userid: str) -> bool:
    """
    校验教师是否有权查看 / 回复某个学生：
    - 要求该学生在 student_class 中有记录；
    - 且 teacher_class 中存在同一班级与该教师的绑定。
    """
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT 1
            FROM student_class AS sc
            JOIN teacher_class AS tc
              ON sc.class_name = tc.class_name
            WHERE sc.userid = %s
              AND tc.teacher_id = %s
            LIMIT 1
            """,
            [student_userid, teacher_id],
        )
        return cursor.fetchone() is not None


class StudentMessageView(APIView):
    """
    学生端：查看 / 发送与班主任之间的私信。

    - GET  /api/student/messages/      → 拉取与当前教师的对话历史
    - POST /api/student/messages/      → 发送新消息给当前教师
    """

    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request):
        user = request.user
        username = getattr(user, "username", None)

        if not username:
            return Response({"error": "未获取到学生账号"}, status=status.HTTP_401_UNAUTHORIZED)

        if getattr(user, "usertype", None) != "student":
            return Response({"error": "当前用户不是学生账号"}, status=status.HTTP_403_FORBIDDEN)

        class_name, teacher_id = _resolve_student_class_and_teacher(username)
        if not class_name or not teacher_id:
            return Response(
                {
                    "username": username,
                    "teacher": None,
                    "class": None,
                    "messages": [],
                    "warning": "当前学生尚未绑定班级或班主任，无法使用私信功能",
                },
                status=status.HTTP_200_OK,
            )

        # 查询完整对话历史（按时间升序）
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, sender, content, is_read_by_student, is_read_by_teacher, created_at
                FROM student_teacher_message
                WHERE student_userid = %s
                  AND teacher_id = %s
                ORDER BY created_at ASC, id ASC
                """,
                [username, teacher_id],
            )
            rows = cursor.fetchall()

        messages = [
            {
                "id": mid,
                "sender": sender,
                "content": content,
                "is_read_by_student": bool(read_s),
                "is_read_by_teacher": bool(read_t),
                "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") and created_at else None,
            }
            for mid, sender, content, read_s, read_t, created_at in rows
        ]

        # 将老师发给该学生且尚未标记为已读的消息，批量标记为已读
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE student_teacher_message
                SET is_read_by_student = 1
                WHERE student_userid = %s
                  AND teacher_id = %s
                  AND sender = 'teacher'
                  AND is_read_by_student = 0
                """,
                [username, teacher_id],
            )

        return Response(
            {
                "username": username,
                "teacher": teacher_id,
                "class": class_name,
                "messages": messages,
            },
            status=status.HTTP_200_OK,
        )

    def post(self, request):
        user = request.user
        username = getattr(user, "username", None)

        if not username:
            return Response({"error": "未获取到学生账号"}, status=status.HTTP_401_UNAUTHORIZED)

        if getattr(user, "usertype", None) != "student":
            return Response({"error": "当前用户不是学生账号"}, status=status.HTTP_403_FORBIDDEN)

        content = (request.data or {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            return Response({"error": "消息内容不能为空"}, status=status.HTTP_400_BAD_REQUEST)

        class_name, teacher_id = _resolve_student_class_and_teacher(username)
        if not class_name or not teacher_id:
            return Response(
                {"error": "当前学生尚未绑定班级或班主任，无法发送消息"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        content = content.strip()

        # 插入消息记录
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO student_teacher_message (
                    student_userid,
                    teacher_id,
                    sender,
                    content,
                    is_read_by_student,
                    is_read_by_teacher,
                    created_at
                )
                VALUES (%s, %s, 'student', %s, 1, 0, NOW())
                """,
                [username, teacher_id, content],
            )
            message_id = cursor.lastrowid

        # 再查询回完整记录，便于前端直接追加
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, sender, content, is_read_by_student, is_read_by_teacher, created_at
                FROM student_teacher_message
                WHERE id = %s
                """,
                [message_id],
            )
            row = cursor.fetchone()

        mid, sender, content, read_s, read_t, created_at = row
        message = {
            "id": mid,
            "sender": sender,
            "content": content,
            "is_read_by_student": bool(read_s),
            "is_read_by_teacher": bool(read_t),
            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") and created_at else None,
        }

        return Response(
            {
                "ok": True,
                "username": username,
                "teacher": teacher_id,
                "class": class_name,
                "message": message,
            },
            status=status.HTTP_201_CREATED,
        )


class TeacherMessageThreadsView(APIView):
    """
    教师端：查看当前教师的学生会话列表（每个学生一条）。
    """

    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request):
        user = request.user
        teacher_id = getattr(user, "username", None)

        if not teacher_id:
            return Response({"error": "未获取到教师账号"}, status=status.HTTP_401_UNAUTHORIZED)

        if getattr(user, "usertype", None) != "teacher":
            return Response({"error": "当前用户不是教师账号"}, status=status.HTTP_403_FORBIDDEN)

        # 1）找出当前教师相关的所有学生（来自消息表）
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT student_userid, MAX(created_at) AS last_time
                FROM student_teacher_message
                WHERE teacher_id = %s
                GROUP BY student_userid
                ORDER BY last_time DESC
                """,
                [teacher_id],
            )
            rows = cursor.fetchall()

        if not rows:
            return Response(
                {
                    "teacher": teacher_id,
                    "threads": [],
                },
                status=status.HTTP_200_OK,
            )

        threads = []
        for student_userid, last_time in rows:
            # 学生基础信息（姓名 + 班级），优先从 student_bulk 读取
            student_name = None
            class_name = None
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT name, class_name
                    FROM student_bulk
                    WHERE username = %s
                    LIMIT 1
                    """,
                    [student_userid],
                )
                info_row = cursor.fetchone()
                if info_row:
                    student_name, class_name = info_row

            # 最近一条消息
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, sender, content, created_at
                    FROM student_teacher_message
                    WHERE student_userid = %s
                      AND teacher_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """,
                    [student_userid, teacher_id],
                )
                msg_row = cursor.fetchone()

            if not msg_row:
                continue

            mid, sender, content, created_at = msg_row

            # 未读消息数（学生 → 教师）
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM student_teacher_message
                    WHERE student_userid = %s
                      AND teacher_id = %s
                      AND sender = 'student'
                      AND is_read_by_teacher = 0
                    """,
                    [student_userid, teacher_id],
                )
                unread_count = cursor.fetchone()[0]

            threads.append(
                {
                    "studentId": student_userid,
                    "studentName": student_name or "",
                    "className": class_name,
                    "lastMessage": {
                        "id": mid,
                        "sender": sender,
                        "content": content,
                        "created_at": created_at.isoformat()
                        if hasattr(created_at, "isoformat") and created_at
                        else None,
                    },
                    "unreadCount": int(unread_count or 0),
                }
            )

        return Response(
            {
                "teacher": teacher_id,
                "threads": threads,
            },
            status=status.HTTP_200_OK,
        )


class TeacherMessageView(APIView):
    """
    教师端：查看 / 回复与某个学生的私信。

    - GET  /api/teacher/messages/?studentId=24120324
    - POST /api/teacher/messages/   { "studentId": "...", "content": "..." }
    """

    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request):
        user = request.user
        teacher_id = getattr(user, "username", None)

        if not teacher_id:
            return Response({"error": "未获取到教师账号"}, status=status.HTTP_401_UNAUTHORIZED)

        if getattr(user, "usertype", None) != "teacher":
            return Response({"error": "当前用户不是教师账号"}, status=status.HTTP_403_FORBIDDEN)

        student_id = request.query_params.get("studentId") or request.query_params.get("student_id")
        if not student_id:
            return Response({"error": "缺少 studentId 参数"}, status=status.HTTP_400_BAD_REQUEST)

        # 权限校验：该学生是否属于当前教师负责的班级
        if not _teacher_has_access_to_student(teacher_id, student_id):
            return Response(
                {"error": "当前教师无权查看该学生的消息"},
                status=status.HTTP_403_FORBIDDEN,
            )

        # 查询消息历史
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, sender, content, is_read_by_student, is_read_by_teacher, created_at
                FROM student_teacher_message
                WHERE student_userid = %s
                  AND teacher_id = %s
                ORDER BY created_at ASC, id ASC
                """,
                [student_id, teacher_id],
            )
            rows = cursor.fetchall()

        messages = [
            {
                "id": mid,
                "sender": sender,
                "content": content,
                "is_read_by_student": bool(read_s),
                "is_read_by_teacher": bool(read_t),
                "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") and created_at else None,
            }
            for mid, sender, content, read_s, read_t, created_at in rows
        ]

        # 标记学生发给该教师的消息为已读
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE student_teacher_message
                SET is_read_by_teacher = 1
                WHERE student_userid = %s
                  AND teacher_id = %s
                  AND sender = 'student'
                  AND is_read_by_teacher = 0
                """,
                [student_id, teacher_id],
            )

        return Response(
            {
                "teacher": teacher_id,
                "studentId": student_id,
                "messages": messages,
            },
            status=status.HTTP_200_OK,
        )

    def post(self, request):
        user = request.user
        teacher_id = getattr(user, "username", None)

        if not teacher_id:
            return Response({"error": "未获取到教师账号"}, status=status.HTTP_401_UNAUTHORIZED)

        if getattr(user, "usertype", None) != "teacher":
            return Response({"error": "当前用户不是教师账号"}, status=status.HTTP_403_FORBIDDEN)

        data = request.data or {}
        student_id = data.get("studentId") or data.get("student_id")
        content = data.get("content", "")

        if not student_id:
            return Response({"error": "缺少 studentId 字段"}, status=status.HTTP_400_BAD_REQUEST)

        if not isinstance(content, str) or not content.strip():
            return Response({"error": "消息内容不能为空"}, status=status.HTTP_400_BAD_REQUEST)

        if not _teacher_has_access_to_student(teacher_id, student_id):
            return Response(
                {"error": "当前教师无权给该学生发送消息"},
                status=status.HTTP_403_FORBIDDEN,
            )

        content = content.strip()

        # 插入消息记录
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO student_teacher_message (
                    student_userid,
                    teacher_id,
                    sender,
                    content,
                    is_read_by_student,
                    is_read_by_teacher,
                    created_at
                )
                VALUES (%s, %s, 'teacher', %s, 0, 1, NOW())
                """,
                [student_id, teacher_id, content],
            )
            message_id = cursor.lastrowid

        # 查询完整记录
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, sender, content, is_read_by_student, is_read_by_teacher, created_at
                FROM student_teacher_message
                WHERE id = %s
                """,
                [message_id],
            )
            row = cursor.fetchone()

        mid, sender, content, read_s, read_t, created_at = row
        message = {
            "id": mid,
            "sender": sender,
            "content": content,
            "is_read_by_student": bool(read_s),
            "is_read_by_teacher": bool(read_t),
            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") and created_at else None,
        }

        return Response(
            {
                "ok": True,
                "teacher": teacher_id,
                "studentId": student_id,
                "message": message,
            },
            status=status.HTTP_201_CREATED,
        )


# =====================================================
# 2026-03 管理员端接口
# =====================================================

_ADMIN_ALLOWED_TYPES = {"admin", "manager"}


def _format_datetime(value):
    if hasattr(value, "strftime") and value is not None:
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return None


def _has_admin_access(user) -> bool:
    usertype = getattr(user, "usertype", None)
    if usertype in _ADMIN_ALLOWED_TYPES:
        return True

    username = getattr(user, "username", None)
    if not username:
        return False

    try:
        return Manager.objects.filter(userid=username, isdeleted=0).exists()
    except Exception:
        return False


def _normalize_text(value, default=""):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _normalize_optional_text(value):
    text = _normalize_text(value)
    return text or None


def _parse_optional_int(value, field_name: str):
    if value in (None, ""):
        return None

    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{field_name}格式错误")


def _parse_score_to_int(value):
    if value in (None, ""):
        raise ValueError("成绩不能为空")

    try:
        score = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("成绩格式错误")

    if score < 0 or score > 100:
        raise ValueError("成绩必须在 0 到 100 之间")

    return int(score.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _validate_email_if_present(email: str):
    if not email:
        return
    try:
        validate_email(email)
    except Exception:
        raise ValueError("邮箱格式错误")


def _ensure_school_class(cursor, class_name: str):
    if not class_name:
        return

    cursor.execute("SELECT id FROM school_class WHERE name = %s LIMIT 1", [class_name])
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO school_class (name, created_at, updated_at) VALUES (%s, NOW(), NOW())",
            [class_name],
        )


def _ensure_legacy_class(cursor, class_name: str, remark: str = ""):
    if not class_name:
        return

    cursor.execute(
        """
        SELECT classid
        FROM class
        WHERE classname = %s AND COALESCE(isdeleted, 0) = 0
        LIMIT 1
        """,
        [class_name],
    )
    if cursor.fetchone() is None:
        cursor.execute(
            """
            INSERT INTO class (classname, teacherid, createtime, updatetime, isdeleted, remark)
            VALUES (%s, %s, NOW(), NOW(), 0, %s)
            """,
            [class_name, 0, remark or ""],
        )


def _normalize_class_ref(class_id):
    if class_id in (None, ""):
        raise ValueError("缺少 classId")

    class_ref = str(class_id).strip()
    if not class_ref:
        raise ValueError("缺少 classId")

    if class_ref.startswith("legacy-"):
        raw_id = class_ref.split("-", 1)[1]
        try:
            return {"source": "class", "id": int(raw_id), "raw": class_ref}
        except (TypeError, ValueError):
            raise ValueError("classId 格式错误")

    try:
        return {"source": "school_class", "id": int(class_ref), "raw": class_ref}
    except (TypeError, ValueError):
        raise ValueError("classId 格式错误")


def _resolve_admin_class(cursor, class_id):
    class_ref = _normalize_class_ref(class_id)

    if class_ref["source"] == "class":
        cursor.execute(
            """
            SELECT classid, classname, teacherid, createtime, remark
            FROM class
            WHERE classid = %s AND COALESCE(isdeleted, 0) = 0
            LIMIT 1
            """,
            [class_ref["id"]],
        )
        row = cursor.fetchone()
        if row:
            classid, classname, teacherid, createtime, remark = row
            return {
                "classid": f"legacy-{classid}",
                "classname": classname,
                "teacherid": teacherid,
                "createtime": createtime,
                "remark": remark or "",
                "source": "class",
                "source_id": classid,
            }
        return None

    cursor.execute(
        "SELECT id, name, created_at FROM school_class WHERE id = %s LIMIT 1",
        [class_ref["id"]],
    )
    row = cursor.fetchone()
    if row:
        classid, classname, createtime = row
        cursor.execute(
            """
            SELECT teacherid, remark
            FROM class
            WHERE classname = %s AND COALESCE(isdeleted, 0) = 0
            ORDER BY classid ASC
            LIMIT 1
            """,
            [classname],
        )
        legacy_row = cursor.fetchone()
        teacherid = legacy_row[0] if legacy_row else 0
        remark = legacy_row[1] if legacy_row else ""
        return {
            "classid": classid,
            "classname": classname,
            "teacherid": teacherid,
            "createtime": createtime,
            "remark": remark or "",
            "source": "school_class",
            "source_id": classid,
        }

    return None


def _list_admin_classes():
    classes = []
    school_class_names = set()
    legacy_by_name = {}

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT teacher_id, class_name
            FROM teacher_class
            ORDER BY id ASC
            """
        )
        teacher_class_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT id, username, first_name
            FROM users
            WHERE COALESCE(isdeleted, 0) = 0
              AND usertype = 'teacher'
            ORDER BY id ASC
            """
        )
        teacher_user_map = {
            str(username): {
                "user_id": user_id,
                "first_name": first_name or "",
            }
            for user_id, username, first_name in cursor.fetchall()
        }

        teacher_profile_map = {}
        teacher_user_ids = [item["user_id"] for item in teacher_user_map.values() if item.get("user_id")]
        if teacher_user_ids:
            placeholders = ", ".join(["%s"] * len(teacher_user_ids))
            cursor.execute(
                f"""
                SELECT userid, name
                FROM teacher
                WHERE COALESCE(isdeleted, 0) = 0
                  AND userid IN ({placeholders})
                """,
                teacher_user_ids,
            )
            teacher_profile_map = {
                str(userid): name or ""
                for userid, name in cursor.fetchall()
            }

        teacher_binding_map = {}
        for teacher_id, class_name in teacher_class_rows:
            if class_name and class_name not in teacher_binding_map:
                teacher_user = teacher_user_map.get(str(teacher_id), {})
                teacher_name = (
                    teacher_profile_map.get(str(teacher_user.get("user_id")), "")
                    or teacher_user.get("first_name", "")
                    or teacher_id
                )
                teacher_binding_map[class_name] = {
                    "teacherId": teacher_id or "",
                    "teacherName": teacher_name or "无",
                }

        cursor.execute(
            """
            SELECT classid, classname, teacherid, createtime, remark
            FROM class
            WHERE COALESCE(isdeleted, 0) = 0
            ORDER BY classid ASC
            """
        )
        for classid, classname, teacherid, createtime, remark in cursor.fetchall():
            if classname and classname not in legacy_by_name:
                legacy_by_name[classname] = {
                    "classid": classid,
                    "teacherid": teacherid or 0,
                    "createtime": createtime,
                    "remark": remark or "",
                }

        cursor.execute(
            "SELECT id, name, created_at FROM school_class ORDER BY id ASC"
        )
        for classid, classname, createtime in cursor.fetchall():
            school_class_names.add(classname)
            legacy = legacy_by_name.get(classname, {})
            teacher_binding = teacher_binding_map.get(classname, {})
            classes.append(
                {
                    "classid": classid,
                    "classname": classname,
                    "teacherid": legacy.get("teacherid", 0),
                    "teacherId": teacher_binding.get("teacherId", ""),
                    "teacherName": teacher_binding.get("teacherName", "无"),
                    "createtime": _format_datetime(createtime),
                    "remark": legacy.get("remark", ""),
                }
            )

        for classname, legacy in legacy_by_name.items():
            if classname in school_class_names:
                continue
            teacher_binding = teacher_binding_map.get(classname, {})
            classes.append(
                {
                    "classid": f"legacy-{legacy['classid']}",
                    "classname": classname,
                    "teacherid": legacy.get("teacherid", 0),
                    "teacherId": teacher_binding.get("teacherId", ""),
                    "teacherName": teacher_binding.get("teacherName", "无"),
                    "createtime": _format_datetime(legacy.get("createtime")),
                    "remark": legacy.get("remark", ""),
                }
            )

    classes.sort(key=lambda item: (str(item["classid"]), item["classname"] or ""))
    return classes


def _resolve_admin_teacher_binding(cursor, teacher_id_value):
    teacher_id = _normalize_optional_text(teacher_id_value)
    if not teacher_id:
        return {
            "teacherId": "",
            "teacherName": "无",
            "legacyTeacherId": 0,
        }

    cursor.execute(
        """
        SELECT u.username,
               COALESCE(t.teacherid, 0) AS legacy_teacherid,
               COALESCE(NULLIF(t.name, ''), NULLIF(u.first_name, ''), u.username) AS teacher_name
        FROM users AS u
        LEFT JOIN teacher AS t
          ON t.userid = u.id
         AND COALESCE(t.isdeleted, 0) = 0
        WHERE COALESCE(u.isdeleted, 0) = 0
          AND u.usertype = 'teacher'
          AND u.username = %s
        LIMIT 1
        """,
        [teacher_id],
    )
    row = cursor.fetchone()
    if row is None:
        raise ValueError("未找到对应教师")

    teacher_username, legacy_teacher_id, teacher_name = row
    return {
        "teacherId": teacher_username or "",
        "teacherName": teacher_name or teacher_username or "无",
        "legacyTeacherId": legacy_teacher_id or 0,
    }


def _sync_admin_class_teacher_binding(cursor, class_name: str, teacher_id: str):
    cursor.execute("DELETE FROM teacher_class WHERE class_name = %s", [class_name])
    if not teacher_id:
        return

    cursor.execute(
        """
        INSERT INTO teacher_class (teacher_id, class_name, created_at, updated_at)
        VALUES (%s, %s, NOW(), NOW())
        """,
        [teacher_id, class_name],
    )


class AdminGuardedAPIView(APIView):
    authentication_classes = [RemoteIntrospectJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def initial(self, request, *args, **kwargs):
        super().initial(request, *args, **kwargs)
        if not _has_admin_access(request.user):
            raise PermissionDenied("无权限访问")


class AdminClassListView(AdminGuardedAPIView):
    def get(self, request):
        return Response({"classes": _list_admin_classes()}, status=status.HTTP_200_OK)


class AdminClassCreateView(AdminGuardedAPIView):
    def post(self, request):
        class_name = _normalize_text(request.data.get("className"))
        remark = _normalize_text(request.data.get("remark"))
        teacher_id = _normalize_optional_text(request.data.get("teacherId"))

        if not class_name:
            return Response({"error": "班级名称不能为空"}, status=status.HTTP_400_BAD_REQUEST)

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM class
                WHERE classname = %s AND COALESCE(isdeleted, 0) = 0
                LIMIT 1
                """,
                [class_name],
            )
            if cursor.fetchone() is not None:
                return Response({"error": "班级名称已存在"}, status=status.HTTP_400_BAD_REQUEST)

            cursor.execute("SELECT 1 FROM school_class WHERE name = %s LIMIT 1", [class_name])
            if cursor.fetchone() is not None:
                return Response({"error": "班级名称已存在"}, status=status.HTTP_400_BAD_REQUEST)

            try:
                teacher_binding = _resolve_admin_teacher_binding(cursor, teacher_id)
            except ValueError as exc:
                return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO class (classname, teacherid, createtime, updatetime, isdeleted, remark)
                    VALUES (%s, %s, NOW(), NOW(), 0, %s)
                    """,
                    [class_name, teacher_binding["legacyTeacherId"], remark],
                )
                _ensure_school_class(cursor, class_name)
                _sync_admin_class_teacher_binding(cursor, class_name, teacher_binding["teacherId"])

                cursor.execute(
                    "SELECT id, name, created_at FROM school_class WHERE name = %s LIMIT 1",
                    [class_name],
                )
                school_row = cursor.fetchone()

        classid, classname, createtime = school_row
        return Response(
            {
                "success": True,
                "data": {
                    "classid": classid,
                    "classname": classname,
                    "teacherid": teacher_binding["legacyTeacherId"],
                    "teacherId": teacher_binding["teacherId"],
                    "teacherName": teacher_binding["teacherName"],
                    "createtime": _format_datetime(createtime),
                    "remark": remark or "",
                },
            },
            status=status.HTTP_201_CREATED,
        )


class AdminClassUpdateView(AdminGuardedAPIView):
    def put(self, request):
        class_id = request.data.get("classId")
        class_name = _normalize_text(request.data.get("className"))
        remark = _normalize_text(request.data.get("remark"))
        teacher_id = _normalize_optional_text(request.data.get("teacherId"))

        if not class_name:
            return Response({"error": "班级名称不能为空"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            class_ref = _normalize_class_ref(class_id)
        except ValueError as exc:
            return Response({"error": "classId 格式错误"}, status=status.HTTP_400_BAD_REQUEST)

        with connection.cursor() as cursor:
            snapshot = _resolve_admin_class(cursor, class_ref["raw"])
            if snapshot is None:
                return Response({"error": "未找到对应班级"}, status=status.HTTP_404_NOT_FOUND)

            old_name = snapshot["classname"]

            try:
                teacher_binding = _resolve_admin_teacher_binding(cursor, teacher_id)
            except ValueError as exc:
                return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

            if class_name != old_name:
                cursor.execute(
                    "SELECT 1 FROM school_class WHERE name = %s LIMIT 1",
                    [class_name],
                )
                if cursor.fetchone() is not None:
                    return Response({"error": "班级名称已存在"}, status=status.HTTP_400_BAD_REQUEST)

                cursor.execute(
                    """
                    SELECT 1
                    FROM class
                    WHERE classname = %s AND COALESCE(isdeleted, 0) = 0
                    LIMIT 1
                    """,
                    [class_name],
                )
                if cursor.fetchone() is not None:
                    return Response({"error": "班级名称已存在"}, status=status.HTTP_400_BAD_REQUEST)

        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE class
                    SET classname = %s, teacherid = %s, remark = %s, updatetime = NOW()
                    WHERE classname = %s AND COALESCE(isdeleted, 0) = 0
                    """,
                    [class_name, teacher_binding["legacyTeacherId"], remark, old_name],
                )

                if cursor.rowcount == 0:
                    cursor.execute(
                        """
                        INSERT INTO class (classname, teacherid, createtime, updatetime, isdeleted, remark)
                        VALUES (%s, %s, NOW(), NOW(), 0, %s)
                        """,
                        [class_name, teacher_binding["legacyTeacherId"], remark],
                    )

                cursor.execute(
                    "UPDATE school_class SET name = %s, updated_at = NOW() WHERE name = %s",
                    [class_name, old_name],
                )
                if cursor.rowcount == 0:
                    _ensure_school_class(cursor, class_name)

                if class_name != old_name:
                    cursor.execute(
                        "UPDATE teacher_class SET class_name = %s, updated_at = NOW() WHERE class_name = %s",
                        [class_name, old_name],
                    )
                    cursor.execute(
                        "UPDATE student_class SET class_name = %s, updated_at = NOW() WHERE class_name = %s",
                        [class_name, old_name],
                    )
                    cursor.execute(
                        "UPDATE student_bulk SET class_name = %s WHERE class_name = %s",
                        [class_name, old_name],
                    )

                _sync_admin_class_teacher_binding(cursor, class_name, teacher_binding["teacherId"])

        return Response({"success": True, "message": "班级信息已更新"}, status=status.HTTP_200_OK)


class AdminClassDeleteView(AdminGuardedAPIView):
    def delete(self, request):
        class_id = request.query_params.get("classId")

        try:
            class_ref = _normalize_class_ref(class_id)
        except ValueError:
            return Response({"error": "classId 格式错误"}, status=status.HTTP_400_BAD_REQUEST)

        with connection.cursor() as cursor:
            snapshot = _resolve_admin_class(cursor, class_ref["raw"])
            if snapshot is None:
                return Response({"error": "未找到对应班级"}, status=status.HTTP_404_NOT_FOUND)

            class_name = snapshot["classname"]

            cursor.execute("SELECT COUNT(*) FROM student_class WHERE class_name = %s", [class_name])
            student_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM teacher_class WHERE class_name = %s", [class_name])
            teacher_count = cursor.fetchone()[0]

            if student_count > 0 or teacher_count > 0:
                return Response(
                    {
                        "error": f"班级仍有关联数据，学生 {student_count} 人、教师 {teacher_count} 人，暂不能删除"
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE class SET isdeleted = 1, updatetime = NOW() WHERE classname = %s AND COALESCE(isdeleted, 0) = 0",
                    [class_name],
                )
                cursor.execute("DELETE FROM school_class WHERE name = %s", [class_name])

        return Response({"success": True, "message": "班级删除成功"}, status=status.HTTP_200_OK)


class AdminBulkImportView(AdminGuardedAPIView):
    def post(self, request):
        user_type = _normalize_text(request.data.get("userType"))
        users = request.data.get("users") or []

        if user_type not in {"student", "teacher"}:
            return Response({"error": "userType 仅支持 student 或 teacher"}, status=status.HTTP_400_BAD_REQUEST)
        if not isinstance(users, list) or not users:
            return Response({"error": "users 必须为非空数组"}, status=status.HTTP_400_BAD_REQUEST)

        success_count = 0
        errors = []

        for payload in users:
            username = _normalize_text((payload or {}).get("username"))
            name = _normalize_text((payload or {}).get("name"))
            gender = _normalize_text((payload or {}).get("gender"))
            class_name = _normalize_text((payload or {}).get("class_name"))
            phone = _normalize_text((payload or {}).get("phone"))
            email = _normalize_text((payload or {}).get("email"))

            try:
                if not username or not name:
                    raise ValueError("缺少 username 或 name")

                _validate_email_if_present(email)
                age = _parse_optional_int((payload or {}).get("age"), "年龄")

                with transaction.atomic():
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT 1 FROM users WHERE username = %s LIMIT 1", [username])
                        if cursor.fetchone() is not None:
                            raise ValueError("用户名已存在")

                        cursor.execute(
                            """
                            INSERT INTO users (
                                username, password, email, phone, first_name, last_name,
                                createtime, updatetime, isdeleted, remark, usertype,
                                totp_setup_completed, totp_enabled, is_active, is_staff,
                                is_superuser, date_joined, is_authorized
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s,
                                NOW(), NOW(), 0, %s, %s,
                                0, 0, 1, 0,
                                0, NOW(), 1
                            )
                            """,
                            [
                                username,
                                make_password("123456"),
                                _normalize_optional_text(email),
                                _normalize_optional_text(phone),
                                name,
                                "",
                                "管理员批量导入",
                                user_type,
                            ],
                        )

                        if class_name:
                            _ensure_school_class(cursor, class_name)
                            _ensure_legacy_class(cursor, class_name, "管理员导入创建")

                        if user_type == "student":
                            cursor.execute(
                                "SELECT id FROM student_bulk WHERE username = %s LIMIT 1",
                                [username],
                            )
                            row = cursor.fetchone()
                            if row is None:
                                cursor.execute(
                                    """
                                    INSERT INTO student_bulk (
                                        username, name, gender, age, phone, email, class_name, created_at
                                    )
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                                    """,
                                    [
                                        username,
                                        name,
                                        _normalize_optional_text(gender),
                                        age,
                                        _normalize_optional_text(phone),
                                        _normalize_optional_text(email),
                                        class_name,
                                    ],
                                )
                            else:
                                cursor.execute(
                                    """
                                    UPDATE student_bulk
                                    SET name = %s,
                                        gender = %s,
                                        age = %s,
                                        phone = %s,
                                        email = %s,
                                        class_name = %s
                                    WHERE username = %s
                                    """,
                                    [
                                        name,
                                        _normalize_optional_text(gender),
                                        age,
                                        _normalize_optional_text(phone),
                                        _normalize_optional_text(email),
                                        class_name,
                                        username,
                                    ],
                                )

                            if class_name:
                                cursor.execute(
                                    "SELECT id FROM student_class WHERE userid = %s LIMIT 1",
                                    [username],
                                )
                                if cursor.fetchone() is None:
                                    cursor.execute(
                                        """
                                        INSERT INTO student_class (userid, class_name, created_at, updated_at)
                                        VALUES (%s, %s, NOW(), NOW())
                                        """,
                                        [username, class_name],
                                    )
                                else:
                                    cursor.execute(
                                        "UPDATE student_class SET class_name = %s, updated_at = NOW() WHERE userid = %s",
                                        [class_name, username],
                                    )

                        if user_type == "teacher" and class_name:
                            cursor.execute(
                                """
                                SELECT id
                                FROM teacher_class
                                WHERE teacher_id = %s AND class_name = %s
                                LIMIT 1
                                """,
                                [username, class_name],
                            )
                            if cursor.fetchone() is None:
                                cursor.execute(
                                    """
                                    INSERT INTO teacher_class (teacher_id, class_name, created_at, updated_at)
                                    VALUES (%s, %s, NOW(), NOW())
                                    """,
                                    [username, class_name],
                                )

                success_count += 1
            except ValueError as exc:
                errors.append(f"{username or '未知用户'}: {str(exc)}")
            except Exception as exc:
                errors.append(f"{username or '未知用户'}: {str(exc)}")

        return Response(
            {
                "success_count": success_count,
                "failed_count": len(errors),
                "errors": errors,
            },
            status=status.HTTP_200_OK,
        )


class AdminStudentListView(AdminGuardedAPIView):
    def get(self, request):
        students = []

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT bs.username,
                       bs.name,
                       bs.gender,
                       bs.age,
                       bs.class_name,
                       bs.phone,
                       bs.email
                FROM student_bulk AS bs
                ORDER BY bs.id ASC
                """
            )
            bulk_rows = cursor.fetchall()

            cursor.execute(
                """
                SELECT id,
                       username,
                       phone,
                       email
                FROM users
                WHERE COALESCE(isdeleted, 0) = 0
                  AND usertype = 'student'
                """
            )
            active_student_rows = cursor.fetchall()
            active_student_usernames = {row[1] for row in active_student_rows}
            active_student_map = {
                str(username): {
                    "userId": user_id,
                    "phone": phone or "",
                    "email": email or "",
                }
                for user_id, username, phone, email in active_student_rows
            }
            seen_student_ids = set()

            for student_id, name, gender, age, class_name, phone, email in bulk_rows:
                if active_student_usernames and student_id not in active_student_usernames:
                    continue
                seen_student_ids.add(str(student_id))
                user_info = active_student_map.get(str(student_id), {})
                students.append(
                    {
                        "studentId": student_id,
                        "name": name,
                        "gender": gender or "",
                        "age": age,
                        "class": class_name or "",
                        "phone": phone or user_info.get("phone", ""),
                        "email": email or user_info.get("email", ""),
                    }
                )

            for _user_id, username, phone, email in active_student_rows:
                username = str(username)
                if username in seen_student_ids:
                    continue
                students.append(
                    {
                        "studentId": username,
                        "name": username,
                        "gender": "",
                        "age": None,
                        "class": "",
                        "phone": phone or "",
                        "email": email or "",
                    }
                )

        return Response({"students": students}, status=status.HTTP_200_OK)


class AdminTeacherListView(AdminGuardedAPIView):
    def get(self, request):
        teachers = []

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, username, first_name, phone, email
                FROM users
                WHERE COALESCE(isdeleted, 0) = 0
                  AND usertype = 'teacher'
                ORDER BY id ASC
                """
            )
            user_rows = cursor.fetchall()

            cursor.execute(
                """
                SELECT userid, name, gender, phone, email, hiredate
                FROM teacher
                WHERE COALESCE(isdeleted, 0) = 0
                """
            )
            teacher_profile_map = {
                str(userid): {
                    "name": name or "",
                    "gender": gender or "",
                    "phone": phone or "",
                    "email": email or "",
                    "hiredate": _format_datetime(hiredate),
                }
                for userid, name, gender, phone, email, hiredate in cursor.fetchall()
            }

            cursor.execute(
                """
                SELECT teacher_id, class_name
                FROM teacher_class
                ORDER BY id ASC
                """
            )
            teacher_classes_map = {}
            for teacher_id, class_name in cursor.fetchall():
                key = str(teacher_id)
                teacher_classes_map.setdefault(key, [])
                if class_name and class_name not in teacher_classes_map[key]:
                    teacher_classes_map[key].append(class_name)

            for user_id, username, first_name, phone, email in user_rows:
                profile = teacher_profile_map.get(str(user_id), {})
                classes = teacher_classes_map.get(username, [])
                teachers.append(
                    {
                        "teacherId": username,
                        "name": profile.get("name") or first_name or username,
                        "gender": profile.get("gender", ""),
                        "phone": profile.get("phone") or phone or "",
                        "email": profile.get("email") or email or "",
                        "classes": classes,
                        "classText": "、".join(classes),
                        "hiredate": profile.get("hiredate"),
                    }
                )

        return Response({"teachers": teachers}, status=status.HTTP_200_OK)


class AdminStudentScoresView(AdminGuardedAPIView):
    def get(self, request):
        student_id = _normalize_text(request.query_params.get("studentId"))
        if not student_id:
            return Response({"error": "缺少 studentId"}, status=status.HTTP_400_BAD_REQUEST)

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM users
                WHERE username = %s AND COALESCE(isdeleted, 0) = 0
                LIMIT 1
                """,
                [student_id],
            )
            row = cursor.fetchone()
            if row is None:
                return Response({"error": "未找到对应学生"}, status=status.HTTP_404_NOT_FOUND)

            user_id = row[0]
            candidate_user_ids = [user_id]
            try:
                numeric_student_id = int(student_id)
                if numeric_student_id not in candidate_user_ids:
                    candidate_user_ids.append(numeric_student_id)
            except (TypeError, ValueError):
                pass

            in_placeholder = ", ".join(["%s"] * len(candidate_user_ids))
            cursor.execute(
                f"""
                SELECT tr.testid,
                       tr.itemid,
                       ti.name,
                       tr.score0,
                       COALESCE(tr.testtime, tr.createtime) AS score_time
                FROM testrecord AS tr
                LEFT JOIN testitem AS ti
                  ON ti.itemid = tr.itemid
                WHERE tr.userid IN ({in_placeholder})
                  AND COALESCE(tr.isdeleted, 0) = 0
                ORDER BY score_time DESC, tr.testid DESC
                """,
                candidate_user_ids,
            )
            scores = [
                {
                    "testid": testid,
                    "itemName": ({0: "仰卧起坐", 1: "引体向上"}.get(item_id) or item_name or (str(item_id) if item_id is not None else "")),
                    "score": float(score) if score is not None else None,
                    "testtime": _format_datetime(score_time),
                }
                for testid, item_id, item_name, score, score_time in cursor.fetchall()
            ]

        return Response({"scores": scores}, status=status.HTTP_200_OK)


class AdminScoreAddView(AdminGuardedAPIView):
    def post(self, request):
        student_id = _normalize_text(request.data.get("studentId"))
        item_name = _normalize_text(request.data.get("itemName"))

        if not student_id:
            return Response({"error": "缺少 studentId"}, status=status.HTTP_400_BAD_REQUEST)
        if not item_name:
            return Response({"error": "缺少 itemName"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            score_value = _parse_score_to_int(request.data.get("score"))
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM users WHERE username = %s AND COALESCE(isdeleted, 0) = 0 LIMIT 1",
                    [student_id],
                )
                row = cursor.fetchone()
                if row is None:
                    return Response({"error": "未找到对应学生"}, status=status.HTTP_404_NOT_FOUND)
                user_id = row[0]

                cursor.execute(
                    "SELECT itemid FROM testitem WHERE name = %s AND COALESCE(isdeleted, 0) = 0 LIMIT 1",
                    [item_name],
                )
                row = cursor.fetchone()
                if row is None:
                    return Response({"error": "未找到对应测试项目"}, status=status.HTTP_404_NOT_FOUND)
                item_id = row[0]

                cursor.execute(
                    """
                    INSERT INTO testrecord (
                        userid, itemid, score0, score1, videourl, dataurl,
                        testtime, createtime, updatetime, isdeleted, remark
                    )
                    VALUES (%s, %s, %s, NULL, '', '', NOW(), NOW(), NOW(), 0, %s)
                    """,
                    [user_id, item_id, score_value, "管理员手动录入"],
                )

        return Response({"success": True, "message": "成绩添加成功"}, status=status.HTTP_201_CREATED)


class AdminScoreUpdateView(AdminGuardedAPIView):
    def put(self, request):
        test_id = request.data.get("testid")
        if not test_id:
            return Response({"error": "缺少 testid"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            test_id = int(test_id)
        except (TypeError, ValueError):
            return Response({"error": "testid 格式错误"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            score_value = _parse_score_to_int(request.data.get("score"))
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE testrecord
                SET score0 = %s, updatetime = NOW()
                WHERE testid = %s AND COALESCE(isdeleted, 0) = 0
                """,
                [score_value, test_id],
            )
            if cursor.rowcount == 0:
                return Response({"error": "未找到对应成绩记录"}, status=status.HTTP_404_NOT_FOUND)

        return Response({"success": True, "message": "成绩修改成功"}, status=status.HTTP_200_OK)


class AdminScoreDeleteView(AdminGuardedAPIView):
    def delete(self, request):
        test_id = request.query_params.get("testid")
        if not test_id:
            return Response({"error": "缺少 testid"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            test_id = int(test_id)
        except (TypeError, ValueError):
            return Response({"error": "testid 格式错误"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            resp = _auth_http.delete(
                settings.AUTH_REMOTE_MEDIA_DELETE,
                params={"testid": test_id},
                timeout=max(settings.AUTH_REMOTE_TIMEOUT, 10),
                proxies={"http": None, "https": None},
            )
        except Exception as exc:
            return Response(
                {"error": f"远端媒体删除失败: {exc}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = {"error": resp.text or "remote media delete error"}
            return Response(detail, status=resp.status_code)

        payload = resp.json()
        return Response(
            {
                "success": True,
                "message": "成绩删除成功",
                "deleted_files": payload.get("deleted_files", []),
                "missing_files": payload.get("missing_files", []),
                "file_errors": payload.get("file_errors", []),
            },
            status=status.HTTP_200_OK,
        )


class AdminStudentsDeleteView(AdminGuardedAPIView):
    def post(self, request):
        student_ids = request.data.get("studentIds") or []
        if not isinstance(student_ids, list) or not student_ids:
            return Response({"error": "studentIds 必须为非空数组"}, status=status.HTTP_400_BAD_REQUEST)

        student_ids = [_normalize_text(student_id) for student_id in student_ids if _normalize_text(student_id)]
        if not student_ids:
            return Response({"error": "studentIds 必须为非空数组"}, status=status.HTTP_400_BAD_REQUEST)

        placeholders = ", ".join(["%s"] * len(student_ids))
        deleted_count = 0

        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT username, id
                    FROM users
                    WHERE username IN ({placeholders})
                    """,
                    student_ids,
                )
                user_rows = cursor.fetchall()
                username_to_user_id = {username: user_id for username, user_id in user_rows}

                cursor.execute(
                    f"SELECT username FROM student_bulk WHERE username IN ({placeholders})",
                    student_ids,
                )
                bulk_usernames = {row[0] for row in cursor.fetchall()}

                affected_usernames = set(username_to_user_id.keys()) | bulk_usernames
                deleted_count = len(affected_usernames)

                if username_to_user_id:
                    cursor.execute(
                        f"UPDATE users SET isdeleted = 1, updatetime = NOW() WHERE username IN ({placeholders})",
                        student_ids,
                    )

                    user_id_placeholders = ", ".join(["%s"] * len(username_to_user_id))
                    cursor.execute(
                        f"UPDATE testrecord SET isdeleted = 1, updatetime = NOW() WHERE userid IN ({user_id_placeholders})",
                        list(username_to_user_id.values()),
                    )

                cursor.execute(
                    f"DELETE FROM student_bulk WHERE username IN ({placeholders})",
                    student_ids,
                )
                cursor.execute(
                    f"DELETE FROM student_class WHERE userid IN ({placeholders})",
                    student_ids,
                )

        return Response(
            {"success": True, "message": f"已删除 {deleted_count} 位学生"},
            status=status.HTTP_200_OK,
        )


def situp_start(request):

    uid = request.GET.get("uid")
    if not uid:
        print("DEBUG: Username is missing or empty, returning error")
        return JsonResponse(
            {"status": "error", "message": "uid is required"},
            status=400,
        )

    res = UserManager().start_sport_test(uid, "situp")

    if res:
        return JsonResponse({"success": True, "message": "开始体测"}, status=status.HTTP_200_OK)
    else:
        return JsonResponse({"success": False, "message": "体测开始失败"}, status=status.HTTP_400_BAD_REQUEST)

def situp_stop(request):

    uid = request.GET.get("uid")
    if not uid:
        print("DEBUG: Username is missing or empty, returning error")
        return JsonResponse(
            {"status": "error", "message": "uid is required"},
            status=400,
        )

    UserManager().stop_sport_test(uid)

    return JsonResponse({"success": True, "message": "体测结束"}, status=status.HTTP_200_OK)

def pullup_start(request):

    uid = request.GET.get("uid")
    if not uid:
        print("DEBUG: Username is missing or empty, returning error")
        return JsonResponse(
            {"status": "error", "message": "uid is required"},
            status=400,
        )

    UserManager().start_sport_test(uid, "pullup")

    return JsonResponse({"success": True, "message": "体测开始"}, status=status.HTTP_200_OK)

def pullup_stop(request):

    uid = request.GET.get("uid")
    if not uid:
        print("DEBUG: Username is missing or empty, returning error")
        return JsonResponse(
            {"status": "error", "message": "uid is required"},
            status=400,
        )

    UserManager().stop_sport_test(uid)

    return JsonResponse({"success": True, "message": "体测结束"}, status=status.HTTP_200_OK)

def sitreach_start_view(request):
    uid = request.GET.get("uid") or request.GET.get("username")
    if not uid:
        print("DEBUG: Username is missing or empty, returning error")
        return JsonResponse(
            {"status": "error", "message": "uid is required"},
            status=400,
        )

    UserManager().start_sport_test(uid, "sitreach")

    return JsonResponse({"success": True, "message": "体测开始"}, status=status.HTTP_200_OK)


def sitreach_stop_view(request):
    uid = request.GET.get("uid") or request.GET.get("username")
    if not uid:
        print("DEBUG: Username is missing or empty, returning error")
        return JsonResponse(
            {"status": "error", "message": "uid is required"},
            status=400,
        )

    UserManager().stop_sport_test(uid)

    return JsonResponse({"success": True, "message": "体测结束"}, status=status.HTTP_200_OK)


def sitreach_fetch_inc_data_view(request):
    uid = request.GET.get("uid") or request.GET.get("username")
    return JsonResponse(sitreach_fetch_inc_data(uid))


def sitreach_get_img_view(request):
    uid = request.GET.get("uid") or request.GET.get("username")
    draw = request.GET.get("draw", "1") != "0"

    img = sitreach_get_latest_frame(uid, draw=draw)

    if img is None:
        return HttpResponse(status=404)

    return HttpResponse(img, content_type="image/jpeg")

# 可删
def sitreach_start_local_view(request):
    uid = request.GET.get("uid") or request.GET.get("username")
    return JsonResponse(sitreach_start_local_camera(uid, CAMERA_INDEX))


def sitreach_stop_local_view(request):
    uid = request.GET.get("uid") or request.GET.get("username")
    return JsonResponse(sitreach_stop_local_camera(uid))

def standjump_start_view(request):
    uid = request.GET.get("uid") or request.GET.get("username")
    if not uid:
        print("DEBUG: Username is missing or empty, returning error")
        return JsonResponse(
            {"status": "error", "message": "uid is required"},
            status=400,
        )

    UserManager().start_sport_test(uid, "jump")

    return JsonResponse({"success": True, "message": "体测开始"}, status=status.HTTP_200_OK)


def standjump_stop_view(request):
    uid = request.GET.get("uid") or request.GET.get("username")
    if not uid:
        print("DEBUG: Username is missing or empty, returning error")
        return JsonResponse(
            {"status": "error", "message": "uid is required"},
            status=400,
        )

    UserManager().stop_sport_test(uid)

    return JsonResponse({"success": True, "message": "体测结束"}, status=status.HTTP_200_OK)