from django.urls import path,include
from . import views
from .student_bulk import StudentBulkSaveView, StudentBulkGetView
from .uwb_views import (
    UwbBindingView,
    UwbDirectDbFetchIncDataView,
    UwbFetchIncDataView,
    UwbIngestView,
    UwbLatestView,
    UwbReplayView,
    UwbReplayWindowView,
    UwbSessionView,
    UwbStartView,
    UwbStopView,
    UwbStreamView,
)

from .uwb_views import (
    UwbBindingView,
    UwbFetchIncDataView,
    UwbIngestView,
    UwbLatestView,
    UwbReplayView,
    UwbReplayWindowView,
    UwbSessionView,
    UwbStartView,
    UwbStopView,
    UwbStreamView,
)


app_name = 'api'

urlpatterns = [
    path('api/totp/setup_info/', views.TOTPSetupInfoView.as_view(), name='totp_setup_info'),
    path('api/totp/setup/', views.SetupTOTPView.as_view(), name='totp_setup'),
    path('api/totp/verify/', views.VerifyTOTPView.as_view(), name='totp_verify'),
    path("api/get_img/", views.get_img, name="get_img"),
    path("api/open_latest_report/", views.open_latest_report, name="open_latest_report"),
    path("api/gen_pdf/", views.gen_pdf, name="gen_pdf"),
    path("api/gen_situp_pdf/", views.gen_situp_pdf, name="gen_situp_pdf"),
    path("api/gen_pullup_pdf/", views.gen_pullup_pdf, name="gen_pullup_pdf"),
    path("api/situp/start/", views.situp_start, name="situp-start"),
    path("api/situp/stop/",  views.situp_stop,  name="situp-stop"),
    path('api/latest_data/', views.latest_data, name='latest_data'), 
    path("api/pullup/start/", views.pullup_start, name="pullup-start"),
    path("api/pullup/stop/",  views.pullup_stop,  name="pullup-stop"),
    path("api/pullup/latest_data/", views.latest_pullup_data, name="pullup-latest_data"),
    path("api/sitreach/start", views.sitreach_start_view),
    path("api/sitreach/stop", views.sitreach_stop_view),
    path("api/sitreach/fetch_inc_data", views.sitreach_fetch_inc_data_view),
    path("api/sitreach/start_local", views.sitreach_start_local_view),
    path("api/sitreach/stop_local", views.sitreach_stop_local_view),
    path("api/sitreach/get_img", views.sitreach_get_img_view),
    path("api/standjump/start/", views.standjump_start_view),
    path("api/standjump/stop/", views.standjump_stop_view),
    path('api/uwb/start/', UwbStartView.as_view(), name='uwb_start'),
    path('api/uwb/stop/', UwbStopView.as_view(), name='uwb_stop'),
    path('api/uwb/ingest/', UwbIngestView.as_view(), name='uwb_ingest'),
    path('api/uwb/stream/', UwbStreamView.as_view(), name='uwb_stream'),
    path('api/uwb/fetch_inc_data/', UwbFetchIncDataView.as_view(), name='uwb_fetch_inc_data'),
    path('api/uwb/latest/', UwbLatestView.as_view(), name='uwb_latest'),
    path('api/uwb/bindings/', UwbBindingView.as_view(), name='uwb_bindings'),
    path('api/uwb/sessions/', UwbSessionView.as_view(), name='uwb_sessions'),
    path('api/uwb/replay/', UwbReplayView.as_view(), name='uwb_replay'),
    path('api/uwb/replay/window/', UwbReplayWindowView.as_view(), name='uwb_replay_window'),
    path('api/get_classes/', views.get_classes, name='get_classes'),
    path('api/get_students/', views.get_students, name='get_students'),
    path('api/register/', views.RegisterView.as_view(), name='login_view'),  # 学生注册
    path('api/login/', views.LoginView.as_view(), name='login_view'),  # 学生登录
    path('api/teacher/info/', views.TeacherInfoView.as_view(), name='teacher_info'),  # 获取教师信息
    path('api/teacher/update/', views.TeacherUpdateView.as_view(), name='teacher_update'),  # 更新教师信息
    path('api/teacher/join-class/', views.TeacherJoinClassView.as_view(), name='teacher_join_class'),  # 创建班级
    path('api/teacher/user/', views.TeacherInfoIDView.as_view(), name='teacher_infoid'),  # 教师发送ID给学生
    path('api/inform/add/', views.AddInformView.as_view(), name='add_inform'),  # 新增通知路由 
    path('api/manager/info/', views.ManagerInfoView.as_view(), name='manager_info'),      # 新增查询管理员路由
    path('api/manager/update/', views.ManagerUpdateView.as_view(), name='manager_update'),# 新增更新管理员路由
    path('api/manager/change-password/', views.ManagerChangePasswordView.as_view(), name='manager_change_password'),  # 新增修改密码路由
    path('api/user/totp-status', views.TOTPStatusView.as_view(), name='totp-status'),
    path('api/token/refresh/', views.RemoteTokenRefreshView.as_view(), name='remote_token_refresh'),
    path('api/student/userInfo/', views.UserInfoView.as_view(), name='user_info'),
    path('api/student/userTestingInfo/', views.UserTestingInfoView.as_view(), name='user_testing_info'),
    path('api/student/userTestingImg/', views.UserTestingImgView.as_view(), name='user_testing_img'),
    path('api/student/getStudentInfo/', views.getStudentInfoView.as_view(), name='get_student_info'),
    path("api/pullup/start/", views.pullup_start, name="pullup-start"),
    path("api/pullup/stop/", views.pullup_stop, name="pullup-stop"),
    path("api/pullup/latest_data/", views.latest_pullup_data, name="pullup-latest_data"),
    path("api/auth/heartbeat/", views.HeartbeatView.as_view(), name="auth-heartbeat"),
    path('api/student/bulk-import/', views.BulkStudentImportView.as_view(), name='bulk_student_import'),  # 2025-12-02 批量导入 student_bulk
    path('api/student/bulk-info/', views.BulkStudentInfoView.as_view(), name='bulk_student_info'),        # 2025-12-02 批量导入查询接口
    path('api/student/bulk-list/', views.BulkStudentListView.as_view(), name='bulk_student_list'),        # 2025-12-05 教师端学生列表
    path('api/student_bulk/save/', StudentBulkSaveView.as_view(), name='student_bulk_save'),
    path('api/student_bulk/get/', StudentBulkGetView.as_view(), name='student_bulk_get'),
    path('api/student/score-history/', views.StudentScoreHistoryView.as_view(), name='student_score_history'),  # 2025-12-20 成绩分析：按 username 查询 testrecord 成绩历史
    path('api/student/score-video/', views.StudentBestVideoView.as_view(), name='student_score_video'),        # 2026-01-08 视频下载：按 username + itemid 下载最佳动作视频
    path('api/uwb/start/', UwbStartView.as_view(), name='uwb_start'),
    path('api/uwb/stop/', UwbStopView.as_view(), name='uwb_stop'),
    path('api/uwb/ingest/', UwbIngestView.as_view(), name='uwb_ingest'),
    path('api/uwb/stream/', UwbStreamView.as_view(), name='uwb_stream'),
    path('api/uwb/fetch_inc_data/', UwbFetchIncDataView.as_view(), name='uwb_fetch_inc_data'),
    path('api/uwb/direct_db/fetch_inc_data/', UwbDirectDbFetchIncDataView.as_view(), name='uwb_direct_db_fetch_inc_data'),
    path('api/uwb/latest/', UwbLatestView.as_view(), name='uwb_latest'),
    path('api/uwb/bindings/', UwbBindingView.as_view(), name='uwb_bindings'),
    path('api/uwb/sessions/', UwbSessionView.as_view(), name='uwb_sessions'),
    path('api/uwb/replay/', UwbReplayView.as_view(), name='uwb_replay'),
    path('api/uwb/replay/window/', UwbReplayWindowView.as_view(), name='uwb_replay_window'),
# 2026-01-27 教师端学生管理：单个学生增删改
    path('api/teacher/student-update/', views.TeacherBulkStudentUpdateView.as_view(), name='teacher_student_update'),
    path('api/teacher/student-delete/', views.TeacherBulkStudentDeleteView.as_view(), name='teacher_student_delete'),        # 2026-01-08 视频下载：按 username + itemid 下载最佳动作视频
    # 2026-01 班级维度查询相关接口（预留）
    path('api/teacher/classes/', views.TeacherClassListView.as_view(), name='teacher_class_list'),
    path('api/teacher/class-students-scores/', views.TeacherClassStudentScoreView.as_view(), name='teacher_class_students_scores'),
    path('api/student/class-scores/', views.StudentClassmateScoreView.as_view(), name='student_class_scores'),
    # 2026-03 师生私信相关接口
    path('api/student/messages/', views.StudentMessageView.as_view(), name='student_messages'),
    path('api/teacher/message-threads/', views.TeacherMessageThreadsView.as_view(), name='teacher_message_threads'),
    path('api/teacher/messages/', views.TeacherMessageView.as_view(), name='teacher_messages'),
    # 2026-03 管理员端接口
    path('api/admin/class-list/', views.AdminClassListView.as_view(), name='admin_class_list'),
    path('api/admin/class-create/', views.AdminClassCreateView.as_view(), name='admin_class_create'),
    path('api/admin/class-update/', views.AdminClassUpdateView.as_view(), name='admin_class_update'),
    path('api/admin/class-delete/', views.AdminClassDeleteView.as_view(), name='admin_class_delete'),
    path('api/admin/bulk-import/', views.AdminBulkImportView.as_view(), name='admin_bulk_import'),
    path('api/admin/student-list/', views.AdminStudentListView.as_view(), name='admin_student_list'),
    path('api/admin/teacher-list/', views.AdminTeacherListView.as_view(), name='admin_teacher_list'),
    path('api/admin/student-scores/', views.AdminStudentScoresView.as_view(), name='admin_student_scores'),
    path('api/admin/score-add/', views.AdminScoreAddView.as_view(), name='admin_score_add'),
    path('api/admin/score-update/', views.AdminScoreUpdateView.as_view(), name='admin_score_update'),
    path('api/admin/score-delete/', views.AdminScoreDeleteView.as_view(), name='admin_score_delete'),
    path('api/admin/students-delete/', views.AdminStudentsDeleteView.as_view(), name='admin_students_delete'),
]
  
