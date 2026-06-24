from django.urls import re_path
from api.websocket import WebRTCConsumer

websocket_urlpatterns = [
    re_path(r"ws/webrtc/(?P<uid>\w+)/$", WebRTCConsumer.as_asgi()),
]