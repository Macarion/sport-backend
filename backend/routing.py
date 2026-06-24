from django.urls import re_path
from api.websocket import DataConsumer, WebRTCConsumer

websocket_urlpatterns = [
    re_path(r"ws/webrtc/(?P<uid>\w+)/$", WebRTCConsumer.as_asgi()),
    re_path(r"ws/(?P<sport_type>\w+)_data_(?P<uid>\w+)/$", DataConsumer.as_asgi()),
]