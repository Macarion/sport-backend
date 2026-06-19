from django.urls import re_path
from api.websocket import VideoConsumer, DataConsumer

websocket_urlpatterns = [
    re_path(r"ws/(?P<sport_type>\w+)_video_(?P<uid>\w+)/$", VideoConsumer.as_asgi()),
    re_path(r"ws/(?P<sport_type>\w+)_data_(?P<uid>\w+)/$", DataConsumer.as_asgi()),
]