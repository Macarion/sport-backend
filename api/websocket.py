from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer
from api.user_manager import UserManager
from api.sport_manager import sport_dict


class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        kwargs = self.scope["url_route"]["kwargs"]
        self.sport_type = kwargs.get("sport_type")
        self.uid = kwargs.get("uid")

        if self.sport_type not in sport_dict or self.uid is None:
            await self.close()
            return

        # 连接时的操作
        await self.accept()

    async def disconnect(self, close_code):
        UserManager().stop_sport_test(self.uid)

    async def receive(self, text_data=None, bytes_data=None):
        UserManager().handle_video_stream(self.uid, bytes_data)


class DataConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        kwargs = self.scope["url_route"]["kwargs"]
        self.sport_type = kwargs.get("sport_type")
        self.uid = kwargs.get("uid")

        self.group_name = f"data_{self.uid}"

        if self.sport_type not in sport_dict or self.uid is None:
            await self.close()
            return
        
        channel_layer = get_channel_layer()
        await channel_layer.group_add(
            self.group_name,
            self.channel_name
        )

        # 连接时的操作
        await self.accept()

    async def disconnect(self, close_code):
         # 离开消息组
        channel_layer = get_channel_layer()
        await channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def receive(self, text_data=None, bytes_data=None):
        # 处理接收到的消息
        pass

    async def send_data(self, event):
        """组消息回调，推送数据到前端"""
        text_data = event["text"]
        # 直接发送文本，不需要外层type包装
        await self.send(text_data=str(text_data))
    
    async def close_client(self, text_data=None, bytes_data=None):
        await self.close()