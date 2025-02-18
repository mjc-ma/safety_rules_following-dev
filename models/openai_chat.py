from typing import List, Dict, Any, Literal
import openai
import yaml
#from mmte.utils.registry import registry
from models.base import BaseChat, Response
from utils import get_abs_path
import os
import base64
import time
import io
from PIL import Image
from openai import OpenAI

os.environ["openai_apikey"] = "sk-C23FdDZCEKSLEBdejPROyWMfkAl1qfSCq0cbtqool7ZDeVh4"
#@registry.register_chatmodel()
class OpenAIChat(BaseChat):
    """
    Chat class for OpenAI models, e.g., gpt-4-vision-preview
    """

    MODEL_CONFIG = {
        "gpt-4-1106-vision-preview": "configs/models/openai/openai.yaml",
        "gpt-4-1106-preview": "configs/models/openai/openai.yaml",
        "gpt-3.5-turbo": "configs/models/openai/openai.yaml",
        "gpt-4-0613": "configs/models/openai/openai.yaml",
        "gpt-4o": "configs/models/openai/openai.yaml",
    }

    model_family = list(MODEL_CONFIG.keys())

    model_arch = "gpt"

    def __init__(self, model_id: str = "gpt-4o", **kargs):
        super().__init__(model_id=model_id)
        config = self.MODEL_CONFIG[self.model_id]
        with open(get_abs_path(config)) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        # use_proxy = self.model_config.get('proxy')
        use_proxy = None
        if use_proxy is not None:
            os.environ["http_proxy"] = self.model_config.get("proxy")
            os.environ["https_proxy"] = self.model_config.get("proxy")
        #base_url = "https://api.claudeshop.top/v1"
        self.api_key = os.getenv("openai_apikey", None)
        assert self.api_key, "openai_apikey is empty"
        self.max_retries = self.model_config.get("max_retries", 10)
        self.timeout = self.model_config.get("timeout", 1)
        openai.api_key = self.api_key

    def chat(self, messages: List, **generation_kwargs):
        conversation = []
        base_url = "https://api.claudeshop.top/v1"
        client = OpenAI(
        base_url=base_url,
        api_key=self.api_key
        )
        assert len(messages) == 1, "Only support one-turn conversation currently"
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if isinstance(message["content"], dict):
                    if len(conversation) == 0 and message["role"] == "system":
                        raise AttributeError(
                            "Currently OpenAI doesn't support images in the first system message but this may change in the future."
                        )

                    # multimodal content
                    text = message["content"]["text"]
                    image_path = message["content"]["image_path"]
                    local_image = os.path.exists(image_path)
                    #print(image_path, local_image)
                    content = [
                        {
                            "type": "text",
                            "text": text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:image/jpeg;base64,{self.encode_image(image_path)}"
                                    if local_image
                                    else image_path
                                )
                            },
                        },
                    ]
                else:
                    content = message["content"]
                conversation.append({"role": message["role"], "content": content})
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )

        # Create Completion Request
        raw_request: Dict[str, Any] = {
            "model": self.model_id,
            "messages": conversation,
        }

        # Generation Configuration
        raw_request["temperature"] = generation_kwargs.get("temperature", 1.0)
        raw_request["max_tokens"] = generation_kwargs.get("max_new_tokens", 100)
        raw_request["n"] = generation_kwargs.get("num_return_sequences", 1)
        if "stop_sequences" in generation_kwargs:
            raw_request["stop"] = generation_kwargs.get("stop_sequences")
        if "do_sample" in generation_kwargs and not generation_kwargs.get("do_sample"):
            raw_request["temperature"] = 0.0
        if "output_scores" in generation_kwargs and "vision" not in self.model_id:
            raw_request["logprobs"] = generation_kwargs.get("output_scores", False)

        from pprint import pp

        pp(raw_request)

        for i in range(self.max_retries):
            try:
                response = client.chat.completions.create(**raw_request)
                break
            except Exception as e:
                print(f"Error in generation: {e}")
                response = f"Error in generation: {e}"
                time.sleep(self.timeout)
        if isinstance(response, str):
            print(response)
            return Response(self.model_id, response, None, None)

        response_message = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        logprobs = response.choices[0].logprobs

        return Response(self.model_id, response_message, logprobs, finish_reason)

    # Function to encode the image
    @classmethod
    # def encode_image(cls, image_path: str):
    #     buffer = io.BytesIO()
    #     with open(image_path, "rb") as image_file:
    #         img_data = base64.b64encode(image_file.read())

    #         img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("RGB")
    #         print(img.size)
    #         if img.width > 400 or img.height > 400:
    #             if img.width > img.height:
    #                 new_width = 400
    #                 concat = float(new_width / float(img.width))
    #                 size = int((float(img.height) * float(concat)))
    #                 img = img.resize((new_width, size), Image.LANCZOS)
    #             else:
    #                 new_height = 400
    #                 concat = float(new_height / float(img.height))
    #                 size = int((float(img.width) * float(concat)))
    #                 img = img.resize((size, new_height), Image.LANCZOS)
    #             img.save(buffer, format="JPEG")
    #             img_data = base64.b64encode(buffer.getvalue())
    #         return img_data.decode("utf-8")
    @classmethod
    def encode_image(cls, image_path: str):
        # 检查图片路径是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # 读取原始二进制数据
        with open(image_path, "rb") as image_file:
            img_bytes = image_file.read()

        try:
            # 解码为图片对象以调整尺寸
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            print(f"Original size: {img.size}")

            # 调整尺寸逻辑
            if img.width > 400 or img.height > 400:
                # 计算新尺寸
                if img.width > img.height:
                    new_width = 400
                    scale = new_width / img.width
                    new_height = int(img.height * scale)
                else:
                    new_height = 400
                    scale = new_height / img.height
                    new_width = int(img.width * scale)

                # 调整尺寸并保存为 JPEG
                buffer = io.BytesIO()
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img.save(buffer, format="JPEG", quality=85)  # 调整质量避免数据过大
                img_bytes = buffer.getvalue()
                print(f"Resized to: {new_width}x{new_height}")

            # 编码为 Base64
            encoded_data = base64.b64encode(img_bytes).decode("utf-8")
            
            # 验证 Base64 数据
            decoded_check = base64.b64decode(encoded_data)
            Image.open(io.BytesIO(decoded_check))  # 确认可解码
            return encoded_data
        except Exception as e:
            raise ValueError(f"图片处理失败: {str(e)}")