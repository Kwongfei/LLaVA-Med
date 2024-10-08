import argparse
import json
import datetime
import os
import hashlib
from llava.constants import LOGDIR
from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
import time

import requests
from PIL import Image
import base64
from io import BytesIO
import imghdr

from llava.conversation import conv_templates

def add_image_and_message_to_conversation(conv, image, message):
    # 将图像转换为 base64 编码字符串
    buffered = BytesIO()
    # image.save(buffered, format=image.format)
    image.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    # 添加带有图像编码的消息到对话
    conv.append_message(conv.roles[0], (message, img_b64_str, "Default"))
    return conv

def main():
    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(controller_addr + "/get_worker_address",
            json={"model": args.model_name})
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        return

    conv = conv_templates["mistral_instruct"].copy()
    conv.append_message(conv.roles[0], args.message)
    prompt = conv.get_prompt()

    headers = {"User-Agent": "LLaVA Client"}
    pload = {
        "model": args.model_name,
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.7,
        "stop": conv.sep2,
    }
    response = requests.post(worker_addr + "/worker_generate_stream", headers=headers,
            json=pload, stream=True)

    print(prompt, end="")
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"].split("[/INST]")[-1]
            print(output, end="\r")
    print("")

def test_image():
    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(controller_addr + "/get_worker_address",
            json={"model": args.model_name})
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        return

    conv = conv_templates["llava_llama_2"].copy()
    # 假设你有一个图片对象和一个文字消息
    image = Image.open("/data/guangfei/medical/classification_models/data/3segcrop_15_622_NoRandom/train/class 2/" + "L3-0001-1.jpg")
    message = "Please identify this picture and give a description of it clearly."
    image_process_mode = "Default"

    text = (message, image, image_process_mode)
    conv.append_message(conv.roles[0], text)

    # 添加图片和文字消息到对话
    # updated_conv = add_image_and_message_to_conversation(conv, image, message)

    prompt = conv.get_prompt()

    all_images = conv.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    headers = {"User-Agent": "LLaVA Client"}
    # Make requests
    pload = {
        "model": args.model_name,
        "prompt": prompt,
        "temperature": float(0.7),
        "top_p": float(0.7),
        "max_new_tokens": min(int(args.max_new_tokens), 1536),
        "stop": conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2,
        "images": f'List of {len(conv.get_images())} images: {all_image_hash}',
    }

    pload['images'] = conv.get_images()

    response = requests.post(worker_addr + "/worker_generate_stream", headers=headers, json=pload, stream=True)

    print(prompt, end="")
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            # data = json.loads(chunk.decode("utf-8"))
            # output = data["text"].split("[/INST]")[-1]
            # print(output, end="\r")
            data = json.loads(chunk.decode())
            if data["error_code"] == 0:
                output = data["text"][len(prompt):].strip()
                conv.messages[-1][-1] = output + "▌"
            else:
                output = data["text"] + f" (error_code: {data['error_code']})"
                conv.messages[-1][-1] = output
                return
            time.sleep(0.03)
            print(output, end="\r")
    print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--message", type=str, default=
        "Tell me a story with more than 1000 words.")
    args = parser.parse_args()

    main()
    # test_image()
