import torch
from io import BytesIO
import base64
from diffusers import DiffusionPipeline
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Configuration information for inference."""

    num_inference_steps: int = 100
    controlnet_conditioning_scale = [0.45, 0.25]
    guidance_scale: int = 9


class InferlessPythonModel:
    config = InferenceConfig()

    def initialize(self):
        from accelerate.utils import write_basic_config
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
        import torch

        write_basic_config(mixed_precision="fp16")

        brightness_controlnet = ControlNetModel.from_pretrained(
            "ioclab/control_v1p_sd15_brightness", torch_dtype=torch.float16
        )

        tile_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1e_sd15_tile",
            torch_dtype=torch.float16,
            use_safetensors=False,
        )

        controller = StableDiffusionControlNetPipeline.from_pretrained(
            "Lykon/DreamShaper",
            controlnet=[brightness_controlnet, tile_controlnet],
            torch_dtype=torch.float16,
            use_safetensors=False,
            safety_checker=None,
        ).to("cuda")

        controller.enable_xformers_memory_efficient_attention()

        self.pipe = controller

    @staticmethod
    def resize_for_condition_image(input_image, resolution: int = 512):
        import PIL.Image

        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=PIL.Image.LANCZOS)
        return img
    

    def infer(self, inputs):
        text = inputs["text"]
        input_image = inputs["input_image"]
        import base64
        import io

        import PIL.Image

        print(input_image[:10])
        print(input_image[-10:])
        if "base64," in input_image:
            input_image = input_image.split("base64,")[1]
        input_image = PIL.Image.open(io.BytesIO(base64.b64decode(input_image))).convert(
            "RGB"
        )
        input_image = input_image.resize((512, 512), resample=PIL.Image.LANCZOS)
        tile_input_image = self.resize_for_condition_image(input_image)
        output_image = self.pipe(
            text,
            image=[input_image, tile_input_image],
            height=512,
            width=512,
            num_inference_steps=self.config.num_inference_steps,
            controlnet_conditioning_scale=self.config.controlnet_conditioning_scale,
            guidance_scale=self.config.guidance_scale,
        )["images"][0]

        # blend the input QR code with the output image to improve scanability
        output_image = PIL.Image.blend(input_image, output_image, 0.9)

        buffered = BytesIO()
        output_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

        return {"generated_image_base64": img_str.decode("utf-8")}


    def finalize(self):
        self.pipe = None
