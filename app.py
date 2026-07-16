import re
import torch
import gradio as gr
import spaces
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
SYSTEM_PROMPT = "Convert the mathematical equation in this image into valid LaTeX code. Output ONLY the LaTeX code. Do not use markdown blocks."
MAX_NEW_TOKENS = 512

print(f"Carregando {MODEL_ID}...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Modelo pronto!")


def clean_latex_output(text: str) -> str:
    """Remove blocos markdown e delimitadores \\[ \\] de display math do output do modelo."""
    pattern = r"```(?:latex)?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1)
    text = text.strip()
    # Remove apenas \[ no início e \] no fim, não em qualquer posição
    text = re.sub(r"^\\\[|\\\]$", "", text).strip()
    return text


@spaces.GPU
def image_to_latex(image: Image.Image) -> str:
    """Converte uma imagem contendo uma equação matemática em código LaTeX.

    Args:
        image: Imagem (PNG/JPG) contendo uma equação matemática manuscrita ou impressa.

    Returns:
        Código LaTeX correspondente à equação, sem blocos markdown.
    """
    if image is None:
        raise gr.Error("Nenhuma imagem foi enviada.")

    try:
        image = image.convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return clean_latex_output(output_text)

    except torch.cuda.OutOfMemoryError:
        raise gr.Error("GPU Out Of Memory: a imagem é muito complexa.")
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Erro interno de inferência: {str(e)}")


demo = gr.Interface(
    fn=image_to_latex,
    inputs=gr.Image(type="pil", label="Upload Equation"),
    outputs=gr.Textbox(label="LaTeX Code"),
    title="Qwen2-VL LaTeX OCR",
    description="Envie uma imagem de uma equação matemática e receba o código LaTeX correspondente. Também disponível como MCP Server.",
    api_name="predict",
)

if __name__ == "__main__":
    demo.launch(mcp_server=True, show_error=True)