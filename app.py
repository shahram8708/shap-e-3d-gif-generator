import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif
from PIL import Image
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_id = "openai/shap-e"
pipe = ShapEPipeline.from_pretrained(ckpt_id).to(device)

def generate_3d_gif(prompt: str, guidance_scale: float, num_steps: int):
    try:
        result = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps
        )
        images = result.images 

        flat_images = [img for sublist in images for img in sublist] if isinstance(images[0], list) else images

        gif_path = export_to_gif(flat_images, "shark_3d.gif")
        return gif_path
    except Exception as e:
        return str(e)

with gr.Blocks(theme=gr.themes.Default(), css=".block {padding: 1rem;}") as demo:
    gr.Markdown(
        """
        ## 🦈 **3D Object Generation with Shap-E**
        Create stunning 3D objects and export them as animated GIFs using OpenAI's Shap-E model!
        """
    )
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter the object description (e.g., 'a shark')",
                lines=1
            )
            guidance_input = gr.Slider(
                label="Guidance Scale",
                minimum=1.0,
                maximum=20.0,
                value=15.0,
                step=0.1
            )
            steps_input = gr.Slider(
                label="Number of Inference Steps",
                minimum=20,
                maximum=100,
                value=64,
                step=1
            )
            generate_btn = gr.Button("Generate 3D GIF")
        with gr.Column(scale=1):
            gif_output = gr.Image(label="Generated GIF", type="filepath")
    
    debug_toggle = gr.Checkbox(label="Enable Debugging", value=True)
    
    generate_btn.click(
        generate_3d_gif,
        inputs=[prompt_input, guidance_input, steps_input],
        outputs=gif_output
    )
    if debug_toggle:
        gr.Textbox(value="Debugging Enabled: Any errors will be displayed here.", label="Debug Status")

demo.launch(debug=True)
