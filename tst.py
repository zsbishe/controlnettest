import torch
from pytorch_lightning import LightningModule
from annotator.midas import MidasDetector
import gradio as gr

from annotator.util import resize_image, HWC3
checkpoint_path = "C:\\Users\\Administrator\\Desktop\\control\\ControlNet-main\\lightning_logs\\version_4\\checkpoints\\epoch=999-step=999.ckpt"

model_midas = None

def midas(img, res, a):
    img = resize_image(HWC3(img), res)
    
    global model_midas
    if model_midas is None:
        model_midas = MidasDetector()
        
        # 加载模型权重（假设检查点中的'date_dict'键下存储了模型权重）
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model_midas.load_state_dict(checkpoint['state_dict'])
        
        # 设置模型为评估模式
        model_midas.eval()
    
    results = model_midas(img, a)
    return results

block = gr.Blocks().queue()

with block:
    with gr.Row():
        gr.Markdown("## MIDAS Depth and Normal")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            alpha = gr.Slider(label="alpha", minimum=0.1, maximum=20.0, value=6.2, step=0.01)
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=384, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=midas, inputs=[input_image, resolution, alpha], outputs=[gallery])

block.launch(server_name='0.0.0.0', share=True)