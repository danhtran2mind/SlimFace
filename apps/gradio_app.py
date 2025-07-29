import gradio as gr
from gradio_app.inference import run_inference

def create_gradio_interface():
    return gr.Interface(
        fn=run_inference,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.File(label="Reference Dict JSON File"),
            gr.File(label="Index to Class Mapping JSON File"),
            gr.File(label="Classifier Model (.pth) File"),
            gr.Textbox(label="EdgeFace Model Name", value="edgeface_base"),
            gr.Textbox(label="EdgeFace Model Directory", value="ckpts/idiap"),
            gr.Dropdown(choices=["yolo", "mtcnn"], label="Face Detection Algorithm", value="yolo"),
            gr.Dropdown(choices=["auto", "cpu", "gpu"], label="Accelerator", value="auto"),
            gr.Slider(minimum=112, maximum=448, step=1, value=224, label="Resolution"),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.6, label="Similarity Threshold")
        ],
        outputs="text",
        title="Face Classification with EdgeFace Validation",
        description="Upload an image and required files to perform face classification with EdgeFace embedding validation."
    )

if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch()