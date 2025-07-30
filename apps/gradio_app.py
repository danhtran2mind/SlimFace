import gradio as gr
from PIL import Image
from gradio_app.inference import run_inference
from gradio_app.components import (
    CONTENT_DESCRIPTION, CONTENT_IN, CONTENT_OUT,
     list_reference_files, list_mapping_files,
      list_classifier_files, list_edgeface_files
)

def create_image_input_column():
    """Create the column for image input and output display."""
    with gr.Column():
        image_input = gr.Image(type="pil", label="Upload Image")
        output = gr.HTML(label="Inference Results", elem_classes=["results-container"])
    return image_input, output

def create_model_files_column():
    """Create the column for model file selection."""
    with gr.Column():
        with gr.Group(elem_classes=["section-group"]):
            gr.Markdown("### Model Files", elem_classes=["section-title"])
            ref_dict = gr.Dropdown(
                choices=["Select a file"] + list_reference_files(),
                label="Reference Dict JSON",
                value="data/reference_data/reference_image_data.json"
            )
            index_map = gr.Dropdown(
                choices=["Select a file"] + list_mapping_files(),
                label="Index to Class Mapping JSON",
                value="ckpts/index_to_class_mapping.json"
            )
            classifier_model = gr.Dropdown(
                choices=["Select a file"] + list_classifier_files(),
                label="Classifier Model (.pth)",
                value="ckpts/SlimFace_efficientnet_b3_full_model.pth"
            )
            edgeface_model = gr.Dropdown(
                choices=["Select a file"] + list_edgeface_files(),
                label="EdgeFace Model (.pt)",
                value="ckpts/idiap/edgeface_s_gamma_05.pt"
            )
    return ref_dict, index_map, classifier_model, edgeface_model

def create_settings_column():
    """Create the column for advanced settings."""
    with gr.Column():
        with gr.Group(elem_classes=["section-group"]):
            gr.Markdown("### Advanced Settings", elem_classes=["section-title"])
            algorithm = gr.Dropdown(
                choices=["yolo", "mtcnn", "retinaface"],
                label="Detection Algorithm",
                value="yolo"
            )
            accelerator = gr.Dropdown(
                choices=["auto", "cpu", "cuda", "mps"],
                label="Accelerator",
                value="auto"
            )
            resolution = gr.Slider(
                minimum=128,
                maximum=512,
                step=32,
                label="Image Resolution",
                value=300
            )
            similarity_threshold = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                step=0.05,
                label="Similarity Threshold",
                value=0.3
            )
    return algorithm, accelerator, resolution, similarity_threshold

def create_interface():
    """Create the Gradio interface for SlimFace."""
    with gr.Blocks(css="gradio_app/static/styles.css", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# SlimFace Demonstration")
        gr.Markdown(CONTENT_DESCRIPTION)
        gr.HTML(CONTENT_IN)
        
        with gr.Row():
            image_input, output = create_image_input_column()
            ref_dict, index_map, classifier_model, edgeface_model = create_model_files_column()
        
        with gr.Row():
            algorithm, accelerator, resolution, similarity_threshold = create_settings_column()
        
        with gr.Row():
            submit_btn = gr.Button("Run Inference", variant="primary", elem_classes=["centered-button"])
        
        submit_btn.click(
            fn=run_inference,
            inputs=[
                image_input,
                ref_dict,
                index_map,
                classifier_model,
                edgeface_model,
                algorithm,
                accelerator,
                resolution,
                similarity_threshold
            ],
            outputs=output
        )
        gr.Markdown(CONTENT_OUT)
    return demo

def main():
    """Launch the Gradio interface."""
    demo = create_interface()
    demo.launch()

if __name__ == "__main__":
    main()