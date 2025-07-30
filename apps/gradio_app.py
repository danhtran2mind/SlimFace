import gradio as gr
from PIL import Image
from gradio_app.inference import run_inference
from gradio_app.components import (
    CONTENT_DESCRIPTION, CONTENT_OUTTRO,
    CONTENT_IN_1, CONTENT_IN_2,
    CONTENT_OUT_1, CONTENT_OUT_2,
    list_reference_files, list_mapping_files,
    list_classifier_files, list_edgeface_files
)
from glob import glob
import os

def create_image_io_row():
    """Create the row for image input and output display."""
    with gr.Row(elem_classes=["image-io-row"]):
        image_input = gr.Image(type="pil", label="Upload Image")
        output = gr.HTML(label="Inference Results", elem_classes=["results-container"])
    return image_input, output

def create_model_settings_row():
    """Create the row for model files and settings."""
    with gr.Row():
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
    return ref_dict, index_map, classifier_model, edgeface_model, algorithm, accelerator, resolution, similarity_threshold

# Load local CSS file
CSS = open("apps/gradio_app/static/styles.css").read()

def create_interface():
    """Create the Gradio interface for SlimFace."""
    with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# SlimFace Demonstration")
        gr.Markdown(CONTENT_DESCRIPTION)
        gr.Markdown(CONTENT_IN_1)
        gr.HTML(CONTENT_IN_2)

        image_input, output = create_image_io_row()
        ref_dict, index_map, classifier_model, edgeface_model, algorithm, accelerator, resolution, similarity_threshold = create_model_settings_row()
        
        # Add example image gallery as a row of columns
        with gr.Group():
            gr.Markdown("### Example Images")
            example_images = glob("apps/assets/examples/*.[jp][pn][gf]")
            if example_images:
                with gr.Row(elem_classes=["example-row"]):
                    for img_path in example_images:
                        with gr.Column(min_width=120):
                            gr.Image(
                                value=img_path,
                                label=os.path.basename(img_path),
                                type="filepath",
                                height=100,
                                elem_classes=["example-image"]
                            )
                            gr.Button(f"Use {os.path.basename(img_path)}").click(
                                fn=lambda x=img_path: Image.open(x),
                                outputs=image_input
                            )
            else:
                gr.Markdown("No example images found in apps/assets/examples/")

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
        gr.Markdown(CONTENT_OUTTRO)
        gr.HTML(CONTENT_OUT_1)
        gr.Markdown(CONTENT_OUT_2)
    return demo

def main():
    """Launch the Gradio interface."""
    demo = create_interface()
    demo.launch()

if __name__ == "__main__":
    main()