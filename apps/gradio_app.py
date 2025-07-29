import os
import sys
import gradio as gr
from gradio_app.inference import run_inference

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def create_gradio_interface():
    def update_reference_input(choice):
        if choice == "Use Existing":
            return gr.Textbox(label="Reference Dict JSON File Path", value="tests/reference_image_data.json", visible=True), gr.File(label="Upload Reference Dict JSON File", visible=False)
        else:
            return gr.Textbox(visible=False), gr.File(label="Upload Reference Dict JSON File", visible=True)

    def update_mapping_input(choice):
        if choice == "Use Existing":
            return gr.Textbox(label="Index to Class Mapping JSON File Path", value="ckpts/index_to_class_mapping.json", visible=True), gr.File(label="Upload Index to Class Mapping JSON File", visible=False)
        else:
            return gr.Textbox(visible=False), gr.File(label="Upload Index to Class Mapping JSON File", visible=True)

    def update_classifier_input(choice):
        if choice == "Use Existing":
            return gr.Textbox(label="Classifier Model (.pth) File Path", value="ckpts/SlimFace_efficientnet_b3_full_model.pth", visible=True), gr.File(label="Upload Classifier Model (.pth) File", visible=False)
        else:
            return gr.Textbox(visible=False), gr.File(label="Upload Classifier Model (.pth) File", visible=True)

    def update_edgeface_input(choice):
        if choice == "Use Existing":
            return gr.Textbox(label="EdgeFace Model (.pt) File Path", value="ckpts/idiap/edgeface_base.pt", visible=True), gr.File(label="Upload EdgeFace Model (.pt) File", visible=False)
        else:
            return gr.Textbox(visible=False), gr.File(label="Upload EdgeFace Model (.pt) File", visible=True)

    def run_inference_wrapper(image, ref_choice, ref_file, ref_upload, map_choice, map_file, map_upload, clf_choice, clf_file, clf_upload, ef_choice, ef_file, ef_upload, face_detection, accelerator, resolution, threshold):
        ref_input = ref_file if ref_choice == "Use Existing" else ref_upload
        map_input = map_file if map_choice == "Use Existing" else map_upload
        clf_input = clf_file if clf_choice == "Use Existing" else clf_upload
        ef_input = ef_file if ef_choice == "Use Existing" else ef_upload
        return run_inference(image, ref_input, map_input, clf_input, ef_input, face_detection, accelerator, resolution, threshold)

    with gr.Blocks(title="Face Classification with EdgeFace Validation") as iface:
        gr.Markdown("# Face Classification with EdgeFace Validation")
        gr.Markdown("Upload an image and required files or use existing file paths to perform face classification with EdgeFace embedding validation.")

        image_input = gr.Image(type="pil", label="Upload Image")

        with gr.Row():
            ref_choice = gr.Dropdown(choices=["Use Existing", "Upload New"], label="Reference Dict JSON File", value="Use Existing")
            ref_file = gr.Textbox(label="Reference Dict JSON File Path", value="tests/reference_image_data.json", visible=True)
            ref_upload = gr.File(label="Upload Reference Dict JSON File", visible=False)
        ref_choice.change(fn=update_reference_input, inputs=ref_choice, outputs=[ref_file, ref_upload])

        with gr.Row():
            map_choice = gr.Dropdown(choices=["Use Existing", "Upload New"], label="Index to Class Mapping JSON File", value="Use Existing")
            map_file = gr.Textbox(label="Index to Class Mapping JSON File Path", value="ckpts/index_to_class_mapping.json", visible=True)
            map_upload = gr.File(label="Upload Index to Class Mapping JSON File", visible=False)
        map_choice.change(fn=update_mapping_input, inputs=map_choice, outputs=[map_file, map_upload])

        with gr.Row():
            clf_choice = gr.Dropdown(choices=["Use Existing", "Upload New"], label="Classifier Model (.pth) File", value="Use Existing")
            clf_file = gr.Textbox(label="Classifier Model (.pth) File Path", value="ckpts/SlimFace_efficientnet_b3_full_model.pth", visible=True)
            clf_upload = gr.File(label="Upload Classifier Model (.pth) File", visible=False)
        clf_choice.change(fn=update_classifier_input, inputs=clf_choice, outputs=[clf_file, clf_upload])

        with gr.Row():
            ef_choice = gr.Dropdown(choices=["Use Existing", "Upload New"], label="EdgeFace Model (.pt) File", value="Use Existing")
            ef_file = gr.Textbox(label="EdgeFace Model (.pt) File Path", value="ckpts/idiap/edgeface_base.pt", visible=True)
            ef_upload = gr.File(label="Upload EdgeFace Model (.pt) File", visible=False)
        ef_choice.change(fn=update_edgeface_input, inputs=ef_choice, outputs=[ef_file, ef_upload])

        face_detection = gr.Dropdown(choices=["yolo", "mtcnn"], label="Face Detection Algorithm", value="yolo")
        accelerator = gr.Dropdown(choices=["auto", "cpu", "gpu"], label="Accelerator", value="auto")
        resolution = gr.Slider(minimum=112, maximum=448, step=1, value=224, label="Resolution")
        threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.6, label="Similarity Threshold")

        submit_button = gr.Button("Run Inference")
        output = gr.Textbox(label="Result")

        submit_button.click(
            fn=run_inference_wrapper,
            inputs=[
                image_input,
                ref_choice, ref_file, ref_upload,
                map_choice, map_file, map_upload,
                clf_choice, clf_file, clf_upload,
                ef_choice, ef_file, ef_upload,
                face_detection, accelerator, resolution, threshold
            ],
            outputs=output
        )

    return iface

if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch()