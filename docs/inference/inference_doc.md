```bash
python src/slim_face/inference/inference.py \
    --input_path <image_path> \
    --model_path <model_path> \
    --index_to_class_mapping_path <index_to_class_mapping_json_path>
```

## Example Usage

```bash
python src/slim_face/inference/inference.py \
    --input_path "assets/test_images/Elon_Musk.jpg" \
    --model_path "ckpts/slim_face_regnet_y_800mf_full_model.pth" \
    --index_to_class_mapping_path ckpts/index_to_class_mapping.json
```