import os

# File listing functions
def list_reference_files():
    ref_dir = "data/reference_data/"
    try:
        files = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".json")]
        return files if files else ["No .json files found in data/reference_data/"]
    except FileNotFoundError:
        return ["Directory data/reference_data/ not found"]
    except Exception as e:
        return [f"Error listing files: {str(e)}"]

def list_mapping_files():
    map_dir = "ckpts/"
    try:
        files = [os.path.join(map_dir, f) for f in os.listdir(map_dir) if f.endswith(".json")]
        return files if files else ["No .json files found in ckpts/"]
    except FileNotFoundError:
        return ["Directory ckpts/ not found"]
    except Exception as e:
        return [f"Error listing files: {str(e)}"]

def list_classifier_files():
    clf_dir = "ckpts/"
    try:
        files = [os.path.join(clf_dir, f) for f in os.listdir(clf_dir) if f.endswith(".pth")]
        return files if files else ["No .pth files found in ckpts/"]
    except FileNotFoundError:
        return ["Directory ckpts/ not found"]
    except Exception as e:
        return [f"Error listing files: {str(e)}"]

def list_edgeface_files():
    ef_dir = "ckpts/idiap/"
    try:
        files = [os.path.join(ef_dir, f) for f in os.listdir(ef_dir) if f.endswith(".pt")]
        return files if files else ["No .pt files found in ckpts/idiap/"]
    except FileNotFoundError:
        return ["Directory ckpts/idiap/ not found"]
    except Exception as e:
        return [f"Error listing files: {str(e)}"]
