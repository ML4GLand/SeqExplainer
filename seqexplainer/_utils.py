import os
import logging

def _make_dirs(
    output_dir,
    overwrite=False,
):
    if os.path.exists(output_dir):
        if overwrite:
            logging.info("Overwriting existing directory: {}".format(output_dir))
            os.system("rm -rf {}".format(output_dir))
        else:
            print("Output directory already exists: {}".format(output_dir))
            return
    os.makedirs(output_dir)

def _path_to_image_html(path):
    return '<img src="'+ path + '" width="240" >'

def _model_to_device(model, device="cpu"):
    """
    """
    model.eval()
    model.to(device)
    return model