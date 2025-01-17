from ultralytics import YOLO
import argparse, glob, os

def list_image_files(folder):
    """
    Lists all image files within a given folder.

    Args:
      folder: The path to the directory containing the images.

    Returns:
      A list of strings, where each string is the full path to an image file.
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]  # what extensions have I missed?
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder, f"*{ext}")))

    return image_files

def predict_folder(folder, yolo_model):
    """
    Use YOLO Predict mode to detect objects in images in a specified folder using a specified YOLO pretrained model

    :param folder: all images in this folder will be used for the detection task
    :param yolo_model:  pretrained YOLO model filename
    :return:
    """
                                            # Load a model
    print(f"Using model file {yolo_model}")
    model = YOLO(yolo_model)                # pretrained YOLO11n model

                                            # Run batched inference on a list of images
    all_images = list_image_files(folder)
    print(f"Collected all images in {folder}")
    results = model(all_images, stream=True)    # return a list of Results objects

                                            # Process results list
    for result in results:
        boxes = result.boxes                # Boxes object for bounding box outputs
        masks = result.masks                # Masks object for segmentation masks outputs
        keypoints = result.keypoints        # Keypoints object for pose outputs
        probs = result.probs                # Probs object for classification outputs
        obb = result.obb                    # Oriented boxes object for OBB outputs
        result.show()                       # display to screen
        result.save(filename="result.jpg")  # save to disk

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test YOLO11 Predict mode with all image files in a folder")
    parser.add_argument("image_folder", type=str, nargs="?", help="Path to the folder of image files",
                        default="/home/sunrise/PycharmProjects/d-robotics/images")
    parser.add_argument("model_file", type=str, nargs="?", help="Model filename",
                        default="yolo11n.pt")
    args = parser.parse_args()
    image_folder = args.image_folder
    model_file = args.model_file
    predict_folder(image_folder, model_file)
