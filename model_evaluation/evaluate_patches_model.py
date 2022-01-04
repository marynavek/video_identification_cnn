import os

import argparse

from dataset_creation.prepare_dataset_patches import DataSetGeneratorPatches
from predictions.patches_prediction_stats import PatchesPredictionStatistics
from predictions.patches_predictions_vis import PatchesPredictionVis
from predictions.patches_predictor import PredictPatches

parser = argparse.ArgumentParser(
    description='Make predictions with signature network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_dir', type=str, required=True, help='Path to directory consisting of .h5-models (to use for predicting)')
parser.add_argument('--models', type=str, required=False, help='Models within input dir (*.h5) to evaluate. Separate models by a ",". ')
parser.add_argument('--constrained', type=int, required=False, default=1, help='Constrained layer included (0=no, 1=yes)')
# parser.add_argument('--dataset', type=str, required=True, help='Dataset to use to make predictions')
parser.add_argument('--batch_size', type=int, required=False, default=64, help='Batch size')
parser.add_argument('--height', type=int, required=False, default=128, help='Height of CNN input dimension [default=480]')
parser.add_argument('--width', type=int, required=False, default=128, help='Width of CNN input dimension [default=800]')
parser.add_argument('--dataset_path', type=str, required=True, help='Path where both noise and frames datasets are located')
parser.add_argument('--noise_dataset_name', type=str, required=True, help='Folder name where noise datasets are located')
parser.add_argument('--frames_dataset_name', type=str, required=True, help='Folder name where frames are located')



#this model returns the path for the dataset containing the test/ data
def get_test_datasets_path(datasets_path, noise_dataset_name, frames_dataset_name):
    test_noise_ds = os.path.join(datasets_path, noise_dataset_name)
    test_frames_ds = os.path.join(datasets_path, frames_dataset_name)
    return test_noise_ds, test_frames_ds

#this folder gets the name of all devices
def get_devices(dataset_path):
    test_dir = os.path.join(dataset_path, "test")
    devices = [item for item in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, item))]
    return devices

#this function gets and returns the list of the saved trained models as .h5 files
def get_models_files(input_dir, models):
    # Get all files (i.e. models) from input directory
    files_list = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files_list = sorted(files_list)
    print(f"Found {len(files_list)} files in {input_dir}: {files_list}")

    if models:
        print(f"'Models' argument is set. Only the following models will be evaluated: {models.split(',')}")
        model_split = models.split(',')
        return [file for file in files_list if file in model_split]

    return files_list

#this function creates and sets directories with predictions results
def get_result_dir(input_dir):
    output_dir = os.path.join(input_dir, "predictions")

    # Create output directory is not exists
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(e)
            raise ValueError(f"Error during creation of output directory")

    frames_output_dir = os.path.join(output_dir, "patches")
    videos_output_dir = os.path.join(output_dir, "videos")
    plots_output_dir = os.path.join(output_dir, "plots")
    if not os.path.isdir(frames_output_dir):
        os.makedirs(frames_output_dir)
    if not os.path.isdir(videos_output_dir):
        os.makedirs(videos_output_dir)
    if not os.path.isdir(plots_output_dir):
        os.makedirs(plots_output_dir)

    return output_dir, frames_output_dir, videos_output_dir, plots_output_dir


if __name__ == "__main__":
    args = parser.parse_args()
    model_input_dir = args.input_dir
    models_to_process = args.models
    batch_size = args.batch_size
    height = args.height
    width = args.width
    constrained = args.constrained == 1

    noise_ds_path, frames_ds_path = get_test_datasets_path(args.dataset_path, args.noise_dataset_name, args.frames_dataset_name)

    model_name = model_input_dir.split(os.path.sep)[-1]
    model_files = get_models_files(model_input_dir, models_to_process)
    _, frames_res_dir, videos_res_dir, plots_res_dir = get_result_dir(model_input_dir)

    print(f"Found {len(model_files)} files for model {model_name}")

    for model_file in model_files:
        print(f"{model_file} | Start prediction process")

        dataset_factory = DataSetGeneratorPatches(input_dir_frames=frames_ds_path,
                            input_dir_noiseprint = noise_ds_path)

        valid_dataset_dict = dataset_factory.create_test_dataset()

        print(f"{model_file} | Start predicting frames")
        # Predict Frames
        frame_predictor = PredictPatches(model_dir=model_input_dir, model_fname=model_file, results_dir=frames_res_dir,
                                         constrained=constrained)
        
        files_with_predictions = frame_predictor.start_predictions(test_dictionary=valid_dataset_dict)

        print(f"{model_file} | Predicting frames completed")

    print(f"Creating Statistics and Visualizations ...")
    # Create Frame Prediction Statistics
    fps =  PatchesPredictionStatistics(result_dir=frames_res_dir)
    frame_stats = fps.start()
    print(f"Frame Prediction Statistics Completed")

    fpv = PatchesPredictionVis(result_dir=plots_res_dir)
    fpv.start(frame_stats)
    print(f"Frame Prediction Visualization Completed")
