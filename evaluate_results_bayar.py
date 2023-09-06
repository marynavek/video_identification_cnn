import os

import argparse

from frame_prediction_stats import FramePredictionStatistics
from new_predict_frames import PredictFrames
from frame_prediction_vizualization import FramePredictionVis
from prepare_csv_dataset_noise_patch import DataSetGeneratorNoisePatch
from prepare_csv_dataset_singular import DataSetGeneratorSingular
from preparing_csv_dataset import DataSetGenerator

parser = argparse.ArgumentParser(
    description='Make predictions with signature network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model_input_dir', type=str, required=True, help='Path to directory consisting of .h5-models (to use for predicting)')
parser.add_argument('--batch_size', type=int, required=True, default=64, help='Batch size')
parser.add_argument('--dataset_path', type=str, required=False, help='Path where both prnu and frames datasets are located')
parser.add_argument('--dataset_name', type=str, required=False, help='Folder name where frames are located')
parser.add_argument('--type_of_dataset', type=str, required=True)


#this model returns the path for the dataset containing the test/ data
def get_test_datasets_path(datasets_path, frames_dataset_name):
    test_frames_ds = os.path.join(datasets_path, frames_dataset_name)
    return test_frames_ds

#this folder gets the name of all devices
def get_devices(dataset_path):
    test_dir = os.path.join(dataset_path, "test")
    devices = [item for item in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, item))]
    return devices

#this function gets and returns the list of the saved trained models as .h5 files
def get_models_files(input_dir):
    # Get all files (i.e. models) from input directory
    files_list = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and not f.startswith(".")]
    files_list = sorted(files_list)
    print(f"Found {len(files_list)} files in {input_dir}: {files_list}")

    return files_list

#this function creates and sets directories with predictions results
def get_result_dir(input_dir, batch_size, type_of_dataset):
    directory_dir = input_dir.split("/")
    input_dir_final = input_dir.replace(directory_dir[len(directory_dir)-1], "")
    
    output_dir = os.path.join(input_dir_final, "predictions", type_of_dataset, str(batch_size))

    # Create output directory is not exists
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(e)
            raise ValueError(f"Error during creation of output directory")

    frames_output_dir = os.path.join(output_dir, "frames")
    videos_output_dir = os.path.join(output_dir, "videos")
    plots_output_dir = os.path.join(output_dir, "plots")
    total_stats_output_dir = os.path.join(output_dir, "total_stats")
    if not os.path.isdir(frames_output_dir):
        os.makedirs(frames_output_dir)
    if not os.path.isdir(videos_output_dir):
        os.makedirs(videos_output_dir)
    if not os.path.isdir(plots_output_dir):
        os.makedirs(plots_output_dir)
    if not os.path.isdir(total_stats_output_dir):
        os.makedirs(total_stats_output_dir)

    return output_dir, frames_output_dir, videos_output_dir, plots_output_dir, total_stats_output_dir


if __name__ == "__main__":
    args = parser.parse_args()
    model_input_dir = args.model_input_dir
    batch_size = args.batch_size
    type_of_dataset = args.type_of_dataset

    # frames_ds_path = get_test_datasets_path(args.dataset_path, args.dataset_name)

    model_name = model_input_dir.split(os.path.sep)[-1]
    model_files = get_models_files(model_input_dir)
    _, frames_res_dir, videos_res_dir, plots_res_dir, total_stats_dir = get_result_dir(model_input_dir, batch_size, type_of_dataset)

    print(f"Found {len(model_files)} files for model {model_name}")
    noise_dataset = "/Users/marynavek/Projects/files/equalized_noise_iframes_minimum_3rd_sector_all"
    reg_patches_dataset = "/Users/marynavek/Projects/files/equalized_patches_iframes_minimum_4th_sector_all"
    
    for model_file in model_files:
        print(f"{model_file} | Start prediction process")

        # dataset_factory = DataSetGeneratorSingular(input_dir_frames=frames_ds_path)
        data_factory = DataSetGeneratorNoisePatch(input_dir_frames=noise_dataset,
                            input_dir_noiseprint=reg_patches_dataset)

        valid_dataset_dict = data_factory.create_test_dataset()


        ###test_ds_filenames will be extracted from csv

        print(f"{model_file} | Start predicting frames")
        # Predict Frames
        frame_predictor = PredictFrames(model_dir=model_input_dir, model_fname=model_file, results_dir=frames_res_dir, constrained=True)
        
        files_with_predictions = frame_predictor.start_predictions(test_dictionary=valid_dataset_dict, batch_size=batch_size)
        # frame_predictor.another_predict(test_dictionary=valid_dataset_dict)

        print(f"{model_file} | Predicting frames completed")

    print(f"Creating Statistics and Visualizations ...")
    # Create Frame Prediction Statistics
    fps =  FramePredictionStatistics(result_dir=frames_res_dir, save_dir=total_stats_dir)
    frame_stats = fps.start()
    print(f"Frame Prediction Statistics Completed")


    fpv = FramePredictionVis(result_dir=plots_res_dir)
    fpv.start(frame_stats)
    print(f"Frame Prediction Visualization Completed")














