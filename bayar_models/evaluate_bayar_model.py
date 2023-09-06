import os
from patch_predictions_stats_bayar import PatchPredictionStatisticsBayar
from vizualiza_predicition_patches_bayar import PatchPredictionVisBayar

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
    files_list = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files_list = sorted(files_list)
    print(f"Found {len(files_list)} files in {input_dir}: {files_list}")

    return files_list

#this function creates and sets directories with predictions results
def get_result_dir(input_dir):
    # directory_dir = input_dir.split("/")
    # input_dir_final = input_dir.replace(directory_dir[len(directory_dir)-1], "")
    
    output_dir = os.path.join(input_dir, "predictions")

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
    model_input_dir = "/Users/marynavek/Projects/video_identification_cnn/ensemble_results/Rong_experiment"
    dataset_path = "/Users/marynavek/Projects/files"
    patches_dataset_name = "vcmi_patches"

    patches_ds_path = get_test_datasets_path(dataset_path, patches_dataset_name)

    model_name = model_input_dir.split(os.path.sep)[-1]
    model_files = get_models_files(model_input_dir)
    model_input_dir = "/Users/marynavek/Projects/video_identification_cnn/ensemble_results/Rong_experiment"
    _, frames_res_dir, videos_res_dir, plots_res_dir, total_stats_dir = get_result_dir(model_input_dir)
    # frames_res_dir = "/Users/marynavek/Projects/video_identification_cnn/ensemble_results/real_exp_1"
    # total_stats_dir = "/Users/marynavek/Projects/video_identification_cnn/ensemble_results/real_exp_1/total_stats"
    # plots_res_dir = "/Users/marynavek/Projects/video_identification_cnn/ensemble_results/real_exp_1/plots"
    print(f"Found {len(model_files)} files for model {model_name}")

    # for model_file in model_files:
    #     if "fm-e00038" in model_file:
    #         print(f"{model_file} | Start prediction process")

    #         dataset_factory = DataSetGeneratorBayar(input_dir_patchs=patches_ds_path)

    #         valid_dataset_dict = dataset_factory.create_test_dataset()
    #         num_classes = len(dataset_factory.get_class_names())

    #         ##test_ds_filenames will be extracted from csv

    #         print(f"{model_file} | Start predicting frames")
    #         # Predict Frames
    #         frame_predictor = PredictBayarPatches(model_dir=model_input_dir, num_classes=num_classes, model_fname=model_file, results_dir=frames_res_dir)
            
    #         files_with_predictions = frame_predictor.start_predictions(test_dictionary=valid_dataset_dict)
    #         # frame_predictor.another_predict(test_dictionary=valid_dataset_dict)

    #         print(f"{model_file} | Predicting frames completed")

    print(f"Creating Statistics and Visualizations ...")
    # Create Frame Prediction Statistics
    fps =  PatchPredictionStatisticsBayar(result_dir=frames_res_dir, save_dir=total_stats_dir)
    frame_stats = fps.start()
    print(f"Frame Prediction Statistics Completed")


    fpv = PatchPredictionVisBayar(result_dir=plots_res_dir)
    fpv.start(frame_stats)
    print(f"Frame Prediction Visualization Completed")














