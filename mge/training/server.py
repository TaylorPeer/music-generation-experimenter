from os import listdir
from os.path import isfile, join
from json.decoder import JSONDecodeError
import sys
import shutil
import datetime
from pandarallel import pandarallel

from mge.dataloaders import *
from mge.filters import *
from mge.transformers import *
from mge.models import *
from mge.evaluators import *

JOBS_BASE_DIR = "jobs/"
JOBS_NEW = "new/"
JOBS_RUNNING = "running/"
JOBS_ERROR = "error/"
JOBS_COMPLETED = "completed/"

MODELS_BASE_DIR = "resources/models"


class ConfigException(BaseException):
    pass


class PipelineException(BaseException):
    pass


def get_models_dir(model_type):
    dt = datetime.datetime.now()
    return "{}/{}/{}/{}/{}/{}/{}/{}/".format(MODELS_BASE_DIR, dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second,
                                             model_type.lower())


def run_job(job_config):
    path = JOBS_BASE_DIR + JOBS_RUNNING + job_config
    with open(path) as config_file:
        job = json.load(config_file)

        # Check that required settings are present
        validate_configuration(job, ["dataloader", "model", "evaluators"], ["pipeline"], ["type", "settings"])

        # Set model directory dynamically based on timestamp and model type
        job["model"]["settings"]["model_directory"] = get_models_dir(job["model"]["type"])

        # Instantiate job pipeline items
        dataloader = instantiate_object(job["dataloader"]["type"], job["dataloader"]["settings"])
        model = instantiate_object(job["model"]["type"], job["model"]["settings"])
        pipeline = []
        if "pipeline" in job:
            for item_config in job["pipeline"]:
                pipeline_item = instantiate_object(item_config["type"], item_config["settings"])
                pipeline.append(pipeline_item)
        evaluators = []
        for evaluator_config in job["evaluators"]:
            pipeline_item = instantiate_object(evaluator_config["type"], evaluator_config["settings"])
            evaluators.append(pipeline_item)

        # TODO extend status print outs

        # Load data
        data = dataloader.load(job["data_directory"])

        # Apply evaluators to raw input data
        job["results"] = {}
        midi_paths = [path for (path, df) in data]
        job["results"]["input_dataset"] = evaluate(midi_paths, evaluators)

        # Apply pipeline items:
        for p in pipeline:
            if issubclass(type(p), Filter):
                data = p.filter(data)
            elif issubclass(type(p), Transformer):
                data = p.transform(data)
            else:
                raise PipelineException("Configured pipeline item was neither a Filter nor Transformer")

        # Apply evaluators to processed input data
        midi_paths = [path for (path, df) in data]
        job["results"]["processed_dataset"] = evaluate(midi_paths, evaluators)

        # Train model
        model.train(data)

        # Generate sample output
        for _ in range(10):  # TODO parameterize
            try:
                model.generate()
            except GenerationException as e:
                print("Error occurred during sequence generation. {}".format(str(e)))

        # Apply evaluators to generated output
        output_dir = model.get_output_dir()
        midi_paths = []
        for file in os.listdir(output_dir):
            if file.endswith(".mid"):
                path_to_midi = os.path.join(output_dir, file)
                midi_paths.append(path_to_midi)
        job["results"]["generated"] = evaluate(midi_paths, evaluators)

        # Write job status to disk
        update_running_job(path, job)

        # TODO write model training time to disk


def update_running_job(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def move_job_running(job):
    shutil.move(JOBS_BASE_DIR + JOBS_NEW + job, JOBS_BASE_DIR + JOBS_RUNNING + job)


def move_job_error(job):
    shutil.move(JOBS_BASE_DIR + JOBS_RUNNING + job, JOBS_BASE_DIR + JOBS_ERROR + job)


def move_job_completed(job):
    shutil.move(JOBS_BASE_DIR + JOBS_RUNNING + job, JOBS_BASE_DIR + JOBS_COMPLETED + job)


def instantiate_object(obj_type, settings=None):
    if obj_type not in globals():
        raise ConfigException("Configured object is not defined: {}".format(obj_type))
    class_name = globals()[obj_type]
    if settings is None:
        obj = class_name()
    else:
        obj = class_name(settings)
    return obj


def validate_configuration(config, required_items, optional_items, sub_items):
    missing_fields = []
    if "data_directory" not in config:
        missing_fields.append("data_directory")
    for item in required_items:
        if item not in config:
            missing_fields.append(item)
        else:
            missing_fields.extend(validate_item_configuration(config[item], item, sub_items))
    for item in optional_items:
        if item in config:
            missing_fields.extend(validate_item_configuration(config[item], item, sub_items))
    if len(missing_fields) > 0:
        template = "The following required fields were missing from job configuration file: {}"
        message = template.format(", ".join(missing_fields))
        raise ConfigException(message)


def evaluate(midi_paths, evaluators):
    results = {"evaluation": [], "file_count": len(midi_paths)}
    for midi_path in midi_paths:
        filename = midi_path.split("/")[-1]
        evaluation_results = {"file": filename, "scores": {}}
        for evaluator in evaluators:
            evaluation_name = evaluator.__class__.__name__
            score = evaluator.evaluate(midi_path)
            evaluation_results["scores"][evaluation_name] = score
        results["evaluation"].append(evaluation_results)
    # TODO: note_count, unique_note_count, time_signatures, tempos
    return results


def validate_item_configuration(config, item, sub_items):
    missing_fields = []
    if isinstance(config, list):
        for i in config:
            for sub_item in sub_items:
                if sub_item not in i:
                    missing_fields.append("{}.{}".format(item, sub_item))
    else:
        for sub_item in sub_items:
            if sub_item not in config:
                missing_fields.append("{}.{}".format(item, sub_item))
    return missing_fields


if __name__ == "__main__":

    tqdm.pandas()
    pandarallel.initialize(progress_bar=False, nb_workers=12)

    if not os.path.exists(JOBS_BASE_DIR + JOBS_NEW):
        print("Required directory for checking for new jobs is missing: {}".format(JOBS_BASE_DIR + JOBS_NEW))
        sys.exit()

    # Create required paths if they do not already exist
    for required_path in [JOBS_RUNNING, JOBS_ERROR, JOBS_COMPLETED]:
        path = JOBS_BASE_DIR + required_path
        if not os.path.exists(path):
            print("Created missing required path: {}".format(path))
            os.mkdir(path)

    # Check if new jobs have been added
    new_jobs_dir = JOBS_BASE_DIR + JOBS_NEW
    new_job_configs = [f for f in listdir(new_jobs_dir) if isfile(join(new_jobs_dir, f)) and f.endswith(".json")]
    new_job_count = len(new_job_configs)
    if new_job_count > 0:
        print("Found {} new jobs to execute".format(new_job_count))

        for job_config in new_job_configs:
            try:
                move_job_running(job_config)
                run_job(job_config)
                move_job_completed(job_config)
            except JSONDecodeError as e:
                print("Error occurred while trying to read job config. JSON seems invalid: {}. Aborting job.".format(
                    str(e)))
                move_job_error(job_config)
            except ConfigException as e:
                print("Error occurred while trying to start job. {}. Aborting job.".format(str(e)))
                move_job_error(job_config)
            except DataLoadingException as e:
                print("Error occurred during data loading. {}. Aborting job.".format(str(e)))
                move_job_error(job_config)
            except TrainingException as e:
                print("Error occurred during training. {}. Aborting job.".format(str(e)))
                move_job_error(job_config)
            except Exception as e:
                print("Unknown error occurred: '{}'. Aborting job.".format(str(e)))
                move_job_error(job_config)
