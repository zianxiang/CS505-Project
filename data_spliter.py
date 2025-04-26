import os
import json
from transformers import pipeline
from collections import defaultdict
import copy

def analyze_sentiment_and_preserve_structure(list_of_file_paths):
    """
    Reads JSON dataset files, applies sentiment analysis to input text,
    and returns a dictionary mapping file paths to metadata and analyzed instances.

    Args:
        list_of_file_paths (list): Paths to the JSON dataset files.

    Returns:
        dict: Dictionary with original file paths as keys and dicts containing
              metadata and analyzed instances as values.
              Returns empty dict if errors occur during pipeline init.
    """
    print("Initializing Sentiment Analysis pipeline...")
    try:
        sentiment_analyzer = pipeline("sentiment-analysis")
        print("Pipeline initialized successfully.")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("Make sure 'transformers' and a backend (torch/tensorflow) are installed.")
        return {}

    processed_data_by_file = {}
    total_instances_processed = 0

    for file_path in list_of_file_paths:
        print(f"\n--- Analyzing file: {os.path.basename(file_path)} ---")
        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}. Skipping.")
            continue

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)

            metadata = {k: v for k, v in data.items() if k != "Instances"}
            analyzed_instances_list = []

            if "Instances" not in data or not isinstance(data["Instances"], list):
                print(f"Warning: No 'Instances' list found in {file_path}. Skipping file processing.")
                processed_data_by_file[file_path] = {
                    "metadata": metadata,
                    "analyzed_instances": []
                }
                continue

            num_instances = len(data["Instances"])
            print(f"Found {num_instances} instances.")

            file_instances_processed = 0
            for i, instance in enumerate(data["Instances"]):
                input_text = instance.get("input")
                instance_id = instance.get("id", f"instance_{i}")

                if not input_text:
                    print(f"Warning: Instance {instance_id} in {os.path.basename(file_path)} has no 'input' text. Skipping analysis for this instance.")
                    predicted_label = "UNKNOWN"
                    predicted_score = 0.0
                else:
                    try:
                        max_length = 512
                        sentiment_result = sentiment_analyzer(input_text[:max_length])[0]
                        predicted_label = sentiment_result['label'].upper()
                        predicted_score = sentiment_result['score']

                    except Exception as e_analyze:
                        print(f"Error analyzing instance {instance_id} in {file_path}: {e_analyze}")
                        predicted_label = "UNKNOWN"
                        predicted_score = 0.0

                instance_data = copy.deepcopy(instance)
                instance_data["sentiment_label"] = predicted_label
                instance_data["sentiment_score"] = predicted_score

                analyzed_instances_list.append(instance_data)
                total_instances_processed += 1
                file_instances_processed +=1

                if (i + 1) % 50 == 0 or (i + 1) == num_instances:
                     print(f"   Analyzed instance {i + 1}/{num_instances}...")

            print(f"Finished analyzing {file_instances_processed} instances for this file.")
            processed_data_by_file[file_path] = {
                "metadata": metadata,
                "analyzed_instances": analyzed_instances_list
            }

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Skipping.")
        except Exception as e_file:
            print(f"An unexpected error occurred while processing {file_path}: {e_file}")

    print(f"\nAnalysis complete. Total instances processed across all files: {total_instances_processed}")
    return processed_data_by_file

def save_sentiment_datasets(processed_data, output_dir):
    """
    Saves separate JSON files for each sentiment category for each input file,
    preserving the original metadata structure.

    Args:
        processed_data (dict): Output from analyze_sentiment_and_preserve_structure.
        output_dir (str): Directory path to save the new dataset files.
    """
    print(f"\n--- Saving datasets grouped by sentiment to: {output_dir} ---")
    os.makedirs(output_dir, exist_ok=True)

    total_files_saved = 0
    sentiment_categories = ["POSITIVE", "NEGATIVE", "NEUTRAL", "UNKNOWN"]

    for original_filepath, data_dict in processed_data.items():
        base_filename = os.path.basename(original_filepath)
        filename_root, filename_ext = os.path.splitext(base_filename)
        print(f"Processing file: {base_filename}")

        metadata = data_dict['metadata']
        analyzed_instances = data_dict['analyzed_instances']

        sentiment_groups = defaultdict(list)
        for instance in analyzed_instances:
            sentiment = instance.get("sentiment_label", "UNKNOWN")
            sentiment_groups[sentiment].append(instance)

        for sentiment in sentiment_categories:
            instances_for_sentiment = sentiment_groups[sentiment]

            if not instances_for_sentiment:
                continue

            output_data = copy.deepcopy(metadata)
            output_data["Instances"] = instances_for_sentiment

            clean_filename_root = filename_root.replace('_baseline', '')
            output_filename = f"{clean_filename_root}_{sentiment.lower()}{filename_ext}"
            output_filepath = os.path.join(output_dir, output_filename)

            print(f"  Saving {len(instances_for_sentiment)} instances to {output_filename}...")
            try:
                with open(output_filepath, "w", encoding="utf-8") as outfile:
                    json.dump(output_data, outfile, indent=2)
                total_files_saved += 1
            except Exception as e_save:
                print(f"  Error saving file {output_filepath}: {e_save}")

    print(f"\nSaving complete. Total new dataset files created: {total_files_saved}")


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, "data", "test")
    output_directory = os.path.join(current_dir, "sentiment_split_datasets")

    datasets_to_analyze = [
        os.path.join(data_dir, "task843_financial_phrasebank_classification_baseline.json"),
        os.path.join(data_dir, "task512_twitter_emotion_classification_baseline.json"),
        os.path.join(data_dir, "task835_mathdataset_answer_generation_baseline.json"),
        os.path.join(data_dir, "task1564_triviaqa_answer_generation_baseline.json"),
        os.path.join(data_dir, "task1346_glue_cola_grammatical_correctness_classification_baseline.json"),
        os.path.join(data_dir, "task116_com2sense_commonsense_reasoning_baseline.json"),
        os.path.join(data_dir, "task828_copa_commonsense_cause_effect_baseline.json"),
    ]
    datasets_to_analyze = [f for f in datasets_to_analyze if os.path.exists(f)]
    if not datasets_to_analyze:
         print(f"Error: No existing dataset files found in the specified list or derived from {data_dir}")
         exit()

    print(f"Found {len(datasets_to_analyze)} dataset files to process.")

    all_processed_data = analyze_sentiment_and_preserve_structure(datasets_to_analyze)

    if all_processed_data:
        save_sentiment_datasets(all_processed_data, output_directory)
    else:
        print("No data was processed, skipping saving step.")

    print("\nScript finished.")