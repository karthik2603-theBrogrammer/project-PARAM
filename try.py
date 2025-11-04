# # import os
# # from datasets import load_dataset
# # os.environ['DOWNLOADED_DATASETS_PATH'] = '/scratch/karthick/pretrain/param-7b'
# # os.environ['HF_DATASETS_CACHE'] = '/scratch/karthick/pretrain/param-7b'
# # # Load specific Parquet files by pattern
# # ds = load_dataset(
# #     "wikimedia/wikipedia", 
# #     "20231101.en", 
# #     data_files=[
# #         "train-00006-of-00041.parquet",
# #         "train-00007-of-00041.parquet",
# #         "train-00008-of-00041.parquet"
# #     ]
# # )


# from datasets import load_dataset, Dataset
# from huggingface_hub import hf_hub_download

# # Specify the local directory for downloads
# local_dir = '/scratch/karthick/pretrain/param-7b/wikimedia/wikipedia/20231101.en/'

# # Download specific files
# file_paths = []
# for file_num in range(40,41):
#     file_path = hf_hub_download(
#         repo_id="wikimedia/wikipedia",
#         filename=f"train-{file_num:05d}-of-00041.parquet",
#         repo_type="dataset",
#         subfolder="20231101.en",
#         local_dir=local_dir
#     )
#     file_paths.append(file_path)

# # Load the dataset from the downloaded files
# ds = load_dataset(
#     "parquet", 
#     data_files=file_paths
# )


import os
import json
import pyarrow.parquet as pq
import pandas as pd

def convert_parquet_directory_to_jsonl(parquet_directory, output_jsonl_file):
    # Get all Parquet files in the directory
    parquet_files = [
        os.path.join(parquet_directory, f) 
        for f in os.listdir(parquet_directory) 
        if f.endswith('.parquet')
    ]
    
    # Sort files to ensure consistent processing order
    parquet_files.sort()
    
    # Open output JSONL file in append mode
    with open(output_jsonl_file, 'a') as jsonl_file:
        # Process each Parquet file
        for file_index, parquet_path in enumerate(parquet_files, 1):
            try:
                # Read Parquet file
                df = pd.read_parquet(parquet_path)
                
                # Convert and write each row to JSONL
                for _, row in df.iterrows():
                    json.dump(row.to_dict(), jsonl_file)
                    jsonl_file.write('\n')
                
                print(f"Processed file {file_index}/{len(parquet_files)}: {os.path.basename(parquet_path)}")
            
            except Exception as e:
                print(f"Error processing {parquet_path}: {e}")
    
    print(f"Conversion complete. Saved to {output_jsonl_file}")

# Usage example
parquet_directory = '/scratch/karthick/pretrain/param-7b/wiki-ds'
output_jsonl_file = '/scratch/karthick/pretrain/param-7b/wiki-ds/wiki_ds_output.jsonl'

# Optionally, remove existing output file before starting
if os.path.exists(output_jsonl_file):
    os.remove(output_jsonl_file)

# Run the conversion
convert_parquet_directory_to_jsonl(parquet_directory, output_jsonl_file)