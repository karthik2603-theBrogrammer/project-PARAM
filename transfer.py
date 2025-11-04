import os
import shutil

def transfer_files(source_files, source_dir, destination_dir):
    """
    Transfer a list of files from source directory to destination directory.
    
    Args:
    source_files (list): List of filenames to transfer
    source_dir (str): Path to the source directory
    destination_dir (str): Path to the destination directory
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Counter for successful and failed transfers
    successful_transfers = 0
    failed_transfers = 0
    
    # Transfer each file
    for filename in source_files:
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)
        
        try:
            # Check if file exists
            if not os.path.exists(source_path):
                print(f"File not found: {source_path}")
                failed_transfers += 1
                continue
            
            # Transfer the file
            shutil.copy2(source_path, destination_path)
            successful_transfers += 1
            print(f"Transferred: {filename}")
        
        except Exception as e:
            print(f"Error transferring {filename}: {e}")
            failed_transfers += 1
    
    # Print summary
    print("\nTransfer Summary:")
    print(f"Total files attempted: {len(source_files)}")
    print(f"Successful transfers: {successful_transfers}")
    print(f"Failed transfers: {failed_transfers}")

# Example usage
if __name__ == "__main__":
    # List of files to transfer
    files_to_transfer = [
        # "train-00000-of-00041.parquet",
        # "train-00001-of-00041.parquet",
        "train-00002-of-00041.parquet",
        "train-00003-of-00041.parquet",
        "train-00004-of-00041.parquet",
        "train-00005-of-00041.parquet"
    ]
    
    # Source and destination directories
    source_directory = "/home/karthik/.cache/huggingface/hub/datasets--wikimedia--wikipedia/snapshots/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en"
    destination_directory = "/scratch/karthick/pretrain/param-7b/wiki-ds"
    
    # Call the transfer function
    transfer_files(files_to_transfer, source_directory, destination_directory)