#!/bin/bash

# Set the target directory where you want to download the files
TARGET_DIR="redpajama_partial"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Download the urls.txt file first
wget 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt' -O "$TARGET_DIR/urls.txt"

# Specify the fraction of URLs to download (e.g., 0.1 means 10% of the URLs)
DOWNLOAD_FRACTION=0.005

# Shuffle the urls.txt file and save the result
shuf "$TARGET_DIR/urls.txt" > "$TARGET_DIR/urls_shuffled.txt"

# Count the total number of URLs in the shuffled file
TOTAL_URLS=$(wc -l < "$TARGET_DIR/urls_shuffled.txt")
# Calculate how many URLs to download (rounded down to an integer)
NUM_TO_DOWNLOAD=$(awk -v total="$TOTAL_URLS" -v frac="$DOWNLOAD_FRACTION" 'BEGIN { printf "%d", total * frac }')

echo "Total URLs available: $TOTAL_URLS"
echo "Downloading $NUM_TO_DOWNLOAD URLs (fraction: $DOWNLOAD_FRACTION)"

# Process only the first NUM_TO_DOWNLOAD lines from the shuffled file
COUNT=0
head -n "$NUM_TO_DOWNLOAD" "$TARGET_DIR/urls_shuffled_fin.txt" | while read -r line; do
    ((COUNT++))
    
    # Remove the base URL prefix to get the relative download path
    dload_loc=${line#https://data.together.xyz/redpajama-data-1T/v1.0.0/}
    
    # Create the subdirectory structure inside the target directory
    mkdir -p "$TARGET_DIR/$(dirname "$dload_loc")"
    
    echo "Downloading $dload_loc (file $COUNT)"
    wget "$line" -O "$TARGET_DIR/$dload_loc"
done

echo "Download complete. Files saved to $TARGET_DIR/"
