def alternate_jsonl(file1_path, file2_path, output_path):
    with open(file1_path, 'r', encoding='utf-8') as file1, \
         open(file2_path, 'r', encoding='utf-8') as file2, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        while True:
            line1 = file1.readline()
            line2 = file2.readline()
            
            # If both files have been completely read, break the loop
            if not line1 and not line2:
                break

            # Write a line from file1 if available
            if line1:
                # Remove any extra newline and add one back for consistency
                outfile.write(line1.rstrip('\n') + '\n')
                
            # Write a line from file2 if available
            if line2:
                outfile.write(line2.rstrip('\n') + '\n')

# Usage example
file1_path = '/scratch/karthick/pretrain/param-7b/wiki-ds/wiki_ds_output.jsonl'
file2_path = '/scratch/karthick/pretrain/param-7b/redpajama_partial/stackexchange/stackexchange.jsonl'
output_path = 'paramds.jsonl'
alternate_jsonl(file1_path, file2_path, output_path)
print(f"Alternated data has been written to {output_path}")
