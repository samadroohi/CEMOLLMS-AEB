import json

def update_json_types(json_file_path, new_type, start_target_row, end_target_row):
    # Read the file line by line
    with open(json_file_path, 'r') as file:
        lines = file.readlines()
    
    # Process each line
    for index, line in enumerate(lines):
        # Only process items within target range
        if index >= start_target_row and index <= end_target_row:
            try:
                # Parse the JSON object from the line
                item = json.loads(line)
                # Update type if it's "unknown"
                if item.get('ds_type') == 'SST':
                    item['ds_type'] = new_type
                # Convert back to string and update the line
                lines[index] = json.dumps(item) + '\n'
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {index}")
    
    # Write all lines back to the file
    with open(json_file_path, 'w') as file:
        file.writelines(lines)

# Example usage
json_file_path = 'data/AEB.json'
new_type = 'V-reg'
start_target_row = 9073
end_target_row = 10009

# Usage
update_json_types(json_file_path, new_type, start_target_row, end_target_row)
