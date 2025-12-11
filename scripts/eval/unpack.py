import os
import json
import base64
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
from io import BytesIO

def process_image(img_url: str, image_path: str):
    header, encoded = img_url.split(",", 1)
    # Decode
    img_bytes = base64.b64decode(encoded)
    # Load as PIL image
    image = Image.open(BytesIO(img_bytes))
    image.save(image_path)


def unpack_file_to_dir(file_path: str, dir_path: str, skip_save=False):
    """
    Unpack a file to a directory.
    """
    file_name = file_path.split('/')[-1].split('.')[0]
    os.makedirs(f"{dir_path}/{file_name}", exist_ok=True)
    with open(file_path, 'r') as f:
        data = json.load(f)

    result = data['resolved']

    if os.path.exists(f"{dir_path}/{file_name}/result.txt") or skip_save:
        return result

    conversation = []
    img_count = 0
    for message in data['messages']:
        role = message['role']
        img = None
        tool_calls = None
        if role == 'system':
            content = message['content'][0]['text']
        elif role == 'user' or role == 'tool':
            content = message['content'][0]['text']
            if len(message['content']) > 1:
                img = f"{img_count}.png"
                process_image(message['content'][1]['image_url']['url'], f"{dir_path}/{file_name}/{img}")
                img_count += 1
        else:
            content = message['content'][0]['text']
            if message['tool_calls'] is not None:
                tool_calls = []
                for tool_call in message['tool_calls']:
                    tool_calls.append({
                        'name': tool_call['function']['name'],
                        'arguments': tool_call['function']['arguments']
                    })

        if tool_calls is not None:
            conversation.append({
                'role': role,
                'content': content,
                'tool_calls': tool_calls
            })
        else:
            conversation.append({
                'role': role,
                'content': content,
                'img': img
            })
    with open(f"{dir_path}/{file_name}/conversation.json", 'w') as f:
        json.dump(conversation, f)

    with open(f"{dir_path}/{file_name}/result.txt", 'w') as f:
        f.write(str(result))
    
    return result

def process_single_file(file_path, dir_path, skip_save=False):
    """Process a single JSON file and return the result."""
    file_name = os.path.basename(file_path)
    print(f"Processing {file_name}...")
    result = unpack_file_to_dir(file_path, dir_path, skip_save)
    print(f"Finished {file_name}")
    return result


def process_all_json_files(dir_path, max_workers=4, skip_save=False):
    """
    Process all JSON files in the directory using multiprocessing.
    
    Args:
        dir_path: Directory containing JSON files to process
        max_workers: Maximum number of processes to use
    
    Returns:
        tuple: (total_correct, total_total, accuracy)
    """
    # Get all JSON files
    json_files = [
        os.path.join(dir_path, f) 
        for f in os.listdir(dir_path) 
        if f.endswith('.json')
    ]
    
    if not json_files:
        print(f"No JSON files found in {dir_path}")
        return 0, 0, 0.0
    
    print(f"Found {len(json_files)} JSON files to process")
    print(f"Using {max_workers} worker processes")
    
    total_correct = 0
    total_total = 0
    
    # Process files in parallel using multiprocessing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, file_path, dir_path, skip_save): file_path 
            for file_path in json_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                total_correct += result
                total_total += 1
            except Exception as exc:
                print(f"Error processing {os.path.basename(file_path)}: {exc}")
    
    accuracy = total_correct / total_total if total_total > 0 else 0.0
    
    print(f"\n{'='*50}")
    print(f"Total correct: {total_correct}")
    print(f"Total total: {total_total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"{'='*50}")
    
    return total_correct, total_total, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unpack JSON files to directories')
    parser.add_argument(
        '--dir_path', 
        type=str, 
        default=os.getcwd(),
        help='Directory path containing JSON files (default: current directory)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=64,
        help='Number of worker processes (default: 64)'
    )
    parser.add_argument(
        '--skip_save',
        action='store_true',
        help='Skip saving the result.txt file'
    )
    
    args = parser.parse_args()
    
    process_all_json_files(args.dir_path, max_workers=args.workers, skip_save=args.skip_save)