import json
import requests
import time
import copy
import subprocess
import os

# ComfyUI API endpoint
api_url = "http://127.0.0.1:8188"
prompt_api_url = f"{api_url}/prompt"
history_api_url = f"{api_url}/history"

# --- ComfyUI Process Management ---
# Path to your ComfyUI batch file
comfyui_bat_path = r"D:\Work\ComfyUI_windows_portable\run_nvidia_gpu.bat"
# Working directory for the batch file
comfyui_working_dir = r"D:\Work\ComfyUI_windows_portable"

def start_comfyui(timeout=120):
    """
    Starts the ComfyUI server and waits for it to become responsive.
    
    Args:
        timeout (int): The maximum number of seconds to wait for the server.
    
    Returns:
        subprocess.Popen: The process object if successful, otherwise None.
    """
    print("Starting ComfyUI server...")
    try:
        # Start the process non-blocking
        process = subprocess.Popen(
            [comfyui_bat_path], 
            cwd=comfyui_working_dir, 
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        
        start_time = time.time()
        print("Waiting for ComfyUI server to become responsive...", end='', flush=True)

        while time.time() - start_time < timeout:
            try:
                # Poll the API to check if it's running
                response = requests.get(history_api_url, timeout=5)
                if response.status_code == 200:
                    print("\nComfyUI server is running and ready!")
                    return process
            except requests.exceptions.RequestException:
                # Server is not yet responsive, continue polling
                pass
            
            print('.', end='', flush=True)
            time.sleep(2) # Wait a couple of seconds before polling again
        
        # If the loop finishes without returning, it timed out
        print(f"\nTimeout: ComfyUI server did not start within {timeout} seconds. Check for errors.")
        process.kill() # Try to clean up the process
        return None

    except FileNotFoundError:
        print(f"\nError: The file '{comfyui_bat_path}' was not found.")
        return None
    except Exception as e:
        print(f"\nAn unexpected error occurred while trying to start ComfyUI: {e}")
        return None
def stop_comfyui(process):
    """Stops the ComfyUI server process."""
    if process:
        print("Stopping ComfyUI server...")
        process.kill()
        print("ComfyUI server stopped.")

def is_comfyui_running():
    """Checks if the ComfyUI server is already running."""
    try:
        response = requests.get(f"{api_url}/history", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# --- Image Generation Functions (from previous script) ---
prompts_file_path = 'prompts.json'
try:
    with open(prompts_file_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{prompts_file_path}' was not found.")
    exit()
except json.JSONDecodeError:
    print(f"Error: The file '{prompts_file_path}' contains invalid JSON.")
    exit()

image_counter = 1

def create_api_prompt_from_workflow(workflow):
    api_prompt = {}
    for node in workflow['nodes']:
        node_id = str(node['id'])
        class_type = node['type']
        inputs = {}
        for input_info in node.get('inputs', []):
            if 'link' in input_info:
                link_id = input_info['link']
                source_node_id = None
                source_output_index = None
                for link in workflow['links']:
                    if link[0] == link_id:
                        source_node_id = str(link[1])
                        source_output_index = link[2]
                        break
                if source_node_id and source_output_index is not None:
                    inputs[input_info['name']] = [source_node_id, source_output_index]
        if 'widgets_values' in node:
            if class_type == "CLIPTextEncode":
                inputs['text'] = node['widgets_values'][0]
            elif class_type == "CheckpointLoaderSimple":
                inputs['ckpt_name'] = node['widgets_values'][0]
            elif class_type == "EmptySD3LatentImage":
                inputs['width'] = node['widgets_values'][0]
                inputs['height'] = node['widgets_values'][1]
                inputs['batch_size'] = node['widgets_values'][2]
            elif class_type == "KSampler":
                inputs['seed'] = node['widgets_values'][0]
                inputs['steps'] = node['widgets_values'][2]
                inputs['cfg'] = node['widgets_values'][6]
                inputs['sampler_name'] = node['widgets_values'][4]
                inputs['scheduler'] = node['widgets_values'][5]
                inputs['denoise'] = node['widgets_values'][3]
            elif class_type == "SaveImage":
                if len(node['widgets_values']) > 0:
                    inputs['filename_prefix'] = node['widgets_values'][0]
                else:
                    inputs['filename_prefix'] = "ComfyUI"
        
        api_prompt[node_id] = {
            'class_type': class_type,
            'inputs': inputs
        }
    return api_prompt

def get_node_id_by_type(workflow, node_type, index=0):
    found_nodes = [node for node in workflow['nodes'] if node['type'] == node_type]
    if len(found_nodes) > index:
        return found_nodes[index]['id']
    return None

def run_workflow_with_prompt(prompt, base_workflow):
    global image_counter
    current_workflow = copy.deepcopy(base_workflow)
    positive_prompt_node_id = get_node_id_by_type(current_workflow, "CLIPTextEncode", index=0)
    if positive_prompt_node_id is None:
        print("Error: Could not find the positive prompt node.")
        return None
    for node in current_workflow['nodes']:
        if node['id'] == positive_prompt_node_id:
            node['widgets_values'][0] = prompt
            break
    save_image_node_id = get_node_id_by_type(current_workflow, "SaveImage")
    if save_image_node_id is not None:
        for node in current_workflow['nodes']:
            if node['id'] == save_image_node_id:
                node['widgets_values'][0] = str(image_counter)
                break
    api_prompt = create_api_prompt_from_workflow(current_workflow)
    payload = {'prompt': api_prompt}
    print(f"\nSubmitting prompt: '{prompt}'")
    try:
        response = requests.post(prompt_api_url, json=payload)
        response.raise_for_status()
        response_data = response.json()
        print(f"Workflow submitted successfully. Prompt ID: {response_data['prompt_id']}")
        return response_data['prompt_id']
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
        try:
            print("Response from server:", response.json())
        except json.JSONDecodeError:
            print("Response content is not valid JSON. Response text:", response.text)
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}. Is ComfyUI server running?")
    except requests.exceptions.RequestException as err:
        print(f"An unexpected error occurred: {err}")
    return None

def wait_for_completion(prompt_id, timeout=300, poll_interval=1):
    start_time = time.time()
    print(f"Waiting for prompt {prompt_id} to finish...", end='', flush=True)
    while True:
        try:
            response = requests.get(history_api_url)
            response.raise_for_status()
            history = response.json()
            if prompt_id in history:
                prompt_info = history.get(prompt_id, {})
                if 'outputs' in prompt_info:
                    print("\nImage generation complete!")
                    return True
            if time.time() - start_time > timeout:
                print(f"\nTimeout: Prompt {prompt_id} did not finish within {timeout} seconds.")
                return False
        except requests.exceptions.RequestException as e:
            print(f"\nError polling history: {e}")
            return False
        print('.', end='', flush=True)
        time.sleep(poll_interval)


# --- Main script logic ---
comfyui_process = None
try:
    if not is_comfyui_running():
        comfyui_process = start_comfyui()
        if not comfyui_process:
            print("Failed to start ComfyUI. Exiting.")
            exit()
    else:
        print("ComfyUI server is already running.")

    workflow_file_path = r"C:\Users\Ali Haider\Downloads\flux_schnell.json"
    with open(workflow_file_path, 'r', encoding='utf-8') as f:
        base_workflow_data = json.load(f)
    print("Workflow loaded successfully.")
    
    for p in prompts:
        prompt_id = run_workflow_with_prompt(p, base_workflow_data)
        if prompt_id:
            if wait_for_completion(prompt_id):
                image_counter += 1

except FileNotFoundError:
    print(f"Error: The workflow file '{workflow_file_path}' was not found.")
except json.JSONDecodeError:
    print(f"Error: The workflow file '{workflow_file_path}' contains invalid JSON.")
finally:
    stop_comfyui(comfyui_process)