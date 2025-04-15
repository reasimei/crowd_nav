import os

def read(api_key_file: str):
    print(f"Looking for file: {api_key_file}")
    print(f"Current directory: {os.getcwd()}")
    
    try:
        with open(api_key_file, 'r') as f:
            api_keys = f.read().splitlines()
        
        if not api_keys:
            print("Warning: The file is empty!")
        else:
            print(f"Found {len(api_keys)} keys:")
            for key in api_keys:
                print(f"Using key: {key}")
    except FileNotFoundError:
        print(f"Error: File '{api_key_file}' not found!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    read("api_key.txt")