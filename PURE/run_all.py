import subprocess

# List of Python scripts to run in sequence
scripts = [
    'pretrained_model_poisoning.py',
    'attention_head_pruning.py',
    'attention_normalization.py',
]

# Iterate over the list of scripts and run them one by one
for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(['python', script])

    # Check if the current script executed successfully
    if result.returncode != 0:
        print(f"{script} failed. Exiting.")
        exit(1)
    else:
        print(f"{script} executed successfully.")

print("All scripts executed successfully.")
