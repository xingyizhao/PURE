import subprocess

scripts = [
    'model_poisoning.py',
    'model_fine_tuning.py',
    'attention_visualization.py',
    'demo_attention.py'
]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(['python', script])

    if result.returncode != 0:
        print(f"{script} failed. Exiting.")
        exit(1)
    else:
        print(f"{script} executed successfully.")

print("All scripts executed successfully.")
