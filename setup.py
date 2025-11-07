# import subprocess
# import sys

# requirements = [
#     "streamlit",
#     "torch",
#     "torchvision", 
#     "Pillow",
#     "numpy"
# ]

# print("Installing dependencies...")
# for package in requirements:
#     print(f"Installing {package}...")
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# print("\nSetup complete!")


# import subprocess
# import sys

# streamlit_file = "coin_classifier.py"

# subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_file])


# COIN VALUE CLASSIFIER - SETUP INSTRUCTIONS

# FILES NEEDED:
# - coin_classifier.py (your Streamlit app)
# - model.pth (trained model)
# - setup.py
# - launch.py

# FIRST TIME SETUP:
# 1. Run: python setup.py
# 2. Wait for all dependencies to install

# RUNNING THE APP:
# 1. Run: python launch.py
# 2. App opens in browser automatically
# 3. Press Ctrl+C to stop

# TROUBLESHOOTING:
# - If launch.py fails: ensure setup.py ran successfully
# - If model not found: place model.pth in same folder
# - If wrong filename: edit launch.py line 4 with your filename