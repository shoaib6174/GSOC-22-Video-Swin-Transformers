import os
import sys

sys.path.append("..")

from VideoSwinTransformer import model_configs


def main():
    for model_name in model_configs.MODEL_MAP:
        
        
        command = f"python convert.py -m {model_name}"
        os.system(command)
        



if __name__ == "__main__":
    main()