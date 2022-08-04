import os
import sys

sys.path.append("..")

from VideoSwinTransformer import model_configs


def main():
    i = 1
    for model_name in model_configs.MODEL_MAP:
        #print()
        #print()
        
        
        command = f"python convert.py -m {model_name}"

        print(command)
        os.system(command)
        
        if i == 1:
            break 



if __name__ == "__main__":
    main()