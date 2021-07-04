import sys

sys.path.extend(['C:\\Users\\SoapClancy\\OneDrive\\PhD\\01-PhDProject\\02-Wind\\MyProject\\Code',
                 'C:\\Users\\SoapClancy\\OneDrive\\PhD\\01-PhDProject\\Python_Project_common_package',
                 'C:/Users/SoapClancy/OneDrive/PhD/01-PhDProject/02-Wind/MyProject/Code'])
from stage_nn import train_nn_model

if __name__ == "__main__":
    for now_zone_num in [5, 1, 6, 7, 2, 3, 4, 8, 9, 10]:
        train_nn_model(task_num=4, zone_num=now_zone_num,
                       continue_training=True if now_zone_num in {1, 7} else False)
