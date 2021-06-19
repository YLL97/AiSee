import os
import itertools
import shutil
import random
import matplotlib.pyplot as plt


def main():
    print(os.getcwd())
    os.chdir('D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/')
    rootpath = 'D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/'

    pathlist = ['validate', 'test1']
    rmlist = ['rm1', 'rm5', 'rm10', 'rm20', 'rm50', 'rm100']

    # for path in pathlist:
    #     for rm in rmlist:
    #         name_list = random.sample(os.listdir(''.join([rootpath, f'train/{rm}'])),15)
    #         for name in name_list:
    #             shutil.move(''.join([rootpath, f'train/{rm}/{name}']), ''.join([rootpath, f'{path}/{rm}']))

    # for rm in rmlist:
    #     name_list = random.sample(os.listdir(f'D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/train/{rm}'),20)
    #     for name in name_list:
    #         shutil.move(f'D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/train/{rm}/{name}', f'D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/extras/{rm}')


if __name__ == "__main__":
    main()
