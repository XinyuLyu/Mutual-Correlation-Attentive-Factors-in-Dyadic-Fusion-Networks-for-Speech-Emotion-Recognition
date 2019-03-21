# coding=utf-8
import os
import shutil
import sys


def get_folder_name(path):
    foldername = []
    for i, j, k in os.walk(path):
        for name in j:
            foldername.append(os.path.join(i, name))
    return foldername


def move_file(src_dir, target_dir):
    for item in os.listdir(src_dir):
        src_name = os.path.join(src_dir, item)
        target_name = os.path.join(target_dir, item)
        shutil.move(src_name, target_name)
        print(str(item) + 'finish')


def move_audio():
    s = [r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session1\sentences\wav',
         r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session2\sentences\wav',
         r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session3\sentences\wav',
         r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session4\sentences\wav',
         r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session5\sentences\wav']
    d = r'C:\Users\Han\Desktop\wav'
    for ss in s:
        folder_list = get_folder_name(ss)
        for folder_name in folder_list:
            move_file(folder_name, d)


def rename_wav():
    f = open(r'C:\Users\Han\Desktop\sentence_name_order.txt', 'r')
    i = 0
    for name in f:
        os.rename('C:\\Users\\Han\\Desktop\\test\\' + name.strip() + '.wav',
                  'C:\\Users\\Han\\Desktop\\test\\' + str(i) + '.wav')
        i += 1


# !/usr/bin/python
# -*- coding: UTF-8 -*-

def rename():
    sentense_index = []
    map = open(r'C:\Users\Han\Desktop\new_map.txt', 'r')
    for line in map:
        sentense_index.append(line.split()[0])
    s = [r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session1\sentences\wav_copy',
         r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session2\sentences\wav_copy',
         r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session3\sentences\wav_copy',
         r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session4\sentences\wav_copy',
         r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session5\sentences\wav_copy']
    for ss in s:
        folder_list = get_folder_name(ss)
        for folder in folder_list:
            for i, j, k in os.walk(folder):
                for name in k:
                    if not name.endswith('.wav'):
                        pass
                    else:
                        num = sentense_index.index(name.strip('-.wav'))
                        # os.rename('C:\\Users\\Han\\Desktop\\test\\' + name.strip() + '.wav','C:\\Users\\Han\\Desktop\\test\\' + str(i) + '.wav')
                        os.rename(os.path.join(i, name), os.path.join(i, str(num) + '.wav'))
                        print(name + ' finish')


if __name__ == '__main__':
    # move_audio()
    #rename()
    pass
