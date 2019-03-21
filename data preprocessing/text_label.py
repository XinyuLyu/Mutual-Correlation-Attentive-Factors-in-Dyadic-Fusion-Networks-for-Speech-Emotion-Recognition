import linecache
import os
import collections

import re


def get_sentense_name_from_transcriptions():
    transcriptions = r'C:\Users\Han\Desktop\transcripts\\'
    transcriptions_name = []
    for i, j, k in os.walk(transcriptions):
        for name in k:
            transcriptions_name.append(os.path.join(i, name))

    sentense_name = []
    for name in transcriptions_name:
        f = open(name, 'r')
        for line in f:
            if line.split()[0].startswith("Ses"):
                sentense_name.append(line.split()[0])
        f.close()

    write_to = r'C:\Users\Han\Desktop\sentense_name_map.txt'
    f = open(write_to, 'a')
    i = 0
    for n in sentense_name:
        f.writelines(n + '  ' + str(i) + '\n')
        i += 1
    f.close()


def count_check():
    write_to = r'C:\Users\Han\Desktop\sentense_name_map1.txt'
    f = open(write_to, 'r')
    list = []
    for line in f:
        list.append(line.strip())
    print([item for item, count in collections.Counter(list).items() if count > 1])


def get_sentense_name_from_emotion():
    # get emotion file names
    emotion_path = r'C:\Users\Han\Desktop\emotions\\'
    emotion_filename = []
    for i, j, k in os.walk(emotion_path):
        for name in k:
            emotion_filename.append(os.path.join(i, name))
    # count how much emotions
    sentense_name = []
    for emotion_files in emotion_filename:
        f = open(emotion_files, 'r')
        for line in f:
            if len(line.split("\t")) == 4 and line.split("\t")[1].startswith('Ses'):
                sentense_name.append(line.split("\t")[1])
    return sentense_name


def write_map(sentense_name):
    write_to = r'C:\Users\Han\Desktop\sentense_name_map.txt'
    f = open(write_to, 'a')
    i = 0
    for n in sentense_name:
        f.writelines(n + '  ' + str(i) + '\n')
        i += 1
    f.close()


def get_emotions_document_name():
    emotions_path = r'C:\Users\Han\Desktop\emotions\\'
    emotions_name = []
    for i, j, k in os.walk(emotions_path):
        for name in k:
            emotions_name.append(os.path.join(i, name))
    return emotions_name


def match_test(emotions_name):
    emotion_label = []
    for emotion_file in emotions_name:
        f = open(emotion_file, 'r')
        i = 0
        for line in f:
            if len(line.split("\t")) == 4 and line.split("\t")[1].startswith('Ses'):
                flag = True
                sub_label = []
                j = 2
                while (flag):
                    line1 = linecache.getline(emotion_file, i + j)
                    if re.search('C-', line1):
                        sub_label.append(line1.split('\t')[1].replace(';', ''))
                        j += 1
                    else:
                        flag = False
                emotion_label.append(sub_label)
            i += 1
        f.close()
    return emotion_label


def match_test1():
    sentense_name = []
    f1 = open(r'C:\Users\Han\Desktop\sentence_name_order.txt', 'r')
    for line in f1:
        sentense_name.append(line.split()[0])

    emotion_label = []
    for name in sentense_name:
        f = open(r'C:\Users\Han\Desktop\emotion.txt', 'r')
        i = 0
        for line in f:
            if len(line.split("\t")) == 4 and line.split("\t")[1] == name:
                flag = True
                sub_label = []
                j = 2
                while flag:
                    line1 = linecache.getline(r'C:\Users\Han\Desktop\emotion.txt', i + j)
                    if re.search('C-', line1):
                        sub_label.append(line1.split('\t')[1].replace(';', ''))
                        j += 1
                    else:
                        flag = False
                emotion_label.append(sub_label)
            i += 1
        f.close()
        print(name + ' fininsh')
    return emotion_label


def write_emotion_label(label_list):
    new_label = []
    for label in label_list:
        sub_dict = {'Neutral': 0, 'Excited': 0, 'Sadness': 0, 'Frustration': 0, 'Happiness': 0, 'Anger': 0, 'Other': 0,
                    'Surprise': 0, 'Disgust': 0, 'Fear': 0}
        for sub_label in label:
            for emotion in sub_label.split():
                sub_dict[emotion] += 1
        sum = 0
        label1 = []
        for key in sub_dict:
            sum += sub_dict[key]
        for key in sub_dict:
            sub_dict[key] /= sum
        label1.append(sub_dict['Neutral'])
        label1.append(sub_dict['Excited'])
        label1.append(sub_dict['Sadness'])
        label1.append(sub_dict['Frustration'])
        label1.append(sub_dict['Happiness'])
        label1.append(sub_dict['Anger'])
        label1.append(sub_dict['Other'])
        label1.append(sub_dict['Surprise'])
        label1.append(sub_dict['Disgust'])
        label1.append(sub_dict['Fear'])
        new_label.append(label1)

    write_path = r'C:\Users\Han\Desktop\label_new.txt'
    f = open(write_path, 'a')
    for l in new_label:
        for num in l:
            f.writelines(str(num) + ' ')
        f.writelines('\n')
    f.close()


def write_emotion():
    emotion_path = r'C:\Users\Han\Desktop\emotions\\'
    emotion_filename = []
    for i, j, k in os.walk(emotion_path):
        for name in k:
            emotion_filename.append(os.path.join(i, name))
    write = open(r'C:\Users\Han\Desktop\emotion.txt', 'a')
    i = 0
    for file in emotion_filename:
        f = open(file, 'r')
        for line in f:
            write.writelines(line)
        f.close()
        print(file + ' finish ' + str(i))
        i += 1
    write.close()


def deletes():
    path = r'C:\Users\Han\Desktop\transcriptions'
    filename = []
    for i, j, k in os.walk(path):
        for name in k:
            filename.append(os.path.join(i, name))

    sentense_name = []
    map = open(r'C:\Users\Han\Desktop\sentence_name_order.txt', 'r')
    for line in map:
        sentense_name.append(line.strip())

    folder_name = [r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session1\dialog\transcriptions',
                   r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session2\dialog\transcriptions',
                   r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session3\dialog\transcriptions',
                   r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session4\dialog\transcriptions',
                   r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session5\dialog\transcriptions']
    for folder in folder_name:
        for i, j, k in os.walk(folder):
            for name in k:
                f = open(os.path.join(i, name), 'r')
                f1 = open(os.path.join(i.strip('-transcriptions') + 'transcriptions_new', name), 'a')
                for line in f:
                    if line.split()[0].startswith('Ses'):
                        f1.write(line)
                    else:
                        continue
                f.close()
                f1.close()
                print(name + ' finish')


'''
    for file in filename:
        transcrips = open(file,'r')
        i = 1
        for line in transcrips:
            if line.split()[0].strip() not in sentense_name:
                print(file+' '+str(i)+' '+line)
            i += 1
'''


def count():
    folder_name = [r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session1\dialog\transcriptions_new',
                   r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session2\dialog\transcriptions_new',
                   r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session3\dialog\transcriptions_new',
                   r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session4\dialog\transcriptions_new',
                   r'E:\Yue\Entire Data\IEMOCAP\IEMOCAP_full_release\Session5\dialog\transcriptions_new']
    sentense_name = []
    map = open(r'C:\Users\Han\Desktop\sentence_name_order.txt', 'r')
    for line in map:
        sentense_name.append(line.strip())
    counter = 0
    for folder in folder_name:
        for i, j, k in os.walk(folder):
            for name in k:
                f1 = open(os.path.join(i, name), 'r')
                for line in f1:
                    if line.split()[0].strip() not in sentense_name:
                        print(name + ' ' + str(i) + ' ' + line)
                        counter += 1
                f1.close()
    print(counter)


def motify():
    folder_name = [r'C:\Users\Han\Desktop\1\S',
                   r'C:\Users\Han\Desktop\2\S',
                   r'C:\Users\Han\Desktop\3\S',
                   r'C:\Users\Han\Desktop\4\S',
                   r'C:\Users\Han\Desktop\5\S']
    for folder in folder_name:
        for i, j, k in os.walk(folder):
            for name in k:
                f1 = open(os.path.join(i, name), 'r')
                f = open(os.path.join(i.strip('-S') + 'S_new', name), 'a')
                for line in f1:
                    string = ''
                    for n in line.split()[2:]:
                        string += n + ' '
                    f.write(string + '\n')
                f1.close()
                f.close()


def new_map():
    f = open(r'C:\Users\Han\Desktop\sentence_name_order.txt', 'r')
    f1 = open(r'C:\Users\Han\Desktop\new_map.txt', 'a')
    i = 0
    for line in f:
        f1.write(line.strip() + ' ' + str(i) + '\n')
        i += 1
    f.close()
    f1.close()


def label():
    label_category = ['neu', 'exc', 'sad', 'fru', 'hap', 'ang', 'oth', 'sur', 'dis', 'fea']
    f = open(r'C:\Users\Han\Desktop\label_multi.txt', 'r')
    f1 = open(r'C:\Users\Han\Desktop\label_new.txt', 'a')
    for label in f:
        label_list = []
        for labels in label.split():
            label_list.append(labels)
        int_label_list = list(map(float, label_list))
        max_index = max(int_label_list)
        same_list = [i for i, v in enumerate(int_label_list) if v == max_index]
        if len(same_list) == 1:
            for i in range(10):
                if i not in same_list:
                    f1.write('0.0')
                else:
                    f1.write('1.0')
            f1.write(label_category[same_list[0]]+'\n')
        else:
            pass
            #f1.write(label_category[6]+'\n')
    f.close()
    f1.close()
def find_same_label():
    read = open(r'E:\Yue\Entire Data\iemocap_ACMMM_2019\label_multi.txt','r')
    write = open(r'E:\Yue\Entire Data\iemocap_ACMMM_2019\same_label2.txt','a')
    same_label = []
    i=0
    for line in read:
        for label in line.strip().split():
            if label != str(0.0) and line.split().count(label) > 1:
            #if line.strip().split().count(max(line.strip().split()))>1:
                same_label.append(str(i)+' '+line)
                write.write(str(i)+' '+line)
                break
            i += 1
    print(len(same_label))
    print(same_label)
'''
def classification_label():
    read = open(r'E:\Yue\Entire Data\iemocap_ACMMM_2019\label_multi.txt','r')
    for line in read:
        for label in line.strip()
'''
if __name__ == "__main__":
    label()
    # new_map()
    # emotion_label = match_test1()
    # print(emotion_label)
    # write_emotion_label(emotion_label)
    #label()
