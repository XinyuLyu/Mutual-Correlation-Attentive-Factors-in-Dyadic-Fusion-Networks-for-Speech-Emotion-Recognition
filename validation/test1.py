test = open ('test_data.txt')
train = open ('train_data.txt')
count = 0
for test_lines in test:
    for train_lines in train:
        if test_lines == train_lines:
            count += 1
print(count)