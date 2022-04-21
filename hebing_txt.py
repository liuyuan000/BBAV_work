file1 = 'dota_bbav/trainval.txt'
file2 = 'trainval.txt'
 
def merge(file1, file2):
    f1 = open(file1, 'a+', encoding='utf-8')
    with open(file2, 'r', encoding='utf-8') as f2:
        for i in f2:
            f1.write(i)
 
 
merge(file1, file2)
 