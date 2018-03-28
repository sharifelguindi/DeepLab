from __future__ import print_function
import os, fnmatch

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def create_train_val(structure_path, num_examples, split):

    planeList = ['ax', 'cor', 'sag']
    planeDir = ['Axial', 'Coronal', 'Sag']
    filename_train = 'train_'
    filename_val = 'val_'
    i = 0

    for plane in planeList:

        file_base = os.path.join(structure_path, 'processed', 'ImageSets',planeDir[i])
        if not os.path.exists(file_base):
            os.makedirs(file_base)
        f = open(os.path.join(file_base, filename_train + plane + '.txt'), 'a')
        f.truncate()
        k = 0
        path = os.path.join(structure_path, 'processed', 'PNGImages')
        pattern = plane + '*.png'
        files = find(pattern, path)
        for file in files:
            if file.find(plane) > 0 and (file.find(plane + '1_') < 1 and file.find(plane + '2_') < 1):
                h = file.split(os.sep)
                f.write(h[-1].replace('.png','') +'\n')
                k = k + 1
        f.close()
        print(filename_train + plane, k)

        file_base = os.path.join(structure_path, 'processed', 'ImageSets', planeDir[i])
        if not os.path.exists(file_base):
            os.makedirs(file_base)
        f = open(os.path.join(file_base, filename_val + plane + '.txt'), 'a')
        f.truncate()
        k = 0
        for file in files:
            if file.find(plane) > 0 and (file.find(plane + '1_') > 0 or file.find(plane + '2_') > 0):
                h = file.split(os.sep)
                f.write(h[-1].replace('.png','') +'\n')
                k = k + 1
        f.close()
        print(filename_val + plane, k)
        i = i + 1
    return

def main():
    structure_path = 'datasets\\urethra'
    num_examples = 20
    split = 0.8
    create_train_val(structure_path, num_examples, split)

if __name__ == '__main__':
    main()