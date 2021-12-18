'''
This is the implementation of FH semantic segmentation masks
papers: https://arxiv.org/pdf/1206.2807.pdf
Link References: 
'''

import skimage
from skimage import data
from skimage.segmentation import felzenszwalb
from skimage.color import label2rgb
from skimage import io
import matplotlib.pyplot as plt
import os, threading

# Call this function to generate FH semantic segmentation masks
def generate(dataset_root_path='/data/1Knew/train', root_write_path='/data/1Knew/FH_binary'):
    subfolder_names = [x for x in os.listdir(dataset_root_path) if '.' not in x]
    # subfolder_names = subfolder_names[:334]
    for num in range(334, 667, 2):
        subfolder_threads = []
        subfolder_to_run = [subfolder_names[num], subfolder_names[num + 1]]
        print(num , ' , ' , subfolder_to_run)
        for i, subfolder in enumerate(subfolder_to_run):
            subfolder_threads.append(threading.Thread(target=subfolder_task, args=(subfolder, root_write_path, dataset_root_path,)))
            subfolder_threads[i].start()
        
        for subfolder_thread in subfolder_threads:
            subfolder_thread.join()



# Parallel mask generation(using CPU)
def generate_FH_mask(dir_name, image_name, root_write_path, dataset_root_path):
    image_read_path = os.path.join(dataset_root_path, dir_name, image_name)
    image_write_path = os.path.join(root_write_path, dir_name, image_name)
    if not os.path.isfile(image_write_path):
        image = io.imread(image_read_path)
        scale_dynamic = int(((image.shape[0] + image.shape[1]) / 8))
        FH_mask = felzenszwalb(image, scale=scale_dynamic, sigma=0.5, min_size=1000)
        FH_mask = label2rgb(FH_mask)
        io.imsave(image_write_path, FH_mask)

def subfolder_task(dir_name, root_write_path, dataset_root_path):
    read_dir_path = os.path.join(dataset_root_path, dir_name)
    write_dir_path = os.path.join(root_write_path, dir_name)
    if not os.path.isdir(write_dir_path):
        print(dir_name, " is a new directory, creating...")
        os.mkdir(write_dir_path)
    
    image_threads = []
    image_names = os.listdir(read_dir_path)
    for i , image_name in enumerate(image_names):
        image_threads.append(threading.Thread(target=generate_FH_mask, args=(dir_name, image_name, root_write_path, dataset_root_path)))
        image_threads[i].start()

    for image_thread in image_threads:
        image_thread.join()




