import pymeshlab
import os

ms = pymeshlab.MeshSet()
rootdir = '/home/user/GPIS/YCB_Video_Models/models'
destDir = '/home/user/GPIS/data/'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if(file.endswith('.obj') and not file.endswith('_simple.obj')):
            print(os.path.join(subdir, file))
            objectName = subdir.split('/')[-1]
            ms.load_new_mesh(os.path.join(subdir, file))
            ms.poisson_disk_sampling(samplenum = 2000)
            #ms.save_current_mesh(file_name = destDir + objectName + '.off', save_face_color = False)