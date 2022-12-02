import glob 
import os

CURDIRPATH = os.path.dirname(__file__)

DATASET = ['Fluo-N2DL-HeLa','PhC-C2DH-U373']
idx = 1

IM_PATH = sorted(glob.glob(os.path.join(CURDIRPATH,f'{DATASET[idx]}/01/*.tif')) + glob.glob(os.path.join(CURDIRPATH,f'{DATASET[idx]}/02/*.tif')))
TG_PATH = sorted(glob.glob(os.path.join(CURDIRPATH,f'{DATASET[idx]}/01_ST/SEG/*.tif')) + glob.glob(os.path.join(CURDIRPATH,f'{DATASET[idx]}/02_ST/SEG/*.tif')))

for path in IM_PATH :
    l = os.path.normpath(path)
    l = l.split(os.sep)
    new_path = os.path.normpath(l[0])
    for i in range(1,len(l)-2):
        new_path = os.path.join(new_path,l[i])
    new_path = os.path.join(new_path, 'IMG', l[-2] + '_' + l[-1])
    new_path = new_path.replace(':',':/')
    os.rename(path, new_path)

for path in TG_PATH :
    l = os.path.normpath(path)
    l = l.split(os.sep)
    new_path = os.path.normpath(l[0])
    for i in range(1,len(l)-3):
        new_path = os.path.join(new_path,l[i])
    new_path = os.path.join(new_path, 'TARGET', l[-3] + '_' + l[-1])
    new_path = new_path.replace(':',':/')
    os.rename(path, new_path)

