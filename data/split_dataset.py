import glob 
import os
from sklearn.model_selection import train_test_split

CURDIRPATH = os.path.dirname(__file__)

DATASET = ['Fluo-N2DL-HeLa','PhC-C2DH-U373']
idx = 1

IM_PATH = sorted(glob.glob(os.path.join(CURDIRPATH,f'{DATASET[idx]}/IMG/*.tif')))
TG_PATH = sorted(glob.glob(os.path.join(CURDIRPATH,f'{DATASET[idx]}/TARGET/*.tif')))

IM_TRAIN, IM_TEST, TG_TRAIN, TG_TEST = train_test_split(IM_PATH,TG_PATH,test_size=0.2,shuffle=True,random_state=42)

for path in IM_TRAIN :
    l = os.path.normpath(path)
    l = l.split(os.sep)
    new_path = os.path.normpath(l[0])
    for i in range(1,len(l)-2):
        new_path = os.path.join(new_path,l[i])
    new_path = os.path.join(new_path, 'IMG_TRAIN', l[-1])
    new_path = new_path.replace(':',':/')
    os.rename(path, new_path)

for path in IM_TEST :
    l = os.path.normpath(path)
    l = l.split(os.sep)
    new_path = os.path.normpath(l[0])
    for i in range(1,len(l)-2):
        new_path = os.path.join(new_path,l[i])
    new_path = os.path.join(new_path, 'IMG_TEST', l[-1])
    new_path = new_path.replace(':',':/')
    os.rename(path, new_path)

for path in TG_TRAIN :
    l = os.path.normpath(path)
    l = l.split(os.sep)
    new_path = os.path.normpath(l[0])
    for i in range(1,len(l)-2):
        new_path = os.path.join(new_path,l[i])
    new_path = os.path.join(new_path, 'TARGET_TRAIN', l[-1])
    new_path = new_path.replace(':',':/')
    os.rename(path, new_path)

for path in TG_TEST :
    l = os.path.normpath(path)
    l = l.split(os.sep)
    new_path = os.path.normpath(l[0])
    for i in range(1,len(l)-2):
        new_path = os.path.join(new_path,l[i])
    new_path = os.path.join(new_path, 'TARGET_TEST', l[-1])
    new_path = new_path.replace(':',':/')
    os.rename(path, new_path)

