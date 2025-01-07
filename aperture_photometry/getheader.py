from astropy.io import fits
import os
dirpath = os.path.dirname(os.path.abspath(__file__))

imgpath = '/Volumes/PortableSSD/ISAAC'
image_files = []
for item in os.listdir(imgpath):
    if item[-5:]=='.fits':
        image_files.append(item)


for filename in image_files:
    path = f'{imgpath}/{filename}'

    try:
        hdul = fits.open(path)
    except OSError:
        with open(f'{imgpath}/{filename}.txt', 'w') as f:
            f.write('DOES NOT OPEN!!!')
        continue
    
    header = hdul[0].header

    # instrument = hdul[0].header['INSTRUME']
    # name = hdul[0].header['OBJECT']
    with open(f'{imgpath}/{filename}.txt', 'w') as f:
        f.write(str(header))