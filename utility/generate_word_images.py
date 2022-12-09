import os
import json
from PIL import Image, ImageChops

def crop_surrounding_whitespace(image):
    """Remove surrounding empty space around an image.

    This implemenation assumes that the surrounding space has the same colour
    as the top leftmost pixel.

    :param image: PIL image
    :rtype: PIL image
    """
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if not bbox:
        return image
    else:
        bboxlist = list(bbox)
        bboxlist[0] = 0
        bboxlist[1] = 0
        bboxlist[2] = bboxlist[2] + 20
        bboxlist[3] = bboxlist[3] + 20
        bbox = tuple(bboxlist)
        return image.crop(bbox)

if __name__ == '__main__':

    i=1
    labels = {}
    with open('unique_words','r') as f:
        for line1 in f:
            words=line1.split(' ')
            for text in words:
                f1= open("temp.txt","w+")
                f1.write(text)
                f1.close() 
                width=1200
                height=250
                if not os.path.exists('output'):
                    os.system('mkdir output')
                convert='text2image --text temp.txt --outputbase output/ --fonts_dir /usr/share/fonts/truetype/lohit-devanagari --font \'Lohit Devanagari\' --xsize {0} --ysize {1} --margin 20 --resolution 300'.format(width, height)
                os.system(convert)
                filename= 'output/' + str(i)+ '.tif'
                outfile= 'output/tel_' + str(i)+ '.jpg'
                rename = 'mv output/.tif ' + filename
                os.system(rename)
                img = Image.open(filename)
                out = img.convert("RGB")
                out = crop_surrounding_whitespace(out)
                out.save(outfile, "JPEG", quality=90)
                os.remove(filename)
                os.remove('temp.txt')
                key = str(i)+ '.jpg'
                value = text.rstrip(text[-1])
                labels.update({key: value})
                print(i)
                i=i+1
    
    with open("labels.json", "w") as outfile:
        json.dump(labels, outfile)
