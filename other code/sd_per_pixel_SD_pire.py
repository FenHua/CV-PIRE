import numpy as np

#progressbar
from tqdm import tqdm

#added for extention
from skimage import color

#multi threading
import multiprocessing as mp
from itertools import repeat

#Calculates the standard deviation of a pixel based on the paper "Towards Imperceptible and Robust Adversarial Example Attacks against Neural Networks"
def calcSDBlackWhite(x, y, variation, img):
    counter = 0
    mu = 0

    #first calculate the average
    for vary in range(-variation, variation + 1):
        for varx in range(-variation, variation + 1):
            if(x + varx < 0 or x + varx >= img.shape[1] 
               or y + vary < 0 or y + vary >= img.shape[0]):
                #out of bounds do not take into average
                #do nothing
                continue
            else:
                mu += img[y+vary][x+varx]
                counter += 1                  

    #calculte the average value
    mu = mu/counter

    value = 0

    #now calculate the distance
    for vary in range(-variation, variation + 1):
        for varx in range(-variation, variation + 1):
            if(x + varx < 0 or x + varx >= img.shape[1] 
               or y + vary < 0 or y + vary >= img.shape[0]):
                #out of bounds do not take into average
                #do nothing
                continue
            else:
                value += (img[y+vary][x+varx] - mu)**2      
    
    #calculate and save standard deviation of the pixel
    return np.sqrt(value/(counter**2))


def calcSDBlackWhiteRows(rowStart, rowEnd, variation, img):
    values = np.empty([rowEnd - rowStart, img.shape[1]])
    for y in tqdm(range(rowStart, rowEnd)):
        for x in range(img.shape[1]):
            values[y-rowStart][x] = calcSDBlackWhite(x, y, variation, img)
    return values

#we dont want to lose edge pixels, thus in edge cases we only take the pixels inside the image
#input rgb image and kernel size
#output standard deviation as in the paper  "Towards Imperceptible and Robust Adversarial Example Attacks against Neural Networks"
def calculateSDBlackWhite(img, n):
    if(n%2==0 or n < 3):
        print("error no such n is allowed")
        return -1
    
    #setup multitreath poolling
    threads = mp.cpu_count()
    pool = mp.Pool(threads)
    
    #since the paper uses the intensity of the pixels. It is not defined how the intensity is calculated
    #in this case I transform the image to a greyscale image. It is defined that the intensity is between 0 and 1. Thus devide by 255

    imgGreyscale = img.convert('L')
    imgGreyscale = np.array(imgGreyscale)/255.0
    print(imgGreyscale.shape)
    variation = int(n/2)
    values = np.empty([imgGreyscale.shape[0],imgGreyscale.shape[1]])
    
    #devide the rows based on the amount of rows and threads
    startEndY = list(chunks(range(0, imgGreyscale.shape[0]), int(imgGreyscale.shape[0]/threads)))
    rowStart = [x[0] for x in startEndY]
    rowEnd = [x[1] for x in startEndY]
    
    #calculate every x rows in a threaded pool. Increasing performance quite significantly on bigger systems
    results = pool.starmap(calcSDBlackWhiteRows, zip(rowStart,rowEnd,repeat(variation),repeat(imgGreyscale)))
    values = np.concatenate(results, axis = 0)       
            
    #no normalization is done in the paper
    return values/np.max(values)