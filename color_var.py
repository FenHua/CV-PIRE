import numpy as np

#progressbar
from tqdm import tqdm

#added for extention
from skimage import color

#multi threading
import multiprocessing as mp
from itertools import repeat

def calcVarCiede(x, y, variation, img, gaussKernel):
    counter = 0
    value = 0
    for vary in range(-variation, variation + 1):
        for varx in range(-variation, variation + 1):
            if(x + varx < 0 or x + varx >= img.shape[1] 
               or y + vary < 0 or y + vary >= img.shape[0]):
                #out of bounds do not take into average
                #do nothing
                continue
            elif(vary == 0 and varx == 0):
                #do not compare with yourself
                continue
            else:
                diff = color.deltaE_ciede2000(img[y][x],img[y+vary][x+varx])
                #multipli with the gaussian value
                gausDiff = diff*gaussKernel[vary + variation][varx + variation]

                #take the power of it for the euclidian distance
                value += gausDiff**2
                #print(str(color.deltaE_ciede2000(img[y][x],img[y+vary][x+varx])) + "value")
                counter += 1                  

    #calculate and save the euclidian difference with its surroundings
    #values[y][x] = value/counter
    return np.sqrt(value/counter)


def calcVarCiedeRows(rowStart, rowEnd, variation, img, gaussKernel):
    values = np.empty([rowEnd - rowStart, img.shape[1]])
    for y in tqdm(range(rowStart, rowEnd)):
        for x in range(img.shape[1]):
            values[y-rowStart][x] = calcVarCiede(x, y, variation, img, gaussKernel)
    return values
            
def chunks(l, n):
    #Yield successive n-sized chunks from l
    for i in range(0, len(l), n):
        if(i+n < len(l)):
            yield i,i + n
        else:
            yield i,len(l)


#returns a gaussian kernel based on the size n and normal distribution sigma
def calculateGaussianKernel(n, sigma):
    kernel = np.zeros((n,n))

    variation = int(n/2)
    for x in range(-variation, variation + 1):
        for y in range(-variation, variation + 1):
            kernel[x+variation][y+variation] = 1.0/(2*np.pi*sigma**2)*np.exp(-1*((x**2+y**2)/(2*sigma**2)))

    return kernel

#we dont want to lose edge pixels, thus in edge cases we only take the pixels inside the image
#input rgb image and kernel size
#output relative color difference using CIEDE2000 between 0 and 1
#sigma indicates the distrubtion of the gaussian kernel, if you take a really big sigma it will
#be a flat distribution
def calculateColorDifference(img, n, sigma):
    if(n%2==0 or n < 3):
        print("error no such n is allowed")
        return -1
    
    #setup multitreath poolling
    threads = mp.cpu_count()
    pool = mp.Pool(threads)
    
    img = color.rgb2lab(img)
    variation = int(n/2)

    #get gaussian kernel
    gaussKernel = calculateGaussianKernel(n, sigma)

    values = np.empty([img.shape[0],img.shape[1]])
    
    #devide the rows based on the amount of rows and threads
    startEndY = list(chunks(range(0, img.shape[0]), int(img.shape[0]/threads)))
    rowStart = [x[0] for x in startEndY]
    rowEnd = [x[1] for x in startEndY]
    
    #calculate every x rows in a threaded pool. Increasing performance quite significantly on bigger systems
    results = pool.starmap(calcVarCiedeRows, zip(rowStart,rowEnd,repeat(variation),repeat(img),repeat(gaussKernel)))
    values = np.concatenate(results, axis = 0)       
    
    return values/np.max(values)