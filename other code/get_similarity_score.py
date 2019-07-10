import os
import argparse
from tqdm import tqdm

from skimage import img_as_float
from skimage.io import imread
from skimage.measure import compare_ssim 
from skimage.measure import compare_mse 
from skimage.measure import compare_nrmse 
from skimage.measure import compare_psnr

parser = argparse.ArgumentParser(description = "Given images it will calculate the mAP score")
parser.add_argument("-org_dir", "--org_dir", help="Directory of the original images.", default="./images/oxbuild_testquery/")
parser.add_argument("-adver_dir", "--adver_dir", help="Directory of generated adversarial queries.", default="./img_output/")
args = parser.parse_args()

path_org = args.org_dir
path_pert = args.adver_dir + "/"
printIntermediateRes = False

files = os.listdir(path_pert)

#remove all non directories
for file in files:
    if(not os.path.isdir(path_pert + file)):
        #its not a direcotry remove it
        files.remove(file)
        
for directory in tqdm(files):

    outfilename = "imageQualityComparison.txt"

    #first remove the existing files
    try:
        os.remove(path_pert + directory + "/" + outfilename)
    except: 
        pass

    #open new file
    compFile = open(path_pert + directory + "/"  + outfilename, 'w+')

    names_org = os.listdir(path_org)

    ssimTot = 0
    mseTot = 0
    nrmseTot = 0
    psnrTot = 0
    counter = 0

    #calulate the difference for every file
    for name in names_org:
        if(name.split(".")[-1] != "jpg"):
            #it is not a jpg skip this file
            print(name)
            continue
            
        img_org = img_as_float(imread(path_org+name))
        img_pert = img_as_float(imread(path_pert + directory + "/"  + name))

        ssim = compare_ssim(img_org,img_pert,multichannel=True)
        mse = compare_mse(img_org,img_pert)
        nrmse = compare_nrmse(img_org,img_pert)
        psnr = compare_psnr(img_org,img_pert)

        ssimTot += ssim
        mseTot += mse
        nrmseTot += nrmse
        psnrTot += psnr
        counter += 1

        compFile.write(name + "\n")
        compFile.write("ssim: " + str(ssim) + "\n")
        compFile.write("mse: " + str(mse) + "\n")
        compFile.write("nrmse: " + str(nrmse) + "\n")
        compFile.write("psnr: " + str(psnr) + "\n")
        compFile.write("\n")

        if(printIntermediateRes):
            print(name)
            print("ssim: " + str(ssim))
            print("mse: " + str(mse))
            print("nrmse: " + str(nrmse))
            print("psnr: " + str(psnr))
            print("")

    ssimAvg = ssimTot/float(counter)
    mseAvg = mseTot/float(counter)
    nrmseAvg = nrmseTot/float(counter)
    psnrAvg = psnrTot/float(counter)

    compFile.write("-----------------" + "\n")
    compFile.write("\n")
    compFile.write("ssim avg: " + str(ssimAvg) + "\n")
    compFile.write("mse avg: " + str(mseAvg) + "\n")
    compFile.write("nrmse avg: " + str(nrmseAvg) + "\n")
    compFile.write("psnr avg: " + str(psnrAvg) + "\n")
    compFile.close()

    print("-----------------")
    print("")
    print("ssim avg: " + str(ssimAvg))
    print("mse avg: " + str(mseAvg))
    print("nrmse avg: " + str(nrmseAvg))
    print("psnr avg: " + str(psnrAvg))


    print("")
    print("All file written to file: " + path_pert + directory + "/" + outfilename)