########Python script that compares two images from two folders with the metrics
########L1 Difference, Mean-IoU, MSE, SSIM



from numpy import asarray
import numpy
from PIL import Image
import cv2
import csv
from skimage.metrics import structural_similarity as ssim
from skimage import data, img_as_float

# import tensorflow as tf
# from tensorflow.keras import backend as K




## IOU in pure numpy
def numpy_iou(y_true, y_pred, n_class=2):
    def iou(y_true, y_pred, n_class):
        # IOU = TP/(TP+FN+FP)
        IOU = []
        #print(range(n_class))
        for c in range(n_class):

            TP = numpy.sum((y_true == c) & (y_pred == c))
            FP = numpy.sum((y_true != c) & (y_pred == c))
            FN = numpy.sum((y_true == c) & (y_pred != c))
            TN = numpy.sum((y_true != c) & (y_pred !=c))

            n = TP
            d = float(TP + FP + FN + 1e-12)

            iou = numpy.divide(n, d)
            IOU.append(iou)

        return numpy.mean(IOU)

    batch = y_true.shape[0]
    y_true = numpy.reshape(y_true, (batch, -1))
    y_pred = numpy.reshape(y_pred, (batch, -1))

    score = []
    for idx in range(batch):
        iou_value = iou(y_true[idx], y_pred[idx], n_class)
        score.append(iou_value)
    return numpy.mean(score)

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = numpy.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err    


count = 1
imageID = 1
MIoU_pix = 0 
MIoU_cyc = 0
p_dif = 0
c_dif = 0
L1_dif = []
MIoU_dif = []
i = 1
fields = ['ID', 'IoU', 'L1']

#Pix2Image.show()
#CycleImage.show()
#GT.show()
while count <2:
   #cleanImage, messyImage = [Image.open(x) for x in ['clean/clean ('+str(count)+').png', 'messy/messy ('+str(count)+').png']]
    cleanImage, messyImage = [Image.open(x) for x in ['images1/image'+str(count)+'.png', 'images2/image'+str(count)+'.png']]
    cleanImage = cleanImage.convert("RGB")
    messyImage = messyImage.convert("RGB")
    NpClean = asarray(cleanImage)
    NpMessy = asarray(messyImage)
    absClean = numpy.mean(abs(NpClean))
    absMessy = numpy.mean(abs(NpMessy))
    squeezeClean = numpy.squeeze(NpClean)
    squeezeMessy = numpy.squeeze(NpMessy)

    ##L1 mean
    dif = 0
    dif = numpy.mean(abs(NpClean - NpMessy)) / 255
    #import pdb; pdb.set_trace()

    #MIoU
    MIoU_val = 0
    MIoU_val = numpy_iou(NpClean, NpMessy)

    #MSE
    mse_val = mse(NpClean,NpMessy)

    #ssimapparently dozed off and didn't notice the vandals sneaking in
    ssim_val = ssim(squeezeClean,squeezeMessy, multichannel=True)

    print(" calculating scores for image "+str(count), end='\r')
    L1_dif = dif
    MIoU_dif = MIoU_val
    #L1_dif.insert(count,dif)
    #MIoU_dif.insert(count,MIoU_val)
    print(L1_dif , end='\r')
    #print(MIoU_dif)

    #write to CSV
    if ssim_val > 0.80:
        #save image if score is good
        #cleanImage = cleanImage.save('a_clean/image' +str(imageID) + '.png')
        #meesyImage = messyImage.save('a_messy/image' +str(imageID)+ '.png')

        row = [imageID, L1_dif, mse_val, MIoU_val, ssim_val]
        column = ["id", "L1 Difference" ,"MSE", "MIOU", "SSIM"]

        scores = [count, absClean, absMessy]
        score_column = ["id", "Clean", "Messy"]

        imageID = count
        #write to csv
        with open('data.csv', 'a', newline='') as f:
            write = csv.writer(f, quoting=csv.QUOTE_ALL)
            #if count <= 1:
            #  write.writerow(column)
           # else:
            write.writerow(row)
    

        #write scores to csv
        #places average of the image matrix

#        with open('scores.csv','a', newline='') as f: 
#            write = csv.writer(f, quoting=csv.QUOTE_ALL)
#            if count < 1:
#                write.writerow(score_column)
#            else:
                

 #               write.writerow(scores)

        count = count + 1
    else:
        print('Skipping image' +str(count) + '----\n')
        count = count + 1



    










#mean IOU
# def numpy_mean_iou(y_true, y_pred):
#     prec = []
#     for t in numpy.arange(0.5, 1.0, 0.5):
#         y_pred_ = tf.cast(y_pred > t, tf.int32)
#         score = tf.numpy_function(numpy_iou, [y_true, y_pred_], tf.float64)
#         prec.append(score)
#     return K.mean(K.stack(prec), axis=0)



#MIoU_pix2 = numpy_mean_iou(NpGT, NpPix)
#MIoU_cyc2 = numpy_mean_iou(NpGT, NpCycle)

#print(MIoU_pix2)
#print(MIoU_cyc2)




