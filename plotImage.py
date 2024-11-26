import PIL.Image
import PIL.ImageFile
import ultralytics.utils.plotting as plotting
import PIL
import numpy as np
data=np.loadtxt("/home/till/git/ultralytics/data_gen/Dataset/labels/test/image_0279.txt").T
cls = data[0]
bboxes = data[1:5]
img = PIL.Image.open("/home/till/git/ultralytics/data_gen/Dataset/images/test/image_0279.jpg")
img=np.array(img).reshape(1,3,512,512)
plotting.plot_images(img,0,cls,bboxes.squeeze().T,)
print("ok")