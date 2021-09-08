import cv2 as cv
import numpy as np
from math import dist
import time
import math
from matplotlib import pyplot as plt


def mod(a):
  return math.sqrt(a[0]**2+a[1]**2)

def dotproduct(a,b):
  return a[0]*b[0]+a[1]*b[1]


def quadrant(origin, cords):
    ox = origin[0]
    oy = origin[1]
    cx = cords[0]
    cy = cords[1]

    if cx <= ox and cy >= oy:
        return 1
    if cx >= ox and cy >= oy:
        return 2
    if cx <= ox and cy <= oy:
        return 3
    if cx >= ox and cy <= oy:
        return 4


def kangle(origin,b):
  q = quadrant(origin,b)
  #print(q)
  #vector form of -x axis
  oa =[-1,0]
  #calculating directional vectors oa,ob
  #oa=[a[0]-origin[0],a[1]-origin[1]]
  ob=[b[0]-origin[0],b[1]-origin[1]]
  #finding dot product of two vectors
  dp = dotproduct(oa,ob)
  #finding the angle
  angle= math.acos(dp/(mod(oa)*mod(ob)))
  angle = math.degrees(angle)
  if q>2:
    angle = 360-angle
  return angle



st=time.time()
def array2bool(a):
    if a[0] and a[1] and a[2]:
        return True
    else:
        return False

def filter(img,clr_le,clr_ue):
    asi=0
    aki=np.zeros([np.shape(img)[0],np.shape(img)[1],1])
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if array2bool(img[i,j]>=clr_le) and array2bool(img[i,j]<=clr_ue):
                aki[i,j] =255
                asi+=1
            else:
                aki[i,j]=0
    return aki,asi
#searching columwise
def frame_gen(img):
    aki=np.zeros(np.shape(img),'uint8')
    flag=0
    temp=0
    flag_r=0
    temp_r=0
    cords=[]
    for i in range(np.shape(img)[1]):
        for j in range(np.shape(img)[0]):
            if flag==0:
                if temp==1:
                    flag=1
                    temp=0
                if img[j,i]==255:
                    aki[j,i] = 255
                    cords.append([j,i])
                    temp=1

            if flag==1:
                if img[j,i]==0:
                    aki[j-1, i] = 255
                    cords.append([j-1,i])
                    flag=0
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if flag_r==0:
                if temp_r==1:
                    flag_r=1
                    temp_r=0
                if img[i,j]==255:
                    aki[i,j]=255
                    cords.append([i,j])
                    temp_r=1

            if flag_r==1:
                if  img[i,j]==0:
                    aki[i-1,j] =255
                    cords.append([i-1,j])
                    flag_r=0

    return aki,cords

class point:
    def __init__(self,point,angle,distance):
        self.point=point
        self.angle = angle
        self.distance = distance

#reading the image
aki = cv.imread("testGreenSquare2.jpg", cv.IMREAD_COLOR)
cv.imshow('originalimage',aki)
aki,asi=filter(aki,[67,167,25],[87,187,45])
#print(asi)
cv.imshow('greemask',aki)
asi, cord = frame_gen(aki)
cord = np.array(cord)
cord=np.unique(cord,axis=0)

max=np.max(cord,axis=0)
min=np.min(cord,axis=0)
points=[]
center=[(max[0]+min[0])/2,(max[1]+min[1])/2]
#(center)
val=[]
for i in cord:
    points.append(point(i,kangle(center,i),dist(center,i)))

lti = []
lsi = []
#sorting the points by angle
points.sort(key=lambda x: x.angle)
for i in points:
    lsi.append(i.distance)
    lti.append(i.angle)

a=plt.plot(lti,lsi)
print(a)
plt.show()
cv.imshow('squareframe_h',asi)
print(time.time()-st)

cv.waitKey(0)
cv.destroyAllWindows()
