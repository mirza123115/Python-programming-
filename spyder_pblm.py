# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:42:50 2020

@author: lenov
"""

#input output in python
x=int (input("take x="))
x=3
print(x)
y=[3, 4, 5]
#string position
format("hfvk",">66")
format("hfvk","<86")
format("hfvk","^86")
print("hdh \n fhc")
#logical oprration
1 and 1
1|1
5//2
9/6
9//6
9%5
2**3**5
a=input("a=")
x=["fj","yfj","yi"]
x[2]
x[0]+x[1]
y=[[6,7],[7 ,9]]
y[0]
y[1][1]=2.3

#string function

y.insert(1,' uhg')
del y[1]
y.sort()
p=[1,2,3]
z=3*p
z.sort()
z.reverse()
b=((2,3),(6,8))
b[0]
b[0][0]
len(b)
len(z)
c=z[1:10]
name="emon"
name.count('m')
name.index('m')
sum(c)
min(c)
max(c)
x+y
#loop operation

d=[ x**x for x in [1,2,3]]
c[-8]
#numpy library

import numpy as np
e=np.array([[1,2],[3,4]])
f=np.array([5,6])
np.linalg.solve(e,f)
import matplotlib.pyplot as plt
t=0;
while t<7:
    t+=.01
    print(t)
plt.plot(t)
plt.show()
#array making

h=range(1,10)
for i in h:
    print(i)
    
#function defination

def cube(x) :
    return x**3

cube(3)    

def sow():
    print("emon the great progamer")
    
sow()
 
def icmnt():
    global i
    i=i+1
    print(i)
    
icmnt()

#lambda function
ad=lambda a,b:a+b
ad(10,20)
def seq(*x):
    for a in x:
        
        print(a)
seq(1, 2, 3)
abs(-4)
round(3.1416897,3)
import math as m 
#math library

import math 
math.floor(4.33)
math.ceil(3.44)
math.sin(math.pi/2)
math.cosh(3)
math.trunc(3.44)
math.e
math.pi
math.exp(4)
math.sqrt(4)
m.pi
m.log(1)
m.log10(10)
m.modf(3.335)
m.pow(2, 3)
m.degrees(m.pi)
m.radians(180)

#string library function
name="hi emon"
name.capitalize()
name.count('emon')
name.endswith('emon')
name.find('on')
len(name)
name.split()
name.split('h')
name.title()
name.lower()
name.islower()
name.upper()
name.swapcase()
name.replace('emon', 'new')
name.strip('h')
m.factorial(5)
help(math);

import sys
sys.path

#datta type dictonay
dic={}
dic['luba']=8801777012143
dic['emon']=8801521213174
dic
dic1={'bisty':'mirzabristy@gmail.com'}
dic['emon']
dic['luba']    
dic.keys()
dic.values();
m.erf(5)

#different libray importation
import numpy as np
import matplotlib as mp
import scipy as sc
import math as m
import cmath as cm
import imageio as im
import imagesize as ims
import audioop as ado
import pandas as pd
c=complex(2+3j)
c.conjugate()
c.__abs__()
c.__bool__()
c.__class__()
c.__pow__(2)
x=3+4j
x.__abs__()
i.bit_length()
bird={'tia','moina'}
bird.add('doyel')
fruit={'orng','banana'}
fruit | bird
bird.clear()
bird

#set data type
a={1,2,3} 
b={3,4,5}
a-b #intersection
a&b
b-a
a^b



#file handaling
fyl=open("emon.txt",'w')
fyl.write("hi emon")
fyl.close()
fyl=open("emon.txt",'a')
fyl.write(" hi luba")
fyl.close()
fyl=open("emon.txt",'r')
print(fyl.read())
fyl.close()

fyl=open("luba.txt",'r')
print(fyl.read())
fyl.close()

import os
os.renames("luba.txt","momi.txt")
os.getcwd()
os.renames('C:\\Users\\lenov\\.spyder-py3\\emon.mat','C:\\Users\\lenov\\.spyder-py3\\mirza.mat')
os.mkdir('C:\\Users\\lenov\\.spyder-py3\\emon_mirza')
os.mkdir("boss")
os.listdir('C:\\')    
os.listdir()
m.factorial(5)  

#exception handaling  
try:
   m.factorial(-5)    
except:
        print("error")
    

#class ,oop
class engr:
     
     university="Aust"
     dept="EEE"
     def __init__(self, name, age):
         self.name = name
         self.age = age
         print("my university is",self.university,"and i study in the dept of ",self.dept)
         print("i am ",name)
         print("my age is",age)
     def svalue(self,weight,cg,hight):
         print("my weight=",weight,"cg=",cg,"hight=",hight)
     def bio(self):
         print("give bio")
         

emon=engr("mirza emon",24)
emon.bio()
emon.svalue(55, 3.55, 5.8)
luba=engr("momii",16)   
luba.svalue(50, 4.61, 5.4)
luhul=engr("riaubal",35)
luhul.svalue(66, 2.3, 4.2)

print(emon.__dict__)

#inheretance and fncn overriding
class doctr(engr):
      def __init__(self):
          print("i am a dctr")
      def bio(x):
        print("hi")
      def cember(self,pos):
          self.pos=pos
          print("i work in ",pos)

#multilavel and multiple inheritance
class  expt(doctr):
    def __init__(self):
        print("i am an exprt")
    def natre(self):
        print("i am motin who is simultaniusly know eng and ddctr")

kutar=doctr()
kutar.bio()

motin=expt()
motin.bio()
motin.svalue(60, 2.00, 5.8)
motin.cember("gulistan")


class hugur:
    def __init__(self,name ,age):
        self.name=name
        self.age=age
        print("hi i am hugur my name is",name,"a am ",age,"years old")
    def worksop(self,area):
        self.area=area
        print("my dukan is in  ",area)
        
class allrndr(expt,hugur):
    __tamgidincm=0
    def __init__(self):
        self.__tamgidincm=10000
        print("hi i am expert")
        print("my monthly income is",self.__tamgidincm)
    def sowme(self):
        print("i am tamjid and popular as alrounder")
        print("my monthly income is",self.__tamgidincm)
        self.__ips()
    def cngincm(self,newincm):
        self.__tamgidincm=newincm
        print("income update",newincm)
    
    #private method
    def __ips(self):
        print("i dont tell ips mechanism")
    
    
class pagol:
    def __init__(self):
        print("i am pagol")
    def bio(self):
        print("nothing to say about pagol")
        

#polymorpism
def quality(typ):
    typ.bio()


#pagol object  
simul=pagol()
simul.bio()

#hugur object
jamil=hugur("jamil", 27)
tamgid=allrndr()
tamgid.worksop("aminpur")

#private varible can't be changed directly
tamgid.__tamgidincm=100
tamgid.sowme()


emon=engr("emon", 20)
emon.bio()
quality(emon)
quality(simul)


x=2+3j
x.conjugate()
x.__abs__()

#time manipulation
import time as t
t.time()
t.localtime()
t.asctime()
t.sleep(5)
print("delay 5sec")

#calender module
import calendar as c
print(c.month(2019, 10))
print(c.calendar(200))
type(quality)
dir()

#pattern
for x in range(1,10):
     print()
     for y in range(1,x+1):
        print("x",end="")
     #print()

#loop
lst=[5,3,8]

for x in lst:
    print(x)
       

x="emon"
y=iter(x)    

next(y)    
len(x)    
x=567
len(str(x))    
    
import math
dir(math)
dir()    

x=range(1,9,2)
for a  in x:
    print(a)
    
p=[1,7,9]
p.append("ho")    
p.extend("ghu")    
n=eval(input())    

q=lambda x,y:x+y                      
 
q(1,5)  
  
def sqr(x):
    return x**2

#lambda and map,reduce,filter fun
list(map(sqr,p))
list(map(lambda x:x**3,p)) 
r=[5,2,3]
p=[1,2,10]
tuple(map(lambda x,y:x+y,p,r))
list(filter(lambda x:x>20,range(1,35)))
import  functools 
functools.reduce(lambda x,y:x+y,r)

#list comperihention
[2*x for x in range(1,10)]    
[x+y for x in p for y in r ]
[p[i]+r[i] for i in range(len(p))]
[x for x in range(9)]
[x if x%2==3 else x**2 for x in range(9)]


#emon module is a module define any function an d class here
import emon_module as emon

print(__name__)
print(emon.ad(2,3))

bin(4)
~50
23>>2
32<<2    
4<<1
help(emon)

p="emomn"
dic2={"1":"a","2":"b"}          
p.maketrans(dic2)
help(str)
from numpy import *
import turtle
t=turtle.Turtle()
t.shape("turtle")
t.forward(100)
t.backward(300)
t.right(90)


#import defined library and call
import emon_module as e
help(e)
e.ad(2,3)
emon=e.engr("mirza",23)
emon.bio()
emon.svalue(55,3.6,5.8)
tamjid=e.allrndr()
tamjid.sowme()
tamjid.cngincm(1000)
quality(emon)
simul=e.pagol()
quality(simul)
jamil=e.hugur("jamil",33)
tiran=e.doctr()
tiran.bio()
tiran.cember("gulistan")
e.ad(2,3,3)

import turtle 
t=turtle.Turtle()
w=turtle.Screen()
w.bgcolor("green")
t.fillcolor("red")
t.speed(0)
t.begin_fill()
for i in range(4):
    t.forward(100)
    t.left(90)

t.hideturtle()
t.end_fill()

t.reset()
t.color("white")
w=turtle.Screen()
w.bgcolor("green")
t.begin_fill()
t.fillcolor("red")
t.circle(100,steps=6)
t.end_fill()
t.hideturtle()

#cng positn
t.reset()
t.up()
t.goto(0,-100)
t.down()
t.begin_fill()
t.fillcolor("red")
t.circle(100)
t.end_fill()
t.hideturtle()

#function for circle
def drwcircle(x,y,clr,rad):
    t.goto(x,y)
    t.down()
    t.begin_fill()
    t.fillcolor(clr)
    t.circle(rad)
    t.end_fill()
    #t.hideturtle()
    t.up()
    t.home()
    
    
t.reset()
t.up()
drwcircle(100, 100, "red", 50)
drwcircle(-100, 100, "orange", 50)
drwcircle(-100, -100, "blue", 50)
drwcircle(100, -100, "purple", 50)


#closer
x=20
def outer():
    y=10
    def inner():
        nonlocal y
        y=4+y
        return print(x+y)
    return inner
    

a=outer()
print(a.__name__)


#copy
x=[1,7,8]
y=x
y.append(23)
y
x
z=list(x)
x.append(67)
w=x[:]
w
x.append(6)
x
w
p=[a for a in x]
p
p.append(8)
p
x
import copy
n=copy.copy(x)
n
n.append(9)
n
x
id(x)


#array indexing and slicing
#sliceing x[start:end:step,start:end:step,start:end:step]

import numpy as np
y=np.array([1,5,3,4])
x=y/3
y[1]

z=np.array([[1,2],[3,4]])
c=np.array([1,3,8],complex)
p=np.array([[[1,6],[7,9]],[[1,2],[3,4]]])
k=np.arange(2,7,2,dtype=complex)
v=None
z=np.zeros((7,8))
z=np.ones((5,9))
z=np.eye(6)
z=np.linspace(1,5,67)
z=np.empty(7)
z=np.linspace(1,8)
help(np.random)
z=np.random.rand(5,8)
z.ndim
z.shape
z.size             
z.dtype
z.itemsize
z=np.float64(6)
z.itemsize
p[:][:][0]
a=z[2: :,2:4]

#advance indexing
import numpy as np
x=np.array([1,6,8,9,0,8])
indx=[1,3,4]
x[indx]
x[[1,3,4]]
z=np.random.rand(2,2)
w=2*z
z[[0,1],[1,0]]
x[x<6]

x=np.array([1,6,8,9,0,8])
y=2*x
x/2
z+2
y+x
y*x
w+z
z
w
np.add(w,z)
np.subtract(w,z)

import numpy as np
x=np.arange(1,10)
y=np.reshape(x,(5,2),order='f')
z=np.resize(x,(4,3))
z.flatten()
z.ravel()
z.transpose()
p=np.arange(1,7)
q=np.concatenate((p,x))
np.split(x,3)

x=170105000
while x<170105200:
    x=x+1
    y=str(x)+"@aust.edu"
    print(y)
    
np.insert(x,1,13)
np.delete(x,3)
z.dot(z)
np.linalg.inv(z)
np.linalg.matrix_power(z,2)
a=np.array([[1,2],[3,5]])
b=np.array([6,7])
np.linalg.solve(a,b)
np.linalg.det(a)


#matplotlib 
import numpy as np
import matplotlib.pyplot as plt
import math as m
x=np.linspace(0,4*m.pi,1000)
y=[m.sin(a) for a in x]
z=[m.sin(a+m.pi*2/3) for a in x]
w=[m.sin(a+m.pi*4/3) for a in x]
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("first plot in python sin and cos")
plt.plot(x,y,"r",x,z,"y",x,w,"g",zorder=2)
plt.plot(x,z,"g",linewidth=21,antialiased=True,zorder=1)
plt.show(block=False)
plt.xlim(2,6)
plt.plot(x,y,"r",x,z,"y",zorder=2)
plt.legend("sin", "cos")
plt.plot(x,z,"g",x,w,"b",x,y,"r",linewidth=21)

#scipy
import scipy as sc
w=sc.fft.fft(y)
plt.plot(w)
plt.grid(True)


plt.barh(["eee","cse","ece"],[1 ,2, 3])
p=np.random.randn(1000)
plt.hist(p)

#module making
import emon_module



#computer vision
import cv2,time
im=cv2.imread("E:\\python_project\\mirza.jpg",1)
im1=cv2.resize(im,(int(im.shape[1]/2),int(im.shape[0]/2)))
im1=cv2.resize(im,(1280,720))
cv2.imshow("emon",im)
cv2.waitKey(0)
cv2.destroyAllWindows()


#video
video=cv2.VideoCapture(0)
cheak,frame=video.read()
print(cheak)
print(frame)
time.sleep(3)
cv2.imshow("emon",frame)
cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()



#viseo disply
import cv2,time
import numpy as np
x=np.ones((720,1280,3))


video=cv2.VideoCapture(0)
while True:
    cheak,im=video.read()
    print(frame)
    im1=cv2.resize(im,(1280,720))
    cv2.imshow("emon",im1)
    key=cv2.waitKey(1)
    if key==ord('a'):
        break

video.release()
cv2.destroyAllWindows()





#pandas module
import pandas as pn
dta={"roll":[33,44,66],"cg":[2,3,3.2]}
pdta=pn.DataFrame(dta,index=[1,2,3])
print(pdta)
pdta.to_html('dta.html')
pdta.to_csv()



import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr


img=cv2.imread("E:\\python_project\\car9.jpg",1)
cv2.imshow("emon",img)
cv2.waitKey(0)
cv2.destroyAllWindows()



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))


bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))


keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]


location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
location 
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))



(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))


reader = easyocr.Reader(['en','bn'])
result = reader.readtext(cropped_image)
result

text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))