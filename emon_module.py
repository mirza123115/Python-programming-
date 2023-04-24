# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 21:38:21 2020

@author: mirza emon
"""

def ad(x,y):
    return x+y
if __name__=='__main__':
    print(__name__)
    print(ad(2,3)) 



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




