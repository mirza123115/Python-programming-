def volum(x):
    return x**3

print(volum(2))
'''fyl=open("luba.txt",'r')
print(fyl.read())
fyl.close()'''
class engr:
     
     university="Aust"
     dept="EEE"
     def disp(self) :
         print(self.university,self.dept)
     def svalue(self,age,cg,hight):
         print("age=",age,"cg=",cg,"hight=",hight)
 
emon=engr() 
bristy=engr()   
emon.disp()
emon.svalue(24, 3.55, 5.8)
bristy.svalue(18, 4.61, 5.4)