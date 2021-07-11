import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
from numba import jit
N= 53

@jit(nopython=True)
def Gfunc(qs,qi,e): 
    p = 2
    c = 16.
    s = 2.5 * c
    ci = abs(1/(1-e))
    return (30./c)*np.exp(- ((p**2+c**2)*(qs**2+qi**2)+2*qs*qi*c**2)/(2*p**2*c**2))*np.sinc(((2+e)*0.5*abs(qs-ci*qi)**2 + e*0.5*abs(qs+ci*qi)**2)/(np.pi*s**2))#simplified collected JTMA

sigma_ratio=[] #initialising ratio of widths
area=[]#initialising ratio of areas 
for e in np.arange(0,1,0.1):
    Pr_M= np.identity(N) #initialising Pr matrix
    def Pr(qs, qi, e, a_s, a_i):
        if qs>a_s and qi>a_i or qs<a_s and qi<a_i:
            return Gfunc(qs, qi, e)
        else:
            return -Gfunc(qs,qi,e)
    #we don't need to find complete Pr matrix only need for i=0 and j=0
    for i in range(1): #loop for i=0
        for j in range(N):
            a_i = i - (N - 1) / 2
            a_s = j - (N - 1) / 2
            I1=sci.dblquad(Pr,-np.inf,0,-np.inf,0,args=[e,a_s,a_i])[0]
            I2=sci.dblquad(Pr,-np.inf,0,0,np.inf,args=[e,a_s,a_i])[0]
            I3=sci.dblquad(Pr,0,np.inf,-np.inf,0,args=[e,a_s,a_i])[0]
            I4=sci.dblquad(Pr,0,np.inf,0,np.inf,args=[e,a_s,a_i])[0]
            Pr_M[i][j] = abs(I1+I2+I3+I4)**2

    for j in range(1): #loop for j=0
        for i in range(N):
            a_i = i-(N-1)/2
            a_s = j-(N-1)/ 2
            I1=sci.dblquad(Pr,-np.inf,0,-np.inf,0,args=[e,a_s,a_i])[0]
            I2=sci.dblquad(Pr,-np.inf,0,0,np.inf,args=[e,a_s,a_i])[0]
            I3=sci.dblquad(Pr,0,np.inf,-np.inf,0,args=[e,a_s,a_i])[0]
            I4=sci.dblquad(Pr,0,np.inf,0,np.inf,args=[e,a_s,a_i])[0]
            Pr_M[i][j] = abs(I1+I2+I3+I4) ** 2

    ys1=Pr_M[:,0] #slicing 0th column
    ys2=Pr_M[0,:] #slicing 0th row
    xs=np.arange(-(N - 1) / 2, 1 + (N - 1) / 2)

    @jit(nopython=True)
    def find_width(ys):
        mx=ys[0]
        k=0
        for i in range(len(ys)):
            if ys[i]<=0.5*mx :
                k+=1#counts the number of data points below y=0.5mx
        return k
    def find_area(ys):
        area= sci.simps(ys,xs)#finding area
        return area

    sigma_ratio.append(find_width(ys1)/find_width(ys2)) 
    area.append(find_area(ys1)/find_area(ys2))#area1= find_area(ys1)
    plt.plot(xs,ys1,'o-',label= '$a_i$')
    plt.plot(xs,ys2,'o-',label='$a_s$')
    plt.title('$\epsilon$ = %.2f'%e)
    plt.xlabel('$a_s$, $a_i$')
    plt.ylabel('Pr')
    plt.legend()
    plt.show()

plt.plot(np.arange(0,1,0.1),np.array(sigma_ratio),'o-',label='ratio of widths')
plt.title('Asymmetry by finding ratio of widths')
plt.xlabel('$\epsilon$')
plt.ylabel('ratio of widths')
plt.legend()
plt.show()
plt.plot(np.arange(0,1,0.1), np.array(area),'o-',label='ratio of areas')
plt.title('Asymmetry by finding ratio of areas')
plt.xlabel('$\epsilon$')
plt.ylabel('ratio of area')
plt.legend()
plt.show()
