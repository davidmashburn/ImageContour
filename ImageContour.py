import numpy as np
from numpy import sqrt
import pylab

def GetBoundingRect(arr,val):
    x,y=np.where(arr==val)
    return [min(x),max(x)],[min(y),max(y)]

# the four directions:
# right -> [0,1]
# left -> [0,-1]
# up -> [-1,0]
# down -> [1,0]
class directions:
    right=[0,1]
    left=[0,-1]
    up=[-1,0]
    down=[1,0]

def turnL(d):
    if d==directions.right:
        return directions.up
    elif d==directions.left:
        return directions.down
    elif d==directions.up:
        return directions.left
    elif d==directions.down:
        return directions.right
def turnR(d):
    if d==directions.right:
        return directions.down
    elif d==directions.left:
        return directions.up
    elif d==directions.up:
        return directions.right
    elif d==directions.down:
        return directions.left

def addP(p1,p2):
    return [p1[0]+p2[0],p1[1]+p2[1]]
def indA(arr,p):
    if 0<=p[0]<arr.shape[0] and 0<=p[1]<arr.shape[1]:
        return arr[p[0],p[1]]
    else:
        return 0

def GetContourFromSubArray(subArr):
    # Find the first nonzero value on row 0 and set the contour to the right
    for i,p in enumerate(subArr[0]):
        if p==1:
            direction=directions.right
            contour=[[0,i],addP([0,i],direction)]
            break
    # By definition there has to be a turn by the first point
    turns=[1] # 1 if turn, 0 if straight
    perimeter=1
    while contour[-1]!=contour[0]: # Stop when we get back around to the beginning
        left=turnL(direction)
        straight=direction
        right=turnR(direction)
        
        # A contour line is a connection between two contour points
        # A contour point is in between four pixels
        #  with y,x as the point coordinates, these four pixels are:
        NW = addP(contour[-1],[-1,-1])
        NE = addP(contour[-1],[-1,0])
        SW = addP(contour[-1],[0,-1])
        SE = addP(contour[-1],[0,0])
        
        # These map to the test values:
        # right contour: NW,NE  =  0,L
        #                SW,SE  =  1,R
        # up contour:    NW,NE  =  L,R
        #                SW,SE  =  0,1
        # left contour:  NW,NE  =  R,1
        #                SW,SE  =  L,0
        # down contour:  NW,NE  =  1,0
        #                SW,SE  =  R,L
        # Where 1 is the inside point of the contour
        #       0 is the outside point of the last contour
        #       R is the point forward and to the right of the last contour
        # and   L is the point forward and to the left of the last contour    
        if direction==directions.right:
            zerozy,onezy,R,L = NW,SW,SE,NE
        elif direction==directions.up:
            zerozy,onezy,R,L = SW,SE,NE,NW
        elif direction==directions.left:
            zerozy,onezy,R,L = SE,NE,NW,SW
        elif direction==directions.down:
            zerozy,onezy,R,L = NE,NW,SW,SE
        
        if indA(subArr,zerozy)!=0 and indA(subArr,onezy)!=1:
            print 'contour should be 0 on the outside and 1 on the inside!!'
            return
        
        # if L==1:
        #     Turn Left
        # elif R==1:
        #     Go Straight
        # else:
        #     Turn Right
        if indA(subArr,L):
            direction=left
            turns+=[1]
        elif indA(subArr,R):
            direction=straight
            turns+=[0]
        else: # otherwise go right... #indA(subArr,sr):
            direction=right
            turns+=[1]
        contour+=[addP(contour[-1],direction)]
        perimeter+=1
        if perimeter==1000000:
            print 'Error! Perimeter Should not be this large!!!'
            return
        
    return contour,turns

def GetContour(arr,val,boundingRect=None):
    if boundingRect==None:
        b=GetBoundingRect(arr,val)
    else:
        b=boundingRect
    subArr=arr[b[0][0]:b[0][1]+1,b[1][0]:b[1][1]+1]
    return GetContourFromSubArray(subArr==val)

def GetCornersSub(cornerList):
    count=[0]
    for i in cornerList:
        if i==0:
            if count[-1]>0:
                count+=[0]
        else:
            count[-1]+=1
    if count[-1]>0:
        count[0]+=count[-1]
    del(count[-1])
    cornersSub = sum([(i+1)/2 for i in count])
    return cornersSub

def GetIJPerimeter(arr,val,boundingRect=None):
    contour,turns=GetContour(arr,val,boundingRect=boundingRect)
    perimeter=len(contour)-1 # Because first and last point are the same
    cornersSubtract = GetCornersSub(turns)
    #print perimeter,cornersSubtract
    return perimeter - cornersSubtract*(2-sqrt(2))

def PlotArrayAndContour(arr,val):
    [[xmin,xmax],[ymin,ymax]] = GetBoundingRect(arr,val)
    contour = GetContour(arr,1)[0]
    npc = np.array(contour)
    pylab.cla()
    pylab.imshow(arr,cmap=pylab.cm.jet,interpolation='nearest')
    pylab.plot(npc[:,1]-0.5+ymin,npc[:,0]-0.5+xmin,'yo-')
    pylab.xlim(-1,arr.shape[1])
    pylab.ylim(arr.shape[0],-1)

if __name__=='__main__':
    z=np.zeros([10,10])
    z[3,3:5]=1
    z[4,2:6]=1
    z[5,2:6]=1
    b=GetBoundingRect(z,1)
    z2=z[b[0][0]:b[0][1]+1,b[1][0]:b[1][1]+1]

    q=np.array([[0,0,0,0,0],
                [0,0,0,1,0],
                [0,1,0,1,0],
                [0,1,1,1,0]])
    print GetIJPerimeter(z,1)
    print GetIJPerimeter(q,1)
    pylab.figure(1)
    PlotArrayAndContour(z,1)
    pylab.figure(2)
    PlotArrayAndContour(q,1)
    import DelayApp
