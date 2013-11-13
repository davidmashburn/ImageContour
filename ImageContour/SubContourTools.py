#!/usr/bin/env python
''''A useful collection of tools to aid in manipulating contour networks
This is admittedly less coherent/documented and more "bleeding edge" than ImageContour.py'''

import os
from copy import deepcopy
#import cPickle # old pickle version
import json
import itertools
import operator
from operator import itemgetter

import numpy as np

import matplotlib.pyplot as plt

import ImageContour

#from list_utils:
from np_utils import ( totuple, interpGen, flatten, ziptranspose, roll,
                       deletecases,partition, polyCirculationDirection,
                       groupByFunction, getElementConnections,
                       getChainsFromConnections, removeDuplicates,
                       removeAdjacentDuplicates )
#from func_utils
from np_utils import compose
#from np_utils:
from np_utils import ( limitInteriorPoints, limitInteriorPointsInterpolating,
                       addBorder, getValuesAroundPointInArray )

def GetValuesAroundSCPoint(watershed2d,point,wrapX=False,wrapY=False):
    vals = getValuesAroundPointInArray(watershed2d,point,wrapX,wrapY)
    return ( tuple(vals.tolist()) if vals!=None else (None,None,None) )

def _kwdPop(kwds,key,defaultValue):
    '''If a dictionary has a key, pop the value and return it,
       otherwise return defaultValue'''
    return ( kwds.pop(key) if key in kwds else defaultValue )

class SubContour(object):
    '''A class to hold the data for a single SubContour (basically a connected list of points)
       This was designed to replace the old and crufty cVLS (contour's values,length, and subcontour)
       These three fields map to values, adjusted_length, and points respectively (much clearer!)
       This is one sc, not a list of them (that's usually referred to as 'scList')
       State: This class does not manage its own state past construction
       '''
    points = [] # list of (x,y)'s
    numPoints = 0
    adjusted_length = 0 # length as computed by a perimeter-style algorithm
    values = (None,None) # always 2 values
    startPointValues = (None,None,None) # 3 or possibly 4 values around the start point ("triple junction")
    endPointValues = (None,None,None)   # 3 or possibly 4 values around the end point ("triple junction")
    identifier = None # only used sometimes for sorting purposes
    def __init__(self,**kwds):
        '''Create a SubContour
           All arguments are converted to attributes; all are optional,
           but some functions only work when these are set properly:
             * points: list of (x,y)'s
             * values: always 2 cellID values
             * startPointValues: 3 or possibly 4 values around the start point ("triple junction")
             * endPointValues:   3 or possibly 4 values around the end point ("triple junction")'''
        for k in kwds:
            self.__dict__[k] = kwds[k]
        if 'numPoints' not in kwds.keys():
            if 'points' in kwds.keys():
                self.numPoints = len(self.points)
    def cVLS(self):
        '''For legacy purposes, returns a list'''
        return [self.values,self.adjusted_length,self.points]
    
    def cornerConnections(self):
        '''Returns connection values at the endpoints that are not one of the 2 main values'''
        return tuple(sorted( set(self.startPointValues+self.endPointValues).difference(self.values) ))
    
    def plot(self,*args,**kwds):
        x = [ p[0]-0.5 for p in self.points ]
        y = [ p[1]-0.5 for p in self.points ]
        return plt.plot( y,x, *args, **kwds )
    def plotT(self,*args,**kwds):
        x = [ p[0]-0.5 for p in self.points ]
        y = [ p[1]-0.5 for p in self.points ]
        return plt.plot( x,y, *args, **kwds )

class QuadPoint(object):
    values = None
    point = None
    def __init__(self,values,point):
        if values.__class__!=tuple:
            raise TypeError('values must be a tuple')
        if not len(values)==4:
            raise TypeError('values must be a 4-tuple')
        self.values=values
        self.point=point

class DegenerateNetworkException(Exception):
    pass

class CellNetwork(object):
    '''Holds the critical information for a single frame in order to reconstruct any subcontour of full contour
       State: This object manages it's state and the state of all it's members like SubContour
              External functions, however should NOT mutate its state if it is passed as an argument;
              use a deepcopy-modify-return solution instead'''
    subContours = [] # list of SubContours
    quadPoints = [] # list of QuadPoints
    contourOrderingByValue = {} # information about reconstructing full contours; these should each be a tuple like:
                                # (<index into subContours>,<boolean that determines if this is forward (True) or backwards (False)>)
                                # Don't use this directly, use the GetContourPoints method instead
                                # no more negative values... (used to mean reverse the contour when inserting)
    allValues = []
    def __init__(self,**kwds):
        '''Create a CellNetwork
           All arguments are converted to attributes; all are optionally,
           but these are needed for functions to work:
             * subContours: list of SubContour objects
             * contourOrderingByValue: dictionary with keys that are cellIDs and tuple values like:
                                       ( <index into subContours>,
                                         <boolean that determines if this is forward (True) or backwards (False)> )'''
        for k in kwds:
            self.__dict__[k] = kwds[k]
        if 'quadPoints' not in kwds.keys():
            if 'subContours' in kwds.keys():
                self.UpdateQuadPoints()
    
    def UpdateQuadPoints(self):
        '''Update quadPoints from subContours
           State: Changes state of quadPoints variable, but only to be more consistent'''
        quadPoints = sorted(set( [ (sc.startPointValues,tuple(sc.points[0])) for sc in self.subContours
                                                                             if len(sc.startPointValues)==4 ] +
                                 [ (sc.endPointValues,tuple(sc.points[-1])) for sc in self.subContours
                                                                            if len(sc.endPointValues)==4 ] ))
        self.quadPoints = [ QuadPoint(vals,pt) for vals,pt in quadPoints ]
    def GetContourPoints(self,v,closeLoop=True):
        '''Get Contour points around a value v
           State: Access only'''
        def reverseIfFalse(l,direction):
            return ( l if direction else l[::-1] )
        scPointsList = [ reverseIfFalse( self.subContours[index].points, direction ) # reconstruct sc's, flipping if direction is False
                        for index,direction in self.contourOrderingByValue[v] ] # for each index & direction tuple
        contourPoints = [ totuple(pt) for scp in scPointsList for pt in scp[:-1] ] # start point is not end point; assumed to be cyclical...
        if closeLoop:
            contourPoints.append(contourPoints[0]) # Tack on the first point back on at the end to close the loop
        return contourPoints
    
    def GetCvlsListAndOrdering(self):
        '''For legacy purposes, returns a list
           State: Access only'''
        cVLS_List = [ sc.cVLS() for sc in self.subContours ]
        return cVLS_List,self.contourOrderingByValue
    
    def GetCellularVoronoiDiagram(self):
        '''Creates a VFMin frame input structure (CellularVoronoiDiagram (cvd)), of the form:
           [
              [(x1,y1),(x2,y2)...],                      # xyList
              [
                [v,[<list of indexes into xyList...>]],  # polygon
                [v,[<list of indexes into xyList...>]],
                ...
              ]
           ]
           This is really, really similar to GetXYListAndPolyList except that here polyList is a tuple and not a dict...
           State: Access only'''
        xyList = self.GetAllPoints()
        indexMap = { tuple(pt):i for i,pt in enumerate(xyList) } # This should massively speed this up over calling xyList.index over and over
        polyList = []
        for v in self.allValues:
            contourList = self.GetContourPoints(v,closeLoop=False)
            #indexList = [ xyList.index(totuple(pt)) + 1 for pt in contourList ] # THIS IS 1-INDEXED! # this is really slow
            indexList = [ indexMap[totuple(pt)] + 1 for pt in contourList ] # THIS IS 1-INDEXED!
            polyList.append( [ v,indexList ] )
        return [xyList,polyList]
        
    def UpdateAllValues(self):
        '''Go through the values in all the subContours and collect a list of all of them
           State: Changes state of allValues, but only to to be more consistent'''
        self.allValues = sorted(set( [ v for sc in self.subContours 
                                         for v in sc.values
                                         if v!=1 ] ))
    
    def GetAllPoints(self):
        '''Get a sorted set of all points in the subContours
           State: Access only'''
        return sorted(set( [ tuple(pt) for sc in self.subContours for pt in sc.points ] ))
    
    def GetXYListAndPolyList(self,closeLoops=True):
        '''Get a list of points (xyList) and a dictionary of index lists (into xyList) with cellID keys (polyList)
           polyList contains the information that reconstructs each individual contour from points' indices
               (much like contourOrderingByValue does using scs' indices)
           'closeLoops' determines if the first point is also appended to the end of each list to close the loop
                        and makes plotting cleaner, but be cautious of this
           State: Access only'''
        xyList = self.GetAllPoints()
        polyList = {}
        
        for v in self.allValues:
            contourPoints = self.GetContourPoints(v,closeLoop=False)
            polyList[v] = [ xyList.index(totuple(pt)) for pt in contourPoints ] # skip each endpoint
            if closeLoops:
                polyList[v] = polyList[v]+[polyList[v][0]] # Tack on the first point back on at the end to close the loop
                                                           # VFMin doesn't like this format; make sure to remove this last
                                                           # point before saving to a file or passing to VFM...
            #polyList[-1][v] = removeDuplicates(polyList[-1][v])+[polyList[-1][v][0]] # Remove interior duplication... bad idea!
        
        return xyList,polyList
    
    def CheckForDegenerateContours(self):
        '''Get a list off subcontours (indexes) that belong to degenerate contours (either points or lines)'''
        problemSCs = set()
        problemVals = set()
        for v,scInds in self.contourOrderingByValue.iteritems():
            if len(scInds)<3:
                scInds = [i[0] for i in scInds] # before, this also contained subcontour direction information
                if sum([ self.subContours[i].numPoints-1 for i in scInds ]) < 3:
                    # Aka, there are not enough points to make a triangle...
                    problemVals.add(v)
                    problemSCs.update(scInds)
        problemValuePairs = [ self.subContours[i].values for i in problemSCs ]
        return sorted(problemVals),sorted(problemValuePairs)
    
    def LimitPointsBetweenNodes(self,numInteriorPointsDict,interpolate=True,checkForDegeneracy=True):
        '''Operates IN-PLACE, so use cautiously...
           State: Changes subContours and numPoints; leaves quadPoints in an improper state
                  (allValues and contourOrderingByValue are unaffected) '''
        limIntPtsFunc = limitInteriorPointsInterpolating if interpolate else limitInteriorPoints
        for sc in self.subContours:
            sc.points = limIntPtsFunc(sc.points,numInteriorPointsDict[tuple(sc.values)])
            sc.numPoints = len(sc.points)
        
        if checkForDegeneracy:
            problemVals,problemValuePairs = self.CheckForDegenerateContours()
            if problemVals!=[]:
                print 'Values with degenerate contours:',problemVals
                raise DegenerateNetworkException('Degenerate contours (zero area) found between these cellIDs!'+repr(problemValuePairs))
    
    def CleanUpEmptySubContours(self):
        '''If we deleted a bunch of contours, this reindexes everything.
           State: Changes the variables subContours,numPoints,contourOrderingByValue
                  (allValues is not changed, but maybe it should be)
                  This should take most dirty states and clean them up'''
        # First things first, make a mapping from old indices to new:
        scIndexMap = {}
        count = 0
        for i in range(len(self.subContours)):
            if self.subContours[i]!=None:
                scIndexMap[i] = count
                count+=1
                
        # Now go through and delete all the dead sc's
        self.subContours = [ sc for sc in self.subContours if sc!=None ]
        
        # Now, go in and reindex contourOrderingByValue
        for v in self.contourOrderingByValue.keys():
            self.contourOrderingByValue[v] = [ (scIndexMap[i],d) for i,d in self.contourOrderingByValue[v] ]
        
        # And, last-but not least, update the quad points
        self.UpdateQuadPoints()
    
    def RemoveValues(self,valuesToRemove):
        '''Remove all the values from all relevant attributes -- until further notice, don't use this;
           just go back to the raw array and remake a network with fewer values...
           State: (could leave object in an inconsistent state)'''
        ### THIS NEEDS SERIOUS RETHINKING! When you leave segments behind, two or more may become part of a single loop:
        # The steps to remedy this:
        # ID ordered groups of sc's with like value pairs (after cell removal)
        # ID all the places that the sc's are used in contourOrderingByValue
        # join these all together in the correct order:
        #      add the points to end of the first segment, set the other to None
        # remove links to the dead sc and ensure that the remaining one is in the places it should be
        # do the standard cleanup; remove the None's from subContours and reindex conourOrderingByValue
        
        # So, the real key is ID'ing segs that are linked:
        #  * must have same value pair
        #  * must have an end-to-end linkage (end or start) = (end or start) = ...
        #  * must be following one another in the same (or reverse) order in ALL contourOrderings
        #  * connecting points have to be singly-connected; no internal triple junctions
        
        if valuesToRemove==[]:
            return # Do nothing...
        
        if (1 in valuesToRemove):
            raise ValueError("You can't eliminate the background (value=1)!")
        
        # Collect a list of all SC's to be outright removed:
        scsToRemoveInternal = [ (i,sc) for i,sc in enumerate(self.subContours)
                               if len(set(sc.values).intersection(valuesToRemove))==2 ] # aka, this sc is is between 2 values we're removing
        scsToRemoveByBackground = [ (i,sc) for i,sc in enumerate(self.subContours)
                                   if sc.values in [(1,v) for v in valuesToRemove] ] # aka, this sc is between the background and a value to be removed
        # and remove them...
        for i,sc in (scsToRemoveInternal + scsToRemoveByBackground):
            self.subContours[i]=None
        
        # And now, replace occurrences of valuesToRemove in the sc.values by 1 (background) instead
        for sc in self.subContours:
            if sc!=None:
                len_intersect = len(set(sc.values).intersection(valuesToRemove))
                if len_intersect==1:
                    sc.values = tuple(sorted([ (1 if v in valuesToRemove else v)
                                              for v in sc.values]))
                elif len_intersect==2:
                    print 'Now how did that happen? We just filtered those out!'
                    return
                
                startNew = set(sc.startPointValues).difference(valuesToRemove)
                endNew = set(sc.endPointValues).difference(valuesToRemove)
                
                if len(startNew) < len(sc.startPointValues): startNew.add(1)
                if len(endNew) < len(sc.endPointValues):     endNew.add(1)
                
                sc.startPointValues = tuple(sorted(startNew))
                sc.endPointValues = tuple(sorted(endNew))
        
        '''
        # Now, go through and make a list of the junctions and the number of connections for each:
        junctions = [] # list of tuples; format: ( pt , [(scInd,True if startPt or False if endPt), ...] )
        for scInd,sc in enumerate(self.subContours):
            start,end = sc.points[0],sc.points[-1]
            jpts = [ pt for pt,connSCList in junctions ]
            tup = (scInd,True)
            if start in jpts:
                junctions[jpts.index(start)].append(tup)
            else:
                junctions.append( (start,[tup]) )
            jpts = [ pt for pt,connSCList in junctions ]
            tup = (scInd,False)
            if end in jpts:
                junctions[jpts.index(end)].append(tup)
            else:
                junctions.append( (end,[tup]) )
        singleJunctions = [ j for j in junctions if len(j[1])==1] # Shouldn't occur
        selfJunctions = [ j for j in junctions if len(j[1])==2 and j[1][0][0] == j[1][1][0]] # same scInd
        doubleJunctions = [ j for j in junctions if len(j[1])==2 and j[1][0][0] != j[1][1][0]] # diff scInd
        tripleJunctions = [ j for j in junctions if len(j[1])==3 ]
        quadJunctions = [ j for j in junctions if len(j[1])==4 ]
        
        doubleJunctionsByValuePair = {}
        for pt,scIndTupleList in doubleJunctions:
            scInds = [ scInd for scInd,startOrEnd in scIndTupleList ]
            
            if self.subContours[scInds[0]].values == self.subContours[scInds[1]].values:
                # this means the 2 sc's are between the same 2 values; should always be the case
                doubleJunctionsByValuePair[self.subContours[scInds[0]].values].append( scIndTupleList )
            else:
                print "Something went wrong; double junction between sc's with different values!!"
        
        for vals in doubleJunctionsByValuePair:
            djs_v = doubleJunctionsByValues[vals] # All the double junctions between pairs of values
                                                  # Aka, what should be internal points in a single subContour
            for scIndTuple0,scIndTuple1 in djs:
                pass # now somehow order the djs based on start and end points with the same scInd
        
        # update quad points from the quadJunctions instead??
        
        '''
        
        # Remove the values from contourOrderingByValue and allValues
        for v in valuesToRemove:
            del(self.contourOrderingByValue[v])
        
        self.allValues = [ v for v in self.allValues if v not in [1]+valuesToRemove ]
        
        # Clean up and we're done!
        self.CleanUpEmptySubContours()
    
    def RemoveSubContour(self,index,useSimpleRemoval=True,leaveTinyFlipFlopContour=False):
        '''This removes a subcontour by one of 3 methods:
        
        useSimple=True:   Take any connecting contours and shift the connecting points all to the midpoint of the contour to be deleted
        useSimple=False:                    THESE TWO ARE NOT IMPLEMENTED
            leaveTinyFlipFlopContour=False: A lot like simple, except that the contour will contribute multiple points to the connecting contours
            leaveTinyFlipFlopContour=True:  A lot like above except that 2 3-junctions are created instead of a 4-junction;
                                            the two parallel contours are connected at the midpoint by a very tiny contour instead
        
        In any of the 3 cases, the last step is to delete the old sc
        
        THIS OPERATES ON DATA IN-PLACE, so be careful!
        State: Leaves everything in a temporary dirty state
               The clean up phase (CleanUpEmptySubContours) can then happen efficiently all at once (after multiple sc removals)'''
        
        scDel = self.subContours[index]
        
        if useSimpleRemoval:
            # Find the center point of the problematic subcontour:
            #npD2 = scDel.numPoints//2
            scDelMidpoint = interpGen(scDel.points,scDel.numPoints*0.5)
                         #( scDel.points[npD2] if scDel.numPoints%2==1 else
                         #  shallowMul(shallowAdd(scDel.points[npD2-1],scDel.points[npD2]),0.5) )
            
            # Find all the subcountours that share triple junctions with the start and/or end points of scDel:
            connectedSCsToStart = [ (i,sc)
                                   for i,sc in enumerate(self.subContours)
                                   if sc!=scDel and sc!=None and ( scDel.points[0] in [sc.points[0],sc.points[-1]] ) ]
            connectedSCsToEnd = [ (i,sc)
                                 for i,sc in enumerate(self.subContours)
                                 if sc!=scDel and sc!=None and ( scDel.points[-1] in [sc.points[0],sc.points[-1]] ) ]
            #print len(connectedSCsToStart),len(connectedSCsToEnd)
            
            for scDelPtInd,connectedSCs in ((0,connectedSCsToStart),(-1,connectedSCsToEnd)):
                for i,s in connectedSCs:
                    connPtInd = ( 0 if s.points[0]==scDel.points[scDelPtInd] else -1 ) # it has to be either the start or the end of a sc
                    self.subContours[i].points[connPtInd] = scDelMidpoint
        else:
            print 'NOT IMPLEMENTED'
            if leaveTinyFlipFlopContour:
                pass
            else:
                pass
            
            return
            
            ########################################################
            ## Can always try something like this if the simple skip-the-contour solution doesn't work...
            ## This is the more complex, but more flexible way to do this:
            #import ImageContour.ImageContour
            #reload(ImageContour.ImageContour)
            #ImageContour.AdjustPointsAwayFromLine = ImageContour.ImageContour.AdjustPointsAwayFromLine
            #
            #cL, cR = ImageContour.AdjustPointsAwayFromLine(np.array(scDel.points),0.2,pinch=True,usePlot=True)
            #print ind,scDelPtInd.points
            #print cL
            #print cR
            #scTmp=SWHelpers.SubContour(points=cL)
            #scTmp=SWHelpers.SubContour(points=cR)
            #del scTmp
            ########################################################
            
        for v in sorted(set(scDel.startPointValues + scDel.endPointValues)): # Luckily, we only have to check values that were touching the deleted sc
            if v!=1: # as usual, skip the background...
                contourIndices = [ i for i,d in self.contourOrderingByValue[v] ]
                if index in contourIndices:
                    self.contourOrderingByValue[v] = [ (i,d) for i,d in self.contourOrderingByValue[v] if i!=index ]
        
        self.subContours[index] = None # This saves us from having to reindex contourOrderingByValue until later...
                                       # use CleanUpEmptySubContours to clean up
    
    def RemoveMultipleSubContours(self,indexList,useSimpleRemoval=True,leaveTinyFlipFlopContour=False):
        '''Remove a bunch of subcontours and then clean up after ourselves
           State: Changes the state of all variables (except possibly allValues)
                  Should leave everything in a clean state'''
        for i in sorted(set(indexList))[::-1]:
            self.RemoveSubContour(i,useSimpleRemoval,leaveTinyFlipFlopContour)
        self.CleanUpEmptySubContours()
    
    def scPlot(self,*args,**kwds):
        '''Plot the subContours in a way that can be overlaid on an imshow
           State: Access only'''
        for sc in self.subContours:
            _=sc.plot(*args,**kwds)
    
    def scPlotT(self,*args,**kwds):
        '''Plot the subContours in a way that can be overlaid on a transposed imshow
           (matches closely with diagramPlot in VFMLite)
           State: Access only'''
        for sc in self.subContours:
            _=sc.plotT(*args,**kwds)
    
    def cellPlot(self,*args,**kwds):
        '''Plot the full contours in a way that can be overlaid on an imshow
           State: Access only'''
        plotFunction = _kwdPop( kwds, 'plotFunction', plt.plot )
        reverseXY    = _kwdPop( kwds, 'reverseXY'   , False    )
        
        contourPoints = { v:self.GetContourPoints(v) for v in self.contourOrderingByValue.keys() }
        for v in contourPoints.keys():
            x = [ p[0]-0.5 for p in contourPoints[v] ]
            y = [ p[1]-0.5 for p in contourPoints[v] ]
            x,y = (y,x) if reverseXY else (x,y)
            _=plotFunction( y,x, *args, **kwds )
    
    def cellPlotT(self,*args,**kwds):
        '''Plot the full contours in a way that can be overlaid on a transposed imshow
           (matches closely with diagramPlot in VFMLite)
           State: Access only'''
        kwds['reverseXY'] = True
        return self.cellPlot(*args,**kwds)

def GetFlatDataFromCellNetwork(cn):
    '''Convert a CellNetwork with all its nested objects into flat datastructure
       (list,dict,tuple,etc).
       Details:
           Any non-builtin object that is made of only builtin elements
           (aka, no classes) can just be converted easily to a dictionary.
           So, to make a complex object into a dictionary , we just need to
           recursively turn all "contained" objects into dictionaries.'''
    # First, ensure that adjusted_length and values are a normal types and not numpy...
    for sc in cn.subContours:
        sc.adjusted_length = float(sc.adjusted_length)
        sc.values = [ int(v) for v in sc.values]
    
    return { 'subContours' : [ sc.__dict__ for sc in cn.subContours ],
             'quadPoints' : [ qp.__dict__ for qp in cn.quadPoints ],
             'contourOrderingByValue' : cn.contourOrderingByValue,
             'allValues' : cn.allValues, }

def GetCellNetworkFromFlatData(fdcn):
    '''Construct a CellNetwork from a flat data structure(fdcn) (aka dictionaries instead of objects)'''
    for qp in fdcn['quadPoints']:
        qp['values'] = tuple(qp['values']) # Force all QP's to be tuple
    
    return CellNetwork( **{ 'subContours' : [ SubContour(**j) for j in fdcn['subContours'] ],
                            'quadPoints' : [ QuadPoint(**j) for j in fdcn['quadPoints'] ],
                            'contourOrderingByValue' : { int(k):v for k,v in fdcn['contourOrderingByValue'].iteritems() },
                            'allValues': fdcn['allValues'], } )

def SubContourListfromCVLSList(cVLS_List,startPointValues_List=[],endPointValues_List=[]):
    '''Get a list of SubContour objects from an old list of cVLS's'''
    if startPointValues_List==[]:
        startPointValues_List = [[None,None,None] for c in cVLS_List]
    if endPointValues_List==[]:
        endPointValues_List = [[None,None,None] for c in cVLS_List]
    return [ SubContour(points = cvls[2],
                        # numPoints = len(cvls[2]), # happens automatically...
                        adjusted_length = cvls[1],
                        values = tuple(cvls[1]),
                        startPointValues = tuple(startPointValues_List[i]),
                        endPointValues = tuple(endPointValues_List[i]))
            for i,cvls in enumerate(cVLS_List)]

def GetCellNetwork( watershed2d,allValues=None,bgVals=(0,1),scale=1,offset=(0,0), ):
    '''Basically a constructor for CellNetwork based on a watershed array
       
       watershed2d is a watershed-like 2d array (indexed block values)
       allValues is optional and specifies which value in the array to include in the network
       bgVals are the values ignored as background (at least 0 should always be used)
       scale and offset define a non-rotational transform over the output points'''
    if allValues==None:
        allValues = np.unique(watershed2d)
    allValues = np.array(allValues).tolist() # force numpy arrays to lists
    allValues = [ v for v in allValues if v not in bgVals ] # skip the background
    
    identifier=0 # unique id for each subContour
    scList = []
    contourOrderingByValue = {} # For each cellID, an ordered list of index to the scList/direction pairs that reconstruct the full contour
    for v in allValues:
        boundingRect=ImageContour.GetBoundingRect(watershed2d,v)
        # No longer needed: #contour,turns,vals = ImageContour.GetContour(watershed[0],v,boundingRect=boundingRect,byNeighbor=True)
        perimeterVals,perimeterList,scPointsList = ImageContour.GetPerimeterByNeighborVal(watershed2d,v,boundingRect=boundingRect,getSubContours=True)
        numSCs=len(perimeterVals)
        scPointsListAdj = []
        for scp in scPointsList:
            scpAdj = np.array(scp)+[boundingRect[0][0],boundingRect[1][0]] # Will need to - 0.5 to line up on an overlay
            scpAdj = scpAdj*scale + offset
            scPointsListAdj.append(scpAdj.tolist())
        if len(perimeterList)>0:
            contourOrderingByValue[v] = []
            for i in range(numSCs):
                newSC = SubContour( points           = scPointsListAdj[i],
                                  # numPoints        = len(scPointsAdj[i]), # happens automatically
                                    adjusted_length  = perimeterList[i],
                                    values           = tuple(sorted([v,perimeterVals[i]])),
                                    startPointValues = GetValuesAroundSCPoint( watershed2d, scPointsListAdj[i][0] ),
                                    endPointValues   = GetValuesAroundSCPoint( watershed2d, scPointsListAdj[i][-1] ),
                                    identifier=identifier )
                matchingSCs = [ sc for sc in scList if sc.values==newSC.values ] # match any subcoutours in cVLS so far that are for the same pair of cells
                matchingSCs = [ sc for sc in matchingSCs if totuple(sc.points[::-1])==totuple(newSC.points) ] # Only keep subcoutours where the points match the reverse of the points in newSC
                                #sorted([newSC.points[0],newSC.points[-1]]) == sorted([sc.points[0],sc.points[-1]]) ] # Should only possibly find 1 match...
                if matchingSCs==[]: # This is a new subContour, not a duplicate!
                    scList.append(newSC)
                    contourOrderingByValue[v].append( (identifier,True) )
                    identifier+=1
                else:
                    matchingSCs[0].adjusted_length = min( matchingSCs[0].adjusted_length,
                                                          newSC.adjusted_length ) # keep the minimum perimeter length...
                    contourOrderingByValue[v].append( (matchingSCs[0].identifier,False) ) # False means the subcountour is backwards for this cell!
    scList.sort( key = lambda x: x.values ) # was just cVLS.sort()... this works, I hope?
    IDs = [sc.identifier for sc in scList]
    for sc in scList:      # scrub the id's, probably not necessary... 
        sc.identifier=None
    
    # Reindex after sorting...
    for v in allValues:
        contourOrderingByValue[v] = [ (IDs.index(i),d) for i,d in contourOrderingByValue[v] ]
    
    return CellNetwork( subContours=scList , contourOrderingByValue=contourOrderingByValue , allValues=allValues )

def GetCellNetworksByFrame(watershed,allValsByFrame,bgVals=(0,1)):
    '''Get a list of CellNetworks based on a watershed segmentation'''
    return [ GetCellNetwork(watershed[i],allValsByFrame[i],bgVals)
            for i in range(len(watershed)) ]

def GetXYListAndPolyListFromCellNetworkList(cellNetworkList,closeLoops=True):
    '''Get a multi-frame xyList and polyList'''
    ret = [ cn.GetXYListAndPolyList(closeLoops=closeLoops) for cn in cellNetworkList ]
    xyList,polyList = zip(*ret) # effectively like transpose...
    return xyList,polyList

def GetCVDListFromCellNetworkList(cellNetworkList):
    return [ cn.GetCellularVoronoiDiagram() for cn in cellNetworkList ]

def GetCellNetworkListWithLimitedPointsBetweenNodes(cellNetworkList,splitLength=1,fixedNumInteriorPoints=None,interpolate=True,checkForDegeneracy=True):
    '''Based on matching subcontours by value pair, this function defines a fixed number of interior points for each subcontour
       and then applies this "trimming" procedure equitably to each frame in cellNetworkList (uses LimitPointsBetweenNodes)'''
    #allValues = sorted(set( [ v for cn in cellNetworkList for v in cn.allValues ] )) # not used...
    allPairs = sorted(set( [ tuple(sc.values) for cn in cellNetworkList for sc in cn.subContours ] )) # Value pairs...
    
    # Build the numInteriorPointsDict:
    if fixedNumInteriorPoints!=None:
        numInteriorPointsDict = { p:fixedNumInteriorPoints for p in allPairs }
    else:
        # minLength is the number of points of the shortest subcountour between cells p[0] and p[1] from all frames
        minLength = { p : min( [ sc.numPoints
                                for cn in cellNetworkList
                                for sc in cn.subContours
                                if tuple(sc.values)==p ] )
                     for p in allPairs }
        numInteriorPointsDict = { p:(minLength[p]//splitLength) for p in allPairs }
    
    cellNetworkListNew = deepcopy(cellNetworkList) # otherwise, we'd also change the input argument in the outside world!
    for cn in cellNetworkListNew:
        cn.LimitPointsBetweenNodes(numInteriorPointsDict,interpolate=interpolate,checkForDegeneracy=False)
    
    if checkForDegeneracy:
        problemValsList,problemValuePairsList = ziptranspose([ cn.CheckForDegenerateContours() for cn in cellNetworkListNew ])
        if flatten(problemValsList)!=[]:
            print 'All degenerate values:',sorted(set(flatten(problemValsList)))
            print 'Degenerate values by frame:'
            for i,problemVals in enumerate(problemValsList):
                if problemVals!=[]:
                    print ' ',i,':',problemVals
            print 'Degenerate contours (zero area) found between these cellIDs on these frames!'
            for i,problemValuePairs in enumerate(problemValuePairsList):
                if problemValuePairs!=[]:
                    print ' ',i,':',problemValuePairs
            raise DegenerateNetworkException('Degeneracy check failed!')
    
    return cellNetworkListNew

def GetCellNetworkWithLimitedPointsBetweenNodes(cellNetwork,splitLength=1,fixedNumInteriorPoints=None,interpolate=True,checkForDegeneracy=True):
    '''Run GetCellNetworkListWithLimitedPointsBetweenNodes, but only for a single CellNetwork
       Useful if we don't want to limit points based on other frames
       To get a list of independent nets, just use a list comprehension
       (Makes a copy)'''
    
    # Pack and unpack the cn into/out of a list
    return GetCellNetworkListWithLimitedPointsBetweenNodes([cellNetwork],splitLength,fixedNumInteriorPoints,interpolate,checkForDegeneracy)[0]

def collectPreviousNextResults(prv,nxt):
    '''A generic function that joins elements from prv and nxt like so: [prv[0],...,prv[i]+nxt[i-1],...,nxt[-1]]
       prv and nxt are lists of lists and were mapped (with some function)
       from elements of an original list (length n) with this relation:
           prv[0 -> n-1]  maps to  original[0 -> n-1]
           nxt[0 -> n-1]  maps to  original[1 -> n]
       again, prv and nxt must be lists of lists'''
    return [prv[0]] + [prv[i]+nxt[i-1] for i in range(1,len(prv))] + [nxt[-1]]

def GetMatchedCellNetworkListsWithLimitedPointsBetweenNodes(cellNetworkListPrev,cellNetworkListNext,splitLength=1,fixedNumInteriorPoints=None,interpolate=True,checkForDegeneracy=True):
    '''Uses GetCellNetworkListWithLimitedPointsBetweenNodes to match each pair between cellNetworkListPrev and cellNetworkListNext'''
    cnListPrevNew = []
    cnListNextNew = []
    for i in range(len(cellNetworkListPrev)):
        cnl = GetCellNetworkListWithLimitedPointsBetweenNodes([cellNetworkListPrev[i],cellNetworkListNext[i]],splitLength,fixedNumInteriorPoints,interpolate,checkForDegeneracy=False)
        cnListPrevNew.append(cnl[0])
        cnListNextNew.append(cnl[1])
    
    if checkForDegeneracy:
        problemValsListPrev,problemValuePairsListPrev = ziptranspose([ cn.CheckForDegenerateContours() for cn in cnListPrevNew ])
        problemValsListNext,problemValuePairsListNext = ziptranspose([ cn.CheckForDegenerateContours() for cn in cnListNextNew ])
        problemValsList = [ sorted(set(i))
                           for i in collectPreviousNextResults(problemValsListPrev,problemValsListNext) ]
        problemValuePairsList = [ sorted(set(i))
                                 for i in collectPreviousNextResults(problemValuePairsListPrev,problemValuePairsListNext) ]
        
        if flatten(problemValsList)!=[]:
            print 'All degenerate values:',sorted(set(flatten(problemValsList)))
            print 'Degenerate values by frame:'
            for i,problemVals in enumerate(problemValsList):
                if problemVals!=[]:
                    print ' ',i,':',problemVals
            print 'Degenerate contours (zero area) found between these cellIDs on these frames!'
            for i,problemValuePairs in enumerate(problemValuePairsList):
                if problemValuePairs!=[]:
                    print ' ',i,':',problemValuePairs
            raise DegenerateNetworkException('Degeneracy check failed!')
    
    return cnListPrevNew,cnListNextNew

def GetXYListAndPolyListWithLimitedPointsBetweenNodes(cellNetworkList,splitLength=1,fixedNumInteriorPoints=None,interpolate=True):
    '''Get a list of points and a set of polygons network from a cellNetwork limit points between triple junctions
       (Applies GetCellNetworkListWithLimitedPointsBetweenNodes and then GetXYListAndPolyListFromCellNetworkList)'''
    return GetXYListAndPolyListFromCellNetworkList(
             GetCellNetworkListWithLimitedPointsBetweenNodes(cellNetworkList,splitLength,fixedNumInteriorPoints,interpolate) )

def FindMatchesAndRemovals(cnA,cnB):
    '''Take two CellNetworks and get 3 lists of indices for each:
       matched:        There is a direct analog in the other cellnet
       remove:         There is no direct analog in the other cellnet, but
                       collapsing these to 4-junctions will map to the other network
       notRecoverable: There is no analog in the other cellnet'''
    if not (cnA.__class__==cnB.__class__==CellNetwork):
        raise TypeError('cnA and cnB must be CellNetworks!')
    
    if cnA==cnB:
        raise ValueError('cnA and cnB must be different objects!')
    
    # Get a sorted list of indices into A and B sorted with this custom key function:
    def getKeyFun(cn):
        def retFun(x):
            sc = cn.subContours[x]
            return sc.values, sc.startPointValues, sc.endPointValues
        return retFun
    
    indexPoolA = sorted( range(len(cnA.subContours)), key = getKeyFun(cnA) )
    indexPoolB = sorted( range(len(cnB.subContours)), key = getKeyFun(cnB) )
    
    # Make dictionaries that take a values pair and return a set of indices into cnX.subcontours
    pairGroupsA = { k:list(g) for k,g in itertools.groupby( indexPoolA,
                                                            lambda x: cnA.subContours[x].values ) }
    pairGroupsB = { k:list(g) for k,g in itertools.groupby( indexPoolB,
                                                            lambda x: cnB.subContours[x].values ) }
    
    matchedA = []
    matchedB = []
    removeA = []
    removeB = []
    notRecoverableA = []
    notRecoverableB = []
    
    def intersectTest(a,b,nIntersections):
        scA,scB = cnA.subContours[a],cnB.subContours[b]
        startEndA = [scA.startPointValues,scA.endPointValues]
        startEndB = [scB.startPointValues,scB.endPointValues]
        return len( set(startEndA).intersection(startEndB) ) == nIntersections
    
    # First, find all the subContours from A and B that match between the same pair of values or flip-flopped values:
    for nIntersections,multiFailString in [ ( 2, 'with the same start/end point values:' ),
                                            ( 1, 'with one common endpoint:'             ),
                                            ( 0, 'without common start/endpoints:'       ),]:
        commonPairs = set(pairGroupsA.keys()).intersection(pairGroupsB.keys())
        for pair in commonPairs:
            # Look for any and all connections (i,j) between pairGroupsA[pair] and pairGroupsB[pair]
            # each pair here is a successful match between a subcontour in A and another in B
            matchInds = [ (i,j) for i,a in enumerate(pairGroupsA[pair])
                                for j,b in enumerate(pairGroupsB[pair])
                                if intersectTest(a,b,nIntersections) ]
            
            aInds,bInds = ziptranspose(matchInds) if len(matchInds)>0 else ([],[])
            aInds,bInds = sorted(set(aInds)),sorted(set(bInds))
            
            # for A and B, get matchX, the list of indices into cnX.subcontours
            matchA = [ pairGroupsA[pair][i] for i in aInds ]
            matchB = [ pairGroupsB[pair][i] for i in bInds ]
            
            # Once we've matched, take these out of the pool for checking
            for i in aInds[::-1]:
                del(pairGroupsA[pair][i])
            for i in bInds[::-1]:
                del(pairGroupsB[pair][i])
            
            # And then collect the matches into either matchedX, removeX, or notRecoverableX
            if len(matchInds)==1:
                matchedA += matchA
                matchedB += matchB
            elif len(matchInds)>1:
                print 'Not recoverable'
                print "sc's matches multiple sc's in opposing list " + multiFailString,pair,matchA,matchB
                # Print more details...
                print 'matchInds:',matchInds
                for cn,matc,ABstr in [ (cnA,matchA,'A'),
                                       (cnB,matchB,'B'), ]:
                    scs = [ [sc.points[0],sc.points[-1],sc.startPointValues,sc.endPointValues]
                           for i in matc
                           for sc in (cn.subContours[i],) ] # loop over 1-element list; basically local variable assignment
                    print 'Start/end coords/values for '+ABstr+':', scs
                
                scA,scB = cnA.subContours[a],cnB.subContours[b]
                notRecoverableA += matchA
                notRecoverableB += matchB
    
    def flipFlopFun(a,b,opStr):
        if opStr=='and':
            op = operator.and_
        elif opStr=='or':
            op = operator.or_
        scA,scB = cnA.subContours[a],cnB.subContours[b]
        return op( scA.values==scB.cornerConnections(),
                   scB.values==scA.cornerConnections() )
    
    # Next, check for flip-flopped values:
    for opStr,multiFailString in [ ( 'and', 'with flip-flopped values'      ),
                                   ( 'or',  'with flip-flopped values (OR)' ), ]:
        for pairA in pairGroupsA.keys():
            for pairB in pairGroupsB.keys():
                # Look for any and all connections (i,j) between pairGroupsA[pair] and pairGroupsB[pair]
                # each pair here is a successful match between a subcontour in A and another in B
                matchInds = [ (i,j) for i,a in enumerate(pairGroupsA[pairA])
                                    for j,b in enumerate(pairGroupsB[pairB])
                                    if flipFlopFun(a,b,opStr) ]
                
                aInds,bInds = ziptranspose(matchInds) if len(matchInds)>0 else ([],[])
                aInds,bInds = sorted(set(aInds)),sorted(set(bInds))
            
                # for A and B, get matchX, the list of indices into cnX.subcontours
                matchA = [ pairGroupsA[pairA][i] for i in aInds ]
                matchB = [ pairGroupsB[pairB][i] for i in bInds ]
                
                # Once we've matched, take these out of the pool for checking
                for i in aInds[::-1]:
                    del(pairGroupsA[pairA][i])
                for i in bInds[::-1]:
                    del(pairGroupsB[pairB][i])
                
                # And then collect the matches into either matchedX, removeX, or notRecoverableX
                if len(matchInds)==1:
                    removeA += matchA
                    removeB += matchB
                elif len(matchInds)>1:
                    print 'Not recoverable'
                    print "sc's matches multiple sc's in opposing list " + multiFailString,pair,matchA,matchB
                    notRecoverableA += matchA
                    notRecoverableB += matchB
    
    # Lastly, look for a subContours in A that collapsed to a 4-junction in B (and vice versa)
    # (This construct lets us loop instead of duplicating the code below)
    loopVars = [ ( pairGroupsA,cnA,cnB,matchedA,removeA,notRecoverableA,'A','B' ),
                 ( pairGroupsB,cnB,cnA,matchedB,removeB,notRecoverableB,'B','A' ), ]
    for pairGroupsX,cnX,cnO,matchedX,removeX,notRecoverableX,name,otherName in loopVars:
        for pair in pairGroupsX.keys():
            matchInds = [ i for i,x in enumerate(pairGroupsX[pair])
                            for q in cnO.quadPoints
                            if q.values == tuple(sorted(set(cnX.subContours[x].startPointValues +
                                                            cnX.subContours[x].endPointValues))) ]
            match = [ pairGroupsX[pair][i] for i in matchInds ]
            
            for i in sorted(matchInds,reverse=True):
                del(pairGroupsX[pair][i])
            
            # And then collect the matches into either matchedX, removeX, or notRecoverableX
            if len(matchInds)==1:
                removeX += match
            elif len(matchInds)>1:
                print 'Not recoverable'
                print "sc in "+name+" matches multiple 4-junctions in "+otherName+":",pair,match
                notRecoverableX += match
    
    # Anything that didn't get matched (and then deleted) get flagged as notRecoverable
    remainingPairsA = flatten(pairGroupsA.values())
    remainingPairsB = flatten(pairGroupsB.values())
    if len(remainingPairsA)+len(remainingPairsB)>0:
        print 'Not Recoverable! No matches found!'
        print 'A:', [ (i,cnA.subContours[i].values) for i in remainingPairsA ]
        print 'B:', [ (i,cnB.subContours[i].values) for i in remainingPairsB ]
        notRecoverableA += remainingPairsA
        notRecoverableB += remainingPairsB
    
    return matchedA,matchedB, removeA,removeB, notRecoverableA,notRecoverableB

def GetMatchedCellNetworksCollapsing(cnA,cnB):
    '''Make 2 simplified cell networks, making sure that there is a 1-to-1 mapping between all subcontours
       This function removes values that are not common to both networks and collapses
       pairs of subcontours that do not match but are in between the same 4 cells'''
    
    if cnA==cnB: # if we got the same object for some reason, just return 2 shallow clones
        return cnA,cnB
    
    # sharedVals = sorted(set(cnA.allValues+cnB.allValues)) # not used...
    valsNotInA = sorted(set(cnB.allValues).difference(cnA.allValues))
    valsNotInB = sorted(set(cnA.allValues).difference(cnB.allValues))
    
    # Delete any values that are not in both, replacing with background...
    cnA,cnB = deepcopy(cnA),deepcopy(cnB) # Make copies so we don't modify the originals
    cnA.RemoveValues(valsNotInB)
    cnB.RemoveValues(valsNotInA)
    
    matchedA,matchedB, removeA,removeB, notRecoverableA,notRecoverableB = FindMatchesAndRemovals(cnA,cnB)
    
    if len(notRecoverableA)+len(notRecoverableB) > 0:
        print 'Summary of pairs that are not recoverable from A:',notRecoverableA
        print 'Summary of pairs that are not recoverable from B:',notRecoverableB
    else:
        print 'All pairs matched.'
    
    cnA.RemoveMultipleSubContours(removeA)
    cnB.RemoveMultipleSubContours(removeB)
    
    return cnA,cnB,notRecoverableA,notRecoverableB

def GetMatchedCellNetworksCollapsingWithLimitedPoints(cnA,cnB,splitLength=1,fixedNumInteriorPoints=None,interpolate=True):
    '''Make 2 simplified cell networks, making sure that there is a 1-to-1 mapping between all points; this function collapses
       pairs of subcontours that do not match but are in between the same 4 cells'''
    
    cnANew,cnBNew,notRecoverableA,notRecoverableB = GetMatchedCellNetworksCollapsing(cnA,cnB)
    cnALim,cnBLim = GetCellNetworkListWithLimitedPointsBetweenNodes( [cnANew,cnBNew],splitLength,
                                                                     fixedNumInteriorPoints,interpolate)
    
    return cnALim,cnBLim

def _loadCNListStaticFromJsonFile(cnListStaticFile):
    with open(cnListStaticFile,'r') as fid:
        cnList = [ GetCellNetworkFromFlatData(i) for i in json.load(fid) ]
    return cnList

def GetCellNetworkListStatic( waterArr,d,extraRemoveValsByFrame=None,forceRemake=False,
                              bgVals=(0,1),scale=1,offset=(0,0), ):
    '''Get a CellNetwork list from a waterArr, ignoring any differences between frames.
       This function will optionally save and load to a pickle file (extraRemoveValsByFrame MUST be None)'''
    
    allValsByFrame = [ sorted( set(np.unique(i)).difference(bgVals) )
                      for i in waterArr ] # Skip background
    
    ### Ensure that this has enough elements, if not, add more empty lists
    if extraRemoveValsByFrame==None:
        extraRemoveValsByFrame = []
    
    # Ensure we got a list-of-lists for extraRemovalsByFrame
    assert all([ hasattr(vals,'__iter__') for vals in extraRemoveValsByFrame ])
    
    extraRemoveValsByFrame += [[] for i in range(len(waterArr)-len(extraRemoveValsByFrame))]
    
    
    cnListStaticFile = os.path.join(d,'cellNetworksListStatic.json') # Saved cellNetworks file
    #cnListStaticFile = os.path.join(d,'cellNetworksListStatic.pickle') # Saved cellNetworks file # old pickle version
    
    loadCnListFromFile = os.path.exists(cnListStaticFile) and not any(extraRemoveValsByFrame) and not forceRemake
    
    if loadCnListFromFile:
        print 'Reloading cnLists from file:',cnListStaticFile
        
        # Load cnList from JSON file
        cnList = _loadCNListStaticFromJsonFile(cnListStaticFile)
        #cnList = cPickle.load(open(cnListStaticFile,'r')) # old pickle version
        
        if len(cnList)!=len(waterArr):
            print 'cnList is the wrong length in the file:',cnListStaticFile
            print 'Will remake them'
            loadCnListFromFile=False
    if not loadCnListFromFile:
        ### The actual code that makes the cnList from scratch
        cnList = []
        for i in range(len(waterArr)):
            print 'Generating CellNetwork for frame: %i' % i
            water = np.array(waterArr[i])
            valsToKeep = sorted( set(allValsByFrame[i]).difference(extraRemoveValsByFrame[i]) )
            for v in sorted(extraRemoveValsByFrame[i]):
                water[np.where(water==v)] = bgVals[0]
            
            # next, create the CellNetwork and append it to the list:
            cnList.append( GetCellNetwork(water,valsToKeep,bgVals,scale,offset))
        
        # Only save this if we're using all the values; otherwise it gets confusing!
        if not any(extraRemoveValsByFrame):
            print 'Saving cnList to file:',cnListStaticFile
            # Save cnList to JSON file
            with open(cnListStaticFile,'w') as fid:
                json.dump( [ GetFlatDataFromCellNetwork(cn) for cn in cnList ], fid )
            #cPickle.dump(cnList,open(cnListStaticFile,'w')) # old pickle version
    
    return cnList
    
    
def GetCVDListStatic( waterArr,d,useStaticAnalysis,
                      extraRemoveValsByFrame=None,splitLength=20, fixedNumInteriorPoints=None,
                      usePlot=False,forceRemake=False,bgVals=(0,1),
                      scale=1,offset=(0,0),
                    ):
    '''Get cvdLists from a waterArr, ignoring any differences between frames. This function:
        * removes values in extraRemoveValsByFrame,
        * limits points between nodes
        * compresses cellID's to begin at 1 (excludes background)
        * returns a cvdList
       
       Throws error if static flag is not set; in this case,
        use GetMatchedCVDListPrevNext instead'''
    
    # Ensure we're doing a static analysis
    assert useStaticAnalysis, 'useStaticAnalysis is not set! Did you mean to use the function GetMatchedCVDListPrevNext?'
    
    cnList = GetCellNetworkListStatic( waterArr,d,extraRemoveValsByFrame,forceRemake,bgVals,scale,offset )
    
    # Run GetCellNetworkWithLimitedPointsBetweenNodes for each individual CellNetwork;
    #   we don't want GetCellNetworkListWithLimitedPointsBetweenNodes(cnList,splitLength,fixedNumInteriorPoints,interpolate=True)
    #   because it limits the points based on other frames which is not right for static runs
    
    cnListLim = [ GetCellNetworkWithLimitedPointsBetweenNodes(cn,splitLength,fixedNumInteriorPoints,interpolate=True)
                 for cn in cnList ]
    cvdList = GetCVDListFromCellNetworkList(cnListLim)
    
    if usePlot:
        print 'Plotting the subContours:'
        cnList[0].scPlotT('r-')
        cnListLim[0].scPlotT('ro-')
    
    return cvdList

def _loadCNListPrevNextFromJsonFile(cnListPrevAndNextFile):
    with open(cnListPrevAndNextFile,'r') as fid:
        jsdat = json.load(fid)
        cnListPrev = [ GetCellNetworkFromFlatData(i) for i in jsdat[0] ]
        cnListNext = [ GetCellNetworkFromFlatData(i) for i in jsdat[1] ]
    return cnListPrev,cnListNext


def GetMatchedCellNetworkListsPrevNext( waterArr,d,extraRemoveValsByFrame=None,forceRemake=False,
                              bgVals=(0,1),scale=1,offset=(0,0), ):
    '''Get matched before and after CellNetwork lists from a waterArr.
       This function will optionally save and load to a pickle file (extraRemoveValsByFrame MUST be None)'''

    allValsByFrame = [ sorted( set(np.unique(i)).difference(bgVals) )
                      for i in waterArr ] # Skip background
    
    # Ensure that this has enough elements, if not, add more empty lists
    if extraRemoveValsByFrame==None:
        extraRemoveValsByFrame = []
    
    # Ensure we got a list-of-lists for extraRemovalsByFrame
    assert all([ hasattr(vals,'__iter__') for vals in extraRemoveValsByFrame ])
    
    extraRemoveValsByFrame += [[] for i in range(len(waterArr)-len(extraRemoveValsByFrame))]
    
    cnListPrevAndNextFile = os.path.join(d,'cellNetworksListPrevAndNext.json') # Saved cellNetworks file
    #cnListPrevAndNextFile = os.path.join(d,'cellNetworksListPrevAndNext.pickle') # Saved cellNetworks file # old pickle version
    
    loadCnListFromFile = os.path.exists(cnListPrevAndNextFile) and not any(extraRemoveValsByFrame) and not forceRemake
    
    if loadCnListFromFile:
        print 'Reloading cnLists from file:',cnListPrevAndNextFile
        
        # Load cnListPrev and cnListNext from JSON file
        cnListPrev,cnListNext = _loadCNListPrevNextFromJsonFile(cnListPrevAndNextFile)
        #cnListPrev,cnListNext = cPickle.load(open(cnListPrevAndNextFile,'r')) # old pickle version
        
        if len(cnListPrev)!=len(waterArr)-1:
            print 'cnLists are the wrong length in the file:',cnListPrevAndNextFile
            print 'Will remake them'
            loadCnListFromFile=False
    
    allMatched = True
    
    if not loadCnListFromFile:
        cnListPrev = []
        cnListNext = []
        badFramePairs = []
        
        for i in range(len(waterArr)-1):
            print 'Matching frames %i and %i' % (i,i+1)
            # create matched arrays first:
            waterA,waterB = np.array(waterArr[i]) , np.array(waterArr[i+1])
            extraRemovalsAB = set(extraRemoveValsByFrame[i]).union(extraRemoveValsByFrame[i+1])
            commonVals = sorted( set(allValsByFrame[i]).intersection(allValsByFrame[i+1]).difference(extraRemovalsAB) )
            valsRemoveA = set(allValsByFrame[i]).difference(commonVals)
            valsRemoveB = set(allValsByFrame[i+1]).difference(commonVals)
            for v in sorted(valsRemoveA):
                waterA[np.where(waterA==v)] = 1
            for v in sorted(valsRemoveB):
                waterB[np.where(waterB==v)] = 1
            
            # next, create matched CellNetworks:
            cnA = GetCellNetwork(waterA,commonVals,bgVals,scale,offset)
            cnB = GetCellNetwork(waterB,commonVals,bgVals,scale,offset)
            
            cnA,cnB,notRecoverableA,notRecoverableB = GetMatchedCellNetworksCollapsing(cnA,cnB)
            cnListPrev.append(cnA)
            cnListNext.append(cnB)
            if len(notRecoverableA)+len(notRecoverableB) > 0:
                allMatched=False
                badFramePairs.append( (i,i+1) )
        if not allMatched:
            print 'Matching Errors! Will save cnLists to file, but these frames did not match:'
            print badFramePairs
        if not any(extraRemoveValsByFrame):
            # Only save this if we're using all the values; otherwise it gets confusing!
            print 'Saving cnLists to file:',cnListPrevAndNextFile
            
            # Save cnListPrev and cnListNext to JSON file
            with open(cnListPrevAndNextFile,'w') as fid:
                json.dump( [ [ GetFlatDataFromCellNetwork(cn)
                              for cn in cnl ]
                            for cnl in (cnListPrev,cnListNext) ], fid )
            #cPickle.dump([cnListPrev,cnListNext],open(cnListPrevAndNextFile,'w')) # old pickle version
    
    return cnListPrev,cnListNext,allMatched

def GetMatchedCVDListPrevNext( waterArr,d,useStaticAnalysis,
                               extraRemoveValsByFrame=None,splitLength=20, fixedNumInteriorPoints=None,
                               usePlot=False,forceRemake=False,bgVals=(0,1),
                               scale=1,offset=(0,0),
                             ):
    '''Get matched before and after cvdLists from a waterArr. This function:
        * removes values in extraRemoveValsByFrame,
        * matches subContours between CellNetworks
            - since this is the slowest step, these 2 matched networks are saved to a file in d and reloaded if this is run again
              (with the caveat that save/load will not occur if extraRemoveValsByFrame are specified)
              load can be overridden with forceRemake
        * limits points between nodes
        * compresses cellID's to begin at 1 (excludes background)
        * returns 2 cvdLists
       
       Throws error if static flag is set; in this case,
        use GetCVDListStatic instead'''
    
    # Ensure we're doing a dynamic analysis (if you just want to get rid of viscous effects, set viscosityTimeStepRatio to 0)
    assert not useStaticAnalysis, 'useStaticAnalysis is set! Did you mean to use the function GetCVDListStatic?'
    
    cnListPrev,cnListNext,allMatched = GetMatchedCellNetworkListsPrevNext( waterArr,d,extraRemoveValsByFrame,forceRemake,bgVals,scale,offset)
    
    cnListPrevLim,cnListNextLim = GetMatchedCellNetworkListsWithLimitedPointsBetweenNodes(cnListPrev,cnListNext,splitLength,fixedNumInteriorPoints,interpolate=True)
    
    cvdListPrev = GetCVDListFromCellNetworkList(cnListPrevLim)
    cvdListNext = GetCVDListFromCellNetworkList(cnListNextLim)
    
    if usePlot:
        print 'Plotting the subContours:'
        cnListPrev[0].scPlotT('r-')
        cnListNext[0].scPlotT('b-')
        cnListPrevLim[0].scPlotT('ro-')
        cnListNextLim[0].scPlotT('bo-')
    
    return cvdListPrev,cvdListNext

def SaveXYListAndPolyListToMMAFormat(xyList,polyList,filename,bumpIndsUp1=True,removeLastPoint=True):
    '''xyList: nested list of xy pairs for each time point.
       polyList: nested list of dictionaries for each time point where
                 each entry is like: {cellID: [ <indexes into xyList> ]}
       Exports a MMA compatible dataStructure also called "polyList" which looks like lists of:
           {xyList,{{k,listOfIndicesTopointXYList}...}}
           where listOfIndicesTopointXYList is of course 1-indexed'''
    
    outputStr='polyList = {'
    for t,polyDict in enumerate(polyList):
        outputStr+='\n{\n'
        outputStr+=repr(xyList[t]).replace('[','{').replace(']','}').replace('(','{').replace(')','}')
        outputStr+=',\n{'
        for k in sorted(polyDict.keys()):
            inds = polyDict[k]
            if bumpIndsUp1:
                inds = [i+1 for i in inds]
            if removeLastPoint:
                inds=inds[:-1]
            outputStr+='{'+str(k)+','+repr(inds).replace('[','{').replace(']','}').replace('(','{').replace(')','}')+'}, '
        outputStr=outputStr[:-2]
        outputStr+='}\n},'
    outputStr=outputStr[:-1]+'\n}'
    open(filename,'w').write(outputStr)

def SaveCellNetworkListToMMAFormat(cellNetworkList,filename,bumpIndsUp1=True,removeLastPoint=True):
    '''Save a cellNetwork to the MMA format
       (basically just GetXYListAndPolyListFromCellNetworkList followed by SaveXYListAndPolyListToMMAFormat)'''
    xyList,polyList = GetXYListAndPolyListFromCellNetworkList(cellNetworkList)
    SaveXYListAndPolyListToMMAFormat(xyList,polyList,filename,bumpIndsUp1,removeLastPoint)


def ContourPlotFromImage(im,neighborPairs,colors=['b','g','r','c','m','y','k']):
    '''Plot an array as a grayscale image (im)
       and then plot the sub contours from an array (im) based on a set of pixel diffs
       Needs a precomputed set of neighbor pairs, but works WITHOUT ever using ImageContour
       Very useful for plotting specific contours an inspecing them (adjust neighborPairs)'''
    from ValueReceived import imshow_vr # external
    
    if len(colors)<len(neighborPairs): # Make sure there are enough colors!
        lenC = len(colors)
        for i in range(lenC,len(neighborPairs)):
            colors.append( colors[i%lenC] )
    
    _=imshow_vr(im,interpolation='nearest',cmap=plt.cm.gray)
    for i,nPair in enumerate(neighborPairs):
        whX = np.where(  ((im[:-1,:]==nPair[0]) & (im[1:,:]==nPair[1])) |
                         ((im[:-1,:]==nPair[1]) & (im[1:,:]==nPair[0]))  )
        whY = np.where(  ((im[:,:-1]==nPair[0]) & (im[:,1:]==nPair[1])) |
                         ((im[:,:-1]==nPair[1]) & (im[:,1:]==nPair[0]))  )
        for j in range(len(whX[0])):
            x,y = whX[1][j]-0.5 , whX[0][j]+0.5
            _=plt.plot([x,x+1],[y,y],colors[i],linewidth=2)
        for j in range(len(whY[0])):
            x,y = whY[1][j]+0.5 , whY[0][j]-0.5
            _=plt.plot([x,x],[y,y+1],colors[i],linewidth=2)

def ContourPlotFromCVLS(cVLSByFrame,frame=0):
    '''Plot a cVLS'''
    for cvls in cVLSByFrame[frame]:
        cvls=np.array(cvls[2])
        _=plt.plot( cvls[:,0], cvls[:,1] )


#####################################################################
## New section for array-based contour generation                  ##
## Lacks some of the finesse of the ImageContour's crawling method ##
#####################################################################

def getDiffArr(arr,axis,wrap=False):
    '''Create an array that delineates pixel-pixel boundaries along an axis'''
    diffArr = addBorder(np.diff(arr,axis=axis),1,axis=axis)!=0
    if wrap:
        pass
        if axis==0: diffArr[-1] = 0
        else:       diffArr[:,-1] = 0
    return diffArr

def getWhWhp(arr,axis,wrap=False):
    '''Get the start and end points of the boundary lines
       between all differing pixels along an axis
       (as np.where results)'''
    lineArr = getDiffArr(arr,axis,wrap)
    wh = np.where(lineArr)
    p1 = ( (0,1) if axis==0 else (1,0) )
    whp = (wh[0]+p1[0],wh[1]+p1[1])
    return wh,whp

def GetDividingLinesFromArray(arr,axis,wrap=False,whwhp=None):
    '''Get all the lines perpendicular to axis that divide differing pixels
       (as an array of [[x1,y1],[x2,y2]] values)'''
    wh,whp = ( whwhp if whwhp else getWhWhp(arr,axis,wrap) )
    edges = np.transpose([wh,whp],(2,0,1))
    return edges

def GetValuePairsFromArray(arr,axis,wrap=False,whwhp=None):
    '''Given an array and an axis, find all pairs of adjacent points
       Optionally takes a whwhp parameter (generated by getWhWhp) to ensure that
       values generated by this function correlate to edges generated by GetDividingLinesFromArray'''
    wh,whp = ( whwhp if whwhp else getWhWhp(arr,axis,wrap) )
    def s(x,y): return ((x,y) if axis==0 else (y,x)) # Flip arguments when axis==1
    sA,s1A,s1m1 = slice(None),slice(1,None),slice(1,-1) # shorthand for slices [:], [1:], and [1:-1]
    
    arrB = addBorder(arr) # add 1 pixel border to each side along axis
    if wrap:  # wrap the sides around
        arrB[s(0,s1m1)]  = arr[s(-1,sA)]
        arrB[s(-1,s1m1)] = arr[s(0,sA)]
    startValue  = arrB[s(sA,s1A)][wh]
    endValue = arrB[s(s1A,sA)][whp]
    return np.transpose([startValue,endValue]) # valuePairs

def GetDividingLinesAndValuePairsFromArray(arr,axis,wrap=False):
    '''Return the set of lines (in [[x1,y1],[x2,y2]] format) that divide pixels
       of different value as well as a list of all the actual pixel values'''
    whwhp = getWhWhp(arr,axis,wrap)
    edgesAxis      = GetDividingLinesFromArray(arr,axis,wrap,whwhp)
    valuePairsAxis = GetValuePairsFromArray(arr,axis,wrap,whwhp)
    return edgesAxis,valuePairsAxis

def GetEdgesAndValuePairsFromArray(arr,wrapX=False,wrapY=False):
    '''Just like GetDividingLinesAndValuePairsFromArray but applies
       it over both x and y axes'''
    evpXY = [ GetDividingLinesAndValuePairsFromArray(arr,axis,wrap)
             for axis,wrap in [(0,wrapX),(1,wrapY)] ]
    edges,valuePairs = map( np.concatenate,ziptranspose(evpXY) )
    return edges,valuePairs

def GetEdgeGroups(edges,valuePairs,eliminateSameCellBoundaries=True):
    '''Get a list of lines between each cell pair
       
       edges and valuePairs are generated by GetEdgesAndValuePairsFromArray
       
       The "eliminateSameCellBoundaries" option eliminates lines where the values
       on either side are acutally the same; this should only occur at wrapped edges.'''
    edgeGroups = groupByFunction( zip(edges.tolist(),np.sort(valuePairs).tolist()),
                                  compose(tuple,itemgetter(1)), itemgetter(0) )
    return ( { k:v for k,v in edgeGroups.iteritems() if k[0]!=k[1] }
             if eliminateSameCellBoundaries else
             edgeGroups )

def GetAmbiguousCrossingPoints(arrayWithBorder):
    '''Find all spots in an array where an ambiguous intersection occurs,
       namely a point surrrounded by a checkerboard pattern, i.e.:
       ****
       *AB*
       *BA*
       ****
       '''
    a = arrayWithBorder
    return np.transpose(np.where(np.all([ a[:-1,:-1] == a[1:,1:],
                                          a[:-1,1:]  == a[1:,:-1],
                                          a[:-1,:-1] != a[1:,:-1],
                                        ], axis=0 )))

def GetContourOrderingByValue(cellIDs,subContourValues,subContourPoints):
    pointIndicesByCellID = { cellID:[ i for i,values in enumerate(subContourValues)
                                        if cellID in values ]
                            for cellID in cellIDs }
    # Not used:
    ##edgeGroupsByCellID = { cellID: [ subContourPoints[i] for i in pointIndicesByCellID[cellID] ]
    ##                      for cellID in cellIDs }
    contourOrderingByValue = {}
    for cellID in cellIDs:
        scInds = pointIndicesByCellID[cellID]
        scToEndPointConnections = [ ( i, subContourPoints[i][j] )
                                   for i in scInds
                                   for j in (0,-1) ]
        conns = getElementConnections(scToEndPointConnections)
        chainsOut = getChainsFromConnections(conns)[0]
        chainsOut = ( chainsOut[:-1] if chainsOut[0]==chainsOut[-1] else
                      chainsOut )
        chainsOut = ( roll(chainsOut) if hasattr(chainsOut[0],'__iter__') else
                      chainsOut )
        scOrder, scEndPts = ziptranspose(partition(chainsOut,2))
        scDirection = [ ( subContourPoints[sco][-1]==scEndPts[i] ) # test for head to tail connections
                       for i,sco in enumerate(scOrder) ]
        chainDirection = polyCirculationDirection(scEndPts)
        if chainDirection==-1:
            scOrder = scOrder[::-1]
            scDirection = [ (not i) for i in scDirection ][::-1]
        contourOrderingByValue[cellID] = zip(scOrder,scDirection)
    return contourOrderingByValue


def GetCellNetwork_NEW( watershed2d,allValues=None,bgVals=(0,1),scale=1,offset=(0,0),
                        wrapX=False, collapsePoles=False, deleteZeroContours=True ):
    '''Basically a constructor for CellNetwork based on a watershed array
       
       This function gives an equivalent output to GetCellNetwork, but uses an
           array-based methodology to get the main data and then uses a
           chaining scheme to construct subContours and contour ordering
           (the original method uses a crawling scheme instead, which
            obviated the need for chaining)
       This method is ~70% slower but has more available options
       
       watershed2d is a watershed-like 2d array (indexed block values)
       allValues is optional and specifies which value in the array to include in the network
       bgVals are the values ignored as background (at least 0 should always be used)
       scale and offset define a non-rotational transform over the output points
       
       Generally, these default options should be used for the other 3 options:
          wrapX=False, collapsePoles=False, deleteZeroContours=True
       
       Other useful modes are:
         * include boundary mode:
             wrapX=False, collapsePoles=False, deleteZeroContours=False
         * cylindrical mode (include outer boundary and wrap around in the x-dimension):
             wrapX=True, collapsePoles=False, deleteZeroContours=False
         * shperical mode (wrap x and collapse y):
             wrapX=True, collapsePoles=True, deleteZeroContours=True
       '''
    arr = watershed2d # synonym
    cellIDs = deletecases( ( np.unique(arr)
                             if allValues==None else         # rebuild allValues if None is passed
                             np.array(allValues) ).tolist(), # force numpy array to list
                           bgVals )                          # remove background values
    
    # Get Edges
    edges,valuePairs = GetEdgesAndValuePairsFromArray(arr,wrapX=wrapX,wrapY=False)
    
    # (convert points in e to wrapped coordinates)
    if wrapX:
        wh=np.where(edges[:,:,0]==arr.shape[0])
        edges[wh[0],wh[1],0] = 0
        edges[wh[0]] = edges[wh[0]][:,::-1] # flip so the 0 comes first (aka sorted properly...)
    
    #if wrapY:
    #    # Is this right yet?
    #    wh=np.where(e[:,:,1]==arr.shape[1])
    #    edges[wh[0],wh[1],1] = 0
    #    edges[wh[0]] = edges[wh[0]][:,::-1] # flip so the 0 comes first (aka sorted properly...)

    # Ambiguous point testing step
    ambiguousCrossingPoints = GetAmbiguousCrossingPoints(arr)
    assert len(ambiguousCrossingPoints)==0, 'Ambiguous crossing points at: '+str(ambiguousCrossingPoints.tolist())

    # SC step: Group edges by value pair and then form chains that divide each cell pair
    edgeGroups = GetEdgeGroups(edges,valuePairs,eliminateSameCellBoundaries=True)
    cellBoundariesMulti = { k:getChainsFromConnections( getElementConnections(totuple(v)) )
                            for k,v in edgeGroups.items() }
    assert all( len(i)==1 for i in cellBoundariesMulti.values() ),'Some cells touch in multiple places'
    cellBoundaries = { k:v[0] for k,v in cellBoundariesMulti.iteritems() } # We should be able to deal with one chain per cell pair
    scPtsByPairID = cellBoundaries
    if collapsePoles:
        # Eliminate all edges on the poles:
        # 1. Replace points (0,x)->(0,0) and (last,x)->(last,0)
        # 2. Merge any contiguous polar points together using removeAdjacentDuplicates
        _collapsePoles = lambda pts: [ ( (0,0)            if p[1]==0 else
                                         (0,arr.shape[1]) if p[1]==arr.shape[1] else
                                         p )
                                      for p in pts ]
        scPtsByPairID = { k : removeAdjacentDuplicates(_collapsePoles(pts))
                         for k,pts in scPtsByPairID.iteritems() }
    if deleteZeroContours:
        scPtsByPairID = { k:pts for k,pts in scPtsByPairID.iteritems()
                                if 0 not in k }

    subContourValues,subContourPoints = ziptranspose(scPtsByPairID.items())
    contourOrderingByValue = GetContourOrderingByValue(cellIDs,subContourValues,subContourPoints)
    subContours = [ SubContour( points=points,
                                values=values,
                                startPointValues = GetValuesAroundSCPoint( arr, points[0], wrapX=True ),
                                endPointValues   = GetValuesAroundSCPoint( arr, points[-1], wrapX=True ) )
                   for values,points in zip(subContourValues,subContourPoints) ]
    return CellNetwork( subContours=subContours,
                        contourOrderingByValue=contourOrderingByValue,
                        allValues=cellIDs )
