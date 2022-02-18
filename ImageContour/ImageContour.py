from __future__ import print_function

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt


def GetBoundingRect(arr, val):
    """Get the minimum rectangle bounding val in the array arr
    returns [minx,maxx],[miny,maxy]"""
    x, y = np.where(arr == val)
    return [min(x), max(x)], [min(y), max(y)]


# the four directions:
# right -> [0,1]
# left -> [0,-1]
# up -> [-1,0]
# down -> [1,0]
class directions(object):
    right = [0, 1]
    left = [0, -1]
    up = [-1, 0]
    down = [1, 0]


def turnL(d):
    if d == directions.right:
        return directions.up
    elif d == directions.left:
        return directions.down
    elif d == directions.up:
        return directions.left
    elif d == directions.down:
        return directions.right


def turnR(d):
    if d == directions.right:
        return directions.down
    elif d == directions.left:
        return directions.up
    elif d == directions.up:
        return directions.right
    elif d == directions.down:
        return directions.left


def addP(p1, p2):
    return [p1[0] + p2[0], p1[1] + p2[1]]


def indA(arr, p):
    if 0 <= p[0] < arr.shape[0] and 0 <= p[1] < arr.shape[1]:
        return arr[p[0], p[1]]
    else:
        return 0


def GetContourFromSubArray(subArr, realArr=None, offsets=None, getSubsections=False):
    """Take a boolean sub-array that has been tightly cropped around a value
      and get the contour.
    The cropping ensures that there is a value of 1 somewhere along
      each edge of the array.
    Uses a custom turn-tracking algorithm to trace the boundary.

    If getSubsections is True, the tracking also stores the values
      just outside each pixel-pixel boundary.

    Returns contour,turns (and optionally contourVals if getSubsections is True)

    <more explanation needed>"""
    # Find the first nonzero value on row 0 and set the contour to the right
    for i, p in enumerate(subArr[0]):
        if p == 1:
            direction = directions.right
            contour = [[0, i], addP([0, i], direction)]
            if getSubsections:
                contourVal = []
                lastContourVal = indA(realArr, [offsets[0], i - 1 + offsets[1]])
            break
    # By definition there has to be a turn by the first point
    turns = [1]  # 1 if turn, 0 if straight
    perimeter = 1
    while contour[-1] != contour[0]:  # Stop when we get back around to the beginning
        left = turnL(direction)
        straight = direction
        right = turnR(direction)

        # A contour line is a connection between two contour points
        # A contour point is in between four pixels
        #  with y,x as the point coordinates, these four pixels are:
        NW = addP(contour[-1], [-1, -1])
        NE = addP(contour[-1], [-1, 0])
        SW = addP(contour[-1], [0, -1])
        SE = addP(contour[-1], [0, 0])

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
        if direction == directions.right:
            zerozy, onezy, R, L = NW, SW, SE, NE
        elif direction == directions.up:
            zerozy, onezy, R, L = SW, SE, NE, NW
        elif direction == directions.left:
            zerozy, onezy, R, L = SE, NE, NW, SW
        elif direction == directions.down:
            zerozy, onezy, R, L = NE, NW, SW, SE

        if indA(subArr, zerozy) != 0 and indA(subArr, onezy) != 1:
            print("contour should be 0 on the outside and 1 on the inside!!")
            return

        if getSubsections:
            contourVal += [
                indA(realArr, [zerozy[0] + offsets[0], zerozy[1] + offsets[1]])
            ]

        # if L==1:
        #     Turn Left
        # elif R==1:
        #     Go Straight
        # else:
        #     Turn Right
        if indA(subArr, L):
            direction = left
            turns += [1]
        elif indA(subArr, R):
            direction = straight
            turns += [0]
        else:  # otherwise go right... #indA(subArr,sr):
            direction = right
            turns += [1]
        contour += [addP(contour[-1], direction)]
        perimeter += 1
        if perimeter > 1000000:
            print("Error! Perimeter Should not be this large!!!")
            return

    if getSubsections:
        contourVal += [lastContourVal]
        return contour, turns, contourVal
    else:
        return contour, turns


def GetContour(arr, val, boundingRect=None, byNeighbor=False):
    """Build a contour around a set of values in an array
    returns contour,turns,  if byNeighbor is True,
    otherwise returns contour,turns"""
    if boundingRect is None:
        b = GetBoundingRect(arr, val)
    else:
        b = boundingRect
    subArr = arr[b[0][0] : b[0][1] + 1, b[1][0] : b[1][1] + 1]
    if byNeighbor:
        # return 3 parameters
        return GetContourFromSubArray(
            subArr == val, realArr=arr, offsets=[b[0][0], b[1][0]], getSubsections=True
        )
    else:
        # return 2 parameters
        return GetContourFromSubArray(subArr == val)


def GetContourInPlace(arr, val):
    b = GetBoundingRect(arr, val)
    subArr = arr[b[0][0] : b[0][1] + 1, b[1][0] : b[1][1] + 1]
    contour = GetContourFromSubArray(subArr == val)[0]
    return np.array(contour) + [b[0][0], b[1][0]]


def GetCornersSub(cornerList, useWrap=True):
    count = [0]
    for i in cornerList:
        if i == 0:
            if count[-1] > 0:
                count += [0]
        else:
            count[-1] += 1
    if count[-1] > 0:
        if useWrap:
            count[0] += count[-1]
        else:
            count += [0]  # make sure to just delete the 0...
    del count[-1]
    cornersSub = sum([(i + 1) // 2 for i in count])
    return cornersSub


def GetIJPerimeter(arr, val, boundingRect=None):
    contour, turns = GetContour(arr, val, boundingRect=boundingRect)
    perimeter = len(contour) - 1  # Because first and last point are the same
    cornersSubtract = GetCornersSub(turns)
    # print perimeter,cornersSubtract
    return perimeter - cornersSubtract * (2 - sqrt(2))


def GetPerimeterByNeighborVal(arr, val, boundingRect=None, getSubContours=False):
    contour, turns, vals = GetContour(
        arr, val, boundingRect=boundingRect, byNeighbor=True
    )
    oldVal = None
    perimeterList = []
    turnsList = []
    perimeterVals = []
    subContours = []
    for i in range(len(turns)):
        if vals[i] == oldVal:
            perimeterList[-1] += 1
            turnsList[-1].append(turns[i])
            subContours[-1].append(contour[i + 1])
        else:
            perimeterList.append(1)  # length 1 to start...
            turnsList.append([turns[i]])
            subContours.append([contour[i], contour[i + 1]])
            perimeterVals.append(vals[i])
            oldVal = vals[i]

    if perimeterVals[0] == perimeterVals[-1]:
        perimeterList[0] += perimeterList[-1]
        del perimeterList[-1]
        del perimeterVals[-1]
        turnsList[0] = turnsList[-1] + turnsList[0]
        subContours[0] = subContours[-1][:-1] + subContours[0]
        del turnsList[-1]
        del subContours[-1]

    for i in range(len(perimeterList)):
        turnsList[i][
            0
        ] = 0  # Insist that the endpoints for each segment NOT be considered corners...
        turnsList[i][
            -1
        ] = 0  # This WILL change the overall length of the total perimeter,
        # so Watch Out!!!
        cornersSubtract = GetCornersSub(turnsList[i], useWrap=False)
        perimeterList[i] -= cornersSubtract * (2 - sqrt(2))

    if getSubContours:
        return perimeterVals, perimeterList, subContours
    else:
        return perimeterVals, perimeterList


# Highlighting the differences in two similar countour turnLists generated by GPBNV (above)...
# Basically, GetWoundNeighborContourLengths just pick whichever yeilds the shorter value... that is more accurate in comparison to the perimeter...
# t1 = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
# t2 = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
# print t1[1:]
# print t2[::-1][:-1]
# len(t1)
# len(t2)

# Pick out one contour from a byNeighbor GetContour
def GetBoundaryLine(arr, v1, v2):
    [xmin, xmax], [ymin, ymax] = GetBoundingRect(arr, v1)
    contours, turns, contourVals = GetContour(arr, v1, byNeighbor=True)
    for i in range(len(contours)):
        # for j in range(len(contours[i])):
        contours[i][0] += xmin
        contours[i][1] += ymin

    contourValOld = None
    c2 = []
    c2ind = []
    c2v = []
    for i, v in enumerate(contourVals):
        if v != contourValOld:
            start = i
            c2ind.append([start, start])
            c2v.append([v1, contourVals[i]])
            contourValOld = contourVals[i]
        c2ind[-1][-1] += 1

    joinLast = False
    if len(c2v) > 1:
        if c2v[0][1] == v2 and c2v[0][1] == c2v[-1][1]:
            joinLast = True

    for i in range(len(c2ind)):
        if v1 in c2v[i] and v2 in c2v[i]:
            c2.append(contours[c2ind[i][0] : c2ind[i][1] + 1])

    if joinLast:
        c2[0] = c2[-1][:-1] + c2[0]
        del c2[-1]

    return c2


def AdjustPointsAwayFromLine(points, d, pinch, usePlot=False):
    xy = points
    x, y = xy.T
    dxy = xy[1:] - xy[:-1]
    dx, dy = dxy.T
    l = np.sqrt(dx ** 2 + dy ** 2)

    # Some control points for calculating intersections...
    xyls = np.array(
        [x[:-1] - dy / l * d, y[:-1] + dx / l * d]
    ).T  # Set of points to the left of the starting point of each segment
    xyle = np.array(
        [x[1:] - dy / l * d, y[1:] + dx / l * d]
    ).T  # Set of points to the left of the ending point of each segment
    xyrs = np.array(
        [x[:-1] + dy / l * d, y[:-1] - dx / l * d]
    ).T  # Set of points to the right of the starting point of each segment
    xyre = np.array(
        [x[1:] + dy / l * d, y[1:] - dx / l * d]
    ).T  # Set of points to the right of the ending point of each segment

    xls, yls = xyls.T
    xle, yle = xyle.T
    xrs, yrs = xyrs.T
    xre, yre = xyre.T

    # Simply uses all points, interleaved -- no good: stuff crosses too much
    # xyl_inter = np.array([xyls.T,xyle.T]).transpose(2,0,1).reshape([len(xyls)*2,2]).T
    # xyr_inter = np.array([xyrs.T,xyre.T]).transpose(2,0,1).reshape([len(xyrs)*2,2]).T

    # Picks the central point between the interior start and end points -- no good, points could easily overlap and set off the "xy match" problem in converting for VFM
    # xyl_avg = (np.append(xyls,xyle[-1:],0)+np.append(xyls[:1],xyle,0))/2
    # xyr_avg = (np.append(xyrs,xyre[-1:],0)+np.append(xyrs[:1],xyre,0))/2

    # Newer solution is to calculate the intersection of lines xyls[i]->xyle[i] and xyls[i+1]->xyle[i+1] (See math below)
    def CalculateCornerPointsFromStartAndEndPoints(starts, ends):
        """Takes 2 sets of points, starts and ends and computes the intersections between
        all adjecent pairs of lines:  starts[i]->ends[i] intersecting starts[i+1]->ends[i+1]
        Then, it takes returns a new set of points: [starts[0], corners..., ends[-1]]"""
        # Does some "magic" linear algebra that works out to the solution of the intersections:
        # This lives inside AdjustPointsAwayFromLine just to cut down on namespace pollution; it's really only useful in here anyway...
        deltas = ends - starts
        deltaJoin = np.array([deltas[:-1], deltas[1:]]).transpose(
            1, 2, 0
        )  # places the first axis at the end

        startDeltaCross = np.cross((starts[:-1], starts[1:]), (deltas[:-1], deltas[1:]))
        deltaCross = np.cross(deltas[1:], deltas[:-1])

        startDeltaCrossRepeat = np.repeat(
            startDeltaCross.T[:, None], 2, axis=1
        )  # repeat the array twice
        deltaCrossRepeat = np.repeat(
            deltaCross[:, None], 2, axis=1
        )  # make it the same size as the other arrays

        corners = np.cross(startDeltaCrossRepeat, deltaJoin) / deltaCrossRepeat

        newPoints = starts[:1].tolist() + corners.tolist() + ends[-1:].tolist()
        return newPoints

    # The older clunkier (and possibly clearer :P ) way to do it...
    # mult = np.zeros([len(diffs)-1,2])
    # corners = np.zeros([len(diffs)-1,2])
    # for i in range(len(starts)-1):
    #    mult[i] = np.cross((starts[i],starts[i+1]),(diffs[i],diffs[i+1]))
    #    corners[i] = np.cross(mult[i],np.array([diffs[i],diffs[i+1]]).T) / denom[i]
    newPointsL = CalculateCornerPointsFromStartAndEndPoints(xyls, xyle)
    newPointsR = CalculateCornerPointsFromStartAndEndPoints(xyrs, xyre)

    if pinch:
        numPts = len(x)
        if (
            numPts % 2 == 0
        ):  # even number of pts; pinch point is at the midpoint of a segment
            pinchPoint = (xy[numPts // 2 - 1] + xy[numPts // 2]) / 2
            newPointsL.insert(numPts // 2, pinchPoint)
            newPointsR.insert(numPts // 2, pinchPoint)
        else:  # odd number of pts; pinch point is at an existing node
            pinchPoint = xy[numPts // 2]
            newPointsL[numPts // 2] = pinchPoint
            newPointsR[numPts // 2] = pinchPoint

    if usePlot:
        for i in range(len(dx)):
            plt.arrow(
                x[i],
                y[i],
                dx[i],
                dy[i],
                "k-",
                length_includes_head=True,
                head_width=0.05,
                facecolor="k",
                edgecolor="k",
            )

        def point_plot(arr, *args, **kwds):
            arr = np.array(arr)
            return plt.plot(arr[:, 0], arr[:, 1], *args, **kwds)

        point_plot(xyls, "ro")
        point_plot(xyle, "rx")
        point_plot(xyrs, "go")
        point_plot(xyre, "gx")
        point_plot(newPointsL, "mo-")
        point_plot(newPointsR, "bo-")
        # dplot(xyp_avg,'b-')
        # dplot(xym_avg,'y-')
        plt.axis("equal")

    return newPointsL, newPointsR


def PlotArrayAndContour(arr, val):
    [[xmin, xmax], [ymin, ymax]] = GetBoundingRect(arr, val)
    contour = GetContour(arr, val)[0]
    npc = np.array(contour)
    plt.cla()
    plt.imshow(arr, cmap=plt.cm.jet, interpolation="nearest")
    plt.plot(npc[:, 1] - 0.5 + ymin, npc[:, 0] - 0.5 + xmin, "yo-")
    plt.xlim(-1, arr.shape[1])
    plt.ylim(arr.shape[0], -1)


if __name__ == "__main__":
    z = np.zeros([10, 10])
    z[3, 3:5] = 1
    z[4, 2:6] = 1
    z[5, 2:6] = 1
    b = GetBoundingRect(z, 1)
    z2 = z[b[0][0] : b[0][1] + 1, b[1][0] : b[1][1] + 1]

    q = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]])
    print(GetIJPerimeter(z, 1))
    print(GetIJPerimeter(q, 1))
    plt.figure(1)
    PlotArrayAndContour(z, 1)
    plt.figure(2)
    PlotArrayAndContour(q, 1)
    import DelayApp
