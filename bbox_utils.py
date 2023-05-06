import torch


import numpy as np
import math
from scipy.spatial import ConvexHull
from numpy import *

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou

# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d



def calculate_iou(box1,box2):
    int_min = np.maximum(np.min(box1, axis=0), np.min(box2, axis=0))
    int_max = np.minimum(np.max(box1, axis=0), np.max(box2, axis=0))
    int_size = np.maximum(int_max - int_min, 0)

    # Calculate intersection volume and union volume
    int_vol = int_size[0] * int_size[1] * int_size[2]
    vol1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    vol2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    union_vol = vol1 + vol2 - int_vol

    # Calculate IoU
    iou = int_vol / union_vol

    return iou

def get_intersection_area(bbox_0, bbox_1):
    """
    compute the intersection area of 2 bbox
    :param bbox_0:
    :param bbox_1:
    :return: interArea
    """
    x_min = max(bbox_0[0, 0], bbox_1[0, 0])
    y_min = max(bbox_0[0, 1], bbox_1[0, 1])
    z_min = max(bbox_0[0, 2], bbox_1[0, 2])

    x_max = min(bbox_0[1, 0], bbox_1[1, 0])
    y_max = min(bbox_0[1, 1], bbox_1[1, 1])
    z_max = min(bbox_0[1, 2], bbox_1[1, 2])

    interArea = abs(
        max((x_max - x_min, 0)) * max((y_max - y_min), 0) * max(((z_max - z_min), 0))
    )

    return interArea

def separate_boxes(box1, box2, iou_threshold=0.3):
    """
    Separate two colliding 3D bounding boxes represented by numpy arrays of shape (8,3),
    using the given IoU threshold and returning the separated boxes.
    """
    movement = 0.02
    iou= box3d_iou(box1, box2)
    for i in range(2):#while iou>iou_threshold:
        
        tmpx1 = box1-[movement,0,0]
        tmpx2 = box2+[movement,0,0]
        tmpy1 = box1-[0,movement,0]
        tmpy2 = box2+[0,movement,0]
        x_iou = box3d_iou(tmpx1,tmpx2)
        y_iou = box3d_iou(tmpy1,tmpy2)
        if x_iou<iou and x_iou<y_iou:
            box1 = tmpx1 
            box2 = tmpx2 
            iou = x_iou
        elif y_iou<iou and y_iou<x_iou:
            box1 = tmpy1 
            box2 = tmpy2 
            iou = y_iou

    

        
    return box1,box2

def get_object_info_from_bounding_box(bounding_box):
    """
    Calculate the object location, orientation, and size given the 8 corner bounding box.
    """
    # Calculate the center of the bounding box
    center = np.mean(bounding_box, axis=0)

    # Calculate the front vector and the right vector based on the first two points of the bounding box
    front_vector = bounding_box[1] - bounding_box[0]
    right_vector = bounding_box[3] - bounding_box[0]

    # Calculate the orientation of the object
    #orientation = np.arctan2(front_vector[1], front_vector[0])

    # Calculate the size of the object
    size = np.array([
        np.linalg.norm(right_vector),  # Length
        np.linalg.norm(front_vector),  # Width
        np.abs(bounding_box[0][2] - bounding_box[4][2])  # Height
    ])

    return center,size

def detect_collisions(objects):
    collisions = []

    for i in range(objects.shape[0]):
        for j in range(i+1, objects.shape[0]):
            box1 = objects[i]
            box2 = objects[j]

            # Calculate IoU
            iou_score = box3d_iou(box1, box2)

            # Check if objects are colliding
            if iou_score > 0:
                #print(iou_score)
                collisions.append((i, j))

    return collisions


# corners_3d_ground  = get_3d_box((1.497255,1.644981, 3.628938), -1.531692, (2.882992 ,1.698800 ,20.785644)) 
# corners_3d_predict = get_3d_box((1.458242, 1.604773, 3.707947), -1.549553, (2.756923, 1.661275, 20.943280 ))


# print(get_object_info_from_bounding_box(corners_3d_ground))

# IOU_3d=box3d_iou(corners_3d_predict,corners_3d_ground)
# print (IOU_3d) #3d IoU/ 2d IoU of BEV(bird eye's view)

# corners1,corners2 = separate_boxes(corners_3d_predict,corners_3d_ground)
# IOU_3d=box3d_iou(corners1,corners2)
# print (IOU_3d)
# print(get_object_info_from_bounding_box(corners1))

