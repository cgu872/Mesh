# -*- coding:utf-8 -*-
# @data 2025/3/17
# @file MeshDeal.py

import time
import sys,re,os
import numpy as np
from itertools import islice
from numba import jit
import pandas as pd
import csv,json,math

# @jit(nopython=True)
if __name__ == '__main__':
    startTime = time.time()
    # workdir="C:\\Users\\ASUS\\Desktop\\airfoil"
    workdir = "C:\\Users\\cgu872\\Desktop\\airfoil"
    os.chdir(workdir)

    #1.-------------------------------------------------------#
    # read [pointdata,facedata,ownerdata,neighbourdata,boundarydata,
    #       pointN,faceN,InnerfaceN,BoundaryfaceN,
    #       elementN,elementBN,BoundaryTypeN]
    # -------------------------------------------------------#
    meshdir=os.path.join(workdir,"constant\\polyMesh")
    if not os.path.exists(meshdir):
        sys.exit("There is no OpenFOAM mesh.")
    else:
        pointfile=os.path.join(meshdir,"points")
        pointdata=[]
        with open(pointfile, "r", encoding="utf-8") as file:
            line_19 = next(islice(file, 18, 19), None)
            pointN=int(line_19)
            for line in file:
                match = re.findall(r"\(.*\)", line)
                if match:
                    pointdata.append(np.fromstring(match[0].replace("(", "").replace(")", ""), sep=" "))
        if not pointN==len(pointdata):
            sys.exit("Fail to read points file .")

        facefile=os.path.join(meshdir,"faces")
        facedata=[]
        with open(facefile, "r", encoding="utf-8") as file:
            line_19 = next(islice(file, 18, 19), None)
            faceN=int(line_19)          #number of the total faces
            for line in file:
                match = re.findall(r"\d\(.*\)", line)
                if match:
                    facedata.append(np.fromstring(re.search(r'\((.*?)\)', match[0]).group(1), sep=" ", dtype=int))
        if not faceN==len(facedata):
            sys.exit("Fail to read faces file .")

        ownerfile = os.path.join(meshdir, "owner")
        with open(ownerfile, "r", encoding="utf-8") as file:
            lines = file.readlines()
        faceNN=int(lines[19])           #Verify again
        if not faceN==faceNN:
            sys.exit("Fail to read owner file .")
        ownerdata=list(map(int, lines[21:21+faceN]))

        neighbourfile = os.path.join(meshdir, "neighbour")
        with open(neighbourfile, "r", encoding="utf-8") as file:
            lines = file.readlines()
        InnerfaceN=int(lines[19])       #number of the inner faces
        BoundaryfaceN=faceN-InnerfaceN  #number of the boundary faces
        neighbourdata=list(map(int, lines[21:21+InnerfaceN]))
        elementN=len(set(ownerdata)) #number of the elements or np.max(ownerdata)+1 because the numpy comes from 0
        elementBN = BoundaryfaceN        #number of the elements including boundary faces

        boundaryfile=os.path.join(meshdir,"boundary")
        boundarydata= []
        with open(boundaryfile, "r", encoding="utf-8") as file:
            line_18 = next(islice(file, 17, 18), None)
            BoundaryTypeN=int(line_18)
            content = file.read()
        pattern = re.compile(r'(\w+)\s*\{([^}]*)\}', re.DOTALL)
        matches = pattern.findall(content)
        for match in matches:
            bound_name = match[0]
            bound_content = match[1].strip()
            bound_dict = {}
            # 解析块内容
            for line in bound_content.splitlines():
                key, value = line.split()
                bound_dict[key] = value
            bound_dict['name'] = bound_name
            boundarydata.append(bound_dict)

    # 2.-------------------------------------------------------#
    # rewrite data based on cell index
    # elementFaces，elementNeighbours
    # -------------------------------------------------------#
    elementNeighbours=[[] for _ in range(elementN)]
    elementFaces=[[] for _ in range(elementN)]
    for iface in range(0,InnerfaceN):
        own=ownerdata[iface]
        nei=neighbourdata[iface]
        elementNeighbours[own].append(nei) #the face index owned by cell
        elementNeighbours[nei].append(own) #the cell index near cell
        elementFaces[own].append(iface)    #比如根据cell编号找周围的面构成[facedata[i] for i in elementFaces[0]]
        elementFaces[nei].append(iface)
    for iface in range(InnerfaceN, faceN):
        own = ownerdata[iface]
        elementFaces[own].append(iface)   #boundary faces in cell
    '''cell 包含的point
    elementNodes = [[] for _ in range(elementN)]
    for icell in range(0,elementN):
        [facedata[i] for i in elementFaces[icell]]
    upperAnbCoeffIndex=
    '''

    # 3.-------------------------------------------------------#
    # calculate face Centroids,face normal vector, Areas, Cell Centroids,Volumes
    # faceCentroids, faceSf, faceAreas, elementCentroids, elementVolumes
    # faceWeights, faceCF, faceCf, faceFf
    # -------------------------------------------------------#
    faceCentroids,faceSf,faceAreas=[],[],[]
    for iface in range(0, faceN):
        centroid = np.zeros(3)
        Sf = np.zeros(3)#surface normal vector
        area = 0
        NodeIndex=facedata[iface]
        local_centre = np.zeros(3) #rough face centroid
        for iNode in NodeIndex:
            local_centre = local_centre + pointdata[iNode]
        local_centre=local_centre/len(NodeIndex)
        line=[pointdata[iTriangle]-local_centre for iTriangle in NodeIndex]
        line.append(line[0])
        point=[pointdata[iTriangle] for iTriangle in NodeIndex]
        point.append(point[0])
        local_Sf = [0.5 * np.cross(line[iline], line[iline+1]) for iline in range(0,len(NodeIndex))]
        local_centroid=[(local_centre+point[iline]+point[iline+1])/3 for iline in range(0,len(NodeIndex))]
        Sf = np.sum(local_Sf,0)
        area=np.linalg.norm(Sf, ord=2) #Euclidean norm
        centroid=[np.linalg.norm(local_Sf[iTriangle], ord=2)*local_centroid[iTriangle] for iTriangle in range(0,len(NodeIndex))]
        centroid=np.sum(centroid,0)/area
        faceCentroids.append(centroid)
        faceSf.append(Sf)
        faceAreas.append(area)
    elementCentroids, elementVolumes = [], []
    for icell in range(0, elementN):
        FaceIndex = elementFaces[icell]
        local_centre=np.average([faceCentroids[i] for i in FaceIndex], 0) #rough cell centroid
        Cf=[faceCentroids[i] - local_centre for i in FaceIndex]
        local_Sf=[faceSf[i] if icell == ownerdata[i] else -faceSf[i] for i in FaceIndex]
        localVolume = [np.dot(local_Sf[i],Cf[i])/3 for i in range(0, len(FaceIndex))]
        totalVolume = np.sum(localVolume,0)
        localCentroid = [0.75 * faceCentroids[i] + 0.25 * local_centre for i in FaceIndex]
        realCentroids=np.sum([localCentroid[i]*localVolume[i] for i in range(0, len(FaceIndex))],0)/totalVolume
        elementVolumes.append(totalVolume)
        elementCentroids.append(realCentroids)
    faceCF, faceCf, faceFf, faceWeights = [],[],[],[]
    for iface in range(0, InnerfaceN):
        n=faceSf[iface]/np.linalg.norm(faceSf[iface], ord=2)
        own=ownerdata[iface]
        nei=neighbourdata[iface]
        faceCF.append(elementCentroids[nei]-elementCentroids[own])
        faceCf.append(faceCentroids[iface] - elementCentroids[own])
        faceFf.append(faceCentroids[iface] - elementCentroids[nei])
        faceWeights.append(np.dot(faceCf[iface],n)/(np.dot(faceCf[iface],n) - np.dot(faceFf[iface],n)))
    for iface in range(InnerfaceN, BoundaryfaceN):
        n = faceSf[iface] / np.linalg.norm(faceSf[iface], ord=2)
        own = ownerdata[iface]
        faceCF.append(faceCentroids[iface]-elementCentroids[own]) #no F in the boundary
        faceCf.append(faceCentroids[iface] - elementCentroids[own])
        faceWeights.append(1.0)
    a=1+1

    # jsonOutput="C:\\Users\\cgu872\\Desktop\\123.json"
    # with open(jsonOutput, "w") as f:
    #     json.dump(boundarydata, f)
    # numbers = [float(num) for num in numbers]
    # os.remove(tifOutput2)
    # del demdata
    endTime = time.time()
    costTime = (endTime - startTime) / 60.0
    print("Reading mesh cost %.3f minutes" % costTime)