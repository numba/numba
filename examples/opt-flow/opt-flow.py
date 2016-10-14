#=
Copyright (c) 2015, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, 
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
THE POSSIBILITY OF SUCH DAMAGE.
=#

from numba import jit
import numpy as np
import struct
import sys
import time
import pdb

#import scipy.sparse
#import scipy.sparse.linalg

@jit(nopython=True)
def blockJacobiPreconditioner(Ix, Iy, r, lam):
  a = Ix*Ix + 4*lam
  b = Ix*Iy
  c = Iy*Iy + 4*lam
  invdet = 1.0/(a*c - b*b)
  res = np.zeros(r.shape, dtype=np.float32)
  res[:,:,0] = (c*r[:,:,0] - b*r[:,:,1])*invdet
  res[:,:,1] = (a*r[:,:,1] - b*r[:,:,0])*invdet
  return res

#@jit(nopython=True)
#def mvm(Ix, Iy, x, lam):
#  height, width = Ix.shape
#  y = np.zeros(x.shape, dtype=np.float32)
#  y[:,:,0] = Ix*( Ix*x[:,:,0] + Iy*x[:,:,1]) + 4*lam*x[:,:,0] 
#  y[:,:,1] = Iy*( Ix*x[:,:,0] + Iy*x[:,:,1]) + 4*lam*x[:,:,1] 
# 
#  y[:,:,0] += -lam * (np.vstack((np.zeros((1,width), dtype=np.float32), x[:-1,:,0]))  \
#                    + np.hstack((np.zeros((height,1), dtype=np.float32), x[:,:-1,0])) \
#                    + np.vstack((x[1:,:,0], np.zeros((1,width), dtype=np.float32))) \
#                    + np.hstack((x[:,1:,0], np.zeros((height,1), dtype=np.float32))))
#  y[:,:,1] += -lam * (np.vstack((np.zeros((1,width), dtype=np.float32), x[:-1,:,1]))  \
#                    + np.hstack((np.zeros((height,1), dtype=np.float32), x[:,:-1,1])) \
#                    + np.vstack((x[1:,:,1], np.zeros((1,width), dtype=np.float32))) \
#                    + np.hstack((x[:,1:,1], np.zeros((height,1), dtype=np.float32))))
#  return y

@jit(nopython=True)
def mvm(Ix, Iy, x, lam):
  height, width = Ix.shape
  y = np.empty(x.shape, dtype=np.float32)
  for i in range(height):
    for j in range(width):
      ix = Ix[i,j]
      iy = Iy[i,j]
      pu00 = x[i,j,0]
      pv00 = x[i,j,1]
      pul0 = pur0 = pvl0 = pvr0 = 0
      pu0t = pu0d = pv0t = pv0d = 0
      if i > 0:
        pu0t = x[i-1,j,0]
        pv0t = x[i-1,j,1]
      if i < height - 1:
        pu0d = x[i+1,j,0]
        pv0d = x[i+1,j,1]
      if j > 0:
        pul0 = x[i,j-1,0]
        pvl0 = x[i,j-1,1]
      if j < width - 1:
        pur0 = x[i,j+1,0]
        pvr0 = x[i,j+1,1]
      y[i,j,0] = ix * (ix*pu00 + iy*pv00) + lam*(4.0*pu00-(pul0+pur0+pu0t+pu0d))
      y[i,j,1] = iy * (ix*pu00 + iy*pv00) + lam*(4.0*pv00-(pvl0+pvr0+pv0t+pv0d))
  return y


@jit(nopython=True)
def single_scale_optical_flow(im1_data, im2_data, lam, num_iterations):
  height, width = im1_data.shape
  #Ix, Iy = np.gradient(im1_data)
  Ix = np.zeros(im1_data.shape, dtype=np.float32)
  Iy = np.zeros(im1_data.shape, dtype=np.float32)
  for x in range(width):
    for y in range(height):
      if (x > 1 and x < width-2):
        Ix[y,x] = ( -im1_data[y,x+2] + -8*im1_data[y,x+1] + 8 * im1_data[y,x-1] + im1_data[y,x-2]) / 12
      if (y > 1 and y < height-2):
        Iy[y,x] = ( -im1_data[y+2,x] + -8*im1_data[y+1,x] + 8 * im1_data[y-1,x] + im1_data[y-2,x]) / 12
  #print("Ix=",np.sum(Ix)," Iy=",np.sum(Iy))
  It = im1_data - im2_data

  r = np.zeros((height, width, 2), dtype=np.float32)
  x = np.zeros((height, width, 2), dtype=np.float32)
  r[:,:,0] = -Ix*It;  
  r[:,:,1] = -Iy*It;  
  #print("ru=",np.sum(r[:,:,0]),"rv=",np.sum(r[:,:,1]))
  z = blockJacobiPreconditioner(Ix, Iy, r, lam)
  p = z
  rsold = np.sum(r * z)
  #print("zu=",np.sum(z[:,:,0]), " zv=",np.sum(z[:,:,1]), " rsold=", rsold)
  for iter in range(num_iterations):
    Ap = mvm(Ix, Iy, p, lam)
    pTAp = np.sum(p * Ap)
    alpha = rsold/pTAp
    x = x + alpha*p
    r = r - alpha*Ap
    z = blockJacobiPreconditioner(Ix, Iy, r, lam)
    rsnew = np.sum(r * z)
    beta = rsnew/rsold
    p = z + beta*p
    rsold = rsnew

  return x

@jit(nopython=True)
def interp(im, x, y):
  xx = int(np.floor(x))
  yy = int(np.floor(y))
  alpha = x - xx
  beta = y - yy
  result = 0.0
  if alpha > 0:
    if beta > 0:
      result = (1.0-beta)*( alpha*im[yy,xx+1] + (1.0-alpha)*im[yy,xx] )  \
         + beta*( alpha*im[yy+1,xx+1] + (1.0-alpha)*im[yy+1,xx] )
    else:
      result = (1.0-beta)* im[yy,xx] + beta* im[yy+1,xx]
  else:
    if beta > 0:
      result =  alpha*im[yy,xx+1] + (1.0-alpha)*im[yy,xx]  
    else:
      result = im[yy,xx]  
  return result

@jit(nopython=True)
def warp_motion(im, flow, other_im):
  height, width = im.shape
  warpim = np.zeros(im.shape, dtype=np.float32)
  for y in range(height):
    for x in range(width):
      u = x + flow[y,x,0]
      v = y + flow[y,x,1]
      if (u < 0 or u > width-1 or v < 0 or v > height-1):
        warpim[y,x] = other_im[y,x]
      else:
        warpim[y,x] = interp(im, u, v)

  return warpim

@jit(nopython=True)
def interpolate_flow(flow, newwidth, newheight):
  height, width = flow.shape[0:2]  
  if (newwidth == width and newheight==height):
    return flow
  else:
    newflow = np.zeros((newheight, newwidth, 2), dtype=np.float32)
    sx = np.float32(width-1) / np.float32(newwidth)
    sy = np.float32(height-1) / np.float32(newheight)
    for y in range(newheight):
      for x in range(newwidth):
        u = np.float32(x)*sx
        v = np.float32(y)*sy
        newflow[y,x,0] = interp(flow[:,:,0], u, v)*sx
        newflow[y,x,1] = interp(flow[:,:,1], u, v)*sx
    return newflow
     
@jit(nopython=True)
def downsample(im, newwidth, newheight):
  height, width = im.shape  
  if (newwidth == width and newheight==height):
    return im

  else:
    factorX = np.float32(width)/np.float32(newwidth)
    factorY = np.float32(height)/np.float32(newheight)
    newim = np.zeros((newheight, newwidth), dtype=np.float32)
    for y in range(newheight):
      for x in range(newwidth):
        result = np.float32(0.0)
        fi = np.float32(y)*factorY
        fLastI = np.float32(y+1)*factorY
        LastI = int(np.floor(fLastI))
        fj = np.float32(x)*factorX
        fLastJ = np.float32(x+1)*factorX
        LastJ = int(np.floor(fLastJ))
        for i in range(int(np.floor(fi)),1+min(LastI,height-1)):
          for j in range(int(np.floor(fj)),1+min(LastJ,width-1)):
            coeff = np.float32(1.0)
            if (fi-i > 0):
              coeff = (1-fi+i)
            if (fj-j > 0):
              coeff *= (1-fj+j)
            if (i == LastI):
              coeff *= (fLastI-LastI)
            if (j == LastJ):
              coeff *= (fLastJ-LastJ)
            result += coeff*im[i,j]
        newim[y,x] = result/(factorX*factorY)

    return newim
    
     
@jit(nopython=True)
def multi_scale_optical_flow(im1_data, im2_data, lam, num_iterations, nscales):
  if (nscales == 1):
    return single_scale_optical_flow(im1_data, im2_data, lam, num_iterations)

  else:
    height, width = im1_data.shape
    scale = np.float32(pow( width/50.0, -1.0/nscales))
    #print("scale=",scale)

    u = np.zeros((height, width, 2), dtype=np.float32)
    for i in range(nscales, -1, -1):
      #print("i=",i," w=",width*pow(scale,i))
      owidth = int(np.floor(width*pow(scale, i)))
      oheight = int(np.floor(height*pow(scale, i)))
      #print(owidth, oheight)

      small_im1 = downsample(im1_data, owidth, oheight)
      small_im2 = downsample(im2_data, owidth, oheight)

      if ( i < nscales):
        u = interpolate_flow(u, owidth, oheight)
        #print("u=",np.sum(u[:,:,0])," v=",np.sum(u[:,:,1]))
        warped_image = warp_motion(small_im2, u, small_im1)
      else:
        u = np.zeros((oheight, owidth, 2), dtype=np.float32)
        warped_image = small_im2

      if (i == 0):
        num_iterations *= 5

      #print("si1=", np.sum(small_im1), "wi=", np.sum(warped_image))
      du = single_scale_optical_flow(small_im1, warped_image, lam, num_iterations)
      #print("du=", np.sum(du[:,:,0]), "dv=", np.sum(du[:,:,1]))
      u = u + du

      iwidth = owidth
      iheight = oheight

    return u

def write_flo(u, fname):
  height, width = u[:,:,0].shape
  flo = open(fname, 'wb')
  #flo.seek(4)
  #flo.write(struct.pack("i", 100))
  flo.write('PIEH'.encode())
  #width = struct.unpack("i", flo.read(4))[0]
  #height = struct.unpack("i", flo.read(4))[0]
  flo.write(struct.pack("i", width))
  flo.write(struct.pack("i", height))
  #print(width, height)
  for yy in range(height):
    for xx in range(width):
      flo.write(struct.pack("f", u[yy,xx,0]))
      flo.write(struct.pack("f", u[yy,xx,1]))
  f.close()

if (__name__ == "__main__"):

  if (len(sys.argv) < 3):
    print("usage: "+sys.argv[0]+ " <first_image_name.dat> <second_image_name.dat>")
    exit(-1)

  f = open(sys.argv[1], "rb")
  width = struct.unpack("i", f.read(4))[0]
  height = struct.unpack("i", f.read(4))[0]
  print(width, height)
  im1 = np.zeros((height, width), np.float32)
  im2 = np.zeros((height, width), np.float32)
  
  for i in range(height):
    for j in range(width):
      c = struct.unpack("B", f.read(1))[0]
      im1[i,j] = np.float32(c)/255.0
  f.close()

  f = open(sys.argv[2], "rb")
  struct.unpack("i", f.read(4))[0]
  struct.unpack("i", f.read(4))[0]
  for i in range(height):
    for j in range(width):
      c = struct.unpack("B", f.read(1))[0]
      im2[i,j] = np.float32(c)/255.0
  f.close()

  lam = np.float32(0.025)
  #u = single_scale_optical_flow(im1, im2, lam, 100)

  start = time.clock()
  u = multi_scale_optical_flow(im1, im2, lam, 1, 1)
  end = time.clock()
  elapsed = end-start
  print("SELFPRIMED ", elapsed)
  start = time.clock()
  u = multi_scale_optical_flow(im1, im2, lam, 100, 44)
  end = time.clock()
  elapsed = end-start
  print("SELFTIMED ", elapsed)
  print("checksum: ", np.sum(u[:,:,0]), " ",np.sum(u[:,:,1]))
  write_flo(u, "out.flo")
