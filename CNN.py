import numpy as np
import h5py
import matplotlib.pyplot as  plt
#from skimage import data#
plt.rcParams['figure.figsize']=(5.0,4.0)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'

def zero_pad(X,pad):
    X_pad=np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values=(0))
    return X_pad
np.random.seed(1)
x=np.random.randn(4,3,3,2)
x_pad=zero_pad(x,2)
print("x_shape=",x.shape)
print("x_pad.shape=",x_pad.size)
print("x[1,1]=",x[1,1])
print("x_pad[1,1]=",x_pad[1,1])

fig,axarr=plt.subplots(1,2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])
plt.show()#这一句就可以把需要的给画出来了,因为ipython里有专有的magic函数。

def conv_sigle_step(a_slice_prev,w,b):
    s=a_slice_prev*w
    z=np.sum(s)
    z=z+b
    return z
np.random.seed(1)
a_slice_prev=np.random.randn(4,4,3)#前置
w=np.random.randn(4,4,3)#参数  a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    #W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    #b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
b=np.random.randn(1,1,1)#参数
z=conv_sigle_step(a_slice_prev,w,b)
print("Z=",z)#这个z是卷积的结果
def conv_forward(A_prev,w,b,hparameters):#前置过程
    (m,n_H_prev,n_W_prev,n_C_prev)=np.shape(A_prev)
    (f,f,n_C_prev,n_C)=np.shape(w)
    stride=hparameters["stride"]
    pad=hparameters["pad"]
    n_H=int((n_H_prev-f+2*pad)/stride+1)#卷积操作，上课学过的。
    n_W=int((n_W_prev-f+2*pad)/stride+1)
    z=np.zeros((m,n_H,n_W,n_C))
    A_prev_pad=zero_pad(A_prev,pad)
    for i in range(m):
        a_prev_pad=A_prev_pad[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start=h*stride
                    vert_end=vert_start+f
                    horiz_start=w*stride
                    horiz_end=horiz_start+f
                    a_slice_prev=a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    z[i,h,w,c]=conv_sigle_step(a_slice_prev,W[:,:,:,c],b[:,:,:,c])
    assert(z.shape==(m,n_H,n_W,n_C))
    cache=(A_prev,W,b,hparameters)
    return z,cache
np.random.seed(1)
A_prev = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad" : 2,
               "stride": 2}
z, cache_conv = conv_forward(A_prev, W, b, hparameters)
print("Z's mean =", np.mean(z))
print("Z[3,2,1] =", z[3,2,1])
print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
def pool_forward(A_prev,hparameters,mode="max"):#池化层建立
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    f=hparameters["f"]
    stride=hparameters["stride"]
    n_H=int(1+(n_H_prev-f)/stride)#这个也是公式，这里不用扩展，千米那得是1
    n_W=int(1+(n_W_prev-f)/stride)
    n_C=n_C_prev

    A=np.zeros((m,n_H,n_W,n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range (n_W):
                for c in range(n_C):
                    vert_start=h*stride
                    vert_end=vert_start+f
                    horiz_start=w*stride
                    horiz_end=horiz_start+f

                    a_prev_slice=A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    if mode=="max":
                        A[i,h,w,c]=np.max(a_prev_slice)
                    elif mode=="average":
                        A[i,h,w,c]=np.mean(a_prev_slice)
    cache=(A_prev,hparameters)
    assert(A.shape==(m,n_H,n_W,n_C))
    return A,cache
np.random.seed(1)
A_prev=np.random.randn(2,4,4,3)
hparameters={"stride":2,"f":3}

A,cache=pool_forward(A_prev,hparameters)
print("mode=max")
print("A=",A)
print()
A,cache=pool_forward(A_prev,hparameters,mode="average")
print("mode=average")
print("A=",A)

def conv_backward(dZ,cache):#卷积反向过程

     (A_prev,W,b,hparameters)=cache
     (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
     (f,f,n_C_prev,n_C)=W.shape
     stride=hparameters["stride"]
     pad=hparameters["pad"]
     (m,n_H,n_W,n_C)=dZ.shape
     dA_prev=np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
     dW=np.zeros((f,f,n_C_prev,n_C))
     db=np.zeros((1,1,1,n_C))
     A_prev_pad=zero_pad(A_prev,pad)
     dA_prev_pad=zero_pad(dA_prev,pad)
     for i in range(m):
        a_prev_pad=A_prev_pad[i,:,:,:]
        da_prev_pad=dA_prev_pad[i,:,:,:]
        for h in range(n_H):
            for w in range (n_W):
                for c in range(n_C):
                    vert_start=h*stride
                    vert_end=vert_start+f
                    horiz_start=w*stride
                    horiz_end=horiz_start+f
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]+=W[:,:,:,c]*dZ[i,h,w,c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]##一开始没跑通是因为这两货没打出来。
                    db[:, :, :, c] += dZ[i, h, w, c]
        dA_prev[i,:,:,:]=da_prev_pad[pad:-pad,pad:-pad,:]
     assert(dA_prev.shape==(m,n_H_prev,n_W_prev,n_C_prev))
     return dA_prev,dW,db
dA,dW,db=conv_backward(z,cache_conv)
print("dA_mean=",np.mean(dA))
print("dW_mean=",np.mean(dW))
print("db_mean=",np.mean(db))
def create_mask_from_window(x):##反向过程，池化层,max pooing
    mask=x==np.max(x)
    return mask
np.random.seed(1)
x=np.random.randn(2,3)
mask=create_mask_from_window(x)
print('x=',x)
print("mask=",mask)
def distribute_value(dz,shape):#average pooling
    (n_H,n_W)=shape
    average=dz/(n_H*n_W)
    a=np.ones(shape)*average
    return a
a=distribute_value(2,(2,2))
print('distributed value=',a)
def pool_backward(dA,cache,mode="max"):##反向过程池化
    (A_prev,hparameters)=cache
    stride=hparameters["stride"]
    f=hparameters["f"]
    m,n_H_prev,n_W_prev,n_C_prev=A_prev.shape
    m,n_H,n_W,n_C=dA.shape
    dA_prev=np.zeros(A_prev.shape)
    for i in range(m):
      a_prev=A_prev[i,:,:,:]
      for h in range(n_H):
        for w in range(n_W):
            for c in range(n_C):
                vert_start=stride*h
                vert_end=vert_start+f
                horiz_start=stride*w
                horiz_end=horiz_start+f
                if mode =="max":
                    a_prev_slice=a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                    mask=create_mask_from_window(a_prev_slice)
                    dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]+=np.multiply(mask,dA[i,h,w,c])
                elif mode=="average":
                    da=dA[i,h,w,c]
                    shape=(f,f)
                    dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]+=distribute_value(da,shape)
    assert(dA_prev.shape==A_prev.shape)
    return dA_prev
np.random.seed(1)#最后把各种的情况列出来。
A_prev=np.random.randn(5,5,3,2)
hparameters={"stride":1,"f":2}
A,cache=pool_forward(A_prev,hparameters)
dA=np.random.randn(5,4,2,2)
dA_prev=pool_backward(dA,cache,mode="max")
print("mode=max")
print('mean of dA =',np.mean(dA))
print('dA_prev[1,1]=',dA_prev[1,1])
print()
dA_prev=pool_backward(dA,cache,mode="average")
print("mode=average")
print('mean of dA=',np.mean(dA))
print('dA_prev[1,1]=',dA_prev[1,1])
