import os
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def convolution(I, Ker):
    '''
    This function takes a n-dimensional numpy array <I> and performs linear 
    convolution with the n-dimensional numpy array <Ker>. The purpose of this 
    function is for convolution of RGB images with 3 dimensions. 

    Parameters
    ----------
    I: np.ndarray 
        Image values (ex. RGB)
    Ker: np.ndarray 
        Kernal/mask values

    Returns
    -------
    I_final: np.ndarray [n, m, 1]
        Convoluted image values 
    '''
    # normalize
    Ker_normed = Ker/np.linalg.norm(Ker)

    # Image and Kernel size
    Iy, Ix, Iz = I.shape
    Kery, Kerx, Kerz = Ker.shape
    Kerx_rad = np.math.floor(Kerx/2)
    Kery_rad = np.math.floor(Kery/2)

    I_final = np.empty((Iy, Ix))
    for i in range(Kerx_rad, Ix-Kerx_rad):
        for j in range(Kery_rad, Iy-Kery_rad):
            I_sub = I[j-Kery_rad:j+Kery_rad+1, i-Kerx_rad:i+Kerx_rad+1]
            I_sub_normed = I_sub/np.linalg.norm(I_sub)
            I_final[j, i] = np.sum(np.multiply(I_sub_normed, Ker_normed))

    return I_final

def calc_bboxes(binary_mask, kernel):
    '''
    This function takes a numpy array binary mask of the red light predictions and 
    outputs the boundary box coordinates in a list of lists. This is based off a few assumptions: 
    - No red lights are located directly above/below another red light. 
    - The binary mask for red lights has circular predictions.
    - Predicted red light locations are of similar size to the kernel.
    '''
    my, mx = binary_mask.shape
    ky, kx, kz = kernel.shape

    I_boxes = []

    x_mid=0
    while x_mid<mx: 
        if np.sum(binary_mask[:, x_mid])==0:
            x_mid+=1
        else: 
            y_mid = np.mean(np.nonzero(binary_mask[:, x_mid]))
            I_boxes.append([x_mid-np.floor(kx/2), y_mid-np.floor(ky/2), \
                x_mid+np.floor(kx/2), y_mid+np.floor(ky/2)])
            x_mid+=kx

    return I_boxes

def detect_red_light(I, plot=False):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    if np.mean(I[0:240, :, :])<40:
        kernel = night_ker
    elif np.mean(I[0:240, :, :])<120:
        kernel = cloudy_ker
    else:
        kernel = day_ker

    post_conv = convolution(I, kernel)

    thresh = 0.9
    red_lights = post_conv>thresh
    plt.imshow(red_lights)

    bounding_boxes = calc_bboxes(red_lights, kernel)

    if plot:
        plt.imshow(I)
        img = Image.fromarray(I, 'RGB')
        draw = ImageDraw.Draw(img)
        for j in bounding_boxes:
            draw.rectangle(j)
        img.show()
        plt.savefig('foo.png')

    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = './data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = './data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

# kernels
night_arr = np.asarray(Image.open(r'.\data\RedLights2011_Medium\RL-248.jpg'))
night_ker = night_arr[134:171, 498:517, :]

day_arr = np.asarray(Image.open(r'.\data\RedLights2011_Medium\RL-318.jpg'))
day_ker = day_arr[210:227, 300:311, :]

cloudy_arr = np.asarray(Image.open(r'.\data\RedLights2011_Medium\RL-031.jpg'))
cloudy_ker = cloudy_arr[251:272, 404:413, :]

preds = {}
for i in range(len(file_names)):
    print('is it running')
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I, plot=True)


# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
