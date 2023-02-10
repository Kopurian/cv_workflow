import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import ndimage
import cv2
import plotly.express as px
import io
# import diplib as dip

##### Functions

# explicit function to normalize array
def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

def create_gaborfilter(number, kernel_size, sigma, lambd, gamma, psi):
    # This function is designed to produce a set of GaborFilters 
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = number
    ksize = kernel_size  # The local area to evaluate
    sigma = sigma  # Larger Values produce more edges
    lambd = lambd
    gamma = gamma
    psi = psi  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

def apply_filter(img, filters):
# This general function is designed to apply filters to our image
     
    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)
     
    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1 # remain depth same as original image
     
    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv2.filter2D(img, depth, kern)  #Apply filter to image
         
        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage

def KMeans(mat2, k_value):
    # convert to np.float32
    Z = np.float32(mat2)
    # define criteria, number of clusters(K) and apply kmeans()
    k_value = k_value
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,k_value,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((mat2.shape))
    
    return res2

@st.cache
def load_depthlog():
    depthlog = pd.read_csv('depthlog.txt', header=None)
    
    return depthlog

def load_depths(bot, top):
    depthlog = load_depthlog()
    depth_range = depthlog.index[(depthlog[0]==bot) | (depthlog[0]==top)].tolist()
    
    return depth_range
   
@st.cache 
def load_data():
    files_list = ['data1.txt', 'data2.txt', 'data3.txt', 'data4.txt']

    df_list = []
    for file in files_list:
        df_list.append(pd.read_csv(file, header=None, sep=' '))

    big_df = pd.concat(df_list, ignore_index=True)
    
    return big_df

def data_gen(bot, top):
    
    donnee = load_data()
    depthlog = load_depthlog()
    depth_range = load_depths(bot, top)
    depth_bot = depth_range[0]
    depth_top = depth_range[1]
    donnee = donnee.iloc[depth_bot:depth_top+1,:]

    #if amplitudes are considered
    max_donnee = donnee.max().max()
    donnee = max_donnee*np.ones((donnee.shape)) - donnee

    ## Interpolation
    interp_interval = 0.01

    # interpolates the depth values to a constant interval
    # depth = np.arange(depthlog.iloc[depth_bot,0], depthlog.iloc[depth_top, 0], interp_interval)
    depth = np.arange(depthlog.iloc[depth_bot,0], depthlog.iloc[depth_top, 0], interp_interval)
    depthlog_new = depthlog.iloc[depth_bot:depth_top+1,:]
    depthlog_flat = depthlog_new.to_numpy()
    depthlog_flat = depthlog_flat.flatten()

    # defines a new matrix with only the data we are interested in

    mat_list = []

    for col in np.arange(0, len(donnee.columns)):
        values = donnee.iloc[:, col]
        interpolated = interp1d(depthlog_flat, values)(depth)
        mat_list.append(interpolated)

    mat = pd.DataFrame(mat_list)
    mat = mat.transpose()
    mat2 = mat.to_numpy()
    mat2 = mat2.astype(np.uint8)
    
    return mat2

##### Page Setup

st.title('Computer Vision Workflow')
st.write('This dashboard is intended for easier observations when changing different values of the algorithms contained in this workflow.')
st.header('Original Image')
st.sidebar.title('Parameters For Adjustment')
st.sidebar.header('Depth Interval')

##### Load data and set range visualization

bot = st.sidebar.number_input('State top depth in ft. The interval is 2ft for better resolution', value=4051, max_value = 5295)
top = bot + 2
depths = load_depths(bot, top)
mat2 = data_gen(bot, top)
ori_img = mat2.copy()

st.sidebar.header('Set the color range for visualization purposes')
color_range = st.sidebar.slider(
    "Pixel range:",
    value=(float(mat2.min().min()),float(mat2.max().max())))

fig_original = px.imshow(mat2,aspect='auto',color_continuous_scale = 'YlOrBr_r', width=800, height=800, range_color=(color_range[0], color_range[1]))
fig_original.update_xaxes(visible=False)

st.plotly_chart(fig_original)

##### Kernel selection

st.header('Average / Gaussian / Bilateral')
st.write('Choose between the different filtering options on the sidebar or experiment to see which fits the depth interval better.')

filter_option = st.sidebar.radio('Filter Options', ('Averaging Kernel', 'Median Filter', 'Gaussian Blur', 'Bilateral Filter'), index=3) # Boolean

if filter_option == 'Averaging Kernel':
    kernel_avg = np.ones((5,5),np.float32)/25
    output = cv2.filter2D(mat2,-1,kernel_avg)
elif filter_option == 'Gaussian Blur':
    output = cv2.blur(mat2,(7,7))
elif filter_option == 'Bilateral Filter':
    output = cv2.bilateralFilter(mat2,9,75,75)
elif filter_option == 'Median Filter':
    output = cv2.medianBlur(mat2,5)

fig_filter = px.imshow(output,aspect='auto',color_continuous_scale = 'YlOrBr_r', width=800, height=800, range_color=(color_range[0], color_range[1]))
st.plotly_chart(fig_filter)

##### Gabor filters
    
st.header('Gabor Filters')
st.write('This step filters the lines in the image based on a range of angles specified by the parameters in the sidebar.')
st.sidebar.header('Gabor Filter Args')

num_filters = st.sidebar.slider('Number of filters', 3, 50, 16)
ksize = st.sidebar.slider('Kernel size', 3, 9, 5)  # The local area to evaluate
sigma = st.sidebar.slider('Sigma value', 0.5, 10.0, 5.0)  # Larger Values produce more edges
lambd = st.sidebar.slider('Lambda value', 0.5, 50.0, 10.0)
gamma = st.sidebar.slider('Gamma value', 0.0, 5.0, 0.5)
psi = st.sidebar.slider('Psi value', 0, 5, 0)  # Offset value - lower generates cleaner results

gfilters = create_gaborfilter(num_filters, ksize, sigma, lambd, gamma, psi)
image_g = apply_filter(output, gfilters)

fig_gabor = px.imshow(image_g,aspect='auto',color_continuous_scale = 'gray', width=800, height=800)
st.plotly_chart(fig_gabor)

##### KMeans
    
st.header('Kmeans Computation')
st.write('If K-Means is used in this step, the resultant image will be used in the rest of the workflow. Otherwise, this step will be skipped.')
st.sidebar.header('Kmeans Args')
kmeans_check = st.sidebar.checkbox('Would you like to run KMeans?')
if kmeans_check == True:
    k_value = st.sidebar.number_input('What K-value would you want to try?', value = 10, min_value = 2)
    res2 = KMeans(image_g, k_value)

    fig_kmeans = px.imshow(res2,aspect='auto',color_continuous_scale = 'gray', width=800, height=800)
    fig_kmeans.update_xaxes(visible=False)
    st.plotly_chart(fig_kmeans)

##### Canny Edge Detection

st.header('Canny Edge Detection')
st.write('This step detects edges contained in the resultant image and connects qualifying edges based on the threshold values in the sidebar. An optional L2 gradient can also be used.')
st.sidebar.header('Canny Edge Args')

# Defining all the parameters
threshold_values = st.sidebar.slider('Threshold values', 100, 1500, (800, 900))
aperture_size = 5 # Aperture size
L2Gradient = st.sidebar.radio('Enable L2 Gradient?', ('True', 'False'), index=1) # Boolean
  
# Applying the Canny Edge filter 
# with Aperture Size and L2Gradient

if kmeans_check == True:
    if L2Gradient == 'True':
        edge = cv2.Canny(res2, threshold_values[0], threshold_values[1],
                        apertureSize = aperture_size, 
                        L2gradient = True )
    else:
        edge = cv2.Canny(res2, threshold_values[0], threshold_values[1],
                    apertureSize = aperture_size)
else:
    if L2Gradient == 'True':
        edge = cv2.Canny(image_g, threshold_values[0], threshold_values[1],
                        apertureSize = aperture_size, 
                        L2gradient = True )
    else:
        edge = cv2.Canny(image_g, threshold_values[0], threshold_values[1],
                    apertureSize = aperture_size)
    
fig_canny = px.imshow(edge,aspect='auto',color_continuous_scale = 'gray', width=800, height=800)
st.plotly_chart(fig_canny)

##### Hough Transform

st.header('Hough Transform')
st.write('This step will check for continuous lines and optionally remove any lines that exceed an angle threshold.')
st.sidebar.header('Hough Transform Args')

hough_thresh = st.sidebar.slider('Threshold', 1, 100, 10)
minLineLength = st.sidebar.slider('Minimum Line Length', 1, 50, 10)  # Larger Values produce more edges
maxLineGap = st.sidebar.slider('Maximum Line Gap', 1, 50, 5)
angle_check = st.sidebar.checkbox('Keep Only Horizontal Lines?', value = True)
if angle_check == True:
    angle_thresh = st.sidebar.slider('Angle Threshold', 10, 90, 10)

lines = cv2.HoughLinesP(edge, 1, np.pi/180, threshold=hough_thresh, minLineLength=minLineLength, maxLineGap=maxLineGap)
dummy = np.ones(shape=mat2.shape, dtype=np.uint8)

if lines is not None:
    statement = f'There are {len(lines)} lines in this image'
    st.success(statement)
    for line in range(0, len(lines)):
        l = lines[line][0]
        pt1 = (l[0], l[1])
        pt2 = (l[2], l[3])
        angle = np.arctan2(l[3] - l[1], l[2] - l[0]) * 180. / np.pi
        if angle_check == True and angle < angle_thresh:
            cv2.line(dummy, pt1, pt2, (0,0,255), 1)
        elif angle_check == False:
            cv2.line(dummy, pt1, pt2, (0,0,255), 1)
elif lines == None:
    statement = 'There are no lines in this image'
    st.error(statement)
    pass

masked = np.ma.masked_where(dummy == 1, dummy)

fig_hough, ax_hough = plt.subplots(figsize=[10,10])
ax_hough.imshow(ori_img, cmap='YlOrBr_r', aspect='auto', vmin=color_range[0], vmax=color_range[1])
ax_hough.imshow(masked, cmap = 'gray', aspect='auto', alpha=0.7)
plt.xticks([]), plt.yticks([])
st.pyplot(fig_hough)

##### Option for downloading all images in a subplot

fig_total, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(25, 25))

ax1.imshow(ori_img, cmap='YlOrBr_r', aspect='auto', vmin=color_range[0], vmax=color_range[1])
ax2.imshow(output, cmap='gray', aspect='auto')
ax3.imshow(image_g, cmap='gray', aspect='auto')
if kmeans_check == True:
    ax4.imshow(res2, cmap='gray', aspect='auto')
else:
    ax4.imshow(np.ones(shape=mat2.shape, dtype=np.uint8), cmap='gray', aspect='auto')
ax5.imshow(edge, cmap='gray', aspect='auto')
ax6.imshow(ori_img, cmap='YlOrBr_r', aspect='auto', vmin=color_range[0], vmax=color_range[1])
ax6.imshow(masked, cmap = 'gray', aspect='auto', alpha=0.5)

ax1.set_title('Original Image', fontsize=16)
ax2.set_title(f'Filtered With {filter_option}', fontsize=16)
ax3.set_title('Gabor Filters', fontsize=16)
ax4.set_title('KMeans Result', fontsize=16)
ax5.set_title('Canny Edge Detection', fontsize=16)
ax6.set_title('Final Result Overlay', fontsize=16)

fn = 'test.png'
img = io.BytesIO()
plt.savefig(img, format='png')

st.header('Option for Download')
st.write('If the analysis and results are to satisfaction, you can download an image with all analysis appended by clicking on the button below.')
 
btn = st.download_button(
   label="Download image",
   data=img,
   file_name=fn,
   mime="image/png"
)

st.header('Experimental Features')

# min_threshold = 100
# culled_edge = dip.BinaryAreaOpening(edge > 0, min_threshold)

# fig_test = px.imshow(culled_edge,aspect='auto',color_continuous_scale = 'gray', width=800, height=800)
# st.plotly_chart(fig_test)

# Sobel test

# edge_hori = ndimage.sobel(ori_img, 0)
# edge_vert = ndimage.sobel(ori_img, 1)
# magnitude = np.hypot(edge_hori, edge_vert)

# fig_sobel, ax_sobel = plt.subplots(figsize=[10,10])
# ax_sobel.imshow(magnitude, aspect='auto')
# st.pyplot(fig_sobel)

# Detect horizontal lines
# dummy2 = np.ones(shape=mat2.shape, dtype=np.uint8)
# thresh = cv2.threshold(ori_img, color_range[0], color_range[1], cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
# detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
# cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     cv2.drawContours(dummy2, [c], -1, (36,255,12), 2)

# masked2 = np.ma.masked_where(dummy2 == 0, dummy2)
# fig_sobel, ax_sobel = plt.subplots(figsize=[10,10])
# ax_sobel.imshow(ori_img, cmap='YlOrBr_r', aspect='auto', vmin=color_range[0], vmax=color_range[1])
# ax_sobel.imshow(dummy2, aspect='auto', cmap = 'gray', alpha=0.5)
# plt.xticks([]), plt.yticks([])
# st.pyplot(fig_sobel)

# def find_first(item, vec):
#     """return the index of the first occurence of item in vec"""
#     for i in range(len(vec)):
#         if item == vec[i]:
#             return i
#     return -1

# bounds = [750, 1500]
# # Now the points we want are the lowest-index 255 in each row
# window = edge[bounds[1]:bounds[0]:-1].transpose()

# xy = []
# for i in range(len(window)):
#     col = window[i]
#     j = find_first(255, col)
#     if j != -1:
#         xy.extend((i, j))
# # Reshape into [[x1, y1],...]
# data = np.array(xy).reshape((-1, 2))
# # Translate points back to original positions.
# data[:, 1] = bounds[1] - data[:, 1]

# v = np.median(ori_img)
# sigma = 0.33

# #---- apply optimal Canny edge detection using the computed median----
# lower_thresh = int(max(mat2.min().min(), (1.0 - sigma) * v))
# upper_thresh = int(min(mat2.max().max(), (1.0 + sigma) * v))

# edge2 = cv2.Canny(ori_img, lower_thresh, upper_thresh,
#                 apertureSize = aperture_size, 
#                 L2gradient = False )

# fig_canny2 = px.imshow(edge2,aspect='auto',color_continuous_scale = 'gray', width=800, height=800)
# st.plotly_chart(fig_canny2)

# def sobel_filters(img):
#     Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
#     Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
#     Ix = ndimage.filters.convolve(img, Kx)
#     Iy = ndimage.filters.convolve(img, Ky)
    
#     G = np.hypot(Ix, Iy)
#     G = G / G.max() * 255
#     theta = np.arctan2(Iy, Ix)
    
#     return (G, theta)

# convert to np.float32
# img_test = np.float32(mat2)
# # define criteria, number of clusters(K) and apply kmeans()
# k_value_2 = 10
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# ret2,label2,center2=cv2.kmeans(img_test,k_value_2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# # Now convert back into uint8, and make original image
# center2 = np.uint8(center2)
# res2 = center2[label2.flatten()]
# res22 = res2.reshape((mat2.shape))

# fig_k = px.imshow(res22,aspect='auto',color_continuous_scale = 'gray', width=800, height=800)
# fig_k.update_xaxes(visible=False)
# st.plotly_chart(fig_k)

