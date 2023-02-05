import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
import cv2
import plotly.express as px

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
    # donnee = pd.read_csv('for_matlab.txt', sep=" ", header=None)
    
    return big_df

def data_gen(bot, top):
    
    donnee = load_data()
    depthlog = load_depthlog()
    depth_range = load_depths(bot, top)
    # depth_range = depthlog.index[(depthlog[0]==bot) | (depthlog[0]==top)].tolist()
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

st.title('Computer Vision Workflow')
st.write('This dashboard is intended for easier observations when changing different values of the algorithms contained in this workflow.')
st.header('Original Image')
st.sidebar.title('Parameters For Adjustment')
st.sidebar.header('Depth Interval')

bot = st.sidebar.number_input('State top depth in ft. The interval is 2ft for better resolution', value=4051, max_value = 5295)
top = bot + 2
depths = load_depths(bot, top)
mat2 = data_gen(bot, top)

st.sidebar.header('Set the color range for visualization purposes')
color_range = st.sidebar.slider(
    "Pixel range:",
    value=(float(mat2.min().min()),float(mat2.max().max())))

fig = px.imshow(mat2,aspect='auto',color_continuous_scale = 'YlOrBr_r', width=800, height=800, range_color=(color_range[0], color_range[1]))
fig.update_xaxes(visible=False)
# fig.update_yaxes(range=[depths[1],depths[0]])
# fig.update_layout(yaxis_range=[depths[0],depths[1]])

st.plotly_chart(fig)

st.header('Average / Gaussian / Bilateral')
st.write('Choose between the different filtering options on the sidebar or experiment to see which fits the depth interval better.')

filter_option = st.sidebar.radio('Filter Options', ('Averaging Kernel', 'Gaussian Blur', 'Bilateral Filter'), index=2) # Boolean

if filter_option == 'Averaging Kernel':
    kernel_avg = np.ones((5,5),np.float32)/25
    output = cv2.filter2D(mat2,-1,kernel_avg)
elif filter_option == 'Gaussian Blur':
    output = cv2.blur(mat2,(7,7))
elif filter_option == 'Bilateral Filter':
    output = cv2.bilateralFilter(mat2,9,75,75)
    
mat2 = output

fig_filter = px.imshow(output,aspect='auto',color_continuous_scale = 'YlOrBr_r', width=800, height=800, range_color=(color_range[0], color_range[1]))
st.plotly_chart(fig_filter)

st.header('Kmeans Computation')
st.write('This is an experimental step in the workflow. If checked, the image shown is only for visualiation and will not be used for analysis.')
st.sidebar.header('Kmeans Args')
kmeans_check = st.sidebar.checkbox('Would you like to run KMeans?')
if kmeans_check == True:
    k_value = st.sidebar.number_input('What K-value would you want to try?', value = 5, min_value = 2)

    # convert to np.float32
    Z = np.float32(mat2)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,k_value,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((mat2.shape))

    fig2 = px.imshow(res2,aspect='auto',color_continuous_scale = 'gray', width=800, height=800)
    fig2.update_xaxes(visible=False)
    st.plotly_chart(fig2)

    
st.header('Gabor Filters')
st.write('This step filters the lines in the image based on a range of angles specified by the parameters in the sidebar.')
st.sidebar.header('Gabor Filter Args')

num_filters = st.sidebar.slider('Number of filters', 3, 50, 16)
ksize = st.sidebar.slider('Kernel size', 3, 15, 5)  # The local area to evaluate
sigma = st.sidebar.slider('Sigma value', 0.5, 10.0, 5.0)  # Larger Values produce more edges
lambd = st.sidebar.slider('Lambda value', 0.5, 50.0, 10.0)
gamma = st.sidebar.slider('Gamma value', 0.0, 5.0, 0.5)
psi = st.sidebar.slider('Psi value', 0, 5, 0)  # Offset value - lower generates cleaner results

gfilters = create_gaborfilter(num_filters, ksize, sigma, lambd, gamma, psi)
image_g = apply_filter(mat2, gfilters)

fig3 = px.imshow(image_g,aspect='auto',color_continuous_scale = 'gray', width=800, height=800)
# fig.update(layout_showlegend=False)
st.plotly_chart(fig3)

st.header('Canny Edge Detection')
st.write('This step detects edges contained in the resultant image and connects qualifying edges based on the threshold values in the sidebar. An optional L2 gradient can also be used.')
st.sidebar.header('Canny Edge Args')

# Defining all the parameters
threshold_values = st.sidebar.slider('Threshold values', 10, 1000, (500, 600))
aperture_size = 5 # Aperture size
L2Gradient = st.sidebar.radio('Enable L2 Gradient?', ('True', 'False'), index=1) # Boolean
  
# Applying the Canny Edge filter 
# with Aperture Size and L2Gradient
#TODO: mat2 or res2 depending on which image is chosen
if L2Gradient == 'True':
    edge = cv2.Canny(image_g, threshold_values[0], threshold_values[1],
                    apertureSize = aperture_size, 
                    L2gradient = True )
else:
    edge = cv2.Canny(image_g, threshold_values[0], threshold_values[1],
                apertureSize = aperture_size)
    
fig4 = px.imshow(edge,aspect='auto',color_continuous_scale = 'gray', width=800, height=800)
st.plotly_chart(fig4)

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
    statement = 'Lines are available'
    for line in range(0, len(lines)):
        l = lines[line][0]
        pt1 = (l[0], l[1])
        pt2 = (l[2], l[3])
        angle = np.arctan2(l[3] - l[1], l[2] - l[0]) * 180. / np.pi
        if angle_check == True and angle < angle_thresh:
            # if a:
            cv2.line(dummy, pt1, pt2, (0,0,255), 3)
        elif angle_check == False:
            cv2.line(dummy, pt1, pt2, (0,0,255), 3)
elif lines == None:
    statement = 'Lines are not available'
    pass

st.write(f'{statement}')

fig_hough, ax = plt.subplots(figsize=[10,10])
ax.imshow(dummy, cmap = 'gray', aspect='auto')
plt.xticks([]), plt.yticks([])
st.pyplot(fig_hough)