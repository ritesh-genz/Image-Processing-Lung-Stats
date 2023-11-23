import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.measure import label,regionprops
from scipy import ndimage as ndi
from scipy.ndimage import center_of_mass, binary_dilation, zoom
import plotly.graph_objects as go
import numpy as np
import nibabel as nib

def imshow(title, image):#Function for displaying image
    plt.imshow(image)
    plt.title(title)
    plt.show()
#Conversion of 3D CT scanned lung image to 3D numpy array(.npy)
image_path = r"OpenCV\IMG_0059.nii"
npy_path = r"OpenCV\Segmented Lungs"
nii_image = nib.load(image_path)
img_3d = nii_image.get_fdata()
np.save(npy_path, img_3d)
#numpy array's saved and then loaded
img = np.load(r'OpenCV\Segmented Lungs.npy')
print(img.shape)

plt.figure(figsize=(8,8))
for i in range(0,3):
    dim=int(input("Enter your layer"))
    plt.pcolormesh(img[dim])# if grayscale :cmap='Greys_r'
    plt.colorbar()
    imshow("Actual image",img[dim])
#Threshold's set to be lesser than -320 so as to seperate it as a mesh(Honsfield units-Attenuation constant)
mask = img < -320
dim=int(input("Enter your finalized layer"))
plt.pcolormesh(mask[dim])
plt.colorbar()
imshow("HF Segmented image",mask[dim])

#Borders are cleared based on its surrounding attenuation levels
mask = np.vectorize(clear_border, signature='(n,m)->(n,m)')(mask)
plt.pcolormesh(mask[dim])
plt.colorbar()
imshow("border cleared image",mask[dim])

#In order to differentiate the lungs from the artifacts ,the lungs are labelled
mask_labeled = np.vectorize(label, signature='(n,m)->(n,m)')(mask)
plt.pcolormesh(mask_labeled[dim])
plt.colorbar()
imshow("Labelled image",mask[dim])
#Region divison and sorting in reverse to get the max area 
slc = mask_labeled[dim]
rps = regionprops(slc)
areas = [r.area for r in rps]
idxs = np.argsort(areas)[::-1]#We want Largest to smallest mesh
print("Areas",areas)
print("Index Positions",idxs)
#3Top colours by Area is chosed and then made to be either 0/1
def keep_top_3(slc):
    new_slc = np.zeros_like(slc)
    rps = regionprops(slc)
    areas = [r.area for r in rps]
    idxs = np.argsort(areas)[::-1]#We want Largest to smallest mesh
    
    for i in idxs[:3]:
        new_slc[tuple(rps[i].coords.T)] = i+1
    return new_slc
mask_labeled = np.vectorize(keep_top_3, signature='(n,m)->(n,m)')(mask_labeled)
imshow("3TopUpdated",mask_labeled[dim])
#Fill all the small holes in the Lungs
mask = mask_labeled > 0
mask = np.vectorize(ndi.binary_fill_holes, signature='(n,m)->(n,m)')(mask)
plt.pcolormesh(mask[dim])
imshow("Removed Holes",mask[dim])
#In a 512*512 image a trachea typically takes 0.69% so we can remove it all 
plt.pcolormesh(mask[-50])
labels = label(mask[-50],connectivity=1,background=0)
plt.pcolormesh(labels)
imshow("Image with Trachea",mask[-50])
def remove_trachea(slc, c=0.0069):
    new_slc = slc.copy()
    labels = label(slc,connectivity=1,background=0)
    rps = regionprops(labels)
    areas = np.array([r.area for r in rps])
    idxs = np.where(areas/512**2 < c)[0]
    for i in idxs:
        new_slc[tuple(rps[i].coords.T)] = 0
    return new_slc
mask = np.vectorize(remove_trachea, signature='(n,m)->(n,m)')(mask)
plt.pcolormesh(mask[-50])
imshow("After removing trachea",mask[dim])
#Center of mass logic is applied to remove the table if its found in a CT SCAN
center_of_mass(labels==3)[0]
def delete_table(slc):
    new_slc = slc.copy()
    labels = label(slc, background=0)
    idxs = np.unique(labels)[1:]
    COM_ys = np.array([center_of_mass(labels==i)[0] for i in idxs])
    for idx, COM_y in zip(idxs, COM_ys):
        if (COM_y < 0.3*slc.shape[0]):#Table or other artifacts
            new_slc[labels==idx] = 0
        elif (COM_y > 0.6*slc.shape[0]):#nostrils or other artifacts
            new_slc[labels==idx] = 0
    return new_slc
mask_new = np.vectorize(delete_table, signature='(n,m)->(n,m)')(mask)
plt.pcolormesh(mask_new[dim])
plt.colorbar()
imshow("COM",mask_new[dim])
#Binary dilations are used to expand the edges of the lungs and it depends on the iterations
mask_new = binary_dilation(mask_new, iterations=5)
print("Dilated image's shape",mask_new.shape)
plt.figure(figsize=(8,8))
plt.pcolormesh(mask_new[dim], cmap='brg')
imshow("Binary dilated image",mask_new[dim])
im = zoom(1*(mask_new), (0.4,0.4,0.4))#Reduces quality by 40%
print("Reduced quality image's shape",im.shape)
'''
z, y, x = [np.arange(i) for i in im.shape]
z*=4
X,Y,Z = np.meshgrid(x,y,z, indexing='ij')
fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=np.transpose(im,(1,2,0)).flatten(),
    isomin=0.1,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=17, # needs to be a large number for good volume rendering
    ))
fig.write_html("test.html")'''
img_new = mask_new * img
plt.pcolormesh(img_new[dim])
imshow("Final segmented Image",img_new[dim])