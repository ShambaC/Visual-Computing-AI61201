# %% [markdown]
# # Assignment 1

# %% [markdown]
# ## Imports

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

# %%
def show_image(image, dpi=100):
    height, width = image.shape[:2]

    fig = plt.figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi
    )

    ax = fig.add_axes([0, 0, 1, 1])

    if image.ndim == 3 and image.shape[2] == 3:
        ax.imshow(image, interpolation='none')
    else:
        ax.imshow(image, cmap='gray', interpolation='none')

    ax.axis('off')
    plt.show()

# %% [markdown]
# ## Task 1. Loading and retrieving RGB channels

# %%
image_a = cv2.imread('image_a.jpg')

# Convert to RGB and show using matplotlib
image_a_rgb = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)

show_image(image_a_rgb)

# %%
# Scratch method
r = image_a[:, :, 2]
g = image_a[:, :, 1]
b = image_a[:, :, 0]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(r, cmap='gray')
plt.title('Red Channel')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(g, cmap='gray')
plt.title('Green Channel')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(b, cmap='gray')
plt.title('Blue Channel')
plt.axis('off')
plt.show()

# %%
# OpenCV method
b, g, r = cv2.split(image_a)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(r, cmap='gray')
plt.title('Red Channel')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(g, cmap='gray')
plt.title('Green Channel')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(b, cmap='gray')
plt.title('Blue Channel')
plt.axis('off')
plt.show()

# %% [markdown]
# ## Task 2: Color Space Conversion

# %% [markdown]
# ### 2.1 YCbCr operations

# %%
# Scratch method
image_a_ycbcr = np.zeros_like(image_a)
Y = 0.299 * r + 0.587 * g + 0.114 * b       # Y
Cb = 128 + 0.564 * (b - Y)                  # Cb
Cr = 128 + 0.713 * (r - Y)                  # Cr

image_a_ycbcr[:, :, 0] = Y
image_a_ycbcr[:, :, 1] = Cb
image_a_ycbcr[:, :, 2] = Cr

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(Y, cmap='gray')
plt.title('Y Channel')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(Cb, cmap='gray')
plt.title('Cb Channel')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(Cr, cmap='gray')
plt.title('Cr Channel')
plt.axis('off')
plt.show()

# Increase brightness
"""
To turn up image brightness in YCbCr, we need to increase the Y channel values and leave the others as is.
"""
Y_new = np.clip(Y + 5, 0, 255)  # Increase Y channel value and clip to [0, 255]

# Convert back to RGB
r_new = Y_new * 1.0 + 1.403 * (Cr - 128)
g_new = Y_new * 1.0 - 0.344 * (Cb - 128) - 0.714 * (Cr - 128)
b_new = Y_new * 1.0 + 1.772 * (Cb - 128)

# Display RGB image
show_image(np.clip(np.stack((r_new, g_new, b_new), axis=-1).astype(np.uint8), 0, 255))

# %%
# OpenCV method
image_a_ycbcr = cv2.cvtColor(image_a, cv2.COLOR_RGB2YCrCb)
Y = image_a_ycbcr[:, :, 0]
Cb = image_a_ycbcr[:, :, 1]
Cr = image_a_ycbcr[:, :, 2]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(Y, cmap='gray')
plt.title('Y Channel')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(Cb, cmap='gray')
plt.title('Cb Channel')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(Cr, cmap='gray')
plt.title('Cr Channel')
plt.axis('off')
plt.show()

# Increase brightness
"""
To turn up image brightness in YCbCr, we need to increase the Y channel values and leave the others as is.
"""
Y_new = np.clip(Y + 5, 0, 255)  # Increase Y channel value and clip to [0, 255]

image_a_ycbcr_new = np.stack((Y_new, Cb, Cr), axis=-1)
image_a_rgb_brightened = cv2.cvtColor(image_a_ycbcr_new, cv2.COLOR_YCrCb2BGR)

# Display RGB image
show_image(image_a_rgb_brightened)

# %% [markdown]
# ### 2.2 HSV operations

# %%
# Scratch method
# Normalize RGB
r_dash = r / 255.0
g_dash = g / 255.0
b_dash = b / 255.0


# Minimum and maximum
rgb_min = np.min([r_dash, g_dash, b_dash], axis=0)
rgb_max = np.max([r_dash, g_dash, b_dash], axis=0)

value = rgb_max

# Difference / chroma
delta = value - rgb_min

# Conversion

# Saturation
saturation = np.zeros_like(value)

nonzero_value = value != 0

saturation[nonzero_value] = (delta[nonzero_value] / value[nonzero_value])

# Hue
hue_dash = np.zeros_like(value)

# Only calculate hue where delta != 0
nonzero_delta = delta != 0

# R is the maximum channel
mask = (value == r_dash) & nonzero_delta
hue_dash[mask] = ((g_dash[mask] - b_dash[mask]) / delta[mask])

# G is the maximum channel
mask = (value == g_dash) & nonzero_delta
hue_dash[mask] = (2.0 + (b_dash[mask] - r_dash[mask]) / delta[mask])

# B is the maximum channel
mask = (value == b_dash) & nonzero_delta
hue_dash[mask] = ( 4.0 + (r_dash[mask] - g_dash[mask]) / delta[mask])

hue = hue_dash * 60.0
hue[hue < 0] += 360.0

# Max out saturation
saturation_maxed = np.ones_like(saturation)

# Convert back to RGB

chroma = value * saturation_maxed

# Calculate X
x = chroma * (1 - np.abs(((hue / 60.0) % 2) - 1))

# Calculate m
m = value - chroma

r_dash_new = np.zeros_like(value)
g_dash_new = np.zeros_like(value)
b_dash_new = np.zeros_like(value)

# H = 0° to 60°
mask = (hue >= 0) & (hue < 60)

r_dash_new[mask] = chroma[mask]
g_dash_new[mask] = x[mask]

# H = 60° to 120°
mask = (hue >= 60) & (hue < 120)

r_dash_new[mask] = x[mask]
g_dash_new[mask] = chroma[mask]

# H = 120° to 180°
mask = (hue >= 120) & (hue < 180)

g_dash_new[mask] = chroma[mask]
b_dash_new[mask] = x[mask]

# H = 180° to 240°
mask = (hue >= 180) & (hue < 240)

g_dash_new[mask] = x[mask]
b_dash_new[mask] = chroma[mask]

# H = 240° to 300°
mask = (hue >= 240) & (hue < 300)

r_dash_new[mask] = x[mask]
b_dash_new[mask] = chroma[mask]

# H = 300° to 360°
mask = (hue >= 300) & (hue < 360)

r_dash_new[mask] = chroma[mask]
b_dash_new[mask] = x[mask]

# Add m
r_dash_new += m
g_dash_new += m
b_dash_new += m

# Convert [0, 1] -> [0, 255]
r_new = r_dash_new * 255.0
g_new = g_dash_new * 255.0
b_new = b_dash_new * 255.0

# Display image
rgb_new = np.stack((r_new, g_new, b_new), axis=-1)

show_image(np.clip(rgb_new, 0, 255).astype(np.uint8))

# %%
# OpenCV method
image_a_hsv = cv2.cvtColor(image_a, cv2.COLOR_BGR2HSV)

# Max out saturation
image_a_hsv[:, :, 1] = 255

# Convert back to RGB
image_a_rgb_saturated = cv2.cvtColor(image_a_hsv, cv2.COLOR_HSV2RGB)

# Display image
show_image(image_a_rgb_saturated)

# %% [markdown]
# ## Task 3: Bit Plane Operations

# %%
# Image loading
image_b = cv2.imread('image_b.jpg')

plt.axis('off')
plt.imshow(cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB))
plt.show()

# %%
# Scratch method

# Grayscale according to the luminance formula of YCbCr
image_b_gray = 0.299 * image_b[:, :, 2] + 0.587 * image_b[:, :, 1] + 0.114 * image_b[:, :, 0]
image_b_gray = np.clip(image_b_gray, 0, 255).astype(np.uint8)

# Toggle bit planes 7, 5, 3, 1
bit_planes = [7, 5, 3, 1]

mask = sum(1 << bit for bit in bit_planes)
image_b_gray_toggled = image_b_gray ^ mask

# Display toggled image
show_image(image_b_gray_toggled)

# %%
# OpenCV method
image_b_gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

# Toggle bit planes 7, 5, 3, 1
bit_planes = [7, 5, 3, 1]

mask = sum(1 << bit for bit in bit_planes)
image_b_gray_toggled = image_b_gray ^ mask

# Display toggled image
show_image(image_b_gray_toggled)

# %% [markdown]
# ## Task 4: Spatial Transformations and Interpolation

# %%
image_c = cv2.imread('image_c.jpg')

plt.axis('off')
plt.imshow(cv2.cvtColor(image_c, cv2.COLOR_BGR2RGB))
plt.show()

# %% [markdown]
# ### 4.1 Cropping and Rotation
# ### 4.2 Resizing and Interpolation

# %%
# Scratch method

# Crop
h, w = image_c.shape[:2] # Image height and width
c_w = 300                # Crop width
c_h = 200                # Crop height

y_start = (h - c_h) // 2
y_end   = (h + c_h) // 2

x_start = (w - c_w) // 2
x_end   = (w + c_w) // 2

image_c_cropped = image_c[y_start:y_end, x_start:x_end]

# Rotation
center_x = c_w // 2
center_y = c_h // 2

angle = 45
theta = np.radians(angle)

rot_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

# Generate all pixel coordinates
y, x = np.indices((c_h, c_w))

# Shift coordinates so that center is origin
x_centered = x - center_x
y_centered = y - center_y


# Apply rotation
new_x = (
    rot_matrix[0, 0] * x_centered +
    rot_matrix[0, 1] * y_centered
)

new_y = (
    rot_matrix[1, 0] * x_centered +
    rot_matrix[1, 1] * y_centered
)


# Shift back
new_x += center_x
new_y += center_y


# Convert to integer coordinates
new_x = new_x.astype(int)
new_y = new_y.astype(int)

# Check boundaries
valid = (
    (new_x >= 0) &
    (new_x < c_w) &
    (new_y >= 0) &
    (new_y < c_h)
)


# Create output
image_c_rotated = np.zeros_like(image_c_cropped)

image_c_rotated[valid] = image_c_cropped[
    new_y[valid],
    new_x[valid]
]

# Display rotated image
show_image(cv2.cvtColor(image_c_rotated, cv2.COLOR_BGR2RGB))

# Resizing and Interpolation

scale = 2
a = 3 # Lanczos kernel size for 2x upsampling

src_h, src_w = image_c_rotated.shape[:2]
dst_h = src_h * scale
dst_w = src_w * scale

# Output pixel coordinates mapped to source coordinates
y_dst = np.arange(dst_h)
x_dst = np.arange(dst_w)

y_src = (y_dst + 0.5) / scale - 0.5
x_src = (x_dst + 0.5) / scale - 0.5

#==================================================

def lanczos(x, a=3):
    """
    Lanczos kernel function.
    """
    x = np.asarray(x, dtype=np.float64)

    result = np.zeros_like(x)

    # x = 0
    zero = (x == 0)
    result[zero] = 1.0

    # -a <= x < a and x != 0
    valid = (x >= -a) & (x < a) & (~zero)

    result[valid] = (
        a * np.sin(np.pi * x[valid])
        * np.sin(np.pi * x[valid] / a)
        / (np.pi ** 2 * x[valid] ** 2)
    )

    return result

#==================================================


# Calculate source pixel indices

y_base = np.floor(y_src).astype(int)
x_base = np.floor(x_src).astype(int)

# Lanczos-3 uses 2a = 6 neighboring samples
offsets = np.arange(-a + 1, a + 1)

y_indices = y_base[:, None] + offsets
x_indices = x_base[:, None] + offsets

# Calculate Lanczos weights

y_weights = lanczos(y_src[:, None] - y_indices, a)
x_weights = lanczos(x_src[:, None] - x_indices, a)

# Handle image boundaries

y_indices = np.clip(y_indices, 0, src_h - 1)
x_indices = np.clip(x_indices, 0, src_w - 1)


# Normalize interpolation weights
y_weights /= y_weights.sum(axis=1, keepdims=True)
x_weights /= x_weights.sum(axis=1, keepdims=True)

# Vertical interpolation
temp = np.einsum('ik,ik...->i...', y_weights, image_c_rotated[y_indices])

# Horizontal interpolation
image_c_upsampled = np.einsum('jk,ijk...->ij...', x_weights, temp[:, x_indices])

# Convert to original image type
image_c_upsampled = np.clip(image_c_upsampled, 0, 255).astype(image_c_rotated.dtype)

# Display upsampled image
show_image(cv2.cvtColor(image_c_upsampled, cv2.COLOR_BGR2RGB))

# %%
# OpenCV method

# Crop

h, w = image_c.shape[:2]

c_w = 300
c_h = 200

y_start = (h - c_h) // 2
y_end   = (h + c_h) // 2

x_start = (w - c_w) // 2
x_end   = (w + c_w) // 2

image_c_cropped = image_c[y_start:y_end, x_start:x_end]

# Rotation

center = (c_w // 2, c_h // 2)
angle = 45

M = cv2.getRotationMatrix2D(center, angle, 1.0)

image_c_rotated = cv2.warpAffine(
    image_c_cropped,
    M,
    (c_w, c_h),
    flags=cv2.INTER_NEAREST,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0
)

show_image(cv2.cvtColor(image_c_rotated, cv2.COLOR_BGR2RGB))

# 2x Upsampling using Lanczos interpolation

scale = 2
image_c_upsampled = cv2.resize(
    image_c_rotated,
    (c_w * scale, c_h * scale),
    interpolation=cv2.INTER_LANCZOS4
)

# Display upsampled image

show_image(cv2.cvtColor(image_c_upsampled, cv2.COLOR_BGR2RGB))


