import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Membaca dan Menampilkan Gambar
def read_and_display_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Membaca gambar dalam mode RGB
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Mengubah gambar menjadi grayscale
    return image, image_gray

# 2. Menampilkan Histogram Citra
def display_histogram(image):
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title("Histogram")

# 3. Konvolusi Spasial Citra
def apply_convolution(image, kernel):
    convolved_image = cv2.filter2D(image, -1, kernel)
    return convolved_image

# Main function
if __name__ == "__main__":
    image_path = r'D:\SEMESTER 5\PENGOLAHANCITRADIGITAL\Tugasindividu\Tugas2\download.jpeg'
    # Ganti dengan path file gambar Anda
    
    # Membaca dan menampilkan gambar
    image_rgb, image_gray = read_and_display_image(image_path)
    
    # Menampilkan histogram citra
    display_histogram(image_gray)
    
    # Membuat kernel/mask/filter untuk konvolusi
    mask1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    mask2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # Menerapkan konvolusi dengan kernel baru
    convolved_edge1 = apply_convolution(image_gray, mask1)
    convolved_edge2 = apply_convolution(image_gray, mask2)
    
    # Menampilkan gambar dan hasil konvolusi
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
    plt.title('Citra RGB')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Citra Gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(convolved_edge1, cmap='gray')
    plt.title('Extract Edge Kernel 1')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(convolved_edge2, cmap='gray')
    plt.title('Extract Edge Kernel 2')
    plt.axis('off')
    
    plt.show()
