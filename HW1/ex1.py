import cv2 
import matplotlib.pyplot as plt
import numpy as np


# Load an image from file as function
def load_image(image_path):
    """
    Load an image from file, using OpenCV
    """
    img = cv2.imread(image_path)
    return img
    pass

# Display an image as function
def display_image(image, title="Image"):
    """
    Display an image using matplotlib. Rembember to use plt.show() to display the image
    """

    if image is not None:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    pass


# grayscale an image as function
def grayscale_image(image):
    """
    Convert an image to grayscale. Convert the original image to a grayscale image. In a grayscale image, the pixel value of the
    3 channels will be the same for a particular X, Y coordinate. The equation for the pixel value
    [1] is given by:
        p = 0.299R + 0.587G + 0.114B
    Where the R, G, B are the values for each of the corresponding channels. We will do this by
    creating an array called img_gray with the same shape as img
    """
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
   
    gray = 0.299*r + 0.587*g + 0.114*b
    
    image_gray = gray.astype(np.uint8)
    # if image is not None:
    #     image_gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    # if image is not None:
    #     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_gray
    pass


# Save an image as function
def save_image(image, output_path):
    """
    Save an image to file using OpenCV
    """
    cv2.imwrite(output_path, img)
    pass


# flip an image as function 
def flip_image(image):
    """
    Flip an image horizontally using OpenCV
    """
    if image is not None:
        img_flipped = cv2.flip(image, 1)
    
    return img_flipped
    pass 


# rotate an image as function
def rotate_image(image, angle):
    """
    Rotate an image using OpenCV. The angle is in degrees
    """
    if image is not None:
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rotated = cv2.warpAffine(image, M, (w, h))
        return img_rotated
    pass


if __name__ == "__main__":
    # Load an image from file
    img = load_image("D:\\XLA\\hw1\\uet.png")

    # Display the image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "D:\\XLA\\hw1\\lena_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "images/lena_gray_rotated.jpg")

    # Show the images
    plt.show() 
