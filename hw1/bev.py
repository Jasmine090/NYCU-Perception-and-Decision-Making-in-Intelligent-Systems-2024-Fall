import cv2
import numpy as np

points = []
sensor_height_bev = 2500  # mm
sensor_height = 1000

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape
        self.points = points
        self.focal = self.width / (2*np.tan(np.pi/4))
        
    def ex_mat(self, alpha, beta, gamma, tx, ty, tz):
    	r11 = np.cos(alpha)*np.cos(beta)
    	r12 = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
    	r13 = np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)
    	r21 = np.sin(alpha)*np.cos(beta)
    	r22 = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
    	r23 = np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)
    	r31 = -np.sin(beta)
    	r32 = np.cos(beta)*np.sin(gamma)
    	r33 = np.cos(beta)*np.cos(gamma)
    	ex = np.array([[r11, r12, r13, tx],
    				  [r21, r22, r23, ty],
    				  [r31, r32, r33, tz],
    				  [0, 0, 0, 1]])
    	return ex

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """

        ### TODO ###
        in_mat = np.array([[self.focal, 0, 256],
        				   [0, self.focal, 256],
        				   [0, 0, 1]])
        					
        new_pixels = []
        for i in self.points:
        	i.append(1)
        	XYZ = np.dot(np.linalg.inv(in_mat), np.reshape((np.array(i) * sensor_height_bev), (3, 1)))
        	print('XYZ: ', XYZ)
        	ex_mat = self.ex_mat(theta, phi, gamma, dx, dy, dz) 
        	#print("transform: ", ex_mat)
        	tmp = np.array([XYZ[0][0], XYZ[1][0], sensor_height_bev, 1])
        	trans = np.dot(ex_mat, np.reshape(tmp, (4, 1)))
        	
        	convert_back = np.reshape(np.dot(in_mat, trans[:3]), (3))
        	convert_back = convert_back / convert_back[2]
        	convert_back = np.around(convert_back, decimals=0).astype(int)
        	#print(convert_back)
        	new_pixels.append([convert_back[0], convert_back[1]])
        
        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)
    cv2.imwrite('img.png', img)


if __name__ == "__main__":

    pitch_ang = -np.pi/2 #-90

    front_rgb = "f1.png"
    top_rgb = "f1_bev.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(gamma=pitch_ang, dy=sensor_height - sensor_height_bev)
    projection.show_image(new_pixels)
