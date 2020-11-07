import pandas as pd
import cv2
import os

#########################################################################
#                         set parameters                                #
#########################################################################

# final size of images
px = 244

file_path = os.path.dirname(os.path.realpath(__file__))

# location of csv data 
labels_paths = [(file_path+'/labels/labels_unsplash.csv'),(file_path+'/labels/labels_nocaps.csv')]

# location of corresponding images files
image_paths = [(file_path+'/images/raw_images/unsplash/'), (file_path+'/images/raw_images/nocaps/')]

# path for norm images
final_img_path = (file_path+'/images/norm_images/')

# name and path of final csv
final_csv_name = r'labels/train_labels.csv'


#########################################################################
#                           main code                                   #
#########################################################################

# create new dataframe to save the new focus points 
final_df = pd.DataFrame(columns=['name','x_p','y_p'])

# iterate through directories
for csv_path, img_path in zip(labels_paths,image_paths):

  df = pd.read_csv(csv_path, names=['name','x_p','y_p'], header=None)

  for i, file_name in enumerate(df['name']):
    original_img = cv2.imread((img_path+file_name))
    if original_img is None:
        print("Error: {} not found".format(file_name))
        continue

    # fill with black to create square img
    width, height, color = original_img.shape
    
    # check if padding is needed and calculate how much
    if not width == height:
        p_height, p_width = [0, int((height-width)/2)] if width < height else [int((width - height)/2), 0]
        squared_img = cv2.copyMakeBorder(original_img, p_width, p_width, p_height, p_height, cv2.BORDER_CONSTANT, value=[0,0,0])
    else:
        squared_img = original_img
        p_height = 0
        p_width = 0

    # resize image to given px
    scaled_img = cv2.resize(squared_img, (px, px), interpolation = cv2.INTER_AREA)

    #bgr_img = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2BGR)
    # save image in new location
    cv2.imwrite((final_img_path+file_name), scaled_img)

    # save new x and y position of focus point
    x_focus_original = (df['x_p'][i]*width) + p_width
    y_focus_original = (df['y_p'][i]*height) + p_height

    px_original = max(width,height)
    norm_px_x = (x_focus_original * px) / px_original
    norm_px_y = (y_focus_original * px) / px_original

    final_df = final_df.append({
      'name': df['name'][i], 
      'x_p': x_focus_original, 
      'y_p': y_focus_original
    },ignore_index=True)

final_df.to_csv(final_csv_name)