import pandas as pd
import cv2

labels_path = './labels/labels_nocaps.csv'
df = pd.read_csv(labels_path, names=['name','x_p','y_p'], header=None)

# Load image
pxl = 244
focus_points = []
df2 = df
for i, file_name in enumerate(df['name']):
  print(file_name)
  original_img = cv2.imread(('./images/raw_images/nocaps/'+file_name))
  # fill with black to create square img
  width, height, color = original_img.shape
  p_height, p_width = [0, int((height-width)/2)] if width < height else [int((width - height)/2), 0]
  squared_img = cv2.copyMakeBorder(original_img, p_width, p_width, p_height, p_height, cv2.BORDER_CONSTANT, value=[0,0,0])
  # resize
  scaled_img = cv2.resize(squared_img, (pxl, pxl), interpolation = cv2.INTER_AREA)
  #bgr_img = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2BGR)
  cv2.imwrite(("images/norm_images/" + file_name), scaled_img)

  # save new x and y position of focus point
  x_p = df['x_p'][i]
  y_p = df['y_p'][i]
  x_focus_original = (x_p*width) + p_width
  y_focus_original = (y_p*height) + p_height
  # rescale
  pxl_original = max(width,height)
  df2['x_p'][i] = (x_focus_original * pxl) / pxl_original
  df2['y_p'][i] = (y_focus_original * pxl) / pxl_original

df2.to_csv(r'labels/norm_labels_nocaps.csv')