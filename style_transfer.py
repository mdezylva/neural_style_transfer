import os
# Insert your prefered path here
home_dir = '/home/mitchell/Documents/projects/cnn_style_transfer/images/'
style_dir = '/home/mitchell/Documents/projects/cnn_style_transfer/images/style/'
if not os.path.exists(style_dir):
    os.makedirs(style_dir)