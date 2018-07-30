# label2crop

parse_label.py provide four functions to parse MIT labelMe xml into mask.
make_datalist.ipynb for generate dataset txt list.

#### xml2dict function: 
parse MIT LabelMe annotation into polygon list.
#### merge_x_y_pts function:
prepare polygon format needed by PIL.
#### poly2mask function: 
return a mask image, indicate the bounding box area.
#### mask_image_crop function:
crop mask area from original iamge


