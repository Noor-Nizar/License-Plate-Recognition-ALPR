## Converts from xmin, ymax, lp_width, lp_height to normalized xcenter, ycenter, lp_width, lp_height
def normalized_coordinates(width, height, xmin, ymax, lp_width, lp_height):
    xmin, lp_width = xmin / width, lp_width / width
    ymax, lp_height = ymax / height, lp_height/ height
    
    lp_width = lp_width
    lp_height = lp_height
    xcenter = xmin+(lp_width/2)
    ycenter = ymax+(lp_height/2)
    
    return xcenter, ycenter, lp_width, lp_height