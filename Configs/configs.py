from enum import Enum, auto


IMG_SHAPE = (600, 800)

PARAMS = {
    "min_dist_y1" : 290 - 40,
    "max_dist_y1" : 290 + 40,
    "min_dist_y2" : 290 - 60,
    "max_dist_y2" : 290 + 60,
    "min_dist_y3" : 150,
    "max_dist_y3" : 400,
    
    'VALIDATE_Y2': 0.20,
    'VALIDATE_Y3': 0.35,
    'MAX_CACHE_LEN':0,
    'TANGENT':10,
    "MIN_WINDOWS_TO_DETECT":2,
    "MIN_ROADBED_AREA": 10000,
    "RANGE_OF_VISION" :380,
}
# class CLASSES(Enum):
#     SOLID = 7
#     DASH = 8
#     SOLID_SOLID = 3
#     DASH_SOLID = 4
#     SOLID_DASH = 5
#     #WAYSIDE = 6
#     NONE = -1
#     EMPY = 0


class CLASSES(Enum):
    SOLID = 5
    DASH = 6
    SOLID_SOLID = 2
    DASH_SOLID = 3
    SOLID_DASH = 4
    #WAYSIDE = 1
    NONE = -1
    EMPY = 0

class WINDOW_PARAMS(Enum):
    nb_windows = 20
    margin = 40
    minpix = 50
    min_lane_pts  = 10
    MAX_RESIDUAL_FITLINE = 120000
    MAX_RESIDUAL_FITLINE_SOLID_SOLID = 200000
    min_share_toClassify = 3
    MAX_SOLID_WIN_TO_DETECT_SOLID = 7
    MAX_SS_WIN_TO_DETECT_SS = 7
    MIN_E_WIN_TO_DETECT_D = 7
    MAX_NO_E_WIN_TO_DETECT_D = 5
    MAX_dx = 20
    win_colors = {
        CLASSES.SOLID:(0,255,0),
        CLASSES.SOLID_SOLID:(0,255,0),
        CLASSES.DASH:(255,0,0),
        CLASSES.DASH_SOLID :(0,0,255),
        CLASSES.SOLID_DASH:(0,0,255),
        CLASSES.NONE :(255,255,255),
        CLASSES.EMPY:(255,255,255)
    }

class ROADBED_PARAMS(Enum):
    min_lane_pts = 5 # min number of 'hot' pixels needed to fit a 2nd order polynomial
    slice_of_Y_max = 600 #roadbed fited on slice bin[slice_of_Y_min:slice_of_Y_max,:]
    slice_of_Y_min = 100
    
    
    adjustment_width = 40
    adjustment_y_min = 300
    
HIST_PARAMS = {
    'roi_x_start' : 150, # start x point for hist search
    'roi_x_end':650,    #  end x point for hist search
}
class SIDES(Enum):
    left = auto()
    right = auto()
WINDOW_HEITH = int(IMG_SHAPE[0] / WINDOW_PARAMS.nb_windows.value)
