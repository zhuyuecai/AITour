import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data
import matplotlib.animation as animation
import numpy as np
from time import sleep

def get_image(ax,img_path,zoom=0.25):
    fn = get_sample_data(img_path, asfileobj=False)
    arr_img = plt.imread(fn, format='png')
    
    imagebox = OffsetImage(arr_img, zoom)
    imagebox.image.axes = ax
    return imagebox

#plt.ion()
size=8
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, autoscale_on=True, xlim=(0, size), ylim=(0, size))


# set your ticks manually
ax.xaxis.set_ticks(range(1,size+1))
ax.grid(True,which='both')


playerbox = get_image(ax, "/Users/pinghuanliu/github/AITour/wumpus_world/img/player.png")
pathbox = get_image(ax, "/Users/pinghuanliu/github/AITour/wumpus_world/img/facing_1.png",1.5)

path = []
for i in range(1,4):
    ax.add_artist(AnnotationBbox(pathbox, [i-0.5,0.5],
                    xybox=(0, 0),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0))

ax.add_artist(AnnotationBbox(playerbox, [4-0.5,0.5],
                    xybox=(0, 0),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0)
)

#fig.draw()
plt.show()
#plt.pause(2mmmmmmmm)
#Splt.draw()
    #plt.get_current_fig_manager().canvas.draw_idle()
    #sleep(0.5)

#line, = ax.plot([], [], 'o-', lw=2)
plt.show()