import matplotlib.pyplot as plt
from matplotlib import rcParams

def draw_circ(i,images,boxes,centers):
    print("Image id:", i)
    #print("Features:", hands[i])
    #c = box[i]
    img = images[i]
    x, y = centers[i]
    box = boxes[i]
    print("Box: %s" % (box,))
    print("Center: (%d,%d)" % (x,y))
    plt.title("circ%d: open=(%d,%d)" % (i,x,y), fontsize=18, fontweight='bold', y=1.02)
    ticks=[0,8,16,24,32,40, 48]
    plt.xticks(ticks, fontsize=12)
    plt.yticks(ticks, fontsize=12)
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.plot(y, x, 'or')
    plt.grid()

rcParams['figure.figsize'] = 9,9
def draw_group(imgs, cntrs,rows=4, cols=4):
    for i in range(0, rows*cols):
        ax = plt.subplot(rows, cols, i+1)
        img = imgs[i]
        x, y = cntrs[i]
        plt.title("circ%d: open=(%d,%d)" % (i,x,y), fontsize=8, fontweight='bold', y=1.02)
        ticks=[0,8,16,24,32,40, 48]
        plt.xticks(ticks, fontsize=7)
        plt.yticks(ticks, fontsize=7)
        plt.imshow(img, cmap='gray', interpolation='none')
        # adding a red marker for highlighting the open
        plt.plot(y, x, '.r', linewidth=0)
        plt.subplots_adjust(wspace=0.5, hspace=0.1) 
