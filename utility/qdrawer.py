import matplotlib

def draw_axes():
    # dummy points for zooming out
    points = [ [1.3,0], [0,1.3], [-1.3,0], [0,-1.3] ]
    # coordinates for the axes
    arrows = [ [1.1,0], [0,1.1], [-1.1,0], [0,-1.1] ]

    # drawing dummy points
    for p in points: matplotlib.pyplot.plot(p[0],p[1]+0.2)
    # drawing the axes
    for a in arrows: matplotlib.pyplot.arrow(0,0,a[0],a[1],head_width=0.04, head_length=0.08)

def draw_unit_circle():
    unit_circle= matplotlib.pyplot.Circle((0,0),1,color='black',fill=False)
    matplotlib.pyplot.gca().add_patch(unit_circle)

def draw_datapoint(x,y, name=None,color='blue', centroid=False):
    if centroid:
        matplotlib.pyplot.plot(x,y, marker='*', markersize=20, markeredgecolor='k', color=color)
    else:
        matplotlib.pyplot.plot(x,y, marker='o', markersize=7, color=color)
    x2 = 1.15 * x
    y2 = 1.15 * y
    #matplotlib.pyplot.text(x2,y2,name)

def draw_quantum_state(x,y,name, color='blue'):
    # shorten the line length to 0.92
    # line_length + head_length should be 1
    x1 = 0.92 * x
    y1 = 0.92 * y
    matplotlib.pyplot.arrow(0,0,x1,y1,head_width=0.04,head_length=0.08,color=color)
    x2 = 1.15 * x
    y2 = 1.15 * y
    matplotlib.pyplot.text(x2,y2,name)

def draw_qubit():
    # draw a figure
    matplotlib.pyplot.figure(figsize=(6,6), dpi=60)
    # draw the origin
    #matplotlib.pyplot.plot(0,0,'ro') # a point in red color
    # drawing the axes by using one of our predefined functions
    draw_axes()
    # drawing the unit circle by using one of our predefined functions
    draw_unit_circle()
    # drawing |0>
    #matplotlib.pyplot.plot(1,0,"o")
    matplotlib.pyplot.text(1.05,0.05,"|0>")
    # drawing |1>
    #matplotlib.pyplot.plot(0,1,"o")
    matplotlib.pyplot.text(0.05,1.05,"|1>")
    # drawing -|0>
    #matplotlib.pyplot.plot(-1,0,"o")
    matplotlib.pyplot.text(-1.2,-0.1,"-|0>")
    # drawing -|1>
    #matplotlib.pyplot.plot(0,-1,"o")
    matplotlib.pyplot.text(-0.2,-1.1,"-|1>")
