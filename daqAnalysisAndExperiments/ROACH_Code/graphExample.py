"""
Demonstrates some customized mouse interaction by drawing a crosshair that follows 
the mouse.
"""


import numpy as np
import pyqtgraph as pg



labelFont = pg.Qt.QtGui.QFont()
labelFont.setFamily('Arial')
labelFont.setPointSize(12)

#generate layout
app = pg.mkQApp("Data Viewer v0.0")
#win = pg.GraphicsLayoutWidget(show=True)
#win.setWindowTitle('Data Viewer v0.0')
#label = pg.LabelItem(justify='right')
labelx = pg.TextItem(color = 'white')
labely1 = pg.TextItem(color = 'green')
labely2 = pg.TextItem(color = 'red')

#win.addItem(label)
view = pg.GraphicsView()
#view.addItem(label)
view.addItem(labelx)
view.addItem(labely1)
view.addItem(labely2)

layout = pg.GraphicsLayout(border=(100,100,100))
view.setCentralItem(layout)
view.show()
view.setWindowTitle('Data Viewer v0.0')
view.resize(800,600)


p1 = layout.addPlot(colspan = 1)

# customize the averaged curve that can be activated from the context menu:
p1.avgPen = pg.mkPen('#FFFFFF')
p1.avgShadowPen = pg.mkPen('#8080DD', width=10)

p3 = layout.addPlot(colspan = 1)
layout.nextRow()
p2 = layout.addPlot(colspan = 2, border=(50,0,0))

region = pg.LinearRegionItem()
region.setZValue(10)
# Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this 
# item when doing auto-range calculations.
p2.addItem(region, ignoreBounds=True)

#pg.dbg()
p1.setAutoVisible(y=True)


#create numpy arrays
#make the numbers large to show that the range shows data from 10000 to all the way 0
data1 = 10000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
data2 = 15000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
label_font = {'color':'white', 'font-size':'15px', 'font':'Arial'}

p1DataA = p1.plot(data1, pen="r")
p1DataB = p1.plot(data2, pen="g")
p1.setLabel('bottom', 'Frequency', 'Hz', **label_font)
p1.setLabel('left', 'Power', 'dBm', **label_font)

p2.setLabel('bottom', 'Frequency (MHz)', 'Hz', **label_font)
p2.setLabel('left', 'Power', 'dBm', **label_font)

p3.setLabel('bottom', 'Frequency (MHz)', 'Hz', **label_font)
p3.setLabel('left', 'Power', 'dBm', **label_font)

p2d = p2.plot(data1, pen="w")
# bound the LinearRegionItem to the plotted data
region.setClipItem(p2d)

def update():
	region.setZValue(10)
	minX, maxX = region.getRegion()
	p1.setXRange(minX, maxX, padding=0)    

def updateData():
	data1 = 10000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
	data2 = 15000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
	p1DataA.setData(data1)
	p1DataB.setData(data2)
	p2d.setData(data2)

region.sigRegionChanged.connect(update)

def updateRegion(window, viewRange):
	rgn = viewRange[0]
	region.setRegion(rgn)

p1.sigRangeChanged.connect(updateRegion)

region.setRegion([1000, 2000])

#cross hair
vLine = pg.InfiniteLine(angle=90, movable=False)
hLine = pg.InfiniteLine(angle=0, movable=False)
p1.addItem(vLine, ignoreBounds=True)
p1.addItem(hLine, ignoreBounds=True)


vb = p1.vb

def mouseMoved(evt):
	pos = evt[0]  ## using signal proxy turns original arguments into a tuple
	if p1.sceneBoundingRect().contains(pos):
		mousePoint = vb.mapSceneToView(pos)
		index = int(mousePoint.x())
		if index > 0 and index < len(data1):
			#label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
			labelx.setText('x: ' + str(round(mousePoint.x(), 1)))
			labelx.setPos(50, 10)
			labely1.setText('y1: ' + str(round(data1[index], 1)))
			labely1.setPos(130, 10)
			labely2.setText('y2: ' + str(round(data2[index], 1)))
			labely2.setPos(230, 10)
			labelx.setFont(labelFont)
			labely1.setFont(labelFont)
			labely2.setFont(labelFont)
			#"<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), 
		vLine.setPos(mousePoint.x())
		hLine.setPos(mousePoint.y())



proxy = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
timer = pg.QtCore.QTimer()
timer.timeout.connect(updateData)
timer.start(500)
#p1.scene().sigMouseMoved.connect(mouseMoved)


if __name__ == '__main__':
	pg.exec()
