import os
from pyecharts.charts import Bar
import os.path
import xml.dom.minidom
import xml.etree.cElementTree as et
from scipy.ndimage import measurements
from matplotlib import pyplot as plt

path = "D:/DLmode/Data/NEU-DET/Statistic"
files = os.listdir(path)
s = []

square_list = []
side_list = []


# =============================================================================
# extensional filename
# =============================================================================
def file_extension(path):
    return os.path.splitext(path)[1]


for xmlFile in files:
    if not os.path.isdir(xmlFile):
        if file_extension(xmlFile) == '.xml':
            print(os.path.join(path, xmlFile))
            tree = et.parse(os.path.join(path, xmlFile))
            root = tree.getroot()
            filename = root.find('filename').text
            #            print("filename is", path + '/' + xmlFile)
            for Object in root.findall('object'):
                #                name=Object.find('name').text
                #                print("Object name is ", name)
                bndbox = Object.find('bndbox')
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text
                square = (int(ymax) - int(ymin)) * (int(xmax) - int(xmin))
                square_list.append(square)
                #                print(xmin,ymin,xmax,ymax)
                #print(square)

            for Object in root.findall('size'):
                side1 = Object.find('width').text
                side2 = Object.find('height').text
                side_list.append(int(side1))
                side_list.append(int(side2))
                #                print(xmin,ymin,xmax,ymax)
                #print(square)
# print("square is ", square_list)

# =============================================================================
# 画出直方图
# =============================================================================
#num = 40000  # 最大面积
# histogram1 = measurements.histogram(square_list, 1, num, 10)  # 直方图
# histogram1 = list(map(int, histogram1))  # 转换成 int 格式
plt.hist(
    x=square_list,
    bins=20,
    color='steelblue',
)
plt.xlabel('面积')
plt.ylabel('频数')
plt.title('面积分布')
plt.show()