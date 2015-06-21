# -*-coding:utf-8 -*-
'''
author:  hrwhipser
date   :  2015.06
'''
import numpy as np
import skimage.io, re
from skimage import transform, measure
from skimage.filter import threshold_otsu
import matplotlib.pyplot as plt
from sklearn import svm
from math import floor, ceil
from correctWord import CorrectWord


sameSize = 32
wordHigh = 10
wordSpace = 7
lineSpace = 25
trainData_row = 6
trainData_col = 13
trainData_num = trainData_row * trainData_col

# 图像二值化
def readImage(fileName):
    image = skimage.io.imread(fileName, as_grey=True)
    threshold = threshold_otsu(image)
    return image < threshold

def toTheSameSize(image):
    def findMinRange(image):
        up, down, left, right = image.shape[0], 0, image.shape[1], 0
        for i in xrange(image.shape[0]):
            for j in xrange(image.shape[1]):
                if image[i][j]:
                    up, down, left, right = min(up, i), max(down, i), min(left, j), max(right, j)
        return up, down, left, right
    up, down, left, right = findMinRange(image)
    image = image[up:down, left:right]
    image = transform.resize(image, (sameSize, sameSize))
    return image

def getTrainData(image):
    cells = np.array([np.hsplit(row, trainData_col) for row in np.vsplit(image, trainData_row)])
    cells = np.reshape(cells, (trainData_num, cells.shape[ 2 ], cells.shape[3]))
    train_input = []
    for i in xrange(trainData_num):
        test = toTheSameSize(cells[i])
        ax = plt.subplot(trainData_row, trainData_col, i + 1)
        ax.set_axis_off() 
        ax.imshow(test)
        train_input.append(test)
    train_input = np.array(train_input).reshape(trainData_num, sameSize * sameSize)
    
    desired_output = []
    for i in xrange(26):    desired_output.append(chr(i + ord('A')))
    for i in xrange(26):    desired_output.append(chr(i + ord('a')))
    for i in xrange(10):    desired_output.append(i)
    for i in r'<>\,.&@?!#%+-*/=': desired_output.append(i)
    
    # print desired_output
    desired_output = np.array(desired_output)
    return train_input, desired_output

class rangeData:
    def __init__(self, up, down, left, right):
        self.up = floor(up)
        self.down = ceil(down)
        self.left = floor(left)
        self.right = ceil(right)
    
    def __str__(self):
        return ' '.join(str(i) for i in [self.left])
    
    __repr__ = __str__
    
    def getInfo(self):
        return self.up, self.down, self.left, self.right

def getImageWords(image):
    # 删除包含的区域，返回正确的区域
    def removeRange(cells):
        # b in a
        def rangeInclude(a, b):
            return b.up >= a.up and b.down <= a.down and b.left >= a.left and b.right <= a.right
        
        def rangeCmp(rangeDataA, rangeDataB):
            return -1 if rangeDataA.down - rangeDataA.up < rangeDataB.down - rangeDataB.up else 1
               
        cells.sort(rangeCmp)
        n = len(cells)
        ok = [True] * n
        for i in xrange(1, n):
            for j in xrange(i):
                if ok[j] and rangeInclude(cells[i], cells[j]):
                    ok[j] = False
        newCells = [cells[i] for i in xrange(n) if ok[i]]
        return newCells        
    # 零散的字母转为一个单词
    def charaterToWord(cells): 
        def theSameLine(rangeDataA, rangeDataB):
            return abs(rangeDataA.up - rangeDataB.up) < lineSpace and abs(rangeDataA.down - rangeDataB.down) < lineSpace
      
        def mycmp(rangeDataA, rangeDataB):
            if theSameLine(rangeDataA, rangeDataB):
                return -1 if rangeDataA.left < rangeDataB.left else 1
            return -1 if rangeDataA.up < rangeDataB.up else 1
        
        cells.sort(mycmp)
        lines , wordCnt , lineCnt = [] , 0 , 0
        for i , c in enumerate(cells):
            if not i:
                lines.append([[]])
            elif theSameLine(c, lines[lineCnt][wordCnt][-1]):
                if c.left - lines[lineCnt][wordCnt][-1].right >= wordSpace:
                    wordCnt += 1
                    lines[lineCnt].append([])
            else:
                lineCnt += 1
                wordCnt = 0
                lines.append([[]])
            lines[lineCnt][wordCnt].append(c)
        return lines  
    
    contours = measure.find_contours(image, 0.8)
    cells = []
    for contour in contours:
        up, down, left, right = min(contour[:, 0]), max(contour[:, 0]), min(contour[:, 1]), max(contour[:, 1])
        if down - up >= wordSpace or right - left >= wordSpace:
            cells.append(rangeData(up, down, left, right))
    
    cells = removeRange(cells)
    lines = charaterToWord(cells)
    print len(cells)
    
    totWords = sum(len(line) for line in lines)
    print totWords
    return lines

plt.gray()

image = readImage('./data/letter.jpg')
# print image.shape
train_input, desired_output = getTrainData(image)

clf = svm.SVC()
clf.fit(train_input, desired_output)

image = readImage('./data/test2.jpg')

lines = getImageWords(image)
plt.figure(2)

res = []
wordCnt = 0
for words in lines:
    line = []
    for i , word in enumerate(words):
        ans , cnt = [] , 1
        for j , c in enumerate(word):
            nw, sw, ne, se = c.getInfo()
            cur = toTheSameSize(image[nw:sw, ne:se])
            cnt += 1
            ans.append(clf.predict(cur.ravel())[0])
        line.append(''.join(ans)) 
        wordCnt += 1
    res.append(line)
    print ' '.join(line)


print '============after correct============'
sign = '<>,.&@?!#%+-=*\/'
num = re.compile(r'[+-]?\d+$')

cw = CorrectWord()
for line in res:
    ans = []
    for word in line:
        if word in sign or num.search(word): ans.append(word)
        elif word[0] in sign:  ans.append(word[0] + cw.correct(word[1:])) 
        elif word[-1] in sign: ans.append(cw.correct(word[:-1]) + word[-1])    
        else: ans.append(cw.correct(word))
    print ' '.join(ans).lower()