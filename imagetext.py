# coding: utf-8
import numpy as np
import cv2
from pytesseract import *
import re
import pandas as pd
import glob


data1 = []
dataname= pd.DataFrame(columns=['file(time)','product','cleantext','Accuracy'])
resultfile=[]
resuttcleantext=[]
resultproduct=[]
resultper=[]
def data_frame():
    #가지고 있는 파일의 이름정보 빼내기
    for num in range(1, 63):
        #63개의 파일 이름 다 빼오기
        path = ('output\\output%d.0.csv' % num)
        df = pd.read_csv(path, engine='python')
        dataname = df['Item Name']
        dataname = dataname.tolist()
        for i in range(0, len(dataname)):
            data1.append(dataname[i])



def ngram(s, num):
    res = []
    slen = len(s) - num + 1
    for i in range(slen):
        ss = s[i:i + num]
        res.append(ss)
    return res


def diff_ngram(sa, sb, num):
    a = ngram(sa, num)
    b = ngram(sb, num)
    cnt = 0
    for i in a:
        for j in b:
            if i == j:
                cnt += 1
    return cnt / len(a)


def Classification(text):
    global text2
    futext = ''
    k=0
    per=0
    for num in range(len(data1)):

        if len(text)<2 or len(data1[num]):
            #글자수가 2자리 미만인것은 pass
            pass
        elif len(text) == 2 or len(data1[num]) == 2:
            #글자수가 둘중 하나라도 2자리면 글자 2칸씩 비교
            per= diff_ngram(text, data1[num], 2)
        else:
            # 인식된 글자와 제품명3자리씩 비교
            per = diff_ngram(text, data1[num], 3)

        if per > 0.4:
            #정확도40프로 이상 중 젤 높은 값 저장
            if k <= per:
                futext = data1[num]
                k = per

    if futext == '':
        futext = "없음"
        k = 0

    resultproduct.append(futext)
    resultper.append(int(k * 100))
    resuttcleantext.append(text)


def Text_Recognition(frame):
    text = image_to_string(frame, lang='kor+eng')
    # 글자 검출

    text = re.sub('[ㄱ-ㅎㅏ-ㅣ]', '', text)
    # 잘못인식된 모음,자음만 있는것 제거
    text = re.sub('[-=.!_|<>Ｌ.「+#"~;^×@()[ㆍ『%*/”?“:$,{}|]', '', text)
    # 특수문자 제거
    if text == '':

        resultproduct.append('글자인식불가')
        resultper.append('글자인식불가')
        resuttcleantext.append('글자인식불가')


    else:

        Classification(text)


def ROI(img, vertices, color3=(255, 255, 255), color1=255):
    # 관심영역 설정 함수
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        color = color3
    else:
        color = color1

    cv2.fillPoly(mask, vertices, color)
    ROI_image = cv2.bitwise_and(img, mask)

    return ROI_image


def frame_textsave(dirname):

    images = glob.glob(dirname + "\\*.jpg")
    # dirname디렉토리에서 jpg가 들어있는 이름들을 리스트화
    data_frame()
    # csv파일에서 이름정보를 가져온다.
    images.sort()

    # 이미지 파일 이름을 정렬
    for image in images:
        # 한 사진마다 실행

        img = cv2.imread(image, 0)

        h, w = img.shape[:2]

        # vertices1 = np.array([[(0, h * 13 / 16), (0, h * 1 / 4), (w, h * 1 / 4), (w, h * 13 / 16)]], dtype=np.int32)
        # 1.자막이 없는경우
        vertices1 = np.array([[(0, h), (0, h * 1 / 8), (w, h * 1 / 8), (w, h)]], dtype=np.int32)
        # 2.자막이 있는경우
        img = ROI(img, vertices1)
        # 관심영역 설정
        ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
        # 이미지이진화
        img = cv2.resize(img, (4000, 2560))
        # 글자인식을 위한 이미지 확대
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # 모폴로지연산

        Text_Recognition(img)
        #파일이름 정리
        imagename = image.replace('.jpg', "")
        imagename = imagename.replace('\\', '')
        imagename = imagename.replace(dirname, '')
        resultfile.append(imagename)

    dataname['file(time)'] = resultfile
    dataname['product'] = resultproduct
    dataname['Accuracy'] = resultper
    dataname['cleantext'] = resuttcleantext
    dataname.to_csv(dirname + '\\result.csv', index=False)
    #결과 result.csv파일로 저장

frame_textsave('101010')
# 인덱스 =디렉토리 이름