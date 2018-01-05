from watson_developer_cloud import VisualRecognitionV3 as VisualRecognition
from watson_developer_cloud import TextToSpeechV1 as TextToSpeech
from tkinter import font
#from picamera import PiCamera

import os
import math
import numpy as np
import cv2 as cv
import time
import json
import pyaudio
import wave
import tkinter as tk

ID = json.load(open('credentials'+os.sep+'watson_credentials.json', 'r'))['visual']['classifier']

''' Classify a picture using the visual recognition instance

    @params
        visual: visual recognition instance.
        image: (File) Photo to classify.
        image_name: Name of the photo.
        params (optional): JSON with the custom classifier ID and other things less
                           important and optionals. Default: IBM classificator
        language (optional): ISO code for the JSON language, e.g. "es" -> "spanish". Default: "en" -> "english"
    @return
        JSON with the results.
'''
def classifyCoins(visual, image, image_name, params=None, language='en'):
    return visual.classify(images_file=image, parameters=params, \
                           accept_language=language, images_filename=image_name)

''' Train a custom classifier to identify coins (euros)

    Print the classifier data

    @params
        visual: visual recognition instance
    @return
        None
'''
def trainCoinClass(visual):
    print("Training")
    with open('train'+os.sep+'zips'+os.sep+'newCoin.zip', 'rb') as coin,\
            open('train'+os.sep+'zips'+os.sep+'euro1.zip', 'rb') as euro1, \
            open('train'+os.sep+'zips'+os.sep+'euro2.zip', 'rb') as euro2, \
            open('train'+os.sep+'zips'+os.sep+'cent1.zip', 'rb') as cent1, \
            open('train'+os.sep+'zips'+os.sep+'cent2.zip', 'rb') as cent2, \
            open('train'+os.sep+'zips'+os.sep+'cent10.zip', 'rb') as cent10, \
            open('train'+os.sep+'zips'+os.sep+'cent50.zip', 'rb') as cent50, \
            open('train'+os.sep+'zips'+os.sep+'cent5.zip', 'rb') as cent5, \
            open('train'+os.sep+'zips'+os.sep+'notCoins.zip', 'rb') as notCoins:
        print(visual.create_classifier('euros', coin_positive_examples=coin, euro1_positive_examples=euro1,\
                                                 euro2_positive_examples=euro2,cent1_positive_examples=cent1,\
                                                 cent2_positive_examples=cent2, cent5_positive_examples=cent5,\
                                                 cent10_positive_examples=cent10,\
                                                 cent50_positive_examples=cent50,negative_examples=notCoins))

''' Delete the custom classifier

    @params
        visual: visual recognition instance
    @return
        None
'''
def deleteClassifier(visual):
    print("Deleting")
    print(visual.delete_classifier(ID))
    print("Deleted")

''' Update the custom classifier (only for non-free accounts)

    @params
        visual: visual recognition instance
    @return
        None
'''
def updateClass(visual):
    pass
    #with open('train/moreDani1.zip', 'rb') as me,\
    #      open('train/notMoreDani1.zip', 'rb') as notMe:
    #        return visual.update_classifier(ID, daniel_positive_examples=me, negative_examples=notMe)

''' Check the custom classifier status.

    @params
        visual: visual recognition instance.
    @return
        None
'''
def statusClass(visual):
    print(json.dumps(visual.get_classifier(ID), indent=2))
    return None

''' Take a picture using PiCamera.

    @params
        camera: PiCamera instance.
        name (optional): name of the photo to save. Default value: "coin.jpeg"
        resolution (optional): resolution of the photo. Default value (HxW): 1024x720
    @return
        None
'''
def capture(camera, name="coin.jpeg", resolution=(1024,720)):
    camera.resolution = resolution
    #Photo is saved as "photo.jpeg"
    camera.capture(name, format="jpeg", quality=100, use_video_port=True)
    return None

''' Convert the picture to grayscale

    @params
        picture: nam eof the picture to convert
    @return
        grayscale image in OpenCV format
'''
def grayscale(picture):
    img = cv.imread(picture)
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

''' Calculate the money on the picture (only with coins and euros)

    @params
        visual: visual_recognition instance
        picture: name of the picture to process

    @return
        Dict with the results or None if can't find coins
'''
def detectCoins(visual, picture):
    print("Detecting coins")
    original = cv.imread(picture)
    copy = original.copy()

    paramClass = json.dumps({'classifier_ids': [ID]})
    #For some reason this doesn't works on Windows idk why :(
    coinResult = classifyCoins(visual, open(picture, 'rb'), picture, params=paramClass)

    if len(coinResult['images'][0]['classifiers']) == 0:
        coins_found = {'coins': {}, 'unknown': 0, 'total': 0.0}
        textToSpeech(coins_found)
        return coins_found

    for c in coinResult['images'][0]['classifiers'][0]['classes']:
        if c['class']=='coin' and c['class'] > 0.9:
            print("Classify as coins")
            # Grayscale to circle detecting
            imgGS = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)

            # Blur the image a little  to avoid false circles
            imgGS = cv.blur(imgGS, (3, 3), 0)

            uncanny = cv.Canny(imgGS, 75, 205)
            thresh = cv.adaptiveThreshold(uncanny.copy(), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv.THRESH_BINARY_INV, 5, 1)

            cont_img = thresh.copy()
            _, contours, hierarchy = cv.findContours(cont_img, cv.RETR_TREE,
                                                  cv.CHAIN_APPROX_SIMPLE)

            # Lists with tuples with values (x,y,radius)
            real_contoursG = []  # Green circles
            real_contoursB = []  # Blue circles
            real_hough = []  # Red circles

            for cnt in contours:
                #Get all the vertices of the contour
                approx = cv.approxPolyDP(cnt, 0.008 * cv.arcLength(cnt, True), True)
                #Get contour area
                area = cv.contourArea(cnt)
                #A circle has infinite vertices, but this is computer science,
                # so we will asume it has more than 12
                if len(approx) > 12 and area > 750:
                    (x, y), radius = cv.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    radius = int(radius)
                    if area > 8000:
                        real_contoursB.append((center, radius))
                    else:
                        real_contoursG.append((center, radius))

            # detect circles in the image
            circles = cv.HoughCircles(thresh, cv.HOUGH_GRADIENT, dp=1.10, minDist=80, \
                                      param1=200, param2=91, maxRadius=200, minRadius=80)

            # ensure at least some circles were found
            if circles is not None:
                # convert the (x, y) coordinates and radius of the circles to integer
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    real_hough.append(((x, y), r))

            num = 0
            real_hough, real_contoursB, real_contoursG = cleanResults(real_hough, real_contoursG, real_contoursB)

            all_coins = []
            all_coins.extend(real_hough)
            all_coins.extend(real_contoursG)
            all_coins.extend(real_contoursB)

            coins_found = {'coins': {}, 'unknown': 0, 'total': 0.0}

            # loop over the center (x,y) coordinates and radius of the circles
            for (c, r) in all_coins:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                left_x = c[0] - r - 25 if c[0] - r - 25 > 0 else 0
                top_y = c[1] - r - 25 if c[1] - r - 25 > 0 else 0
                bot_y = c[1] + r + 25 if c[1] + r + 25 < original.shape[0] else original.shape[0] - 1
                right_x = c[0] + r + 25 if c[0] + r + 25 < original.shape[1] else original.shape[1] - 1

                coin = original[top_y:bot_y, left_x:right_x]
                cv.imwrite('.tempCoin.jpeg', coin)

                coinResult = classifyCoins(visual, open('.tempCoin.jpeg', 'rb'), '.tempCoin.jpeg', params=paramClass)
                # To save the values
                coin_score = 0.0
                coin_name = ''

                # Check the type of coin
                for coin in coinResult['images'][0]['classifiers'][0]['classes']:
                    # Get the biggest value
                    if float(coin['score']) > 0.9 and float(coin['score']) > coin_score:
                        coin_score = float(coin['score'])
                        coin_name = coin['class']

                # Final results of the coin
                if coin_name != '':
                    value = 0.0
                    if coin_name not in coins_found['coins']:
                        coins_found['coins'][coin_name] = 0
                    coins_found['coins'][coin_name] += 1
                    if coin_name == 'euro1':
                        value = 1.0
                    elif coin_name == 'euro2':
                        value = 2.0
                    elif coin_name == 'cent1':
                        value = 1/100
                    elif coin_name == 'cent2':
                        value = 2/100
                    elif coin_name == 'cent5':
                        value = 5/100
                    elif coin_name == 'cent10':
                        value = 1/10
                    elif coin_name == 'cent50':
                        value = 5/10
                    coins_found['total'] += value
                    cv.putText(copy, ''+str(value), c, cv.FONT_HERSHEY_SIMPLEX,2,(0,255,255),2)
                else:
                    coins_found['unknown'] += 1

            # loop over the center (x,y) coordinates and radius of the circles
            # to draw the circles
            for (c, r) in real_hough:
                # draw the circle in the output image, then draw a rectangle
                #cv.rectangle(copy, (c[0] - 5, c[1] - 5), (c[0] + 5, c[1] + 5), (0, 128, 255), -1)
                cv.circle(copy, c, r, (0, 0, 255), 4)
                num += 1
            for (c, r) in real_contoursB:
                # draw the circle in the output image, then draw a rectangle
                #cv.rectangle(copy, (c[0] - 5, c[1] - 5), (c[0] + 5, c[1] + 5), (0, 128, 255), -1)
                cv.circle(copy, c, r, (255, 0, 0), 4)
                num += 1
            for (c, r) in real_contoursG:
                # draw the circle in the output image, then draw a rectangle
                #cv.rectangle(copy, (c[0] - 5, c[1] - 5), (c[0] + 5, c[1] + 5), (0, 128, 255), -1)
                cv.circle(copy, c, r, (0, 255, 0), 4)
                num += 1
            cv.putText(copy, 'Coins: ' + str(num), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv.putText(copy, 'Total: ' + str(coins_found['total']), (0, 120), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

            os.remove('.tempCoin.jpeg')
            print(coins_found)

            cv.namedWindow('before-after')
            before_after = np.concatenate((original, copy), axis=1)
            cv.imwrite('before-after.jpeg', before_after)
            cv.imshow('before-after', cv.resize(before_after, (before_after.shape[1] // 2, before_after.shape[0] // 2)))

            textToSpeech(coins_found)

            return coins_found
    #Not coins
    return None

def nothing(x):
    pass

''' Interactive detect circles GUI.
    Use Hough Circles and contours.

    Mostly for testing and debugging.

    @params
        visual: visual recognition instance
        picture: array of path to pictures to test
    @return:
        None
'''
def houghCircles(visual, picture):
    # Create a window
    cv.namedWindow('filtros', cv.WINDOW_NORMAL)
    cv.namedWindow('imagen')

    # create trackbars for treshold change
    cv.createTrackbar('kernel', 'filtros', 1, 10, nothing)
    cv.createTrackbar('dp', 'filtros', 10, 20, nothing)
    cv.createTrackbar('param1', 'filtros', 50, 300, nothing)
    cv.createTrackbar('param2', 'filtros', 50, 300, nothing)
    cv.createTrackbar('pict', 'filtros', 0, len(picture), nothing)
    cv.createTrackbar('blur', 'filtros', 0, 2, nothing)
    cv.createTrackbar('mode', 'filtros', 0, 2, nothing)
    cv.createTrackbar('clean', 'filtros', 0, 1, nothing)

    while True:
        # get current positions of four trackbars
        kern = cv.getTrackbarPos('kernel', 'filtros')
        dp = float(cv.getTrackbarPos('dp', 'filtros')) / 10
        param1 = cv.getTrackbarPos('param1', 'filtros')
        param2 = cv.getTrackbarPos('param2', 'filtros')
        pict = cv.getTrackbarPos('pict', 'filtros')
        blur = cv.getTrackbarPos('blur', 'filtros')
        mode = cv.getTrackbarPos('mode', 'filtros')
        clean = cv.getTrackbarPos('clean', 'filtros')

        original = cv.imread(picture[pict - 1])
        copy = original.copy()
        # Grayscale to circle detecting
        imgGS = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)

        if kern <= 0:
            kern = 1
        if kern % 2 == 0:
            kern += 1

        # Blur the image a little  to avoid false circles
        if blur == 0:
            imgGS = cv.GaussianBlur(imgGS, (kern, kern), 0)
        elif blur == 1:
            imgGS = cv.blur(imgGS, (kern, kern), 0)
        else:
            imgGS = cv.bilateralFilter(imgGS, 15, 17, 17)

        uncanny = cv.Canny(image=imgGS, threshold1=75, threshold2=205)

        thresh = cv.adaptiveThreshold(uncanny.copy(), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 5, 1)

        if dp <= 0.0:
            dp = 1
        if param1 <= 0:
            param1 = 200
        if param2 <= 0:
            param2 = 100

        cont_img = thresh.copy()
        _, contours, hierarchy = cv.findContours(image=cont_img, mode=cv.RETR_TREE,
                                              method=cv.CHAIN_APPROX_SIMPLE)

        #Lists with tuples with values (x,y,radius)
        real_contoursG = []    #Green circles
        real_contoursB = []    #Blue circles
        real_hough = []        #Red circles

        for cnt in contours:
            approx = cv.approxPolyDP(cnt, 0.008 * cv.arcLength(cnt, True), True)
            area = cv.contourArea(cnt)
            if len(approx) > 12 and area > 750:
                (x, y), radius = cv.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                if area > 8000:
                    real_contoursB.append((center, radius))
                    #cv.circle(original, center, radius, (255, 255, 0), 4)
                else:
                    real_contoursG.append((center, radius))
                    #cv.circle(original, center, radius, (0, 255, 0), 4)

                    ## detect circles in the image
        circles = cv.HoughCircles(thresh, cv.HOUGH_GRADIENT, dp=dp, minDist=80, \
                                  param1=param1, param2=param2, maxRadius=200, minRadius=80)

        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integer
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                real_hough.append(((x,y),r))

        num = 0
        if clean == 0:
            for cnt in contours:
                approx = cv.approxPolyDP(cnt, 0.008 * cv.arcLength(cnt, True), True)
                area = cv.contourArea(cnt)
                if len(approx) > 12 and area > 750:
                    (x, y), radius = cv.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    radius = int(radius)
                    num += 1
                    if area > 8000:
                        cv.circle(original, center, radius, (255, 0, 0), 4)
                    else:
                        cv.circle(original, center, radius, (0, 255, 0), 4)

            if circles is not None:
                # loop over the (x, y) coordinates and radius of the circles
                for (x, y, r) in circles:
                    # draw the circle in the output image, then draw a rectangle
                    cv.rectangle(original, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    cv.circle(original, (x, y), r, (0, 0, 255), 4)
                    num += 1

        else:
            real_hough, real_contoursB, real_contoursG = cleanResults(real_hough, real_contoursG, real_contoursB)

            for (c,r) in real_hough:
                # draw the circle in the output image, then draw a rectangle
                cv.rectangle(original, (c[0]-5, c[1]-5), (c[0] + 5, c[1] + 5), (0, 128, 255), -1)
                cv.putText(original, str(c), c, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.circle(original, c, r, (0, 0, 255), 4)
                num += 1
            for (c,r) in real_contoursB:
                # draw the circle in the output image, then draw a rectangle
                cv.rectangle(original, (c[0]-5, c[1]-5), (c[0] + 5, c[1] + 5), (0, 128, 255), -1)
                cv.putText(original, str(c), c, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.circle(original, c, r, (255, 0, 0), 4)
                num += 1
            for (c,r) in real_contoursG:
                # draw the circle in the output image, then draw a rectangle
                cv.rectangle(original, (c[0]-5, c[1]-5), (c[0] + 5, c[1] + 5), (0, 128, 255), -1)
                cv.putText(original, str(c), c, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.circle(original, c, r, (0, 255, 0), 4)
                num += 1

        cv.putText(original, 'Coins: ' + str(num), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # Displaying the results
        if mode == 0:
            cv.imshow('imagen', cv.resize(original, (original.shape[1] // 2, original.before_after.shape[0] // 2)))
        if mode == 1:
            cv.imshow('imagen', cv.resize(uncanny, (uncanny.shape[1] // 2, uncanny.shape[0] // 2)))
        if mode == 2:
            cv.imshow('imagen', cv.resize(thresh, (thresh.shape[1] // 2, thresh.shape[0] // 2)))

        # ESC to break
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

    # close all open windows
    cv.destroyAllWindows()

''' Return a list with all the circles needed to recognize
    most of the coins. or all of them.

    Conditions to save the circle:
        If Red circle -> Hough detection. If there is other Red Circle inside it and the size difference
        is not too big, save it and ignore the other. If there is more than 1 Red Circle, ignore this and save the others.
        If Blue circle -> Contour detection with area > 8000 (which means can be various coins)
           If within are more than 2 circles, look conditions for each circle, otherwise, save it.
        If Green circle (Contour with area < 8000) inside a Blue Circle,
           only if radius and center position are very different

    @params:
        hough: List with the radius and center of each red circle
               found with HoughCircles algorithm.
        contourG: List with the radius and center of each green circle
                 found searching contours.
        contourB: List with the radius and center of each blue circle
                 found searching contours.
    @returns:
        hough: List with the final red circles.
        contourG: List with the final green circles.
        contourB: List with the final blue circles
'''
def cleanResults(hough, contourG, contourB):
    # First we check red circles (the fewer)
    aux = [x for x in hough]
    for redC in aux:
        toRemove = False
        removeList = []
        for redC2 in hough:
            # Bigger radius
            r = redC[1] if redC[1] > redC2[1] else redC2[1]
            # If are not the same and the distance between centers are less
            # than the bigger radius, then REMOVE the BIGGER circle
            if redC[0] != redC2[0] and \
                    math.sqrt((redC[0][0] - redC2[0][0])**2 + (redC[0][1] - redC2[0][1])**2) < r:
                c = redC if redC[1] > redC2[1] else redC2
                # If the circle is not in the list, then add it.
                # To avoid repetitions, check in RedCircles too
                if removeList.count(c) == 0 and hough.count(c) != 0:
                    toRemove = True
                    removeList.append(c)
        if toRemove:
            for c in removeList:
                hough.remove(c)

    # Then remove green between green circles and remove the smaller circle
    aux = [x for x in contourG]
    for greenC in aux:
        toRemove = False
        removeList = []
        for greenC2 in contourG:
            # Bigger radius
            r = greenC[1] if greenC[1] > greenC2[1] else greenC2[1]
            # If are not the same and the distance between centers are less
            # than the bigger radius, then REMOVE the SMALLER circle
            if greenC[0] != greenC2[0] and \
                            math.sqrt((greenC[0][0] - greenC2[0][0]) ** 2 + (greenC[0][1] - greenC2[0][1]) ** 2) < (r + r/2):
                c = greenC if greenC[1] < greenC2[1] else greenC2
                # If the circle is not in the list, then add it.
                # To avoid repetitions, check in RedCircles too
                if removeList.count(c) == 0 and contourG.count(c) != 0:
                    toRemove = True
                    removeList.append(c)
        if toRemove:
            for c in removeList:
                contourG.remove(c)

    # Then check if the green circle is inside or almost the same redCircle.
    # In that case, remove it
    for redC in hough:
        toRemove = False
        removeList = []
        for greenC in contourG:
            # Bigger radius
            r = greenC[1] if greenC[1] > redC[1] else redC[1]
            # If the distance between centers are less
            # than the bigger radius, then REMOVE the GREEN circle
            if math.sqrt((greenC[0][0] - redC[0][0]) ** 2 + (greenC[0][1] - redC[0][1]) ** 2) < (greenC[1] + redC[1]):
                # If the circle is not in the list, then add it
                if removeList.count(greenC) == 0:
                    toRemove = True
                    removeList.append(greenC)
        if toRemove:
            for c in removeList:
                contourG.remove(c)

    # Then we check blue circles between blue circles (like the red)
    aux = [x for x in contourB]
    for blueC in aux:
        toRemove = False
        removeList = []
        for blue2C in contourB:
            # Bigger radius
            r = blueC[1] if blueC[1] > blue2C[1] else blue2C[1]
            # If are not the same and the distance between centers are less
            # than the bigger radius, then REMOVE the BIGGER circle
            if blueC != blue2C and \
                    math.sqrt((blueC[0][0] - blue2C[0][0])**2 + (blueC[0][1] - blue2C[0][1])**2) < r:
                c = blueC if blueC[1] > blue2C[1] else blue2C
                # If the circle is not in the list, then add it
                # To avoid repetitions, check in BlueCircles too
                if removeList.count(c) == 0 and contourB.count(c) != 0:
                    toRemove = True
                    removeList.append(c)
        if toRemove:
            for c in removeList:
                contourB.remove(c)

    #Then remove Green Circles from inside Blue Circles.
    #only if there is not more than 1 (case same circle)
    for blueC in contourB:
        toRemove = False
        removeList = []
        for greenC in contourG:
            # Bigger radius
            r = blueC[1] if blueC[1] > greenC[1] else greenC[1]
            # If the distance between centers are less
            # than the bigger radius, then REMOVE the BIGGER circle
            if math.sqrt((blueC[0][0] - greenC[0][0])**2 + (blueC[0][1] - greenC[0][1])**2) < r:
                # If the circle is not in the list, then add it
                # To avoid repetitions, check in BlueCircles too
                if removeList.count(greenC) == 0:
                    toRemove = True
                    removeList.append(greenC)
        if toRemove and len(removeList) <= 2:
            for c in removeList:
                contourG.remove(c)

    #Then remove Blue Circles very very close to Red Circles
    for redC in hough:
        toRemove = False
        removeList = []
        for blueC in contourB:
            # Bigger radius
            r = blueC[1] if blueC[1] > redC[1] else redC[1]
            # If the distance between centers are less
            # than the bigger radius, then REMOVE the Blue circle
            if math.sqrt((blueC[0][0] - redC[0][0])**2 + (blueC[0][1] - redC[0][1])**2) < r:
                # If the circle is not in the list, then add it
                # To avoid repetitions, check in BlueCircles too
                if removeList.count(blueC) == 0:
                    toRemove = True
                    removeList.append(blueC)
        if toRemove:
            for c in removeList:
                contourB.remove(c)

    #In case after all the cleaning, a blue circle has more than 1 circle inside
    #means that the blue circle is wrong.
    aux = [x for x in contourB]
    for blueC in aux:
        inside = []
        for greenC in contourG:
            # Bigger radius
            r = blueC[1] if blueC[1] > greenC[1] else greenC[1]
            # If the distance between centers are less
            # than the bigger radius, then REMOVE the BIGGER circle
            if math.sqrt((blueC[0][0] - greenC[0][0]) ** 2 + (blueC[0][1] - greenC[0][1]) ** 2) < r:
                # If the circle is not in the list, then add it
                # To avoid repetitions, check in "inside" too
                if inside.count(greenC) == 0:
                    inside.append(greenC)
        if len(inside) > 1:
            contourB.remove(blueC)

    return hough, contourB, contourG

''' Text To Speech of the results.

    @params:
        results: JSON with the results.
    @return:
        None
'''
def textToSpeech(results):
    #Init the service
    cred = json.load(open('credentials'+os.sep+'watson_credentials.json', 'r'))['tts']
    text_to_speech = TextToSpeech(url=cred['url'], username=cred['username'], password=cred['password'])

    #Translate the results to text
    #{'coins': {}, 'unknown': 0, 'total': 0.0}

    textToSay = ""
    if len(results['coins']) == 0:
        textToSay = "No se han encontrado monedas."
    else:
        textToSay = "El dinero total es: " + str(results['total']) + "."
        if results['total'] > 0:
            textToSay += "Las monedas encontradas han sido:"
            for k,v in results['coins'].iteritems():
                textToSay += str(v)
                if v > 1:
                    textToSay += " monedas de "
                else:
                    textToSay += " moneda de "
                if k == 'euro1':
                    textToSay += "un euro, "
                elif k == 'euro2':
                    textToSay += "dos euros, "
                elif k == 'cent1':
                    textToSay += "un céntimo, "
                elif k == 'cent2':
                    textToSay += "dos céntimos, "
                elif k == 'cent10':
                    textToSay += "diez céntimos, "
                elif k == 'cent5':
                    textToSay += "cinco céntimos, "
                elif k == 'cent50':
                    textToSay += "cincuenta céntimos, "

            if results['unknown'] != 0:
                textToSay += "Además, no se han podido identificar " + str(results['unknown'])
                if results['unknown'] % 2 == 0:
                    textToSay += " monedas."
                else:
                    textToSay += " moneda."

    #Save and play the audio
    speaker = pyaudio.PyAudio()

    data = text_to_speech.synthesize(
        text=textToSay,
        voice='es-ES_EnriqueVoice', accept='audio/wav')
    wf = wave.open('.tempAudio.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(22050)
    wf.writeframes(data)
    wf.close()
    wf = wave.open('.tempAudio.wav', 'rb')
    data = wf.readframes(1024)

    stream = speaker.open(format=pyaudio.get_format_from_width(wf.getsampwidth()), channels=1, rate=wf.getframerate(),
                          output=True)

    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(1024)

    # stop stream (4)
    stream.stop_stream()
    stream.close()
    wf.close()

    os.remove('.tempAudio.wav')

    return None

def takePicture():
    camera = PiCamera()
    camera.resolution=(1280,720)
    time.sleep(1)
    raw_input("Press <Enter> to take picture")
    camera.capture("coin.jpeg", format="jpeg", use_video_port=True, quality=100)
    camera.close()
    pict = cv.imread("coin.jpeg")
    cv.imshow("coin.jpeg", cv.resize(pict, (pict.shape[1]/3, pict.shape[0]/3)))

def demo(picture, visual):
    # Create a window
    cv.namedWindow('filtros', cv.WINDOW_NORMAL)
    cv.namedWindow('imagen')

    # create trackbars for treshold change
    cv.createTrackbar('kernel', 'filtros', 3, 10, nothing)
    cv.createTrackbar('dp', 'filtros', 11, 20, nothing)
    cv.createTrackbar('param1', 'filtros', 200, 300, nothing)
    cv.createTrackbar('param2', 'filtros', 107, 300, nothing)
    cv.createTrackbar('pict', 'filtros', 0, len(picture), nothing)
    cv.createTrackbar('blur', 'filtros', 1, 2, nothing)
    cv.createTrackbar('mode', 'filtros', 0, 2, nothing)
    cv.createTrackbar('clean', 'filtros', 1, 1, nothing)
    cv.createTrackbar('classify', 'filtros', 0, 1, nothing)

    paramClass = json.dumps({'classifier_ids': [ID]})

    while True:
        # get current positions of four trackbars
        kern = cv.getTrackbarPos('kernel', 'filtros')
        dp = cv.getTrackbarPos('dp', 'filtros') / 10
        param1 = cv.getTrackbarPos('param1', 'filtros')
        param2 = cv.getTrackbarPos('param2', 'filtros')
        pict = cv.getTrackbarPos('pict', 'filtros')
        blur = cv.getTrackbarPos('blur', 'filtros')
        mode = cv.getTrackbarPos('mode', 'filtros')
        clean = cv.getTrackbarPos('clean', 'filtros')
        classify = cv.getTrackbarPos('classify', 'filtros')

        original = cv.imread(picture[pict - 1])
        copy = original.copy()
        # Grayscale to circle detecting
        imgGS = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)

        if kern <= 0:
            kern = 1
        if kern % 2 == 0:
            kern += 1

        # Blur the image a little  to avoid false circles
        if blur == 0:
            imgGS = cv.GaussianBlur(imgGS, (kern, kern), 0)
        elif blur == 1:
            imgGS = cv.blur(imgGS, (kern, kern), 0)
        else:
            imgGS = cv.bilateralFilter(imgGS, 15, 17, 17)

        uncanny = cv.Canny(image=imgGS, threshold1=75, threshold2=205)

        thresh = cv.adaptiveThreshold(uncanny.copy(), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 5, 1)

        if dp <= 0.0:
            dp = 1
        if param1 <= 0:
            param1 = 200
        if param2 <= 0:
            param2 = 100

        cont_img = thresh.copy()
        _, contours, hierarchy = cv.findContours(image=cont_img, mode=cv.RETR_TREE,
                                              method=cv.CHAIN_APPROX_SIMPLE)

        #Lists with tuples with values (x,y,radius)
        real_contoursG = []    #Green circles
        real_contoursB = []    #Blue circles
        real_hough = []        #Red circles

        for cnt in contours:
            approx = cv.approxPolyDP(cnt, 0.008 * cv.arcLength(cnt, True), True)
            area = cv.contourArea(cnt)
            if len(approx) > 12 and area > 750:
                (x, y), radius = cv.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                if area > 8000:
                    real_contoursB.append((center, radius))
                    #cv.circle(original, center, radius, (255, 255, 0), 4)
                else:
                    real_contoursG.append((center, radius))
                    #cv.circle(original, center, radius, (0, 255, 0), 4)

                    ## detect circles in the image
        circles = cv.HoughCircles(thresh, cv.HOUGH_GRADIENT, dp=dp, minDist=80, \
                                  param1=param1, param2=param2, maxRadius=200, minRadius=80)
        # ensure at least some circles were found

        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integer
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                real_hough.append(((x,y),r))

        num = 0
        if clean == 0:
            for cnt in contours:
                approx = cv.approxPolyDP(cnt, 0.008 * cv.arcLength(cnt, True), True)
                area = cv.contourArea(cnt)
                if len(approx) > 12 and area > 750:
                    (x, y), radius = cv.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    radius = int(radius)
                    num += 1
                    if area > 8000:
                        cv.circle(original, center, radius, (255, 0, 0), 4)
                    else:
                        cv.circle(original, center, radius, (0, 255, 0), 4)

            if circles is not None:
                # loop over the (x, y) coordinates and radius of the circles
                for (x, y, r) in circles:
                    # draw the circle in the output image, then draw a rectangle
                    cv.rectangle(original, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    cv.circle(original, (x, y), r, (0, 0, 255), 4)
                    num += 1

        else:
            real_hough, real_contoursB, real_contoursG = cleanResults(real_hough, real_contoursG, real_contoursB)

            for (c,r) in real_hough:
                # draw the circle in the output image, then draw a rectangle
                cv.rectangle(original, (c[0]-5, c[1]-5), (c[0] + 5, c[1] + 5), (0, 128, 255), -1)
                cv.putText(original, str(c), c, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.circle(original, c, r, (0, 0, 255), 4)
                num += 1
            for (c,r) in real_contoursB:
                # draw the circle in the output image, then draw a rectangle
                cv.rectangle(original, (c[0]-5, c[1]-5), (c[0] + 5, c[1] + 5), (0, 128, 255), -1)
                cv.putText(original, str(c), c, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.circle(original, c, r, (255, 0, 0), 4)
                num += 1
            for (c,r) in real_contoursG:
                # draw the circle in the output image, then draw a rectangle
                cv.rectangle(original, (c[0]-5, c[1]-5), (c[0] + 5, c[1] + 5), (0, 128, 255), -1)
                cv.putText(original, str(c), c, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.circle(original, c, r, (0, 255, 0), 4)
                num += 1

            if classify == 1:
                cv.setTrackbarPos("classify", "filtros", 0)
                all_coins = []
                all_coins.extend(real_hough)
                all_coins.extend(real_contoursG)
                all_coins.extend(real_contoursB)

                coins_found = {'coins': {}, 'unknown': 0, 'total': 0.0}

                # loop over the center (x,y) coordinates and radius of the circles
                for (c, r) in all_coins:
                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    left_x = c[0] - r - 25 if c[0] - r - 25 > 0 else 0
                    top_y = c[1] - r - 25 if c[1] - r - 25 > 0 else 0
                    bot_y = c[1] + r + 25 if c[1] + r + 25 < original.shape[0] else original.shape[0] - 1
                    right_x = c[0] + r + 25 if c[0] + r + 25 < original.shape[1] else original.shape[1] - 1

                    coin = copy[top_y:bot_y, left_x:right_x]
                    cv.imwrite('.tempCoin.jpeg', coin)

                    coinResult = classifyCoins(visual, open('.tempCoin.jpeg', 'rb'), '.tempCoin.jpeg', params=paramClass)
                    # To save the values
                    coin_score = 0.0
                    coin_name = ''

                    # Check the type of coin
                    for coin in coinResult['images'][0]['classifiers'][0]['classes']:
                        # Get the biggest value
                        if float(coin['score']) > 0.9 and float(coin['score']) > coin_score:
                            coin_score = float(coin['score'])
                            coin_name = coin['class']

                    # Final results of the coin
                    if coin_name != '':
                        value = 0.0
                        if coin_name not in coins_found['coins']:
                            coins_found['coins'][coin_name] = 0
                        coins_found['coins'][coin_name] += 1
                        if coin_name == 'euro1':
                            value = 1.0
                        elif coin_name == 'euro2':
                            value = 2.0
                        elif coin_name == 'cent1':
                            value = 0.010
                        elif coin_name == 'cent2':
                            value = 0.020
                        elif coin_name == 'cent5':
                            value = 0.050
                        elif coin_name == 'cent10':
                            value = 0.10
                        elif coin_name == 'cent50':
                            value = 0.50
                        coins_found['total'] += value
                        cv.putText(copy, ''+str(value), c, cv.FONT_HERSHEY_SIMPLEX,2,(0,255,255),2)
                    else:
                        coins_found['unknown'] += 1

                # loop over the center (x,y) coordinates and radius of the circles
                # to draw the circles
                for (c, r) in real_hough:
                    # draw the circle in the output image, then draw a rectangle
                    #cv.rectangle(copy, (c[0] - 5, c[1] - 5), (c[0] + 5, c[1] + 5), (0, 128, 255), -1)
                    cv.circle(copy, c, r, (0, 0, 255), 4)
                    num += 1
                for (c, r) in real_contoursB:
                    # draw the circle in the output image, then draw a rectangle
                    #cv.rectangle(copy, (c[0] - 5, c[1] - 5), (c[0] + 5, c[1] + 5), (0, 128, 255), -1)
                    cv.circle(copy, c, r, (255, 0, 0), 4)
                    num += 1
                for (c, r) in real_contoursG:
                    # draw the circle in the output image, then draw a rectangle
                    #cv.rectangle(copy, (c[0] - 5, c[1] - 5), (c[0] + 5, c[1] + 5), (0, 128, 255), -1)
                    cv.circle(copy, c, r, (0, 255, 0), 4)
                    num += 1
                cv.putText(copy, 'Coins: ' + str(num), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv.putText(copy, 'Total: ' + str(coins_found['total']), (0, 120), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                os.remove('.tempCoin.jpeg')
                print(coins_found)

                cv.namedWindow('after')
                cv.imshow('after', cv.resize(copy, (copy.shape[1] // 2, copy.shape[0] // 2)))

        cv.putText(original, 'Coins: ' + str(num), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # Displaying the results
        if mode == 0:
            cv.imshow('imagen', cv.resize(original, (original.shape[1] // 2, original.shape[0] // 2)))
        if mode == 1:
            cv.imshow('imagen', cv.resize(uncanny, (uncanny.shape[1] // 2, uncanny.shape[0] // 2)))
        if mode == 2:
            cv.imshow('imagen', cv.resize(thresh, (thresh.shape[1] // 2, thresh.shape[0] // 2)))

        # ESC to break
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

    # close all open windows
    cv.destroyAllWindows()


def GUI(visual):

    root = tk.Tk()
    window = tk.Frame(root)

    buttonFont = font.Font(family='Helvetica', size=16)

    coins = ['monedas_demo/pict (1).jpg','monedas_demo/pict (2).jpg','monedas_demo/pict (3).jpg',\
            'monedas_demo/pict (4).jpg','monedas_demo/pict (5).jpg','monedas_demo/pict (6).jpg','monedas_demo/pict (7).jpg']
    #Checks the classifier status
    tk.Button(root, font=buttonFont, text="status", command=lambda: statusClass(visual),width=8,height=1).pack()
    #Trains a new custom classifier using the zips in the train folder
    tk.Button(root, font=buttonFont, text="train", command=lambda: trainCoinClass(visual),width=8,height=1).pack()
    #Opens a GUI which allows to change the OpenCV values and find the "perfect ones"
    tk.Button(root, font=buttonFont, text="test", command=lambda: houghCircles(visual,coins),width=8,height=1).pack()
    #Sends the picture taken with the "TakePicture" method to the Visual Recognition classifier.
    #tk.Button(root, font=buttonFont, text="classify", command=lambda: detectCoins(visual, 'coin.jpeg'),width=8,height=1).pack()
    #Takes a new picture using the Camera Module
    tk.Button(root, font=buttonFont, text="Take picture", command=lambda: takePicture(), width=8, height=1).pack()
    #Opens a GUI with default values and allows to classify the picture.
    tk.Button(root, font=buttonFont, text="Demo", command=lambda: demo(coins, visual), width=8, height=1).pack()
    tk.Button(root, text="", state=tk.DISABLED, height=2, relief=tk.FLAT).pack()  # BLANK SPACE
    #Removes the Custom Classifier
    tk.Button(root, font=buttonFont, text="delete", command=lambda: deleteClassifier(visual), width=8, height=1).pack()
    #Closes the program
    tk.Button(root, font=buttonFont, text="QUIT", command=lambda: quit(), width=8).pack()

    return window

''' Main function (executed with main.py)

    @params
        None
    @return
        None
'''
def main():
    #load the visual recognition service from Watson with the right credentials
    creds = json.load(open('credentials'+os.sep+'watson_credentials.json', 'r'))['visual']
    url = creds['url']
    api_key = creds['api_key']
    visual_recognition = VisualRecognition('2016-05-20', url=url, api_key=api_key)

    window = GUI(visual_recognition)
    window.mainloop()

if __name__ == '__main__':
    main()
