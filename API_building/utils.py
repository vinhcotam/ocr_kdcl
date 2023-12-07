import cv2
import numpy as np
from PIL import Image
import json
import torch
import torch
import onnxruntime
import numpy as np
import re 
import os


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def sort_box(boxes):
    sorted_boxes = []
    for box in boxes:
        sorted_boxes.append(order_points_clockwise(box))
    mid_points = []
    for box in sorted_boxes:
        try:
            mid = line_intersection((box[0],box[2]), (box[1], box[3]))
            mid_points.append(mid)
        except:
            continue
    sorted_indices = np.argsort(mid_points, axis=0)
    sorted_boxes = sorted(sorted_boxes , key=lambda sorted_indices: [sorted_indices[0][1], sorted_indices[0][0]]) 
    return sorted_boxes

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
    
def crop_box(img, boxes, sort=False):

    img_list = []
    h,w,c = img.shape

    if sort:
        boxes = sort_box(boxes)

    for i, box in enumerate(boxes):

        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = box
        x1,y1,x2,y2,x3,y3,x4,y4 = int(x1),int(y1),int(x2),int(y2),int(x3),int(y3),int(x4),int(y4)
        x1 = max(0, x1)
        x2 = max(0, x2)
        x3 = max(0, x3)
        x4 = max(0, x4)
        y1 = max(0, y1)
        y2 = max(0, y2)
        y3 = max(0, y3)
        y4 = max(0, y4)
        min_x = max(0, min(x1,x2,x3,x4))
        min_y = max(0, min(y1,y2,y3,y4))
        max_x = min(w, max(x1,x2,x3,x4))
        max_y = min(h, max(y1,y2,y3,y4))

        tw = int(np.sqrt((x1-x2)**2 + (y1-y2)**2))
        th = int(np.sqrt((x1-x4)**2 + (y1-y4)**2))
        pt1 = np.float32([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])
        pt2 = np.float32([[0, 0],
                            [tw - 1, 0],
                            [tw - 1, th - 1],
                            [0, th - 1]])
        matrix = cv2.getPerspectiveTransform(pt1,pt2)
        cropped = cv2.warpPerspective(img, matrix, (tw, th))

        img_list.append(cropped)

    return boxes, img_list


def normalize_usage(s):
    REMOVE_LIST = ['cách dùng', 'Cách dùng', 'cách', 'Cách', 'cách dùng:', 'Cách dùng:', 'ghi chú', 'Ghi chú', 'Uống:', 'uống:', 'Uống', 'uống']
    remove = '|'.join(REMOVE_LIST)
    regex = re.compile(r'\b('+remove+r')\b', flags=re.IGNORECASE)
    out = regex.sub("", s)
    output_string = re.sub(r'^[^a-zA-Z0-9]+', '', out)
    return output_string.rstrip()


def normalize_quantity(s):
    REMOVE_LIST = ['SL:', 'sl:', 'SL', 'sl', 'Số lượng:', 'số lượng:', 'Liều lượng:', 'liều lượng:', 'Liều lượng', 'liều lượng']
    remove = '|'.join(REMOVE_LIST)
    regex = re.compile(r'\b('+remove+r')\b', flags=re.IGNORECASE)
    out = regex.sub("", s)
    output_string = re.sub(r'^[^a-zA-Z0-9]+', '', out)
    return output_string.rstrip()

def normalize_brandname(s):
    output = re.sub(r'^[0-9]+', '', s)
    output = re.sub(r'^[^a-zA-Z0-9]+', '', output)
    return output.rstrip()

def text_to_json(t, l):
    response = {
        'date': '',
        'medicines':[],
        'diagnose':''
    }

    usage_array = []
    brandname = ''
    quantity = []
    date = ''
    dia = []
    for i, item in enumerate(l):
        if item == 'usage':
            usage_array.append(t[i])
        elif item == 'quantity':
            quantity.append(normalize_quantity(t[i]))
        elif item == 'brandname':
            usage = ''
            for item in reversed(usage_array):
                usage += (item + ' ')
            tmp = {
                'brandname':normalize_brandname(t[i]),
                'usage': normalize_usage(usage),
            }
            response['medicines'].append(tmp)
            usage_array = []
            brandname = ''
        elif item == 'date':
            date = t[i]
        elif item =='diagnose':
            dia.append(t[i])

    diagnose = ''
    try:
        for item in reversed(dia):
            diagnose += (item + ' ')
    except:
        diagnose = ''
    
    for i, item in enumerate(response['medicines']):
        try:
            response['medicines'][i]['quantity'] = quantity[i]
        except:
            response['medicines'][i]['quantity'] = ''

        
    response['date'] = date.rstrip()
    response['diagnose'] = diagnose.rstrip()

    return response

def translate_onnx(img, session, max_seq_length=128, sos_token=1, eos_token=2):
    """data: BxCxHxW"""
    cnn_session, encoder_session, decoder_session = session
    
    # create cnn input
    cnn_input = {cnn_session.get_inputs()[0].name: img}
    src = cnn_session.run(None, cnn_input)
    
    # create encoder input
    encoder_input = {encoder_session.get_inputs()[0].name: src[0]}
    encoder_outputs, hidden = encoder_session.run(None, encoder_input)
    translated_sentence = [[sos_token] * len(img)]
    max_length = 0

    while max_length <= max_seq_length and not all(
        np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
    ):
        tgt_inp = translated_sentence
        decoder_input = {decoder_session.get_inputs()[0].name: tgt_inp[-1], decoder_session.get_inputs()[1].name: hidden, decoder_session.get_inputs()[2].name: encoder_outputs}

        output, hidden, _ = decoder_session.run(None, decoder_input)
        output = np.expand_dims(output, axis=1)
        output = torch.Tensor(output)

        values, indices = torch.topk(output, 1)
        indices = indices[:, -1, 0]
        indices = indices.tolist()

        translated_sentence.append(indices)
        max_length += 1

        del output

    translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence



