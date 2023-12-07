import numpy as np
from PIL import Image, ImageFile
from ultralytics import YOLO
ImageFile.LOAD_TRUNCATED_IMAGES = True
import re
from datetime import datetime
import os
import matplotlib.pyplot as plt
import cv2
from utils import crop_box
import config as cf
from pyvi import ViTokenizer
import re
import csv
import pandas as pd
from underthesea import text_normalize
#loading model to use
def loading_model():
    #load model yolov8
    global weights_path
    weights_path = cf.YOLOv8_WEIGHTS
    global yolov8_model
    yolov8_model = YOLO(weights_path)
    # load model vietocr
    cf.CONFIG_VIETOCR
    # define the label
    cf.LABELS_NAME
    cf.CLASSES
    # load model paddle ocr
    cf.DETECTOR

def delete_watermark(img):

    alpha = 2.0
    beta = -160
    new = alpha * img + beta
    new = np.clip(new, 0, 255).astype(np.uint8)
    image_pil = Image.fromarray(cv2.cvtColor(new, cv2.COLOR_BGR2RGB))
    image_np = np.array(image_pil)
    after_deleted = []
    after_deleted.append(image_pil)
    after_deleted.append(image_np)
    return after_deleted

def get_results_detect(results_yolov8):
    bbox = []
    labels = []
    conf = []
    for r in results_yolov8:
        #get Coordinate from yolov8
        bbox.append(r.boxes.xywh)
        #get class
        labels.append(r.boxes.cls)
        #get conf
        conf.append(r.boxes.conf)
    bbbox = bbox[0].tolist()
    labelss = labels[0].tolist()
    conff = conf[0].tolist()
    results_detail = []
    results_detail.append(bbbox)
    results_detail.append(labelss)
    results_detail.append(conff)
    return results_detail

def get_max_confs(bboxes, confs, labels):
    #check conf if have >2 with one class
    max_confs = {}
    for i, box in enumerate(bboxes):
        label_name = cf.LABELS_NAME[labels[i]]
        if label_name not in max_confs or confs[i] > max_confs[label_name]:
            max_confs[label_name] = confs[i]
    return max_confs


def crop_image(results_detail, image_pil, id):
    bboxes = results_detail[0]
    labels = results_detail[1]
    confs = results_detail[2]
    cropped_images = []
    max_confs = get_max_confs(bboxes, confs, labels)
    for i, box in enumerate(bboxes):
        if (confs[i] > 0.3) and (confs[i] == max_confs[cf.LABELS_NAME[labels[i]]]):
            # crop img from Coordinate
            x, y, w, h = box
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            crop_img = image_pil.crop((x1, y1, x2, y2))
            label_name = cf.LABELS_NAME[labels[i]]
            # save img after crop
            os.makedirs(f'./crop_image/{id}/', exist_ok=True)
            save_path = f'./crop_image/{id}/{label_name}.jpg'
            cv2.imwrite(save_path, cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR))
            cropped_images.append((label_name, crop_img, save_path))

    return cropped_images

def extract_text_from_images(cropped_images, id):
    results_dicts = {}
    for class_name, cropped_image, save_path in cropped_images:
        if class_name == "ngay" or class_name == "soqd":
            text_result = cf.PREDICTOR.predict(cropped_image)
            text_result = text_normalize(text_result)

            if class_name not in results_dicts:
                results_dicts[class_name] = []
            results_dicts[class_name].append(text_result)
        else:
            if class_name not in results_dicts:
                results_dicts[class_name] = []
            bounding_boxes = text_detection_v2(np.array(cropped_image))
            new_cropped_images = crop_image_by_bbox_v2(save_path, bounding_boxes)
            crop_text_image = crop_text_images_v2(new_cropped_images, class_name, id)
            results_text = ocr_v2(crop_text_image)
            results_text = text_normalize(results_text)
            results_dicts[class_name].append(results_text)

    return results_dicts

#check pattern
def text_processing(cropped_images, text_sum, keywords, id):
    results = {}
    results_dict = extract_text_from_images(cropped_images, id)
    for class_name, texts in results_dict.items():
        result_text = ' '.join(texts)
        if class_name == "soqd":
            match = re.search(cf.PATTERN_SQD, result_text)
            results[class_name] = result_text if match else ""
        elif class_name == "ngay":
            match = re.search(r'(?:ngày\s+)?\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}', result_text)
            results[class_name] = datetime.strptime(match.group(), 'ngày %d tháng %m năm %Y').strftime('%d-%m-%Y') if match else ""
        else:
            results[class_name] = result_text

    for class_name in cf.CLASSES:
        results.setdefault(class_name, "")
    results['trichyeu'] = text_sum
    results['tukhoa'] = keywords
    return results

def output_encoding(results):
    # check_results(results)
    response = {
        "status": check_results(results),
        "results": results,
        "error": []
    }
    return response

def output_encoding_v2(results, label):
    data = {"status": True,
            "results": {label: results},
            "error": None
            }
    return(data)

def check_results(results):
    if all(results[result] == '' for result in results):
        status = 'fail'
    elif any(results[result] == '' for result in results):
        status = 'part'
    else:
        status = 'excellent'

    return status

def sort_words_by_coordinates(words):
    centers = [np.mean(np.array(coords), axis=0) for coords, _, _ in words]
    sorted_words = sorted(zip(centers, words), key=lambda x: x[0][1])
    lines = []
    current_line = []
    current_y = sorted_words[0][0][1]

    for center, word in sorted_words:
        if abs(center[1] - current_y) < 10:
            current_line.append(word)
        else:
            lines.append(sorted(current_line, key=lambda x: np.mean(np.array(x[0]), axis=0)[0]))
            current_line = [word]
            current_y = center[1]

    lines.append(sorted(current_line, key=lambda x: np.mean(np.array(x[0]), axis=0)[0]))
    return lines





def process_input_image(image_path, id):
    #read file img from input
    img = plt.imread(image_path)
    #delete watermark
    after_deleted = delete_watermark(img)
    image_pil = after_deleted[0]
    image_np = after_deleted[1]
    #get results from yolov8 model
    results_yolov8 = yolov8_model(image_np)
    #processing with results from yolov8
    results_detail = get_results_detect(results_yolov8)
    #crop img
    cropped_image_results = crop_image(results_detail, image_pil, id)
    # save_path = cropped_image_results["save_path"]
    #ocr full image
    ocr_full_um = ocr_full(img)
    text_sum = summary_text(ocr_full_um)
    remove_stopwords_text = remove_stopwords(ocr_full_um)
    tokenize_word = tokenizer_text(remove_stopwords_text)
    # save_csv = save_data_keyword(tokenize_word)
    keywords = get_keyword(ocr_full_um)

    #ocr
    extract_text_list = text_processing(cropped_image_results, text_sum, keywords, id)

    #encoding results
    results = output_encoding(extract_text_list)
    print(results)
    return results

def get_keyword(ocr_full_um):
    df = pd.read_excel(cf.EXCEL_FILE_PATH, usecols=["Matching Tokens"])
    matching_tokens_values = df["Matching Tokens"].tolist()
    matching_tokens = [str(value) for value in df["Matching Tokens"].dropna().values]
    matches = [value for value in matching_tokens if value in ocr_full_um]
    keyword_array = []
    if matches:
        keyword_array.append(matches)
    else:
        print("No matching values found.")
    print(keyword_array)
    return keyword_array
def process_input_pdf(img_array, id):
    content = ""
    first_page = img_array[0]
    img_array.pop(0)
    img = plt.imread(first_page)
    after_deleted = delete_watermark(img)
    image_pil = after_deleted[0]
    image_np = after_deleted[1]
    results_yolov8 = yolov8_model(image_np)
    results_detail = get_results_detect(results_yolov8)
    cropped_image_results = crop_image(results_detail, image_pil, id)
    ocr_full_um = ocr_full(img)
    content += ocr_full_um
    for i in img_array:
        img1 = plt.imread(i)
        content += ocr_full(img1)

    text_sum = summary_text(content)
    remove_stopwords_text = remove_stopwords(text_sum)
    tokenize_word = tokenizer_text(remove_stopwords_text)
    extract_text_list = text_processing(cropped_image_results, text_sum, id)
    results = output_encoding(extract_text_list)
    # save_csv = save_data_keyword(tokenize_word)
    keywords = get_keyword(ocr_full_um)

    # ocr
    extract_text_list = text_processing(cropped_image_results, text_sum, keywords, id)

    # encoding results
    results = output_encoding(extract_text_list)
    return results

def remove_stopwords(textsum):
    with open(cf.STOP_WORDS_PATH, "r", encoding="utf-8") as file:
        vietnamese_stopwords = [line.strip() for line in file]
    # Chuyển đổi chuỗi văn bản thành danh sách các từ
    words = textsum.split()
    # Loại bỏ các từ dừng
    filtered_words = [word for word in words if word not in vietnamese_stopwords]
    # Chuyển đổi danh sách các từ đã lọc thành chuỗi văn bản
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def tokenizer_text(filtered_text):
    tokens = ViTokenizer.tokenize(filtered_text)
    return tokens

def save_data_keyword(tokenize_word):
    matching_tokens = re.findall(cf.TOKENIZER_PATTERN, tokenize_word)
    matching_tokens = [token.replace('_', ' ') for token in matching_tokens]
    try:
        df_existing = pd.read_excel(cf.EXCEL_FILE_PATH)
    except FileNotFoundError:
        df_existing = pd.DataFrame()
    df_new = pd.DataFrame({'Matching Tokens': matching_tokens})
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_excel(cf.EXCEL_FILE_PATH, index=False)

    # with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #
    #     # Write header (optional)
    #     csv_writer.writerow(['Matching Tokens'])
    #
    #     # Write matching tokens on separate rows
    #     for token in matching_tokens:
    #         csv_writer.writerow([token])
    #
    # print(f'Matching tokens with underscores have been saved to {csv_file_path}.')
def ocr_full(img):
    extract_full_text = cf.READER.readtext(img)
    sorted_text_result = sort_words_by_coordinates(extract_full_text)
    tesst = ""
    for line in sorted_text_result:
        tesst += ' '.join([text for _, text, _ in line])
    text_normalize(tesst)
    return text_normalize(tesst)

def summary_text(sorted_text_result):
    results = {}
    input_ids = cf.TOKENIZER(sorted_text_result, return_tensors="pt")["input_ids"]
    outputs = cf.SUMMARY_MODEL.generate(input_ids=input_ids,
                                        max_length=1024,
                                        early_stopping=True)
    output_text = cf.TOKENIZER.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    results['sum'] = output_text
    return output_text
def crop_image_by_bbox_v2(image_path, bounding_boxes):
    image = cv2.imread(image_path)
    cropped_images = []
    for i, bbox in enumerate(bounding_boxes):
        x_coords = [int(point[0]) for point in bbox]
        y_coords = [int(point[1]) for point in bbox]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        cropped = image[y_min:y_max, x_min:x_max]
        # convert numpy to pil
        cropped_pil = Image.fromarray(cropped)
        # Save file img
        output_path = f'results/cropped_image_{i}.jpg'
        cropped_pil.save(output_path)
        cropped_images.append(output_path)
        return cropped_images

def crop_image_by_bbox_v2(image_path, bounding_boxes):
    image = Image.open(image_path)
    cropped_images = []
    for bbox in bounding_boxes:
        # get coodinate
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        # get min of bounding box
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        # crop image depend on bbox
        cropped = image.crop((x_min, y_min, x_max, y_max))
        cropped_images.append(cropped)
    return cropped_images

def add_padding_and_keep_size_v2(bbox, padding):
    # calculate new bbox
    new_bbox = [
        [bbox[0][0] - padding, bbox[0][1] - padding],
        [bbox[1][0] + padding, bbox[1][1] - padding],
        [bbox[2][0] + padding, bbox[2][1] + padding],
        [bbox[3][0] - padding, bbox[3][1] + padding]
    ]
    return new_bbox

def text_detection_v2(img):
    bbox = cf.DETECTOR.ocr(img, rec=False, cls=False)
    bboxes, img_list = crop_box(img, bbox[0])
    bounding_boxes = bboxes
    return bounding_boxes

def ocr_v2(crop_images):
    results = []
    for path in crop_images:
        s = cf.PREDICTOR.predict(path)
        results.append(s)
    results.reverse()
    results_final = ""
    for i in results:
        results_final += " "+i
    return results_final

def crop_text_images_v2(cropped_images, label, id):
    crop_images = []
    for i, cropped in enumerate(cropped_images):
        cropped.save(f'results/cropped_image_{i}.jpg')
        os.makedirs(f'./results/{id}/{label}/{i}', exist_ok=True)

        crop_images.append(cropped)
    return crop_images
def input_processing_v2(save_path, label, id):
    img = cv2.imread(save_path)
    bounding_boxes = text_detection_v2(img)
    cropped_images = crop_image_by_bbox_v2(save_path, bounding_boxes)
    crop_text_image = crop_text_images_v2(cropped_images, label, id)
    results_text = ocr_v2(crop_text_image)
    results = output_encoding_v2(results_text, label)
    return results


def main():
    loading_model()
    image_path = "./image/kehoach_5.jpg"
    process_input_image(image_path, 1000)

if __name__ == "__main__":
    main()
