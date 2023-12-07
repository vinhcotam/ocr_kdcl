import torch
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from paddleocr import PaddleOCR
import easyocr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
#directory of yolov8 weights
YOLOv8_WEIGHTS = "./weights/best_22_08_2023.pt"
#vietocr config
CONFIG_VIETOCR = Cfg.load_config_from_name('vgg_seq2seq')
CONFIG_VIETOCR['cnn']['pretrained'] = False
CONFIG_VIETOCR['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PREDICTOR = Predictor(CONFIG_VIETOCR)
#labels of yolov8 model
LABELS_NAME = {0.0: 'ngay', 1.0: 'noigui', 2.0: 'soqd', 3.0: 'tieude'}
#class
CLASSES = ["noigui", "soqd", "ngay", "tieude"]
#paddleocr config
DETECTOR = PaddleOCR(
        lang='en',
        show_log=False,
        use_space_char='True',
        det_db_box_thresh=0.6,
        drop_score=0.8,
        use_onnx=True,
        det_model_dir='onnx/text_detect.onnx',
        rec_model_dir='onnx/rec_onnx.onnx',
        cls_model_dir='onnx/cls_onnx.onnx'
    )
PATTERN_SQD = r'[Ss][ỐốồỒOoÓóÒò][:\s-]?.+'
#library convert pdf to img
POPPLER_PATH = r"F:/airc/Information_extraction_APIs/Information_extraction_APIs/Release-23.07.0-0/poppler-23.07.0/Library/bin"
#load model for summary text
TOKENIZER = AutoTokenizer.from_pretrained("./vit5_summary_model")
SUMMARY_MODEL = AutoModelForSeq2SeqLM.from_pretrained("./vit5_summary_model")
#load model easyocr
READER = easyocr.Reader(['vi'])
#stop word path
STOP_WORDS_PATH = "./stopwords/vietnamese.txt"
TOKENIZER_PATTERN= r'\w+_\w+'
EXCEL_FILE_PATH = "keywords.xlsx"
#host and port
HOST = '192.168.88.145'
PORT = 5000