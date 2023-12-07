import io
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import flask
import sys
import um_evid_upload as vinh_library
# import paddle_ocr as vinh_library_v2
from werkzeug.exceptions import HTTPException
from flask_cors import CORS
from pdf2image import convert_from_path
import os
import config as cf
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
CORS(app)
sys.setrecursionlimit(40000)

#load lib for crop img from file pdf
cf.POPPLER_PATH

id=0
@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    return flask.jsonify({
        "status": "fail",
        "results": [],
        "error": str(e)
    }), code

@app.route("/UM_evid_upload", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    global result
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        global id
        if flask.request.files.get("image"):
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            open_cv_image = np.array(image)
            id += 1
            data["success"] = True
            #save image
            save_path = f'./save_images/{id}.jpg'
            cv2.imwrite(save_path, open_cv_image)
            #data procesing
            result = vinh_library.process_input_image(save_path, id)
        elif flask.request.files.get("pdf"):
                img_array = []
                pdf = flask.request.files["pdf"]
                pdf_save_path = f'./save_pdf/{id}.pdf'

                with open(pdf_save_path, "wb") as f:
                    f.write(pdf.read())

                # Convert all pages to images
                pages = convert_from_path(pdf_path=pdf_save_path, poppler_path=cf.POPPLER_PATH)

                # Process each page
                for page_num, page in enumerate(pages, start=1):
                    img_name = f"{page_num}.jpeg"
                    jpg_save_path = f'./save_pdf/{id}'
                    os.makedirs(jpg_save_path, exist_ok=True)
                    # Save each page as an image
                    page.save(os.path.join(jpg_save_path, img_name), "JPEG")
                    jpeg_path = f'./save_pdf/{id}/{page_num}.jpeg'
                    img_array.append(jpeg_path)
                    # Process each image
                result = vinh_library.process_input_pdf(img_array, id)
    response = flask.jsonify(result)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route("/UM_evid_upload_v2", methods=["POST"])
def predict_v2():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    if flask.request.method == "POST":
        global id
        image = flask.request.files['image']
        label = flask.request.form.get('label')
        # Đọc nội dung của tệp thành byte
        image_in_bytes = image.read()
        image = Image.open(io.BytesIO(image_in_bytes))
        open_cv_image = np.array(image)

        id += 1
        data["success"] = True
        # save image
        save_path = f'./save_images/{id}.jpg'
        cv2.imwrite(save_path, open_cv_image)
        # data processing
        result = vinh_library.input_processing_v2(save_path, label, id)

    return flask.jsonify(result)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":

    vinh_library.loading_model()
    # vinh_library_v2.loading_model()
    print(("* Loading  model and Flask starting server..." +
        "please wait until server has fully started"))
    app.run(host=cf.HOST, port=cf.PORT, threaded=True)
