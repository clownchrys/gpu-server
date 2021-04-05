import traceback
import numpy as np
from flask import (
    Blueprint,
    request, jsonify,
)

from modules import ocr


app = Blueprint('common', __name__)
ocr_app = ocr.Application(['ch_sim', 'en'], gpu_device=True)


@app.route('/health_check', methods=['GET', 'POST'])
def health_check():
    result = {
        'success': True,
        'data': 'status OK'
    }
    return jsonify(result)


@app.route('/ocr_run', methods=['GET', 'POST'])
def ocr_run():
    result = {}

    try:
        src_img = request.json['src_img']  # it can be list of (string url, numpy array, or byte io)
        include_prob = request.json.get('include_prob', True)

    except:
        result['success'] = False
        result['data'] = 'json should contain "src_img" argument'
        status_code = 401
        return jsonify(result), status_code

    try:
        data = ocr_app.run(src_img, detail=1)
        if not include_prob:
            data = [[coords, text] for coords, text, prob in data]

        result['success'] = True
        result['data'] = data
        status_code = 200

    except:
        result['success'] = False
        result['data'] = traceback.format_exc()
        status_code = 400

    finally:
        return jsonify(result), status_code
