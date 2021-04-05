# -*- coding: utf-8 -*-

from .detection import get_detector, get_textbox
from .recognition import get_recognizer, get_text
from .utils import (
    group_text_box, get_image_list, calculate_md5, get_paragraph,
    diff, reformat_input, make_rotated_img_list, set_result_with_confidence,
    compat_tr,
)
from .config import *
from bidi.algorithm import get_display
import cv2
import torch
import numpy as np
import os
import sys
from PIL import Image
from logging import getLogger
import yaml


if sys.version_info[0] == 2:
    from io import open
    from six.moves.urllib.request import urlretrieve
    from pathlib2 import Path
else:
    from urllib.request import urlretrieve
    from pathlib import Path


LOGGER = getLogger(__name__)


class Application(object):

    def __init__(
        self,
        lang_list,
        gpu_device=True,
        recog_network='standard',
        detector=True,
        recognizer=True,
        verbose=True,
        quantize=True
    ):
        """
        Parameters:
            lang_list (list): Language codes (ISO 639) for languages to be recognized during analysis.
            gpu_device (bool): Enable GPU support (default)
        """
        self.model_storage_directory = os.path.join(MODULE_PATH, 'model')
        self.user_network_directory = os.path.join(MODULE_PATH, 'user_network')
        Path(self.model_storage_directory).mkdir(parents=True, exist_ok=True)
        Path(self.user_network_directory).mkdir(parents=True, exist_ok=True)
        sys.path.append(self.user_network_directory)

        if gpu_device is False:
            self.device = 'cpu'
            LOGGER.warning('Using CPU. Note: This module is much faster with a GPU.')
        elif not torch.cuda.is_available():
            self.device = 'cpu'
            LOGGER.warning('CUDA not available - defaulting to CPU. Note: This module is much faster with a GPU.')
        elif gpu_device is True:
            self.device = 'cuda'
        else:
            self.device = gpu_device

        # detection model
        detector_model = 'craft'
        detector_path = os.path.join(
            self.model_storage_directory,
            detection_models[detector_model]['filename']
        )
        if detector:
            if not os.path.isfile(detector_path):
                raise FileNotFoundError("Missing %s and downloads disabled" % detector_path)
            elif calculate_md5(detector_path) != detection_models[detector_model]['filesize']:
                raise FileNotFoundError("MD5 mismatch for %s and downloads disabled" % detector_path)

        # recognition model
        separator_list = {}
        recog_networks = [
            'standard',
            *[model for model in recognition_models['gen1']],
            *[model for model in recognition_models['gen2']]
        ]

        if recog_network in recog_networks:

            if recog_network in [model for model in recognition_models['gen1']]:
                model = recognition_models['gen1'][recog_network]
                recog_network = 'generation1'
                self.model_lang = model['model_script']

            elif recog_network in [model for model in recognition_models['gen2']]:
                model = recognition_models['gen2'][recog_network]
                recog_network = 'generation2'
                self.model_lang = model['model_script']

            else:
                unknown_lang = set(lang_list) - set(all_lang_list)
                if unknown_lang != set():
                    raise ValueError(unknown_lang, 'is not supported')
                if lang_list == ['en']:
                    self.setModelLanguage('english', lang_list, ['en'], '["en"]')
                    model = recognition_models['gen2']['english_g2']
                    recog_network = 'generation2'
                elif 'ch_tra' in lang_list:
                    self.setModelLanguage('chinese_tra', lang_list, ['ch_tra','en'], '["ch_tra","en"]')
                    model = recognition_models['gen1']['zh_tra_g1']
                    recog_network = 'generation1'
                elif 'ch_sim' in lang_list:
                    self.setModelLanguage('chinese_sim', lang_list, ['ch_sim','en'], '["ch_sim","en"]')
                    model = recognition_models['gen2']['zh_sim_g2']
                    recog_network = 'generation2'
                else:
                    raise NotImplementedError(f"undefined model information: {lang_list!r}")

            self.character = model['characters']

            # check recognition model file
            model_path = os.path.join(self.model_storage_directory, model['filename'])
            if recognizer:
                if not os.path.isfile(model_path):
                    raise FileNotFoundError("Missing %s" % model_path)
                elif calculate_md5(model_path) != model['filesize']:
                    raise FileNotFoundError("MD5 mismatch for %s" % model_path)
            self.setLanguageList(lang_list, model)

        else: # user-defined model
            with open(os.path.join(self.user_network_directory, recog_network+ '.yaml')) as file:
                recog_config = yaml.load(file, Loader=yaml.FullLoader)
            imgH = recog_config['imgH']
            available_lang = recog_config['lang_list']
            self.setModelLanguage(recog_network, lang_list, available_lang, available_lang)
            char_file = os.path.join(self.user_network_directory, recog_network+ '.txt')
            self.character = recog_config['character_list']
            model_file = f"{recog_network}.pth"
            model_path = os.path.join(self.model_storage_directory, model_file)
            self.setLanguageList(lang_list, None)

        dict_list = {}
        for lang in lang_list:
            dict_list[lang] = os.path.join(BASE_PATH, 'dict', lang + ".txt")

        if detector:
            self.detector = get_detector(detector_path, self.device, quantize)

        if recognizer:
            if recog_network == 'generation1':
                network_params = {
                    'input_channel': 1,
                    'output_channel': 512,
                    'hidden_size': 512
                }
            elif recog_network == 'generation2':
                network_params = {
                    'input_channel': 1,
                    'output_channel': 256,
                    'hidden_size': 256
                }
            else:
                network_params = recog_config['network_params']

            self.recognizer, self.converter = get_recognizer(
                recog_network, network_params,
                self.character, separator_list,
                dict_list, model_path,
                device=self.device,
                quantize=quantize
            )

    def setModelLanguage(self, language, lang_list, list_lang, list_lang_string):
        self.model_lang = language
        if set(lang_list) - set(list_lang) != set():
            if language == 'ch_tra' or language == 'ch_sim':
                language = 'chinese'
            raise ValueError(language.capitalize() + ' is only compatible with English, try lang_list=' + list_lang_string)

    def getChar(self, fileName):
        char_file = os.path.join(BASE_PATH, 'character', fileName)
        with open(char_file, "r", encoding="utf-8-sig") as input_file:
            list = input_file.read().splitlines()
            char = ''.join(list)
        return char

    def setLanguageList(self, lang_list, model):
        self.lang_char = []
        for lang in lang_list:
            char_file = os.path.join(BASE_PATH, 'character', lang + "_char.txt")
            with open(char_file, "r", encoding = "utf-8-sig") as input_file:
                char_list =  input_file.read().splitlines()
            self.lang_char += char_list

        if model:
            symbol = model['symbols']
        else:
            symbol = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '

        self.lang_char = set(self.lang_char).union(set(symbol))
        self.lang_char = ''.join(self.lang_char)

    def detect(
        self,
        img, min_size=20,
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.4,
        canvas_size=2560,
        mag_ratio=1.,
        slope_ths=0.1,
        ycenter_ths=0.5,
        height_ths=0.5,
        width_ths=0.5,
        add_margin=0.1,
        reformat=True,
        optimal_num_chars=None
    ):

        if reformat:
            img, img_cv_grey = reformat_input(img)

        text_box = get_textbox(
            self.detector,
            img, canvas_size, mag_ratio,
            text_threshold, link_threshold,
            low_text, False,
            self.device, optimal_num_chars
        )
        horizontal_list, free_list = group_text_box(
            text_box, slope_ths,
            ycenter_ths, height_ths,
            width_ths, add_margin,
            (optimal_num_chars is None)
        )

        if min_size:
            horizontal_list = [i for i in horizontal_list if max(i[1]-i[0],i[3]-i[2]) > min_size]
            free_list = [i for i in free_list if max(diff([c[0] for c in i]), diff([c[1] for c in i]))>min_size]

        return horizontal_list, free_list

    def recognize(
        self,
        img_cv_grey,
        horizontal_list=None,
        free_list=None,
        decoder='greedy',
        beamWidth=5,
        batch_size = 1,
        workers=0,
        allowlist=None,
        blocklist=None,
        detail=1,
        rotation_info=None,
        paragraph=False,
        contrast_ths=0.1,
        adjust_contrast=0.5,
        filter_ths=0.003,
        reformat=True
    ):

        if reformat:
            img, img_cv_grey = reformat_input(img_cv_grey)

        if allowlist:
            ignore_char = ''.join(set(self.character)-set(allowlist))
        elif blocklist:
            ignore_char = ''.join(set(blocklist))
        else:
            ignore_char = ''.join(set(self.character)-set(self.lang_char))

        if self.model_lang in ['chinese_tra','chinese_sim']:
            decoder = 'greedy'

        if (horizontal_list==None) and (free_list==None):
            y_max, x_max = img_cv_grey.shape
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []

        if ((batch_size == 1) or (self.device == 'cpu')) and not rotation_info:
            result = []

            for bbox in horizontal_list:
                h_list = [bbox]
                f_list = []
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
                result0 = get_text(
                    self.character,
                    imgH, int(max_width),
                    self.recognizer, self.converter,
                    image_list, ignore_char,
                    decoder, beamWidth, batch_size,
                    contrast_ths, adjust_contrast, filter_ths,
                    workers, self.device
                )
                result += result0

            for bbox in free_list:
                h_list = []
                f_list = [bbox]
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
                result0 = get_text(
                    self.character,
                    imgH, int(max_width),
                    self.recognizer, self.converter,
                    image_list, ignore_char,
                    decoder, beamWidth, batch_size,
                    contrast_ths, adjust_contrast, filter_ths,
                    workers, self.device
                )
                result += result0

        else:
            image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey, model_height = imgH)
            image_len = len(image_list)
            if rotation_info and image_list:
                image_list = make_rotated_img_list(rotation_info, image_list)
                max_width = max(max_width, imgH)

            result = get_text(
                self.character,
                imgH, int(max_width),
                self.recognizer, self.converter,
                image_list, ignore_char,
                decoder, beamWidth, batch_size,
                contrast_ths, adjust_contrast, filter_ths,
                workers, self.device
            )

            if rotation_info and (horizontal_list+free_list):
                result = set_result_with_confidence(result, image_len)

        direction_mode = 'ltr'
        if paragraph:
            result = get_paragraph(result, mode = direction_mode)
        result = list(map(lambda x: list(x), result))

        if detail == 0:
            result = [item[1] for item in result]
        else:
            result = [[compat_tr(elem[0]), *elem[1:]] for elem in result]

        return result

    def run(
        self,
        image,
        decoder='greedy',
        beamWidth=5,
        batch_size=1,
        workers=0,
        allowlist=None,
        blocklist=None,
        detail=1,
        rotation_info=None,
        paragraph=False,
        min_size=20,
        contrast_ths=0.1,
        adjust_contrast=0.5,
        filter_ths=0.003,
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.4,
        canvas_size=2560,
        mag_ratio=1.,
        slope_ths=0.1,
        ycenter_ths=0.5,
        height_ths=0.5,
        width_ths=0.5,
        add_margin=0.1
    ):
        '''
        Parameters:
            image: file path or numpy-array or a byte stream object
        '''
        img, img_cv_grey = reformat_input(image)

        horizontal_list, free_list = self.detect(
            img, min_size, text_threshold,
            low_text, link_threshold,
            canvas_size, mag_ratio,
            slope_ths, ycenter_ths,
            height_ths,width_ths,
            add_margin, False
        )

        result = self.recognize(
            img_cv_grey, horizontal_list, free_list,
            decoder, beamWidth, batch_size,
            workers, allowlist, blocklist, detail, rotation_info,
            paragraph, contrast_ths, adjust_contrast,
            filter_ths, False
        )

        torch.cuda.empty_cache()
        return result
