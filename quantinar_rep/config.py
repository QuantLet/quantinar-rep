import logging
import json
from quantinar_rep.constants import COURSE_BASE, COURSELET_BASE, ORDER_BASE,\
    PAGE_BASE, REVIEW_BASE, USER_BASE

DATA_DIR_PATH = "./quantinar_rep/data_20221227"
EDGES_HISTORY_PATH = f"{DATA_DIR_PATH}/all_edges.csv"
PV_HISTORY_PATH = f"{DATA_DIR_PATH}/pv_edges.csv"
NODE_TYPE_MAPPER = json.load(open(
    f"{DATA_DIR_PATH}/node_type_mapper.json", "r"))

LOGGER_LEVEL = "debug"
# https://github.com/sourcecred/sourcecred/blob/74fda4e1050a836a4a877c6a5fa7ceb18d4934c1/packages/sourcecred/src/core/credrank/compute.js
DEFAULT_ALPHA = 0.1
DEFAULT_BETA = 0.4
DEFAULT_GAMMA_FORWARD = 0.1
DEFAULT_GAMMA_BACKWARD = 0.1

FREQ_DAYS = 7
NODE_WEIGHTS = {
    PAGE_BASE: 0,
    COURSE_BASE: 0,
    COURSELET_BASE: 8,
    USER_BASE: 0,
    REVIEW_BASE: 1,
    ORDER_BASE: 1
}

MINTING_CONTRI_LABEL = [k for k in NODE_WEIGHTS if NODE_WEIGHTS[k] != 0]
DEFAULT_WEIGHTS = {
    "pv_course": 1e-5,

    "course_user": 1.,
    "user_course": 1 / 8,

    "courselet_user": 1.,
    "user_courselet": 1 / 8,

    "course_courselet": 1.,
    "courselet_course": 1 / 8,

    "order_course": 5.,
    "course_order": 1 / 16,  # 1 / 16,

    "order_user": 1.,
    "user_order": 1 / 16,  # 1 / 16,

    "review_user": 1.,
    "user_review": 1 / 8,  # 1e-5,

    "review_course": 2.,
    "course_review": 1 / 16,  # 1e-5,
}
PERSONALIZATION_METHOD = "seed"

VALID_WEIGHT_KEYS = list(DEFAULT_WEIGHTS.keys())


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    blue = '\x1b[38;5;39m'
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(name)s:%(filename)s:%(lineno)d] - [%(process)d] - %(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name_logger, level="debug"):
    logger_to_ret = logging.getLogger(name_logger)
    if level == "debug":
        level = logging.DEBUG
    elif level == "info":
        level = logging.INFO
    elif level == "warning":
        level = logging.WARNING
    else:
        raise NotImplementedError()
    logger_to_ret.setLevel(level)

    stdout_logger = logging.StreamHandler()
    stdout_logger.setLevel(level)
    stdout_logger.setFormatter(CustomFormatter())
    logger_to_ret.addHandler(stdout_logger)
    logger_to_ret.propagate = False

    return logger_to_ret


LOGGER = get_logger("quantinar_rep-Logger", level=LOGGER_LEVEL)