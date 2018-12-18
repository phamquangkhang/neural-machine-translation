import datetime


def get_current_datetime():
    time = datetime.datetime.now()
    return time.strftime('%Y%m%d_%H%M%S')