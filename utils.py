import logging
from blocks.extensions import SimpleExtension
logger = logging.getLogger('main.utils')


class SaveLog(SimpleExtension):
    def __init__(self, dir, show=None, **kwargs):
        super(SaveLog, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        epoch = self.main_loop.status['epochs_done']
        current_row = self.main_loop.log.current_row
        logger.info("\nIter:%d" % epoch)
        for element in current_row:
            logger.info(str(element) + ":%f" % current_row[element])
