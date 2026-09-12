import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):

    '''
    gives the error in a specifc manner respect to the custom exception
    '''

    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logging.error(self.error_message)

    def __str__(self):
        '''
        when u raise the error this will show the error
        '''
        return self.error_message

