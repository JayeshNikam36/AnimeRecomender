import sys
import traceback

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: Exception):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_detail: Exception) -> str:
        # Get traceback info
        _, _, exc_tb = sys.exc_info()
        if exc_tb:
            line_number = exc_tb.tb_lineno
            file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            line_number = "Unknown"
            file_name = "Unknown"

        detailed_message = (
            f"Error occurred in script: {file_name} "
            f"at line number: {line_number} "
            f"with original message: {str(error_detail)} "
            f"\nContext: {error_message}"
        )
        return detailed_message

    def __str__(self):
        return self.error_message
