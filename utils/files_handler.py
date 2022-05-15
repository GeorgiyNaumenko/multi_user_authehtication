import os


def folder_state(folder_path):
    """

    :param folder_path:
    :return:
    """
    try:
        os.stat(folder_path)
        return True
    except Exception as e:
        print(f"Exception: {e}. {folder_path} does not exist.")
        return False


def create_folder(folder_path):
    """

    :param folder_path:
    :return:
    """
    try:
        os.stat(folder_path)
        print(f"{folder_path} already exists.")
        return True
    except Exception as e:
        print(f"Exception: {e}.\n Created new folder {folder_path}")
        os.mkdir(folder_path)
        return False
