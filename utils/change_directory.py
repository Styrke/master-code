import os

class cd:
    """Context manager for changing the current working directory.
    http://stackoverflow.com/questions/431684/how-do-i-cd-in-python/13197763#13197763
    """
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        if not os.path.exists(self.newPath):
            print("'{0}' does not exist. Creating directory..".format(self.newPath))
            os.makedirs(self.newPath)
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
