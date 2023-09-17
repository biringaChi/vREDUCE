import pathlib

class Configuration:
	"""
	Configurations
	"""
	def __init__(self) -> None:
		self.dir_navigator = pathlib.Path.cwd().parents[0]
		self.d2a_csv_train = self.dir_navigator / "data/d2a/function/train.csv"
		self.d2a_csv_dev = self.dir_navigator / "data/d2a/function/dev.csv"
		self.d2a_csv_test = self.dir_navigator / "data/d2a/function/test.csv"

		self.d2a_cfile_train = self.dir_navigator / "data/d2a/raw/train"
		self.d2a_cfile_dev = self.dir_navigator / "data/d2a/raw/dev"
		self.d2a_cfile_test = self.dir_navigator / "data/d2a/raw/test"