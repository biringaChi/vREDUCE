import os
import gensim
import typing
import pickle
import pathlib
import numpy as np

class Help:
	def __init__(self) -> None:
		pass

	def __len__(self, arg) -> int:
		if (isinstance(arg, (int, float, bool))):
			raise TypeError("Invalid argument. Only text, sequence, mapping and set are accepted")
		else:
			return len(arg)

	def pickle(self, data, file_name):
		with open(file_name, "wb") as file:
			pickle.dump(data, file)
	
	def unpickle(self, data):
		with open(data, "rb") as file:
			loaded = pickle.load(file)
		return loaded

	def get_project(self, project):
		source_files, source_file_names = [], []
		for root, _, files in os.walk(project):
			for file in files:
				if file.endswith(".java"):
					temp = os.path.join(root, file) 
					source_file_names.append(pathlib.Path(temp).stem)
					try:
						with open(temp, "r") as source_file:
							source_files.append(source_file.read())
					except OSError as e:
						raise e
		return source_files, source_file_names
	
	def read_txtfile(self, file):
		with open(file) as f:
			return f.readlines()

class NR(Help):
	def __init__(self, project = None) -> None:
		super().__init__()
		self.project = project
		self.seqlen = self.max_seqlen
	
	def node_selection(self, tree) -> typing.List:
		return [node.__class__.__name__ for _, node in tree if node.__class__.__name__ in self.features]

	def _model(self, data):
		vec_dict = {}
		model = gensim.models.Word2Vec(sentences = data, vector_size = 32).wv
		for key in model.key_to_index.keys():
			vec_dict[key] = model[key]
		out = []
		for embeds in data:
			temp = []
			for embed in embeds:
				if embed in vec_dict:
					temp.append(vec_dict[embed])
			out.append(temp)
		return out
	
	def truncate(self, embeddings):
		return [vector[:self.seqlen] if self.__len__(vector) > self.seqlen else vector for vector in embeddings] 
		
	def zeropad(self, embeddings):
		out  = []		
		for embedding in embeddings:
			if len(embedding) < self.seqlen:
				embedding.extend([0.0] * (self.seqlen - len(embedding)))
		for embedding in embeddings:
			temp = []
			for vector in embedding:
				if not type(vector).__module__ == np.__name__:
					temp.append(np.zeros((32,), dtype = np.float64))
				else: temp.append(vector)
			out.append(temp)
		return out
	
	def flatten(self, embeddings):
		out = []
		for embedding in embeddings:
			if not embedding:
				out.append(embedding)
			else:
				flatten_list = np.concatenate(embedding).ravel().tolist()
				out.append(flatten_list) 
		return out
			
	def __call__(self):
		trees, _ = self.get_trees(self.get_project(self.project))
		features = [self.node_selection(tree) for tree in trees]
		return self.zeropad(self.truncate(self.vector_match(self.model(features, "vectors.kv"))))