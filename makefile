deps:
	pip install -r requirements.txt
	conda install faiss-cpu -c pytorch
	conda install -c anaconda ipykernel
	python -m ipykernel install --user --name=text_mining