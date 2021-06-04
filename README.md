# **Text Mining Medium Posts**

**Author:** Victor Pontes 2020-06-03

> This repository is intended to record the procedures presented in the series of posts about Text Mining on my [Medium Blog Post](https://medium.com/@victoraleff).

### Published Articles:
+ [Mineração de Texto - Framework de Prospecção de Categorias](https://medium.com/@victoraleff)

### **Basic Usage**

+ First of all, download the dataset [News of Brazilian Newspaper](https://www.kaggle.com/marlesson/news-of-the-site-folhauol?select=articles.csv) and extract the articles.csv file into the 'data' directory of this repository.

+ Then it is necessary to configure the environment. I strongly recommend the use of virtual environments with conda. 
Above there is an example creating and activating a conda virtual environment.

```
	conda create -n text_mining python=3.8
	conda activate text_mining
```

+ Install the dependencies:

```
	make deps
```
+ After setting up the environment, you can run the get_base_files.py python script to generate the text mining models. In this step, text pre-processing, bag of words, topic modeling and clustering are performed.

```
	python get_base_files.py
```

+ From now on, if everything went well, the notebooks with the experiments can be run.

> :warning: **Notebooks are being updated and will be available in a few minutes.**