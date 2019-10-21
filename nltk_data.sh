#A folder inside .venv to save nltk resources
mkdir .venv/nltk_data

#Punkt Tokenizer, to use tokenize functions in nltk
#https://groups.google.com/forum/#!topic/nltk-users/LEGj1IHJJ1g
python -m nltk.downloader -d .venv/nltk_data punkt

#Stop words of English lang
python -m nltk.downloader -d .venv/nltk_data stopwords