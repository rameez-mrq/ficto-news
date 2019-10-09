# A tool to classify the writing styles of news articles.
### Steps to run the program:
- Open the terminal in the file location and run  `pip3 install -r requirements.txt`.
- install NLTK Punkt tokenizer using `nltk.download('punkt')`.
- Run `python3 ficto_news.py NEWS_URL`.

Based on the work [Qureshi, Mohammed Rameez, et al. "A Simple Approach to Classify Fictional and Non-Fictional Genres." Proceedings of the Second Workshop on Storytelling. 2019](https://www.aclweb.org/anthology/W19-3409/).

External libraries used : Stanford POS Tagger [(Toutanova et al., 2003)](https://nlp.stanford.edu/software/tagger.shtml).
