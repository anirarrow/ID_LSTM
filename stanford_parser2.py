import stanfordnlp
stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
#doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
doc=nlp("Musicland is trying to embrace the Internet and emulate the retail chains like Starbucks.")
doc.sentences[0].print_dependencies()