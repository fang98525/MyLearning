import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp('Start using Neo4j with built-in guides.Learn the basics of graph database technology ')

type(doc)