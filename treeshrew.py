import os
import numpy as np
import networkx as nx
from keybert import KeyBERT
import yake
from pymorphy2 import MorphAnalyzer
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, Doc
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QComboBox, QHBoxLayout, QPushButton, QLineEdit, QFileDialog, QGridLayout
from PyQt5.QtGui import QDesktopServices, QColor
from PyQt5.QtCore import QUrl
import pyqtgraph as pg

# Preprocessing and other functions
morph = MorphAnalyzer()
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

def preprocess_text(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)

    lemmas = []
    for token in doc.tokens:
        lemmas.append(morph.normal_forms(token.text)[0])
    return " ".join(lemmas)

def extract_key_concepts(texts):
    kw_model = KeyBERT()
    yake_kw_extractor = yake.KeywordExtractor(lan="ru", n=1, top=10)
    yake_kw_extractor_2 = yake.KeywordExtractor(lan="ru", n=2, top=10)
    SW = ["image", "png", "jpg", "рисунок", "com", "screen", "https"]
    all_keywords = []
    for text in texts:
        keybert_keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='russian', top_n=10)
        yake_keywords = yake_kw_extractor.extract_keywords(text)
        yake_keywords_2 = yake_kw_extractor_2.extract_keywords(text)
        keywords = [kw for kw, score in keybert_keywords] + [kw for kw, score in yake_keywords] + [kw for kw, score in yake_keywords_2]
        keywords = [elem for elem in keywords if not elem in SW]
        keywords = list(set(keywords))
        all_keywords.append(keywords)
    for i in range(len(all_keywords) - 1, -1, -1):
        for j in range(len(all_keywords[i]) - 1, -1, -1):
            doc = Doc(all_keywords[i][j])
            doc.segment(segmenter)
            doc.tag_morph(morph_tagger)
            if not ((len(doc.tokens) > 1) or (list(doc.tokens[0])[6] == 'NOUN')):
                all_keywords[i].pop(j)
    return all_keywords

def compute_embeddings(texts):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model.encode(texts)

def compute_similarity_matrix(embeddings):
    return cosine_similarity(embeddings)

class MindMapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mind Map")
        self.resize(1000, 800)

        layout = QVBoxLayout()
        
        # Button to load files
        self.load_button = QPushButton("Load Markdown Files")
        self.load_button.clicked.connect(self.load_files)
        layout.addWidget(self.load_button)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.view = self.plot_widget.addViewBox()
        self.view.setAspectLocked()
        self.view.setMouseEnabled(True, True)
        layout.addWidget(self.plot_widget)

        # Keyword selector and search bar
        self.keyword_selector = QComboBox()
        self.keyword_selector.currentIndexChanged.connect(self.highlight_keywords)
        self.keyword_selector.setMinimumWidth(250)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search by filename or keyword...")
        self.search_bar.textChanged.connect(self.search_nodes)

        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.keyword_selector)
        top_layout.addWidget(self.search_bar)
        top_layout.addWidget(self.reset_button)
        layout.addLayout(top_layout)

        self.info_label = QLabel("Click on a file node to view information here.")
        layout.addWidget(self.info_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        self.info_layout = QGridLayout()
        self.info_layout.setContentsMargins(10, 10, 10, 10)
        layout.addLayout(self.info_layout)

        self.graph = nx.Graph()
        self.text_items = []  # Store text items here
        self.file_keywords = {}  # Store keywords for each file

    def load_files(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select Markdown Files", "", "Markdown Files (*.md);;All Files (*)", options=options)
        
        if files:
            filenames = [os.path.basename(file) for file in files]
            texts = [open(file, "r", encoding="utf-8").read() for file in files]
            preprocessed_texts = [preprocess_text(text) for text in texts]
            keywords = extract_key_concepts(preprocessed_texts)
            self.file_keywords = {file: kw for file, kw in zip(filenames, keywords)}  # Store keywords
            self.keyword_selector.addItems(set(kw for sublist in self.file_keywords.values() for kw in sublist))
            self.keyword_selector.adjustSize()

            file_info = {file: {"path": file, "description": ", ".join(kw)} for file, kw in zip(filenames, keywords)}

            embeddings = compute_embeddings(preprocessed_texts)
            similarity_matrix = compute_similarity_matrix(embeddings)

            self.setup_mind_map(filenames, similarity_matrix, file_info)

    def setup_mind_map(self, filenames, similarity_matrix, file_info, threshold=0.6):
        self.view.setBackgroundColor(QColor(240, 240, 240))
        self.graph.clear()  # Clear previous graph
        self.text_items.clear()  # Clear previous text items

        for file1 in filenames:
            self.graph.add_node(file1)

        for i, file1 in enumerate(filenames):
            self.graph.add_node(file1, **file_info[file1])
            max_num = 0
            for j, file2 in enumerate(filenames):
                if i != j and max_num < similarity_matrix[i][j]:
                    max_num = similarity_matrix[i][j] 
            for j, file2 in enumerate(filenames):
                if i != j and (similarity_matrix[i][j] > threshold) and (similarity_matrix[i][j] > 0.9 * max_num):
                    self.graph.add_edge(file1, file2, weight=similarity_matrix[i][j])

        pos = nx.spring_layout(self.graph, k = 1, scale = 5, seed=0, iterations=25)
        
        for file1 in filenames:
            x, y = pos[file1]
            text_item = pg.TextItem(file1, color=(0, 0, 0), anchor=(0.5, 0.5))
            text_item.setPos(x, y)
            text_item.file = file1
            text_item.mousePressEvent = lambda event, item=text_item: self.node_clicked(item, file_info)
            text_item.hoverEvent = lambda event, item=text_item: self.node_hovered(event, item)
            self.view.addItem(text_item)
            self.text_items.append(text_item)  # Store the item for later access

            for neighbor in self.graph.neighbors(file1):
                x1, y1 = pos[file1]
                x2, y2 = pos[neighbor]
                weight = similarity_matrix[filenames.index(file1)][filenames.index(neighbor)]
                line_width = max(1, 2 * weight)
                brightness = int(255 * (weight - threshold) / (1 - threshold))
                color = QColor(brightness, 100, 255)
                line = pg.PlotDataItem([x1, x2], [y1, y2], pen=pg.mkPen(color, width=line_width))
                self.view.addItem(line)

    def highlight_keywords(self):
        selected_keyword = self.keyword_selector.currentText().lower()
        for item in self.text_items:
            keywords = self.file_keywords.get(item.file, [])
            if any(selected_keyword in kw.lower() for kw in keywords):
                item.setText(item.file, color=(255, 0, 0))  # Highlight matching nodes
            else:
                item.setText(item.file, color=(0, 0, 0))  # Default color

    def search_nodes(self, text):
        text = text.lower()  # Convert search text to lowercase
        for item in self.text_items:
            keywords = self.file_keywords.get(item.file, [])
            if any(text in kw.lower() for kw in keywords):  # Check both filename and keywords
                item.setText(item.file, color=(255, 0, 0))  # Highlight found nodes
            else:
                item.setText(item.file, color=(0, 0, 0))
     
    def reset_view(self):
        self.search_bar.clear()
        self.keyword_selector.setCurrentIndex(-1)
        for item in self.text_items:
            item.setText(item.file, color=(0, 0, 0))

    def node_hovered(self, event, item):
        if event.isEnter():
            self.info_label.setText(f"Hovered over: {item.file}")

    def node_clicked(self, item, file_info):
        file = item.file
        info = file_info.get(file, {})
        file_path = info.get("path", "")
        self.info_label.setText(f"File: {file}\nDescription: {info.get('description', '')}\nPath: {file_path}")

        if os.path.exists(file_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))

# Main execution
def main():            
    app = QApplication([])
    window = MindMapWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()