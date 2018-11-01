# -*- coding: UTF-8 -*-

import logging

logging.basicConfig(level=logging.INFO)


class Doc:
    """
    文本类型
    """

    def __init__(self, content, id=None, title=None, author=None, source=None, label=None):
        self.id = str(id)
        self.title = str(title)
        self.author = str(author)
        self.source = str(source)
        self.content = str(content)
        self.label = str(label)

    def __str__(self):
        doc = ""
        if self.id is not None and self.id != "":
            doc += "id:" + self.id + "\t"
        if self.title is not None and self.title != "":
            doc += "title:" + self.title + "\t"
        if self.author is not None and self.author != "":
            doc += "author:" + self.author + "\t"
        if self.source is not None and self.source != "":
            doc += "source:" + self.source + "\t"
        if self.content is not None and self.content != "":
            doc += "content:" + self.content + "\t"
        if self.label is not None and self.label != "":
            doc += "label:" + self.label + "\t"
        return doc.rstrip()
