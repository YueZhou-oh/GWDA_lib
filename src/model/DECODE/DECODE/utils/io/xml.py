# mypy: ignore-errors
# this is part of pyRXP, version 1.0.7
# see the pyRXP distribution directory for licensing, etc.

"Some XML helper classes."
import os
import string
import sys

# from types import StringType, ListType, TupleType
import pyRXPU as pyRXP

assert pyRXP.version >= "0.5", "get the latest pyRXP!"

IGNOREWHITESPACE = 1


def ignoreWhitespace(list):
    newlist = []
    for elem in list:
        if type(elem) is str:
            short = elem.strip()
            if short == "":
                pass
            else:
                newlist.append(short)
        else:
            newlist.append(elem)
    return newlist


class TagWrapper:
    """Lazy utility for navigating XML.

    The following Python code works:

    tag.attribute      # returns given attribute
    tag.child          # returns first child with matching tag name
    for child in tag:  # iterates over them
    tag[3]             # returns fourth child
    len(tag)           # no of children
    """

    def __init__(self, node, returnEmptyTagContentAsString=1):
        tagName, attrs, children, spare = node
        self.tagName = tagName

        # this option affects tags with no content like <Surname></Surname>.
        # Can either return a None object, which is a pain in a prep file
        # as you have to  put if expressions around everything, or
        # an empty string so prep files can just do {{xml.wherever.Surname}}.
        self.returnEmptyTagContentAsString = returnEmptyTagContentAsString

        if attrs is None:
            self._attrs = {}
        else:
            self._attrs = attrs  # share the dictionary

        if children is None:
            self._children = []
        elif IGNOREWHITESPACE:
            self._children = ignoreWhitespace(children)
        else:
            self._children = children

    def __repr__(self):
        return "TagWrapper<%s>" % self.tagName

    def __str__(self):
        if len(self):
            return str(self[0])
        else:
            if self.returnEmptyTagContentAsString:
                return ""
            else:
                return None

    def __len__(self):
        return len(self._children)

    def _value(self, name, default):
        try:
            return getattr(self, name)[0]
        except (AttributeError, IndexError):
            return default

    def __getattr__(self, attr):
        "Try various priorities"
        if attr in self._attrs:
            return self._attrs[attr]
        else:
            # first child tag whose name matches?
            for child in self._children:
                if type(child) is str:
                    pass
                else:
                    tagName, attrs, children, spare = child
                    if tagName == attr:
                        t = TagWrapper(child)
                        t.returnEmptyTagContentAsString = self.returnEmptyTagContentAsString
                        return t
            # not found, barf
            msg = f'"{attr}" not found in attributes of tag <{self.tagName}> or its children'
            raise AttributeError(msg)

    def keys(self):
        "return list of valid keys"
        result = list(self._attrs.keys())
        for child in self._children:
            if type(child) is str:
                pass
            else:
                result.append(child[0])
        return result

    def has_key(self, k):
        return k in list(self.keys())

    def __getitem__(self, idx):
        try:
            child = self._children[idx]
        except IndexError:
            raise IndexError(f"{self.__repr__()} no index {repr(idx)}")
        if type(child) is str:
            return child
        else:
            return TagWrapper(child)

    def _namedChildren(self, name):
        R = []
        for c in self:
            if type(c) is str:
                if name is None:
                    R.append(c)
            elif name == c.tagName:
                R.append(c)
        return R


def xml2doctree(xml):
    pyRXP_parse = pyRXP.Parser(
        ErrorOnValidityErrors=1,
        NoNoDTDWarning=1,
        ExpandCharacterEntities=0,
        ExpandGeneralEntities=0,
    )
    return pyRXP_parse.parse(xml)


if __name__ == "__main__":
    import os

    xml = open("rml_manual.xml").read()
    parsed = xml2doctree(xml)