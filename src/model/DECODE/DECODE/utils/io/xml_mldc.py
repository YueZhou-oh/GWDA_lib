# mypy: ignore-errors
# import synthlisa

import os.path

# import time
# import string
import re
import sys

# import math
from pathlib import Path

import numpy
import pyRXPU as pyRXP

# require xmlutils from pyRXP examples
from utils import xmlutils

# begin definitions encoding synthLISA syntax

# argumentList = {}
# outputList = {}


class readXML:
    def __init__(self, filename):
        p = pyRXP.Parser()

        f = open(filename)
        lines = f.read()
        f.close()

        try:
            tree = p(lines)
        except pyRXP.error:
            print("XML validation error! (Or perhaps I couldn't access the DTD).")
            print("I'll try to use the file anyway by removing the DTD...")

            lines = re.sub('<!DOCTYPE XSIL SYSTEM ".*">', "", lines)
            tree = p(lines)

        if tree[0] != "XSIL":
            print("Not a LISA XSIL file!")
            raise TypeError

        self.directory = os.path.dirname(filename)

        self.tw = xmlutils.TagWrapper(tree)

    def close(self):
        pass

    def getTime(self, node):
        try:
            # keep Time as string, get Type if provided
            return (str(node), node.Type)
        except AttributeError:
            return (str(node),)

    def getParam(self, node):
        try:
            # convert Param to float, get Unit if provided
            return [str(node), node.Unit]
        except AttributeError:
            return [str(node), None]

    def getDim(self, node):
        return int(str(node))

    def processSeries(self, node):
        timeseries = {}

        timeseries["Type"] = node.Type

        # I suppose 'Name' must be provided!
        timeseries["Name"] = node.Name
        timeseries["Vars"] = node.Name.split(",")

        for node2 in node:
            if node2.tagName == "Time":
                timeseries[node2.Name] = self.getTime(node2)
            elif node2.tagName == "Param":
                timeseries[node2.Name] = self.getParam(node2)
            elif node2.tagName == "Array":
                for node3 in node2:
                    if node3.tagName == "Dim":
                        timeseries[node3.Name] = self.getDim(node3)
                    elif node3.tagName == "Stream":
                        timeseries["Encoding"] = node3.Encoding

                        if node3.Type == "Remote":
                            timeseries["Filename"] = str(node3)

                            if "Binary" in timeseries["Encoding"]:
                                # assume length of doubles is 8 (generic?)
                                readlength = 8 * timeseries["Length"] * timeseries["Records"]

                                # need to catch reading errors here
                                if self.directory:
                                    binaryfile = open(
                                        self.directory + "/" + timeseries["Filename"],
                                        "rb",
                                    )
                                else:
                                    binaryfile = open(timeseries["Filename"], "rb")

                                # readbuffer = numpy.fromstring(binaryfile.read(readlength),'double')
                                readbuffer = numpy.frombuffer(binaryfile.read(readlength), "double")
                                binaryfile.close()

                                if ("BigEndian" in timeseries["Encoding"] and sys.byteorder == "little") or ("LittleEndian" in timeseries["Encoding"] and sys.byteorder == "big"):
                                    readbuffer = readbuffer.byteswap()

                                if timeseries["Records"] == 1:
                                    timeseries["Data"] = readbuffer
                                else:
                                    timeseries["Data"] = numpy.reshape(
                                        readbuffer,
                                        [
                                            timeseries["Length"],
                                            timeseries["Records"],
                                        ],
                                    )
                            else:
                                # remote data, not binary
                                raise NotImplementedError
                        elif node3.Type == "Local":
                            if "Text" in timeseries["Encoding"]:
                                timeseries["Delimiter"] = node3.Delimiter

                                datastring = str(node3)

                                for delchar in timeseries["Delimiter"]:
                                    datastring = str.join(datastring.split(delchar), " ")

                                # there may be a more efficient way to initialize an array
                                datavalues = list(map(float, datastring.split()))

                                if timeseries["Records"] == 1:
                                    timeseries["Data"] = numpy.array(datavalues, "d")
                                else:
                                    timeseries["Data"] = numpy.reshape(
                                        numpy.array(datavalues, "d"),
                                        [
                                            timeseries["Length"],
                                            timeseries["Records"],
                                        ],
                                    )

                                # should try different delimiters
                            else:
                                # local data, not textual
                                raise NotImplementedError

        return timeseries

    def getTDITimeSeries(self):
        result = []

        for node in self.tw:
            # outermost XSIL level container

            if node.tagName == "XSIL":
                if node.Type == "TDIData":
                    # inside TDIData

                    for node2 in node:
                        if node2.tagName == "XSIL":
                            if node2.Type == "TDIObservable":
                                for node3 in node2:
                                    if node3.tagName == "XSIL":
                                        if node3.Type == "TimeSeries":
                                            # got a TimeSeries!

                                            result.append(self.processSeries(node3))

        return result

    def getTDIFrequencySeries(self):
        result = []

        for node in self.tw:
            # outermost XSIL level container

            if node.tagName == "XSIL":
                if node.Type == "TDIData":
                    # inside TDIData

                    for node2 in node:
                        if node2.tagName == "XSIL":
                            if node2.Type == "TDIObservable":
                                for node3 in node2:
                                    if node3.tagName == "XSIL":
                                        if node3.Type == "FrequencySeries":
                                            # got a FrequencySeries!

                                            result.append(self.processSeries(node3))

        return result


def test():
    if len(sys.argv) < 1:
        print("Usage: %s <synthlisa.xml>" % sys.argv[0])

    inputXML = readXML(sys.argv[1])
    obs = inputXML.getTDITimeSeries()
    inputXML.close()

    t = obs[0]["Data"][:, 0]
    X = obs[0]["Data"][:, 1]
    Y = obs[0]["Data"][:, 2]
    Z = obs[0]["Data"][:, 3]
    print(t[:200])
    print(X[:200])


def data_reader_wraper(data_xml_dir):
    # data_folder_dir = "/workspace/zhaoty/GW denoise/mldc/Training-1.1.1a_LISAsim"
    # data_xml_dir = Path(data_folder_dir) / 'Training-1.1.1a.xml'
    data_xml = readXML(data_xml_dir)
    data = data_xml.getTDITimeSeries()
    data_xml.close()
    t = data[0]["Data"][:, 0]
    X = data[0]["Data"][:, 1]
    Y = data[0]["Data"][:, 2]
    Z = data[0]["Data"][:, 3]
    return t, X, Y, Z


# class FlieName(object):
#     # sig_xml_dir = Path(data_folder_dir) / 'Training-1.1.1a_noisefree' / 'Training-1.1.1a.xml'
#     def __init__(self, folder_dir):
#         self.data_folder = Path(folder_dir)
#         self.prefix = 'challenge'
#         self.tail = ['strain', 'frequency', 'nonoise']

#     def _get_tail():
#         pass

#     def _get_suffix(self, bin=False):
#         return '.bin' if bin else '.xml'

#     def __call__(self, challenge_num=[1, 1, '1a'], data=False):
#         fn = self.prefix + '{}'.format()
#         pass


def main():
    pass


if __name__ == "__main__":
    main()
