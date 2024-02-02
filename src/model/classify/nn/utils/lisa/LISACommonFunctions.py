# mypy: ignore-errors
"""
    Copyright (C) 2017 Stas Babak, Antoine Petiteau for the LDC team

    This file is part of LISA Data Challenge.

    LISA Data Challenge is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
"""

##################################################
#                                                #
#            LISA common functions               #
#                 version 1.0                    #
#                                                #
#         A. Petiteau, ...                       #
#      for the LISA Data Challenge team          #
#                                                #
##################################################


import os
import re
import sys

import numpy as np

# import subprocess
import pyfftw


def run(command, disp=False, NoExit=False):
    """
    Run system command
    @param command is a string to run as a system command
    @param disp is true to display the command
    @param is true to continue if the command failed
    """
    commandline = command % globals()
    if disp:
        print("----> %s" % commandline)
    try:
        assert os.system(commandline) == 0
    except:
        print(f'Script {sys.argv[0]} failed at the sytem command "{commandline}".')
        if not NoExit:
            sys.exit(1)
        else:
            print("continue anyway ...")


def makefromtemplate(output, template, keyChar, **kwargs):
    """
    Create an output file identical to the template file with the string
    between the key character KeyChar and corresponding to a keyword in kwargs
    replaced by the argument of the keyword
    @param output is the output file
    @param template is the template file
    @param keyChar is the key character surrounding keywords
    @param kwargs are keywords and corresponding arguments
    """
    fi = open(template)
    fo = open(output, "w")
    for line in fi:
        repline = line
        for kw in kwargs:
            repline = re.sub(keyChar + kw + keyChar, str(kwargs[kw]), repline)
        print(repline, end=" ", file=fo)


def LoadFileNpStruct(FileName):
    """
    Load txt file of data and return a structured numpy array
    @param filename with one column per record and title of the record on the first line "#rec1 rec1 ..."
    @return structured numpy array
    """
    fIn = open(FileName)
    lines = fIn.readlines()
    fIn.close()
    w = re.split(r"\s+", lines[0])
    w[0] = w[0][1:]
    dty = []
    for x in w:
        if x != "":
            dty.append((x, np.float))
    d = np.loadtxt(FileName, dtype=dty)
    return d


def GetStrCodeGitCmd(filepath, options, args):
    """
    Get script history and running informations:
    @param filepath is file path : os.path.realpath(__file__)
    @param options is the dictionary of options : vars(options)
    @param args is the list of arguments : args
    @return string with informations: script, git hash, git branch, commandline
    """
    dirCurrent = os.getcwd() + "/"
    dirScript = os.path.dirname(filepath) + "/"
    nameScript = os.path.basename(filepath)
    os.chdir(dirScript)
    tmp = dirCurrent + "/tmp_GetStrCodeGitCmd.txt"
    run(
        "cd " + dirScript + "; git rev-parse HEAD > " + tmp + " ; git branch >> " + tmp,
        disp=False,
        NoExit=True,
    )
    run("cd " + dirCurrent, disp=False, NoExit=False)
    gitHash = "git hash not found"
    gitBranch = "git branch not found"
    try:
        fIn = open(tmp)
        lines = fIn.readlines()
        fIn.close()
        if len(lines) == 0:
            ### For docker
            tmp2 = "/codes/LDC/default_GetStrCodeGitCmd.txt"
            if os.path.isfile(tmp2):
                fIn = open(tmp2)
                lines = fIn.readlines()
                fIn.close()
        if len(lines) != 0:
            gitHash = lines[0][:-1]
            for x in lines:
                if x[0] == "*":
                    gitBranch = re.split(r"\s+", x)[1]
        if os.path.isfile(tmp):
            run("rm " + tmp)
    except:
        print("WARNING in GetStrCodeGitCmd: cannot recover script informations")
    os.chdir(dirCurrent)
    r = f"#{gitHash} ({gitBranch}): python3 {filepath}"
    for i, k in enumerate(options):
        r = r + " --" + str(k) + "="
        if type(options[k]) == str:
            r = r + options[k]
        else:
            r = r + str(options[k])
    for x in args:
        r = r + " " + x
    return r


def AziPolAngleL2PsiIncl(bet, lam, theL, phiL):
    """
    Convert Polar and Azimuthal angles of zS (typically orbital angular momentum L)
    to polarisation and inclination (see doc)
    @param bet is the ecliptic latitude of the source in sky [rad]
    @param lam is the ecliptic longitude of the source in sky [rad]
    @param theL is the polar angle of zS [rad]
    @param phiL is the azimuthal angle of zS [rad]
    @return polarisation and inclination
    """
    # inc = np.arccos( np.cos(theL)*np.sin(bet) + np.cos(bet)*np.sin(theL)*np.cos(lam - phiL) )
    # up_psi = np.cos(theL)*np.cos(bet) - np.sin(bet)*np.sin(theL)*np.cos(lam - phiL)
    # down_psi = np.sin(theL)*np.sin(lam - phiL)
    # psi = np.arctan2(up_psi, down_psi)

    inc = np.arccos(-np.cos(theL) * np.sin(bet) - np.cos(bet) * np.sin(theL) * np.cos(lam - phiL))
    down_psi = np.sin(theL) * np.sin(lam - phiL)
    up_psi = -np.sin(bet) * np.sin(theL) * np.cos(lam - phiL) + np.cos(theL) * np.cos(bet)
    psi = np.arctan2(up_psi, down_psi)

    return psi, inc


def Window(tm):
    """
    Apply window ...
    @param tm is the time array
    @return the window array
    """
    # TODO Need details about the window
    xl = 1000.0
    ind_r = np.argwhere(tm[-1] - tm <= 1000.0)[0][0]
    xr = tm[ind_r]
    kap = 0.005
    winl = 0.5 * (1.0 + np.tanh(kap * (tm - xl)))
    winr = 0.5 * (1.0 - np.tanh(kap * (tm - xr)))
    return winl * winr


def FourierTransformData(x, dt, wis=None):
    """
    Fourier transform data
    @param x is the input data
    @param dt is the step in the reference array
    @return x transform of the data
    """
    # pyfftw.import_wisdom(ws)
    N = len(x)
    yt = pyfftw.empty_aligned(N, dtype="float64")
    yf = pyfftw.empty_aligned(int(N / 2 + 1), dtype="complex128")
    fft_object = pyfftw.FFTW(yt, yf, flags=("FFTW_ESTIMATE",))
    # fft_object = pyfftw.FFTW(yt, yf)
    yt = np.copy(x)
    yf = np.copy(fft_object(yt * dt))
    # wis = pyfftw.export_wisdom()
    # print ("wis = ", wis)
    return yf


def FormatNpArray(x):
    """
    Make sure the result is list format even if it is a single value
    @param input variable or list
    @return list
    """
    a = np.zeros([1, 1])
    if type(x) != type(a):
        x = np.array([x])
    return x
