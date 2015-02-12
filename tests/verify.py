#!/usr/bin/env python3
import argparse as ap
import os
import sys
import re

# all the warnings GKLEE can emit as well as the program's name
# dict(string -> tuple of regular expressions)
GKLEE = { "prog" : "GKLEE",

          "volatile" : (
        re.compile("These two threads access common memory location, it is "+
                   "better to set shared variables as volatile!"), ),

          "read write" : (
        re.compile("Across different warps, threads \\d+ and \\d+ incur a "+
                   "Read-Write race \(Actual\) on "),
        re.compile("Under the pure canonical schedule, thread \\d+ and \\d+ "+
                   "incur a Write-Read race \(Actual\) on "),
        re.compile("Within a warp, because of branch divergence, threads "+
                   "\\d+ and \\d+ incur a Read-Write race \(Actual\) on"),
        re.compile("One thread at BI \\d+ of Block \\d+ incurs a read-write "+
                   "race with the thread at BI \\d+ of Block \\d+"),
        re.compile("One thread at BI \\d+ of Block \\d+ incurs a write-read "+
                   "race with the thread at BI \\d+ of Block \\d+"),
        re.compile("Under pure canonical schedule, a read-write race is "
                   "found from BI \\d+ of the block \\d+") ),

          "benign read write" : (
        re.compile("incur a Write-Read race with the "+
                   "same value \(Benign\) on"), ), 

          "write write" : (
        re.compile("Within a warp, threads \\d+ and \\d+  incur a "+
                   "Write-Write race \(Actual\) on "),
        re.compile("Across different warps, threads \\d+ and \\d+ incur a "+
                   "Write-Write race \(Actual\) on "),
        re.compile("Under the pure canonical schedule, within a block, "+
                   "thread \\d+ and \\d+ incur a Write-Write race \(Actual\) "+
                   "on "),
        re.compile("Within a warp, because of branch divergence, threads "+
                   "\\d+ and \\d+ incur a Write-Write race \(Actual\) on "),
        re.compile("Under pure canonical schedule, a write-write race is "+
                   "found from BI \\d+ of the block \\d+"),
        re.compile("One thread at BI \\d+ of Block \\d+ incurs a "+
                   "write-write race with the thread at BI \\d+ of Block "+
                   "\\d+") ),
          
          "benign write write" : (
        re.compile("Within a warp, threads \\d+ and \\d+  incur a "+
                   "Write-Write race with the same value \(Benign\) on "),
        re.compile("Across different warps, threads \\d+ and \\d+ incur a "+
                   "Write-Write race with same value\(Benign\) on "),
        re.compile("Across different warps, threads \\d+ and \\d+ incur a "+
                   "Write-Write race with same value \(Benign\) on "),
        re.compile("Under the pure canonical schedule, within a block, "+
                   "thread \\d+ and \\d+ incur a Write-Write race with the "+
                   "same value \(Benign\) on "),
        re.compile("Under the pure canonical schedule, across different "+
                   "blocks, thread \\d+ and \\d+ incur a Write-Write race "+
                   "with the same value \(Benign\) on "),
        re.compile("Within a warp, because of branch divergence, threads "+
                   "\\d+ and \\d+ incur a Write-Write race with the same "+
                   "value \(Benign\) on "),
        re.compile("incur a Write-Write race with the same value \(Benign\)"), ), 
          
          "barrier" : (
        re.compile("violating the property that barriers have to be "+
                   "textually aligned"), 
        re.compile("execution halts on a barrier mismatch") ),

          "bounds" : (
        re.compile("memory error: out of bound pointer"), ),
          }


# all the warnings GKLEEp can emit as well as the program's name
# dict string -> tuple of regualr expressions
GKLEEp = { "prog" : "GKLEEp",

           "volatile" : (
        re.compile("so 'volatile' qualifier required"), ),

           "read write" : (
        re.compile("incur the \(Actual\) read-write race"), ),

           "benign read write" : (
        re.compile("incur the \(Benign\) write-write race"), ),
           
           "write write" : (
        re.compile("incur the \(Actual\) write-write race"), ),

           "benign write write" : (), # no benign write write from GKLEEp ?

           "barrier" : (
        re.compile(" encounter different barrier sequences"),
        re.compile("violating the property that barriers have to be "+
                   "textually aligned"), 
        re.compile("execution halts on a barrier mismatch") ),

           "bounds" : (
        re.compile("memory error: out of bound pointer"), ),
           }


def read_expected(directory, gklee_tanslator):
    """ 
    Reads the expected.txt file and extracts the warnings that GKLEE and
    co should produce. Error is produced on unknown input

    * directory : either a string representation or an os.directory object
    which is the location of the text file

    * gklee_dict : a dict from warning name to tuple of regualr
    expressions for the warning

    returns : a list of tuples containing the warning name to tuples of the
    regualr expressions
    """
    with open(directory+"expected.txt") as e:
        lines = [line.strip() for line in e if line.strip() != ""]
    return [(line, gklee_dict[line]) for line in lines if line != "prog"]


def generate_not_expected(expected, gklee_dict):
    """
    Generates the inverse of the expected list

    * expected : a list of tuples which contain a warning name and a tuple of
    regular expressions for the warnings

    * gklee_dict : a dict from warning name to tuple of regualr
    expressions for the warning

    returns : a list of tuples containing the warning name to tuples of the
    regualr expressions
    """
    full_dict = gklee_dict.copy()
    full_set = set(full_dict.keys())
    not_expected = full_set - {e[0] for e in expected}
    return [(ne, gklee_dict[ne]) for ne in not_expected if ne != "prog"]


def read_actual(directory, gklee_dict):
    """
    Reads in the log file for the given Gklee* implementation

    * directory : either a string representation or an os.directory object
    which is the location of the text file

    * gklee_dict : a dict from warning name to tuple of regualr
    expressions for the warning

    returns : the text of the file
    """
    with open(directory+gklee_dict["prog"].lower()+"_log.txt") as a:
        text = a.read()
    return text


CORE_DUMP_RE = re.compile("Segmentation fault")
def verify(expected, actual, gklee_dict):
    """
    Verifies that the output of Gklee* matches the expected output

    * expected : a list of tuples which contain a warning name and a tuple of
    regular expressions for the warnings

    * actual : the plaintext of Gklee*'s output

    returns : nothing
    """
    passed = True
    if CORE_DUMP_RE.search(actual) != None:
        print("FATAL ERROR IN {} CORE DUMPED".format(gklee_dict["prog"]))

    for e in expected:
        # True means regex not found
        # False means regex found
        matches = map(lambda reg: reg.search(actual) == None, e[1])
        if False not in matches: # if no regex mached
            passed = False
            print("Omission by {} : expected {}".format(gklee_dict["prog"], 
                                                        e[0]))

    not_expected = generate_not_expected(expected, gklee_dict)
    for ne in not_expected:
        matches = map(lambda reg: reg.search(actual) == None, ne[1])
        if False in matches: # if one ore more regex matched 
            passed = False
            print("False alarm by {} : recieved {}".format(gklee_dict["prog"], 
                                                           ne[0]))

    if passed:
        print("Test passed by {}".format(gklee_dict["prog"]))

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Verifies if the output of GKLEE "+
                               "and GKLEEp match the expected values")
    parser.add_argument("directory", 
                        nargs='?', 
                        help="the directory where the test was ran,"+
                        " defaults to current",
                        default="./")
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print("ERROR: {} is not a directory".format(args.directory))
        sys.exit(-1)
        
    for gklee_dict in [GKLEE, GKLEEp]:
        expected = read_expected(args.directory, gklee_dict)
        actual = read_actual(args.directory, gklee_dict)
        verify(expected, actual, gklee_dict)
