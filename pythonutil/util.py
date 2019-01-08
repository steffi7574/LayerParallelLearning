#!/usr/bin/env python

import os
import string
import shutil, glob

def make_link(src,dst):
    """ make_link(src,dst)
        makes a relative link
        Inputs:
            src - source file
            dst - destination to place link
    """
    
    assert os.path.exists(src) , 'source file does not exist \n%s' % src
    
    # find real file, incase source itself is a link
    src = os.path.realpath(src) 
    
    # normalize paths
    src = os.path.normpath(src)
    dst = os.path.normpath(dst)        

    # check for self referencing
    if src == dst: return        
    
    # find relative folder path
    srcfolder = os.path.join( os.path.split(src)[0] ) + '/'
    dstfolder = os.path.join( os.path.split(dst)[0] ) + '/'
    srcfolder = os.path.relpath(srcfolder,dstfolder)
    src = os.path.join( srcfolder, os.path.split(src)[1] )
    
    # make unix link
    if os.path.exists(dst): os.remove(dst)
    os.symlink(src,dst)
    

def comparefiles(refname, testname):
    """ comparefiles(refname, testname)
        compares the two files line by line
        ignoring last element in each line
        ignoring lines that start with '#'
        Inputs:
            refname  - reference filename
            testname - filename to compare
        Output:
            boolean indicating equal files (0) or differing files (1)
    """

    # open the files 
    reffile = open(refname, 'r')
    testfile = open(testname, 'r')

    # count the different elements
    fail = 0
    
    # loop over all lines
    for refline in reffile:

        # read the line in the testfile
        testline = testfile.readline()
        if not testline: 
            fail = 1
            break

        # split lines into elements (space delimiter)
        refwords = string.split(refline, ' ')
        testwords = string.split(testline, ' ')
        
        # ignore lines that start with '#'
        if refwords[0] == '#':
            continue
    
        # loop over all but the last elements
        for ref, test in zip(refwords[:-1], testwords[:-1]):
            if ref != test :
                fail = 1
                break

    # close the files
    reffile.close()
    testfile.close()

    return fail 
