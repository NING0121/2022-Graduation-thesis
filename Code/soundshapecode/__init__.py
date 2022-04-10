import sys
sys.path.append('../')

from soundshapecode.ssc import getHanziStrokesDict, getHanziStructureDict, getHanziSSCDict, getSSC, getSSC_char
from soundshapecode.compute_ssc_similarity import computeSSCSimilaruty


SIMILARITY_THRESHOLD = 0.8
SSC_ENCODE_WAY = 'ALL'#'ALL','SOUND','SHAPE'