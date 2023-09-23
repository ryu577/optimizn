from optimizn.combinatorial.algorithms.suitcase_reshuffle.suitcases\
    import SuitCases
from optimizn.combinatorial.algorithms.suitcase_reshuffle\
    .sim_anneal_suitcase_reshuffle import SuitCaseReshuffle


def test_1():
	config = [[7,5,1],[4,6,1]]
	sc = SuitCases(config)
	scr = SuitCaseReshuffle(params=sc)

def test_2():
	# from optimizn.combinatorial.tst_anneal.suitcase_reshuffle import *

	config = [[7,5,1],[4,6,1]]
	sc = SuitCases(config)
	scr = SuitCaseReshuffle(params=sc)
	candidate = scr.get_candidate()
	scr.anneal()