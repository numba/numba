from __future__ import print_function, division, absolute_import

class ArrayAnalysis(object):
    '''Analysis of array computations such as shapes and equivalence classes
    '''

    def __init__(self, func_ir, type_annotation):
        '''Constructor for the Rewrite class.
        '''
        self.func_ir = func_ir
        self.type_annotation = type_annotation
        self.next_eq_class = 1
        #print("ARRAY ANALYSIS")

    def run(self):
        #print("RUN ARRAY ANALYSIS")
