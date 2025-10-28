/*
 * wmatrix.h
 *
 * Jeff Bilmes  <bilmes@ee.washington.edu>
 * 
 * @(#)$Header: /home/srilm/CVS/srilm/flm/src/wmatrix.h,v 1.7 2006/08/12 06:26:03 stolcke Exp $
 *
 */

#ifndef _WMatrix_h_
#define _WMatrix_h_

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>

#endif
#include <cstring>
#include <stdio.h>

#include <cstring>
#include "Trie.h"
#include <cstring>
#include "Array.h"

#include <cstring>
#include "LMStats.h"
#include <cstring>
#include "FactoredVocab.h"
#include <cstring>
#include "FNgramSpecs.h"

const unsigned maxNumFactors = maxWordsPerLine + maxExtraWordsPerLine;

// matrices for storing factors in parsed string format and wid format
class WidMatrix {
public:
  WidMatrix();
  ~WidMatrix();
  VocabIndex* wid_factors[maxNumFactors];
  VocabIndex* operator[](int i) { return wid_factors[i]; }
};


class WordMatrix {
public:
  WordMatrix();
  ~WordMatrix();
  VocabString* word_factors[maxNumFactors];
  VocabString* operator[](int i) { return word_factors[i]; }
  void print(FILE* f);
};

#endif
