# Natural Language Toolkit: Utility functions
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

import sys
import inspect
import locale
import re
import types
import textwrap
import pydoc
import bisect
import os

from itertools import islice, chain, combinations, tee
from pprint import pprint
from collections import defaultdict, deque
from sys import version_info

from urllib.request import (
    build_opener,
    install_opener,
    getproxies,
    ProxyHandler,
    ProxyBasicAuthHandler,
    ProxyDigestAuthHandler,
    HTTPPasswordMgrWithDefaultRealm,
)

##########################################################################
# Breadth-First Search
##########################################################################


def breadth_first(tree, children=iter, maxdepth=-1):
    """Traverse the nodes of a tree in breadth-first order.
    (No check for cycles.)
    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.
    """
    queue = deque([(tree, 0)])

    while queue:
        node, depth = queue.popleft()
        yield node

        if depth != maxdepth:
            try:
                queue.extend((c, depth + 1) for c in children(node))
            except TypeError:
                pass


##########################################################################
# Breadth-First / Depth-first Searches with Cycle Detection
##########################################################################

import warnings

def acyclic_breadth_first(tree, children=iter, maxdepth=-1):
    """Traverse the nodes of a tree in breadth-first order,
    discarding eventual cycles.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.
    """
    traversed = set()
    queue = deque([(tree, 0)])
    while queue:
        node, depth = queue.popleft()
        yield node
        traversed.add(node)
        if depth != maxdepth:
            try:
                for child in children(node):
                    if child not in traversed:
                        queue.append((child, depth + 1))
                    else:
                        warnings.warn('Discarded redundant search for {0} at depth {1}'.format(child, depth + 1), stacklevel=2)
            except TypeError:
                pass


def acyclic_depth_first(tree, children=iter, depth=-1, cut_mark=None, traversed=None):
    """Traverse the nodes of a tree in depth-first order,
    discarding eventual cycles within any branch,
    adding cut_mark (when specified) if cycles were truncated.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    Catches all cycles:

    >>> import nltk
    >>> from nltk.util import acyclic_depth_first as acyclic_tree
    >>> wn=nltk.corpus.wordnet
    >>> from pprint import pprint
    >>> pprint(acyclic_tree(wn.synset('dog.n.01'), lambda s:s.hypernyms(),cut_mark='...'))
    [Synset('dog.n.01'),
     [Synset('canine.n.02'),
      [Synset('carnivore.n.01'),
       [Synset('placental.n.01'),
        [Synset('mammal.n.01'),
         [Synset('vertebrate.n.01'),
          [Synset('chordate.n.01'),
           [Synset('animal.n.01'),
            [Synset('organism.n.01'),
             [Synset('living_thing.n.01'),
              [Synset('whole.n.02'),
               [Synset('object.n.01'),
                [Synset('physical_entity.n.01'),
                 [Synset('entity.n.01')]]]]]]]]]]]]],
     [Synset('domestic_animal.n.01'), "Cycle(Synset('animal.n.01'),-3,...)"]]
    """
    if traversed is None:
        traversed = {tree}
    out_tree = [tree]
    if depth != 0:
        try:
            for child in children(tree):
                if child not in traversed:
#                   Recurse with a common "traversed" set for all children:
                    traversed.add(child)
                    out_tree += [acyclic_depth_first(child, children, depth - 1, cut_mark, traversed)]
                else:
                    warnings.warn('Discarded redundant search for {0} at depth {1}'.format(child, depth - 1), stacklevel=3)
                    if cut_mark:
                        out_tree += ['Cycle({0},{1},{2})'.format(child, depth - 1, cut_mark)]
        except TypeError:
            pass
    elif cut_mark:
        out_tree += [cut_mark]
    return out_tree


def acyclic_branches_depth_first(tree, children=iter, depth=-1, cut_mark=None, traversed=None):
    """Traverse the nodes of a tree in depth-first order,
    discarding eventual cycles within the same branch,
    but keep duplicate pathes in different branches.
    Add cut_mark (when defined) if cycles were truncated.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    Catches only only cycles within the same branch,
    but keeping cycles from different branches:

    >>> import nltk
    >>> from nltk.util import acyclic_branches_depth_first as tree
    >>> wn=nltk.corpus.wordnet
    >>> from pprint import pprint
    >>> pprint(tree(wn.synset('certified.a.01'), lambda s:s.also_sees(), cut_mark='...', depth=4))
    [Synset('certified.a.01'),
     [Synset('authorized.a.01'),
      [Synset('lawful.a.01'),
       [Synset('legal.a.01'),
        "Cycle(Synset('lawful.a.01'),0,...)",
        [Synset('legitimate.a.01'), '...']],
       [Synset('straight.a.06'),
        [Synset('honest.a.01'), '...'],
        "Cycle(Synset('lawful.a.01'),0,...)"]],
      [Synset('legitimate.a.01'),
       "Cycle(Synset('authorized.a.01'),1,...)",
       [Synset('legal.a.01'),
        [Synset('lawful.a.01'), '...'],
        "Cycle(Synset('legitimate.a.01'),0,...)"],
       [Synset('valid.a.01'),
        "Cycle(Synset('legitimate.a.01'),0,...)",
        [Synset('reasonable.a.01'), '...']]],
      [Synset('official.a.01'), "Cycle(Synset('authorized.a.01'),1,...)"]],
     [Synset('documented.a.01')]]
    """
    if traversed is None:
        traversed = {tree}
    out_tree = [tree]
    if depth != 0:
        try:
            for child in children(tree):
                if child not in traversed:
#                   Recurse with a different "traversed" set for each child:
                    out_tree += [acyclic_branches_depth_first(child, children, depth - 1, cut_mark, traversed.union({child}))]
                else:
                    warnings.warn('Discarded redundant search for {0} at depth {1}'.format(child, depth - 1), stacklevel=3)
                    if cut_mark:
                        out_tree += ['Cycle({0},{1},{2})'.format(child, depth - 1, cut_mark)]
        except TypeError:
            pass
    elif cut_mark:
        out_tree += [cut_mark]
    return out_tree


##########################################################################
# Invert a dictionary
##########################################################################


def invert_dict(d):
    inverted_dict = defaultdict(list)
    for key in d:
        if hasattr(d[key], "__iter__"):
            for term in d[key]:
                inverted_dict[term].append(key)
        else:
            inverted_dict[d[key]] = key
    return inverted_dict


##########################################################################
# Utilities for directed graphs: transitive closure, and inversion
# The graph is represented as a dictionary of sets
##########################################################################


def transitive_closure(graph, reflexive=False):
    """
    Calculate the transitive closure of a directed graph,
    optionally the reflexive transitive closure.

    The algorithm is a slight modification of the "Marking Algorithm" of
    Ioannidis & Ramakrishnan (1998) "Efficient Transitive Closure Algorithms".

    :param graph: the initial graph, represented as a dictionary of sets
    :type graph: dict(set)
    :param reflexive: if set, also make the closure reflexive
    :type reflexive: bool
    :rtype: dict(set)
    """
    if reflexive:
        base_set = lambda k: set([k])
    else:
        base_set = lambda k: set()
    # The graph U_i in the article:
    agenda_graph = dict((k, graph[k].copy()) for k in graph)
    # The graph M_i in the article:
    closure_graph = dict((k, base_set(k)) for k in graph)
    for i in graph:
        agenda = agenda_graph[i]
        closure = closure_graph[i]
        while agenda:
            j = agenda.pop()
            closure.add(j)
            closure |= closure_graph.setdefault(j, base_set(j))
            agenda |= agenda_graph.get(j, base_set(j))
            agenda -= closure
    return closure_graph


def invert_graph(graph):
    """
    Inverts a directed graph.

    :param graph: the graph, represented as a dictionary of sets
    :type graph: dict(set)
    :return: the inverted graph
    :rtype: dict(set)
    """
    inverted = {}
    for key in graph:
        for value in graph[key]:
            inverted.setdefault(value, set()).add(key)
    return inverted



##########################################################################
# FLATTEN LISTS
##########################################################################


def flatten(*args):
    """
    Flatten a list.

        >>> from nltk.util import flatten
        >>> flatten(1, 2, ['b', 'a' , ['c', 'd']], 3)
        [1, 2, 'b', 'a', 'c', 'd', 3]

    :param args: items and lists to be combined into a single list
    :rtype: list
    """

    x = []
    for l in args:
        if not isinstance(l, (list, tuple)):
            l = [l]
        for item in l:
            if isinstance(item, (list, tuple)):
                x.extend(flatten(item))
            else:
                x.append(item)
    return x


##########################################################################
# Ngram iteration
##########################################################################


def pad_sequence(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    """
    Returns a padded sequence of items before ngram extraction.

        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']

    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


# add a flag to pad the sequence so we get peripheral ngrams?


def ngrams(sequence, n, **kwargs):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:

        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]


    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = pad_sequence(sequence, n, **kwargs)
    
    # Creates the sliding window, of n no. of items.
    # `iterables` is a tuple of iterables where each iterable is a window of n items.
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables): # For each window,
        for _ in range(i):                       # iterate through every order of ngrams
            next(sub_iterable, None)             # generate the ngrams within the window.
    return zip(*iterables) # Unpack and flattens the iterables.


def bigrams(sequence, **kwargs):
    """
    Return the bigrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import bigrams
        >>> list(bigrams([1,2,3,4,5]))
        [(1, 2), (2, 3), (3, 4), (4, 5)]

    Use bigrams for a list version of this function.

    :param sequence: the source data to be converted into bigrams
    :type sequence: sequence or iter
    :rtype: iter(tuple)
    """

    for item in ngrams(sequence, 2, **kwargs):
        yield item


def trigrams(sequence, **kwargs):
    """
    Return the trigrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import trigrams
        >>> list(trigrams([1,2,3,4,5]))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Use trigrams for a list version of this function.

    :param sequence: the source data to be converted into trigrams
    :type sequence: sequence or iter
    :rtype: iter(tuple)
    """

    for item in ngrams(sequence, 3, **kwargs):
        yield item


def everygrams(sequence, min_len=1, max_len=-1, pad_left=False, pad_right=False, **kwargs):
    """
    Returns all possible ngrams generated from a sequence of items, as an iterator.

        >>> sent = 'a b c'.split()

    New version outputs for everygrams.
        >>> list(everygrams(sent))
        [('a',), ('a', 'b'), ('a', 'b', 'c'), ('b',), ('b', 'c'), ('c',)]

    Old version outputs for everygrams.
        >>> sorted(everygrams(sent), key=len)
        [('a',), ('b',), ('c',), ('a', 'b'), ('b', 'c'), ('a', 'b', 'c')]

        >>> list(everygrams(sent, max_len=2))
        [('a',), ('a', 'b'), ('b',), ('b', 'c'), ('c',)]

    :param sequence: the source data to be converted into ngrams. If max_len is
        not provided, this sequence will be loaded into memory
    :type sequence: sequence or iter
    :param min_len: minimum length of the ngrams, aka. n-gram order/degree of ngram
    :type  min_len: int
    :param max_len: maximum length of the ngrams (set to length of sequence by default)
    :type  max_len: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :rtype: iter(tuple)
    """

    # Get max_len for padding.
    if max_len == -1:
        try:
            max_len = len(sequence)
        except TypeError:
            sequence = list(sequence)
            max_len = len(sequence)

    # Pad if indicated using max_len.
    sequence = pad_sequence(sequence, max_len, pad_left, pad_right, **kwargs)

    # Sliding window to store grams.
    history = list(islice(sequence, max_len))

    # Yield ngrams from sequence.
    while history:
        for ngram_len in range(min_len, len(history)+1):
            yield tuple(history[:ngram_len])

        # Append element to history if sequence has more items.
        try:
            history.append(next(sequence))
        except StopIteration:
            pass

        del history[0]



def skipgrams(sequence, n, k, **kwargs):
    """
    Returns all possible skipgrams generated from a sequence of items, as an iterator.
    Skipgrams are ngrams that allows tokens to be skipped.
    Refer to http://homepages.inf.ed.ac.uk/ballison/pdf/lrec_skipgrams.pdf

        >>> sent = "Insurgents killed in ongoing fighting".split()
        >>> list(skipgrams(sent, 2, 2))
        [('Insurgents', 'killed'), ('Insurgents', 'in'), ('Insurgents', 'ongoing'), ('killed', 'in'), ('killed', 'ongoing'), ('killed', 'fighting'), ('in', 'ongoing'), ('in', 'fighting'), ('ongoing', 'fighting')]
        >>> list(skipgrams(sent, 3, 2))
        [('Insurgents', 'killed', 'in'), ('Insurgents', 'killed', 'ongoing'), ('Insurgents', 'killed', 'fighting'), ('Insurgents', 'in', 'ongoing'), ('Insurgents', 'in', 'fighting'), ('Insurgents', 'ongoing', 'fighting'), ('killed', 'in', 'ongoing'), ('killed', 'in', 'fighting'), ('killed', 'ongoing', 'fighting'), ('in', 'ongoing', 'fighting')]

    :param sequence: the source data to be converted into trigrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param k: the skip distance
    :type  k: int
    :rtype: iter(tuple)
    """

    # Pads the sequence as desired by **kwargs.
    if "pad_left" in kwargs or "pad_right" in kwargs:
        sequence = pad_sequence(sequence, n, **kwargs)

    # Note when iterating through the ngrams, the pad_right here is not
    # the **kwargs padding, it's for the algorithm to detect the SENTINEL
    # object on the right pad to stop inner loop.
    SENTINEL = object()
    for ngram in ngrams(sequence, n + k, pad_right=True, right_pad_symbol=SENTINEL):
        head = ngram[:1]
        tail = ngram[1:]
        for skip_tail in combinations(tail, n - 1):
            if skip_tail[-1] is SENTINEL:
                continue
            yield head + skip_tail


######################################################################
# Binary Search in a File
######################################################################

# inherited from pywordnet, by Oliver Steele
def binary_search_file(file, key, cache={}, cacheDepth=-1):
    """
    Return the line from the file with first word key.
    Searches through a sorted file using the binary search algorithm.

    :type file: file
    :param file: the file to be searched through.
    :type key: str
    :param key: the identifier we are searching for.
    """

    key = key + " "
    keylen = len(key)
    start = 0
    currentDepth = 0

    if hasattr(file, "name"):
        end = os.stat(file.name).st_size - 1
    else:
        file.seek(0, 2)
        end = file.tell() - 1
        file.seek(0)

    while start < end:
        lastState = start, end
        middle = (start + end) // 2

        if cache.get(middle):
            offset, line = cache[middle]

        else:
            line = ""
            while True:
                file.seek(max(0, middle - 1))
                if middle > 0:
                    file.discard_line()
                offset = file.tell()
                line = file.readline()
                if line != "":
                    break
                # at EOF; try to find start of the last line
                middle = (start + middle) // 2
                if middle == end - 1:
                    return None
            if currentDepth < cacheDepth:
                cache[middle] = (offset, line)

        if offset > end:
            assert end != middle - 1, "infinite loop"
            end = middle - 1
        elif line[:keylen] == key:
            return line
        elif line > key:
            assert end != middle - 1, "infinite loop"
            end = middle - 1
        elif line < key:
            start = offset + len(line) - 1

        currentDepth += 1
        thisState = start, end

        if lastState == thisState:
            # Detects the condition where we're searching past the end
            # of the file, which is otherwise difficult to detect
            return None

    return None



######################################################################
# ElementTree pretty printing from http://www.effbot.org/zone/element-lib.htm
######################################################################


def elementtree_indent(elem, level=0):
    """
    Recursive function to indent an ElementTree._ElementInterface
    used for pretty printing. Run indent on elem and then output
    in the normal way.

    :param elem: element to be indented. will be modified.
    :type elem: ElementTree._ElementInterface
    :param level: level of indentation for this element
    :type level: nonnegative integer
    :rtype:   ElementTree._ElementInterface
    :return:  Contents of elem indented to reflect its structure
    """

    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for elem in elem:
            elementtree_indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


######################################################################
# Mathematical approximations
######################################################################


def choose(n, k):
    """
    This function is a fast way to calculate binomial coefficients, commonly
    known as nCk, i.e. the number of combinations of n things taken k at a time.
    (https://en.wikipedia.org/wiki/Binomial_coefficient).

    This is the *scipy.special.comb()* with long integer computation but this
    approximation is faster, see https://github.com/nltk/nltk/issues/1181

        >>> choose(4, 2)
        6
        >>> choose(6, 2)
        15

    :param n: The number of things.
    :type n: int
    :param r: The number of times a thing is taken.
    :type r: int
    """
    if 0 <= k <= n:
        ntok, ktok = 1, 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


