import numpy as np


def wrap_list(lst):
    """
        Wrap an element in a list or return the given list

        usefull for features and targets:
            lst = [A, B]
            ( A, B === ndarray )
            ==> return [A, B]

            lst = A
            ( A === ndarray )
            ==> return [A]
    """
    if not isinstance(lst, list):
        return [lst]
    return lst


def cwrap_list(lst):
    """
        Wrap an element in a list or return the given list
        and copy all contained array ( ndarray compatible )

        usefull for features and targets:
            lst = [A, B]
            ( A, B === ndarray )
            ==> return [np.array(A), np.array(B)]

            lst = A
            ( A === ndarray )
            ==> return [np.array(A)]
    """
    if not isinstance(lst, list):
        lst = [lst]
    return [np.array(el) for el in lst]


def wrap_tlist(tlst):
    """
        tlst is a tuple of list or element : ([,,], [,])
        for each element/list in the uple:
            Wrap the element in a list if element else continue
    """
    tlst = list(tlst)
    for i in range(len(tlst)):
        tlst[i] = wrap_list(tlst[i])
    return tuple(tlst)


def cwrap_tlist(tlst):
    """
        tlst is a tuple of list or element : ([,,], [,])
        for each element/list in the uple:
            Wrap the element in a list if element else continue
        and copy all contained array ( ndarray compatible )
    """
    tlst = list(tlst)
    for i in range(len(tlst)):
        tlst[i] = cwrap_list(tlst[i])
    return tuple(tlst)
