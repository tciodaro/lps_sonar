
import numpy as np

"""
    Good for nbox >> len(indexes). If not, last box might have a length very different
    from other boxes.
"""
def simple_cross_validation(indexes, trnPerc, valPerc, nbox, nsel):
    indexes = np.array(indexes, 'i')
    ## Create boxes
    box_cnt = int(indexes.shape[0] / float(nbox))
    np.random.shuffle(indexes)
    boxes = {}
    ibef = 0
    for ibox in range(nbox - 1):
        boxes[ibox] = indexes[ibef:(ibox+1)*box_cnt].tolist()
        ibef = (ibox+1)*box_cnt
    boxes[nbox-1] = indexes[ibef:].tolist()
    ## Make random selections
    iboxes = np.array(range(nbox), 'i')
    cvidx = {}
    for isel in range(nsel):
        cvidx[isel] = [None, None, None]
        np.random.shuffle(iboxes)
        ntrn = int(nbox * trnPerc + 0.5)
        nval = int(nbox * valPerc + 0.5)
        itrn = []
        for i in range(0, ntrn):
            itrn += boxes[iboxes[i]]
        ival = []
        for i in range(ntrn, ntrn+nval):
            ival += boxes[iboxes[i]]
        itst = []
        for i in range(ntrn+nval, nbox):
            itst += boxes[iboxes[i]]
        # Final setup
        cvidx[isel][0] = np.array(itrn)
        cvidx[isel][1] = np.array(ival)
        cvidx[isel][2] = np.array(itst)
    return cvidx
# End of file


