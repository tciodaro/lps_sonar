import numpy as np
from scipy import linalg
from scipy import signal


def tpsw(x,npts = None,n = None,p = None,a = None):
    """
    [mx] = tpsw(x,npts = None,n = None,p = None,a = None)
    calcula a media local para o vetor x usando o algoritmo two-pass split window

    onde:
            x	= vetor de entrada
            npts= numero de pontos em x a ser usado
    		n	= tamanho da janela lateral
    		p	= tamanho do gap lateral
    		a	= constante para calculo do limiar

    		mx	= vetor de saida com a media local
    """
    x = np.array(x)
    if npts is None: npts = x.shape[0]
    x = x [:npts,:]
    if n is None: n = np.round(npts * 0.04 / 2 + 1)
    if p is None: p = np.round(n/8.0 + 1.0)
    if a is None: a = 2.0

    # Calcula media local usando janela com gap central
    #
    h = []
    if p > 0:
        h=np.concatenate((np.ones((1,n-p+1)), np.zeros((1,2*p-1)), np.ones((1,n-p+1))), axis=1)[0]
    else:
        h=np.ones((1,2*n+1))
        p=1
    h = np.array([h/linalg.norm(h,1)])

    mx = signal.fftconvolve(h.transpose(), x)


    ix = int(np.fix((h.shape[1])/2.0))
    mx = mx[ix:npts+ix,:]


    # Corrige pontos extremos do espectro
    ixp = ix - p + 1;
    mult = np.array([np.concatenate((np.ones(p-1)*ixp, np.arange(ixp, 2*ixp)))]).transpose() / (2*ixp)
    mx[:ix,:] = mx[ix,:] * (mult * np.ones(x.shape[1]));# Pontos iniciais
    mx[npts-ix:npts,:] = mx[npts-ix:npts,:] * (np.flipud(mult)*np.ones(x.shape[1]));   # Pontos finais

    # Elimina picos para a segunda etapa da filtragem

    ind1 = np.nonzero((x-a*mx)>0);                          # Pontos maiores que a*mx
    x[ind1] = mx[ind1];                               # Substitui pontos por mx
    mx = signal.fftconvolve(h.transpose(),x)
    mx=mx[ix:npts+ix,:];                          # Corrige defasagem

    # Corrige pontos extremos do espectro

    mx[:ix,:] = mx[:ix,:] * (mult * np.ones(x.shape[1]))
    mx[npts-ix:npts,:] = mx[npts-ix:npts,:] * (np.flipud(mult)*np.ones(x.shape[1]))

    return mx
