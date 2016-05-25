# -----------------------------------------------------------------------------
# Self-organizing map
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------

import numpy as np
import time
from threading import Thread

def fromdistance(fn, shape, center=None, dtype=float):
    def distance(*args):
        d = 0
        for i in range(len(shape)):
            d += ((args[i]-center[i])/float(max(1,shape[i]-1)))**2
        return np.sqrt(d)/np.sqrt(len(shape))
    if center == None:
        center = np.array(list(shape))//2
    return fn(np.fromfunction(distance,shape,dtype=dtype))

def Gaussian(shape,center,sigma=0.5):
    ''' '''
    def g(x):
        return np.exp(-x**2/sigma**2)
    return fromdistance(g,shape,center)

class SOM(Thread):
    ''' Self-organizing map '''
    
    '''
    codebook: a map of the all the weights 
    dim: dimension of the map
    tedious: a boolean value indicating something.. 
    epochs: number of epochs 
    sigma: sigma_i, sigma_f 
    lr: lr_i, lr_f 
    samples: samples 
    features: number of features of the sample
    fires: the winner neuron for each sample, number of times and the class of the data point
    test: the neurons hit during test
    ''' 

    def __init__(self, shape, ted=False):
        Thread.__init__(self)
        ''' Initialize som '''
        self.codebook = np.zeros(shape)
        self.dim = len(shape)-1 
        self.tedious = ted
        #self.events = np.full(shape[0:len(shape)-1], [])
        self.reset()

    def reset(self):
        ''' Reset weights '''
        self.codebook = np.random.random(self.codebook.shape)

    def winner( self, data):
        return self.codebook[self.winner_index( data )]
        
    def winner_index(self, data):
        D = ((self.codebook-data)**2).sum(axis=-1)
        return np.unravel_index(np.argmin(D), D.shape)


        
    def start(self, samples, epochs=25000, sigma=(10, 0.001), lrate=(0.5,0.005)):
        ''' Learn samples '''
        self.sigma_i, self.sigma_f = sigma
        self.lrate_i, self.lrate_f = lrate
        self.epochs = epochs
        self.samples = samples
            
        
    def run(self):        
        if self.tedious:
            self.fires = [None for i in self.samples]
            print len(self.fires)
            
        start = time.mktime(time.localtime())
        pickAtRandom = True
        if self.epochs > len(self.samples):
            print "events will be taken sequentially from samples"
            pickAtRandom = False
            
        sample=-1
        while (sample<self.epochs) or (self.tedious and any(map(lambda e : e==None,self.fires))):
        ##for i in range(self.epochs):
            sample+=1
            if sample%(self.epochs/100)==0 and sample:
                now = time.mktime(time.localtime())
                self.eta = (now - start) / float(sample) * float(self.epochs-sample)
                #print "Epoch",sample,"eta %5.2f [min]"% self.eta

            # Adjust learning rate and neighborhood
            t = sample/float(self.epochs)
            lrate = self.lrate_i*(self.lrate_f/float(self.lrate_i))**t
            sigma = self.sigma_i*(self.sigma_f/float(self.sigma_i))**t

            # Get random sample
            if pickAtRandom:
                index = np.random.randint(0,self.samples.shape[0])
            else:
                index = sample%self.samples.shape[0]
            data = self.samples[index]

            # Get index of nearest node (minimum distance)
            D = ((self.codebook-data)**2).sum(axis=-1)
            winner = np.unravel_index(np.argmin(D), D.shape)

            if self.tedious:
                if winner == self.fires[index]:
                    #this event already fired that neuron : skip things
                    #i-=1
                    self.skipped+=1
                    continue
                else:
                    self.fires[index] = winner
            
            ## has the neuron been hit (=0 -> dead neuron)
            self.hits[winner] +=1
            
            # Generate a Gaussian centered on winner
            G = Gaussian(D.shape, winner, sigma)
            G = np.nan_to_num(G)

            # Move nodes towards sample according to Gaussian 
            delta = self.codebook-data
            for i in range(self.codebook.shape[-1]):
                self.codebook[...,i] -= lrate * G * delta[...,i]

        now = time.mktime(time.localtime())
        self.cook_rate = (now - start) / self.epochs  ## s per epochs
        self.cook_time = (now - start) ## s
   
    def control_train(self, samples, types, epochs= 5, sigma=(10, 0.001), lrate=(0.5,0.005), threshold = 4):
        
        # initialise the paramters 
        self.fires = np.zeros((samples.shape[0], self.dim + 2)) #one for number of times and one for type
        self.sigma_i, self.sigma_f = sigma
        self.lrate_i, self.lrate_f = lrate
        self.epochs = epochs
        self.types = types 
        
        features = self.codebook.shape[-1] # one for type, one for weight
        print 'number of features:', features
                
        for i in range(epochs): 
            t = i/float(epochs)
            sigma = self.sigma_i*(self.sigma_f/float(self.sigma_i))**t
            lrate = self.lrate_i*(self.lrate_f/float(self.lrate_i))**t
            
            start = time.mktime(time.localtime())
            print 'epoch:', i
            print 'learning rate:', lrate
            print 'sigma:', sigma
            
            for index in range(samples.shape[0]):
                
                # Adjust learning rate and neighborhood
                weight = samples[index, -1]
                
                data = samples[index, 0:features]
        
                # Forward pass 
                D = ((self.codebook-data)**2).sum(axis=-1)
                winner = np.unravel_index(np.argmin(D), D.shape)

                if np.array_equal(self.fires[index, 0:self.dim], winner):
                    self.fires[index, self.dim] += 1
                
                else:
                    self.fires[index, 0:self.dim] = winner
                    self.fires[index, self.dim] = 0
                    self.fires[index, self.dim + 1] = samples[index, features]
                 
                
                if self.fires[index, self.dim] < threshold: 
                    
                    # Move nodes towards sample according to Gaussian 
                    delta = self.codebook-data 
                
                    G = Gaussian(D.shape, winner, sigma)
                    G = np.nan_to_num(G)
             
                    for j in range(features):
                        delta[..., j] *=  (1 - lrate * G) ** weight
        
                    self.codebook = data + delta
                            
            
            # One epoch of training has ended 
            end = time.mktime(time.localtime())
            #print 'end time:', end 
            print 'time per epoch:', end - start
            #print 'time per sample:', (end - start)/samples.shape[0] 
            print 'number of stable samples:', np.sum(self.fires[:, self.dim] >= threshold) 
            
    def control_test(self, tst):
        start = time.mktime(time.localtime())
        self.test = np.zeros(self.codebook.shape[0:-1] + tuple([self.types]))
        features = self.codebook.shape[-1]
        
        for index in range(tst.shape[0]):
            data = tst[index, 0:features]
            weight = tst[index, -1]

            # Forward pass 
            D = ((self.codebook-data)**2).sum(axis=-1)
            winner = np.unravel_index(np.argmin(D), D.shape)

            self.test[winner + tuple([tst[index, features]])] += weight
        end = time.mktime(time.localtime())
        print 'total time:', end - start
        print 'time per sample:', (end - start)/tst.shape[0] 
        
    # A plotting method that shows 3d histgrams   
    def show_map(self, threshold = 0):
        bins = self.test
        dim = bins.shape[0]
        types = bins.shape[-1]
        x, y, z = np.indices((dim, dim, dim))
        for t in range(types):     
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, c = bins[..., t], norm = LogNorm(), lw = 0)
            plt.show()
                    
                    
    def best_cell(self): #show the amount of contamination in the least contaminated cell, need to add weights etc
        bkg = np.sum(self.test[..., 1:self.test.shape[-1]], -1)
        bkg[bkg == 0] = 0.01 #replace zero with 0.01, add in uncertainties later
        signif = self.test[..., 0]/(np.sum(self.test[..., 0]) * np.sqrt(bkg))
        best_cell = np.unravel_index(np.argmax(signif), self.test.shape[0:-1])
        return best_cell, self.test[best_cell][0] 
        
    def sig_cells(self): #Output a list of signal cells 
        return np.where(DataMap.test[..., 0]!= 0)