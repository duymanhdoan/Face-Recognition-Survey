# 		research_cosface paper
objective: maximum discrimination by maximizing inter-class variance and minimizing intra-class variance
 
### Normalization Approaches. 

- Note: that normalization on feature vectors or weight vectors achieves much lower intra-class angular variability by concentrating( tap trung) more on the angle during traning. Hence the angles between identities can be well optimized. 
- l2 normalizing both features and weight vectors. 

### network
Convnet lear a large cosine margin. 

### Loss functions. 
- rethinking the softmax loss from a cosine perspective. 
Fi meaning activation of fully-connected layer  with weight vector Wj & bias Bj. 
- normalization: Bj = 0  ||Wj|| = 1  ||x|| = s
- Because remove variations in radial directions by fixng ||x|| = 2, the resulting model learns features that are separable in the  angular space. 

- thera(i) denote the  angle between the learned feature vector and the weight vector of class C(i). 

- Note: the NSL forces cos(thera1) > cos(thera2) for C1 as similarly for C2
- Emphasizes: NSL only correct classification. 

### LMCL 
- cos(thera1) - m > cos(thera(2) (where m > 0)
![Formally](/image/formully_LMCL.png)

