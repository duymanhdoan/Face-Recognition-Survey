# 		COSFACE PAPER
objective: maximum discrimination by maximizing inter-class variance and minimizing intra-class variance
 
### Normalization Approaches. 

- Note: that normalization on feature vectors or weight vectors achieves much lower intra-class angular variability by concentrating( tap trung) more on the angle during traning. Hence the angles between identities can be well optimized. 
- l2 normalizing both features and weight vectors. 
- minimize the overall loss. 
- method requires that the entire feature vector set has same L2 norm. so that learning only depends on the cosine value to develop the recognitions ability. 
- feature vectors of the same class are grouped torgether and feature vectors of different classes are pulled apart on the hypersphere. 


### Network
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
- with cos(theraj ,i) = Wj * xi. 
- ![Formally](/image/formully_LMCL.png)
### Comparison on different loss function 
- SOFTMAX. 
![softmax](/image/softmax_cosface.png) 
- NSL 
![NSL](/image/NSL_cosface.png)
- A_softmax 
![A- softmax](/image/A_softmax_cosface.png)
- LMCL
![LMCL DEFINE] (/image/cosin_space.png)
- comparison of loss functions. 
![4 loss functions](/image/comparison_of_different_lf.png)
### Visualization geometrical 
- ![geometrical interpretation](/image/geometrical_cosface.png)

### Train model 

DATA = CASIA-0.49M with 10,575 subjects. 
image_size = [112,112]  , (image RGB in [0,255] - 127,5) / 128
data augmentation [ horizontally flipped ]. 
network: resenet 50 ( 64-layer CNN ). 
s = 64 , margin_m = 0.35

### Testing data set 

DATA = lfw and ytf 13,233 image with 5749 identities. (99.73% on LFW and 97.6% in YTF)

analyst: 
## Motivation 
- same idea of loss recognitions
: Maximizing the distance between classes and minimizing the distance between classes. 
- compared with the euclidean margin and angular margin. angular margin has an inherent consistency with the softmax. 
- A-softmax depends on thera, which results in differnet margins for different categories. 
## Work 
- By maximizing the distance between classes and minimizing the distance within classes, the LMCL loss functions is proposed to learn high discriminative depth features for face recognition. 
- hyperspherical feature distribution based on LMCL . Provide reasonable theoretical analysis. 
- improved performance on LFW, YTF, Megaface test get. 



experment: 1. 

train model cosface: [margin_m = 0.32, margin_s = 64.0, lr = 0.1, validation_split = 0.3, loss = cosface , batch_size = 1 28, net =  IR_50 (50 layer)] 

hours of training = 18h. acc = 0.67, loss <7 && > 3. 
### Code 
```class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self。.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'                   
```

# 				ARCFACE PAPER. 









































