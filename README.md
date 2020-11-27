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

RESENET 64 layer.

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

# 				SPHEREFACE PAPER.

objective: smaller maximal intra-class distance than minimal inter-class, base in softmax fomular with the angular softmax (A-softmax).

### normalization Approaches
- weight norm become 1, and bias become 0.
![formular](/image/nor_softmax_sphereface.png)

### A-Softmax
![modified A-softmax](/image/A-softmax_modified.png)

- Then we can get the decision boundary:
![image](/image/decision_boundaries.png)
- reoresents a geometric point of view from A-softmax loss;
![formular](/image/geometric_sphereface.png)
### explain the nature of A-softmax loss

- The larger the angle of this interval, the smaller the size of the corresponding region manifold, which makes the training task more difficult.
- This property is fairly easy to understand, as shown in Figure 1: The angle of this interval is (m−1)θ1, so the larger the m, the smaller the angle of the interval; Mθ1<π, when the m is larger, the corresponding region manifold θ1
![formular](/image/manifold.png)

### comparision A-softmax,

- L-Softmax loss [16] also implicitly involves the concept of angles. As a regularization method, it shows great improvement on closed-set classification problems. Differently, A-Softmax loss is developed to learn discriminative face embedding. The explicit connections to hypersphere manifold makes our learned features particularly suitable for open-set FR problem, as verified by our experiments. In addition, the angular margin in A-Softmax loss is explicitly imposed and can be quantitatively controlled (e.g. lower bounds to approximate desired feature criterion), while can only be analyzed qualitatively
### The difference with L-Softmax
The biggest difference between A-Softmax and L-Softmax is that the weight of A-Softmax is normalized, while L-Softmax does not. The normalization of the weights of A-Softmax causes the points on the feature to be mapped to the unit hypersphere, while L-Softmax does not have this limitation. This feature makes the geometric interpretation of the two different. As shown in Figure 10, if the features of two categories are input in the same area during training, as shown in Figure 10 below. A-Softmax can only classify these two categories from an angle, that is to say, it only classifies from the direction, and the result of the classification is shown in Figure 11; while L-Softmax can not only distinguish the two categories from the angle, The two classes can also be distinguished from the weight modulus (length), and the classification result is shown in Figure 12. Under the condition of a fixed data set size, L-Softmax can be classified in two ways. Training may not make it separate in both the angle and length directions, resulting in its accuracy may not be as good as A-Softmax.
![figure 10](/image/image_10.png)
![figure 11](/image/image_11.png)
## experiments

![figure 5](/image/experiments.png)
- compared accuracy between A-softmax and softmax
![image](/image/comapre_LFW_YTF.png)


#                     ARCFACE
