# SuperResolution
Course project for 13M054NM

PyTorch implementation of the Super-Resolution approach used in the researh paper [Residual Dense Network for Image Super-Resolution](https://arxiv.org/pdf/1802.08797.pdf)

Resulting images(input-output-target):

<p flaot="left">
<img src="https://user-images.githubusercontent.com/43972534/155841399-1a431999-cc13-422b-ae17-014de4836110.png" width="270" height="270">
<img src="https://user-images.githubusercontent.com/43972534/155841412-8c28a70e-5d38-4fcd-a936-b0bb861801ef.png" width="270" height="270">
<img src="https://user-images.githubusercontent.com/43972534/155841420-5cf61dfb-f359-4b80-94d4-56e8885d7d84.png" width="270" height="270">
</p>

**Further work**:
- **Hyperparameter tuning** - Due to the lack of computing capabilities I was not able to carry out a proper hyperparameter tuning procedure with cross-validation. Most of the values were inherited from the original research paper.
- **Data** - The dataset used in this project is [DIV2K](https://paperswithcode.com/dataset/div2k), adding more data could improve the system performance. Aside from more data, other downsampling methods could be used during training, here I used pixel left-out downsampling which is the simplest approach.
- **Upscale block rework** - This convolutional block (after the global residual connection) can be improved compared to the current solution.
- **Global Contiguous memory** - Since Feature Fusion and Residual Learning are used in both local and global ways, maybe adding a global version of Contiguous memory could enhance the training.
