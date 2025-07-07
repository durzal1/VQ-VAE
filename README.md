# Intro 
Complex Computer Vision Project that consisted of training a VQ-VAE GAN from scratch to create AI Generated flowers.

Here is the evolution of the model from various epochs. The image to the top is AI generated and to the bottom is what it should be.

# Explanation of model

1. VQ-VAE learns discrete representations

The Encoder compresses the input image (flower) into a grid of discrete tokens to use in a learned codebook.
These tokens are quantized representations of the input (flower) and describe a piece of what a flower is. These, combined, make up the AI-generated flower. 

Then, the decoder reconstructs the image using these tokens from the codebook.

2. GAN. Enhancing the image.

These generated image are given to the discriminator which decided if the image is real or AI generated.
This adversarial training enhances the quality of the VQ-VAE (generator) to produce sharper images.

3. Combining it together.

These two elements together make up the VQ-VAE GAN. When tried with an adequate dataset with enough time it is capable of recreating those images from scratch. 

4. Unique Images with transformers 

To make unique AI-generated images, we can implement a transformer (pre-trained GP2 model) and fine-tune it with tokens generated using the dataset. The transformer learns how to create a coherent set of quantized tokens to create a beautiful image. 
We randomly set the first token in the sequence and use the transformer to fill in the rest. We then feed this into the VQ-VAE, and the final result is a unique AI-generated flower. 




# Images
Epoch 1
\\
![epoch_1_ours2](https://github.com/user-attachments/assets/1b311e1d-ee8a-4dc6-8e98-c40a36055f9d)
![epoch_1_samples](https://github.com/user-attachments/assets/789505a7-993e-4996-8c67-f658af144527)


Epoch 12

![epoch_12_ours2](https://github.com/user-attachments/assets/9c07ca4f-206e-4767-a31e-1d82a1bc86a6)
![epoch_12_samples](https://github.com/user-attachments/assets/0516752a-cd88-434e-868a-7dced5f69b54)


Epoch 50

![epoch_50_ours2](https://github.com/user-attachments/assets/c122c8dd-0181-48aa-a85b-d18314344a5d)
![epoch_50_samples](https://github.com/user-attachments/assets/e1c5f972-8dd7-4604-b68f-3db594a5012b)


Epoch 100

![epoch_100_ours2](https://github.com/user-attachments/assets/866074b2-c43a-4cb2-8e81-1499fc650808)
![epoch_100_samples](https://github.com/user-attachments/assets/80e778c9-aa7a-43a8-b392-4c420cff5f06)




